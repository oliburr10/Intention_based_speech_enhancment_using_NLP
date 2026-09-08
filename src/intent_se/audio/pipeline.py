from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from intent_se.audio.imcra import IMCRA
from intent_se.audio.stft import SlidingSTFT
from intent_se.audio.wiener import ParametricWienerFilter
from intent_se.config import AudioConfig, DeviceConfig, IMCRAConfig, WienerConfig

__all__ = ["SpeechEnhancer", "EnhancerStats"]


@dataclass
class EnhancerStats:
    """Lightweight per-frame diagnostics, useful for logging and plots."""

    frames_processed: int = 0
    preroll_complete: bool = False
    mean_gain: float = 1.0
    mean_snr_db: float = 0.0


def to_mono(
    block: np.ndarray,
    left: int | None = None,
    right: int | None = None,
) -> np.ndarray:
    """Collapse a possibly multi-channel block to mono by averaging.

    The hearing aid microphone delivers a two-channel signal on the soundcard,
    but both channels carry the same content. Averaging them into mono before
    the STFT halves the processing cost and is the right call for a
    proof-of-concept; a production system would keep the channels separate for
    binaural processing.

    Parameters
    ----------
    block:
        Shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    left, right:
        Zero-based channel indices to mix. Given both, only those two are used
        and every other channel is ignored -- necessary on a multichannel
        interface where the microphone sits on a specific pair of lines.
        Omitted, all channels are averaged.

    Returns
    -------
    np.ndarray
        Mono block of shape ``(n_samples,)``.
    """
    block = np.asarray(block, dtype=np.float64)
    if block.ndim == 1:
        return block
    if left is not None and right is not None:
        if max(left, right) >= block.shape[1]:
            raise ValueError(
                f"Channels {left} and {right} requested, but the block has only "
                f"{block.shape[1]}. Check input_channels in DeviceConfig."
            )
        return (block[:, left] + block[:, right]) * 0.5
    return block.mean(axis=1)


class SpeechEnhancer:
    """Real-time single-channel speech enhancer.

    Parameters
    ----------
    audio_config, imcra_config, wiener_config:
        Component configurations; defaults are the values used in the thesis.

    Examples
    --------
    >>> import numpy as np
    >>> enh = SpeechEnhancer()
    >>> noisy = np.random.randn(48_000) * 0.1
    >>> clean = enh.process_signal(noisy)
    >>> clean.shape
    (48000,)

    Changing suppression strength mid-stream:

    >>> enh.set_parameters(beta=1.8)
    >>> enh.parameters["beta"]
    1.8
    """

    def __init__(
        self,
        audio_config: AudioConfig | None = None,
        imcra_config: IMCRAConfig | None = None,
        wiener_config: WienerConfig | None = None,
    ) -> None:
        self.cfg = audio_config or AudioConfig()
        self.cfg.validate()

        self.stft = SlidingSTFT(self.cfg)
        self.noise_estimator = IMCRA(self.cfg.n_bins, imcra_config)
        self.filter = ParametricWienerFilter(
            self.cfg.n_bins,
            wiener_config,
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
        )

        self.stats = EnhancerStats()
        self._preroll: list[np.ndarray] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Parameter control
    # ------------------------------------------------------------------

    def set_parameters(self, **kwargs: float) -> None:
        """Thread-safe parameter update. See :meth:`ParametricWienerFilter.set_parameters`."""
        with self._lock:
            self.filter.set_parameters(**kwargs)

    @property
    def parameters(self) -> dict[str, float]:
        """Current DSP parameters."""
        with self._lock:
            return self.filter.parameters

    def reset(self) -> None:
        """Return the enhancer to its pre-startup state."""
        self.stft.reset()
        self.filter.reset()
        self.noise_estimator = IMCRA(self.cfg.n_bins, self.noise_estimator.cfg)
        self._preroll.clear()
        self.stats = EnhancerStats()

    def self_test(self, tolerance: float = 1e-9) -> float:
        """Verify perfect reconstruction of the analysis/synthesis path.

        Applies unit gain to every bin and checks that the overlap-add output
        matches the input. Raises if the WOLA condition is not met, which would
        mean the window or hop is misconfigured.

        Returns
        -------
        float
            Maximum deviation from the WOLA condition.
        """
        error = self.stft.reconstruction_error
        if error > tolerance:
            raise RuntimeError(
                f"WOLA condition violated (max deviation {error:.3e} > {tolerance:.1e}). "
                "Check win_length, hop_length and the window function."
            )
        return error

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Enhance one soundcard block.

        During the first ``preroll_frames`` calls the input is passed through
        unfiltered while background-noise spectra are collected; IMCRA is
        seeded from their per-bin minimum once enough frames have arrived.

        Parameters
        ----------
        block:
            ``block_size`` new samples, mono or multi-channel.

        Returns
        -------
        np.ndarray
            ``block_size`` enhanced mono samples.
        """
        mono = to_mono(block)
        spectrum = self.stft.analyze(mono)
        power = np.abs(spectrum) ** 2

        # -- Pre-roll: seed the noise floor before enhancing anything ----
        if not self.noise_estimator.is_seeded:
            self._preroll.append(power)
            if len(self._preroll) >= self.cfg.preroll_frames:
                self.noise_estimator.seed(np.stack(self._preroll))
                self._preroll.clear()
                self.stats.preroll_complete = True
            self.stats.frames_processed += 1
            # Pass through untouched; the estimate is not trustworthy yet.
            return self.stft.synthesize(spectrum)

        with self._lock:
            noise_psd = self.noise_estimator.update(power, xi=self.filter.a_priori_snr)
            gain = self.filter.compute_gain(power, noise_psd)

        out = self.stft.synthesize(spectrum * gain)

        self.stats.frames_processed += 1
        self.stats.mean_gain = float(gain.mean())
        with np.errstate(divide="ignore"):
            self.stats.mean_snr_db = float(
                10.0 * np.log10(np.maximum(self.filter.a_priori_snr.mean(), 1e-12))
            )
        return out

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """Enhance a complete signal offline, block by block.

        Useful for batch evaluation on recorded material. Uses exactly the same
        code path as the real-time callback, so offline results match what the
        live system does.

        Parameters
        ----------
        signal:
            Mono or multi-channel signal of shape ``(n_samples,)`` or
            ``(n_samples, n_channels)``.

        Returns
        -------
        np.ndarray
            Enhanced mono signal, same length as the input (zero-padded to a
            whole number of blocks, then trimmed).
        """
        mono = to_mono(signal)
        hop = self.cfg.hop_length

        n_blocks = int(np.ceil(len(mono) / hop))
        padded = np.zeros(n_blocks * hop)
        padded[: len(mono)] = mono

        out = np.zeros_like(padded)
        for i in range(n_blocks):
            sl = slice(i * hop, (i + 1) * hop)
            out[sl] = self.process_block(padded[sl])

        return out[: len(mono)]

    # ------------------------------------------------------------------
    # Live audio
    # ------------------------------------------------------------------

    def run_stream(
        self,
        device: int | str | tuple | None = None,
        duration: float | None = None,
        devices: DeviceConfig | None = None,
    ) -> None:
        """Open a duplex audio stream and enhance in real time.

        Requires the optional ``sounddevice`` dependency (``pip install
        '.[realtime]'``). The processed mono output is routed to both output
        channels.

        Parameters
        ----------
        device:
            Overrides the device selection: an index, a name, or an
            ``(input, output)`` pair. ``None`` uses ``devices``.
        duration:
            Seconds to run. ``None`` runs until interrupted.
        devices:
            Soundcard routing -- which devices, how many channels to open, and
            which input lines carry the microphone. Defaults to
            :class:`~intent_se.config.DeviceConfig`.
        """
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Real-time playback needs the 'sounddevice' package. "
                "Install it with: pip install '.[realtime]'"
            ) from exc

        routing = devices or DeviceConfig()
        routing.validate()
        if device is None:
            device = (routing.input_device, routing.output_device)

        self.self_test()

        def callback(indata, outdata, frames, time_info, status):  # noqa: ANN001
            if status:  # pragma: no cover - hardware dependent
                print(f"[audio] {status}")
            # Take only the two lines carrying the microphone, not the mean of
            # every open input channel.
            mono = to_mono(indata, routing.input_left, routing.input_right)
            enhanced = self.process_block(mono)
            # Silence every output channel, then write the processed signal to
            # the headset pair, so nothing leaks onto unused outputs.
            outdata[:] = 0.0
            outdata[:, routing.output_channel] = enhanced
            if routing.output_channel + 1 < outdata.shape[1]:
                outdata[:, routing.output_channel + 1] = enhanced

        with sd.Stream(
            device=device,
            samplerate=self.cfg.sample_rate,
            blocksize=self.cfg.block_size,
            dtype="float32",
            channels=(routing.input_channels, routing.output_channels),
            callback=callback,
        ):
            if duration is None:  # pragma: no cover - interactive
                print("Streaming. Press Ctrl+C to stop.")
                try:
                    threading.Event().wait()
                except KeyboardInterrupt:
                    print("\nStopped.")
            else:
                sd.sleep(int(duration * 1000))
