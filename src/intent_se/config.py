"""Central configuration for the whole system.

Every value here is justified in the thesis; the docstrings record *why* each
one is what it is, because several of them are hardware-derived rather than
free choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Intent classes
# --------------------------------------------------------------------------

#: Canonical class order. Kept fixed so that label ids, confusion-matrix axes
#: and saved models all agree across runs.
CLASS_ORDER: tuple[str, ...] = (
    "TOO_NOISY",
    "SPEECH_UNCLEAR",
    "TOO_LOUD",
    "TOO_QUIET",
    "TOO_SHARP",
    "BALANCED",
)

#: The only class that represents satisfaction rather than a complaint. It
#: exists so the system has a concept of "nothing is wrong" -- without it every
#: utterance reads as a complaint and the controller never stops adjusting.
NEUTRAL_CLASS = "BALANCED"


@dataclass(frozen=True)
class AudioConfig:
    """Parameters of the real-time audio pipeline.

    The chain of reasoning behind the three frame sizes:

    * ``block_size`` (512) is imposed by the soundcard (RME Fireface UC). It is
      a hardware constraint, not a design choice, and everything follows from it.
    * ``win_length`` (1024) is what the STFT needs for adequate frequency
      resolution. A sliding buffer bridges 512 -> 1024, which incidentally
      yields exactly the 50% overlap that WOLA requires (``hop == win/2``).
    * ``n_fft`` (2048) is reached by zero-padding the 1024-sample frame. This
      interpolates between frequency bins, giving IMCRA and the Wiener filter a
      higher-resolution spectral view and hence smoother gain curves.

    The 512 -> 1024 step is *time domain* buffering; the 1024 -> 2048 step is
    *frequency domain* zero-padding. They are independent and both necessary.
    """

    #: Soundcard sample rate. The RME Fireface UC was run at 48 kHz.
    sample_rate: int = 48_000

    #: New samples delivered per soundcard callback. Hardware-imposed.
    block_size: int = 512

    #: STFT window length. Also the sliding-buffer length.
    win_length: int = 1024

    #: FFT length after zero-padding. Yields ``n_fft // 2 + 1 == 1025`` bins.
    n_fft: int = 2048

    #: Analysis/synthesis hop. Equals ``block_size`` and ``win_length // 2``.
    hop_length: int = 512

    #: Frames of pure background noise captured at startup to seed IMCRA.
    #: Without this the recursive estimator starts from zero and either
    #: suppresses everything or nothing for the first few seconds.
    preroll_frames: int = 5

    @property
    def n_bins(self) -> int:
        """Number of non-negative-frequency bins (1025 for a 2048-point FFT)."""
        return self.n_fft // 2 + 1

    def validate(self) -> None:
        """Raise if the configuration violates the WOLA assumptions."""
        if self.hop_length * 2 != self.win_length:
            raise ValueError(
                f"WOLA here assumes 50% overlap: hop_length ({self.hop_length}) "
                f"must be win_length // 2 ({self.win_length // 2})."
            )
        if self.n_fft < self.win_length:
            raise ValueError(
                f"n_fft ({self.n_fft}) must be >= win_length ({self.win_length})."
            )
        if self.block_size != self.hop_length:
            raise ValueError(
                f"The sliding buffer assumes one hop per callback: block_size "
                f"({self.block_size}) must equal hop_length ({self.hop_length})."
            )


@dataclass(frozen=True)
class DeviceConfig:
    """Soundcard routing the system was measured on.

    The hearing aid microphone and the headset were on **different** devices,
    which is why a single device index is not enough::

        [1] Black Gold                   (in:  0, out:  2)   <- headset
        [5] Fireface UC Mac (23913018)   (in: 18, out: 18)   <- hearing aid mic

    On the Fireface the microphone arrives on **hardware lines 5 and 6**, which
    are zero-based channel indices 4 and 5. The stream therefore has to open all
    18 input channels and pick those two -- opening only the first two would
    capture the wrong inputs entirely.

    .. warning::
       Device indices are assigned by the operating system and change when
       interfaces are plugged in or removed. Confirm them with
       ``intent-se-run --list-devices`` before a session; override with
       ``--input-device`` / ``--output-device``.
    """

    input_device: int | None = 5
    """Fireface UC Mac (23913018) -- carries the hearing aid microphone."""

    output_device: int | None = 1
    """Black Gold -- the headset."""

    input_channels: int = 18
    """Channels opened on the input device. The Fireface exposes 18; all are
    opened so that lines 5 and 6 are reachable."""

    output_channels: int = 2
    """Channels opened on the output device."""

    input_left: int = 4
    """Hardware line 5 (zero-based index 4)."""

    input_right: int = 5
    """Hardware line 6 (zero-based index 5)."""

    output_channel: int = 0
    """Hardware line 1 (zero-based index 0). The processed mono signal is
    written here and to the next channel, so both ears of the headset get it."""

    def validate(self) -> None:
        """Raise if the selected input channels lie outside the opened range."""
        for name, ch in (("input_left", self.input_left),
                         ("input_right", self.input_right)):
            if not 0 <= ch < self.input_channels:
                raise ValueError(
                    f"{name}={ch} is outside the {self.input_channels} opened "
                    f"input channels."
                )
        if not 0 <= self.output_channel < self.output_channels:
            raise ValueError(
                f"output_channel={self.output_channel} is outside the "
                f"{self.output_channels} opened output channels."
            )


@dataclass(frozen=True)
class IMCRAConfig:
    """IMCRA parameters, following Cohen (2003).

    ``D == U * V == 120`` is not a coincidence: the minimum tracker is a
    two-stage design that keeps ``U`` subwindows of ``V`` frames each, so the
    effective minimum-search window spans ``U * V`` frames.
    """

    alpha_s: float = 0.9
    """Smoothing for the periodogram used by the minimum tracker."""

    alpha_d: float = 0.85
    """Base smoothing of the noise PSD update. Raised toward 1 when speech is
    likely present, so the estimate freezes instead of absorbing speech."""

    beta: float = 1.47
    """Bias compensation for the minimum statistics estimate."""

    gamma0: float = 4.6
    """A posteriori SNR threshold used by the speech-presence decision."""

    gamma1: float = 3.0
    """Secondary a posteriori SNR threshold."""

    zeta0: float = 1.67
    """A priori SNR threshold used by the speech-presence decision."""

    u: int = 8
    """Number of subwindows in the minimum tracker."""

    v: int = 15
    """Frames per subwindow."""

    @property
    def d(self) -> int:
        """Total minimum-search window length in frames (``u * v`` == 120)."""
        return self.u * self.v


@dataclass(frozen=True)
class WienerConfig:
    """Parametric Wiener filter parameters."""

    beta: float = 1.0
    """Suppression exponent. ``beta == 1`` recovers the standard Wiener filter.

    Higher values steepen the gain curve (more aggressive suppression), lower
    values flatten it (more high-frequency content preserved). **This is the
    parameter the NLP side controls.**
    """

    gain_floor: float = 0.05
    """No bin is attenuated below 5% of its original level. Prevents bins from
    being zeroed out entirely, which is what produces musical-noise artifacts
    (the 'underwater' or 'bubbling' quality of poor noise suppression)."""

    dd_alpha: float = 0.98
    """Decision-directed smoothing for the a priori SNR (Ephraim & Malah)."""

    snr_min_db: float = -25.0
    """Floor on the a priori SNR, for numerical stability."""

    output_gain: float = 1.0
    """Broadband output gain. Controlled by the NLP side for TOO_LOUD/TOO_QUIET."""

    tilt: float = 0.0
    """High-frequency tilt in dB across the spectrum. Negative values tame
    brightness (TOO_SHARP); positive values add presence (SPEECH_UNCLEAR)."""


@dataclass(frozen=True)
class NLPConfig:
    """Sentence-embedding, classifier and severity-scorer settings."""

    embedding_model: str = "all-mpnet-base-v2"
    """768-dim Sentence-BERT model, chosen over the faster 384-dim
    ``all-MiniLM-L6-v2``. Borderline sentences (e.g. "I can't make out what
    people are saying", which could be TOO_NOISY or SPEECH_UNCLEAR) need the
    better semantic placement, and inference speed is not a bottleneck here."""

    embedding_dim: int = 768

    normalize_embeddings: bool = True
    """L2-normalise so only direction matters; cosine similarity then equals
    the dot product."""

    test_size: float = 0.15
    val_size: float = 0.15
    """70 / 15 / 15 stratified split. Cross-validation runs on the combined
    train+val pool (940 sentences); the test set is touched exactly once."""

    cv_folds: int = 5
    random_state: int = 42

    ridge_alphas: tuple[float, ...] = (0.01, 0.05, 0.10, 0.50, 1.00)
    """Candidate regularisation strengths for the severity scorer. alpha = 0.10
    gave the lowest CV MAE (0.1002)."""

    knn_neighbors: int = 15
    mlp_hidden: tuple[int, ...] = (256, 64)


@dataclass(frozen=True)
class Config:
    """Top-level configuration bundle."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    imcra: IMCRAConfig = field(default_factory=IMCRAConfig)
    wiener: WienerConfig = field(default_factory=WienerConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)


DEFAULT = Config()
