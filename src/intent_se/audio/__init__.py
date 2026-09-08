"""Real-time audio processing pipeline.

Signal flow for one soundcard callback::

    512 new samples
        -> sliding buffer (holds the most recent 1024)
        -> sqrt-Hann analysis window
        -> zero-pad to 2048, rFFT            -> 1025 bins
        -> IMCRA noise PSD estimate
        -> parametric Wiener gain, applied to the spectrum
        -> irFFT, discard zero-padding       -> 1024 samples
        -> sqrt-Hann synthesis window
        -> overlap-add into the output buffer
    512 enhanced samples out
"""

from intent_se.audio.imcra import IMCRA
from intent_se.audio.pipeline import SpeechEnhancer
from intent_se.audio.stft import SlidingSTFT, sqrt_hann, wola_check
from intent_se.audio.wiener import ParametricWienerFilter

__all__ = [
    "IMCRA",
    "ParametricWienerFilter",
    "SlidingSTFT",
    "SpeechEnhancer",
    "sqrt_hann",
    "wola_check",
]
