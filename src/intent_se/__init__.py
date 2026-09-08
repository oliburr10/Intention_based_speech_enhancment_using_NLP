"""Intention-Based Speech Enhancement for Hearing Aids Using NLP.

A real-time speech enhancement system whose DSP parameters are controlled by
natural-language complaints from the user.

The system has three parts:

``intent_se.audio``
    Real-time STFT/WOLA pipeline with IMCRA noise estimation and a parametric
    Wiener filter whose suppression exponent ``beta`` is externally tunable.

``intent_se.nlp``
    Sentence-embedding intent classifier (six classes) and a Ridge severity
    regressor, both operating on ``all-mpnet-base-v2`` embeddings.

``intent_se.control``
    The bridge: maps (intent, severity) onto DSP parameter updates - only experimental (not figured out only testing - outside of thesis!!)
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
