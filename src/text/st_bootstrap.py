"""Bootstrap sentence-transformers for text-only use on Windows.

sentence-transformers>=5.6 imports torchcodec (audio/video) at package load.
Without FFmpeg shared DLLs that fails even for text encoding. We install
lightweight stubs *before* importing sentence_transformers when torchcodec is
missing or broken. Text encoding does not use those modalities.
"""

from __future__ import annotations

import importlib.machinery
import sys
from types import ModuleType
from typing import Any


def _mod(name: str) -> ModuleType:
    m = ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__file__ = f"<stub {name}>"
    sys.modules[name] = m
    return m


def install_torchcodec_stubs() -> None:
    """Replace/create torchcodec with inert stubs so ST text models can load."""
    root = _mod("torchcodec")
    for sub in ("decoders", "encoders", "samplers", "transforms", "_core"):
        m = _mod(f"torchcodec.{sub}")
        setattr(root, sub, m)
    # Nested paths ST / datasets may touch
    _mod("torchcodec._core.ops")
    _mod("torchcodec._core._metadata")
    _mod("torchcodec._internally_replaced_utils")
    dec = sys.modules["torchcodec.decoders"]
    dec.AudioDecoder = type("AudioDecoder", (), {})  # type: ignore[attr-defined]
    dec.VideoDecoder = type("VideoDecoder", (), {})  # type: ignore[attr-defined]
    core = sys.modules["torchcodec._core"]
    core.AudioStreamMetadata = type("AudioStreamMetadata", (), {})  # type: ignore[attr-defined]
    core.VideoStreamMetadata = type("VideoStreamMetadata", (), {})  # type: ignore[attr-defined]


def _torchcodec_usable() -> bool:
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import AudioDecoder  # noqa: F401

        return True
    except Exception:
        return False


def import_sentence_transformer() -> Any:
    """Return SentenceTransformer class, stubbing torchcodec if required."""
    if not _torchcodec_usable():
        # Evict broken real package so stubs win
        for key in list(sys.modules):
            if key == "torchcodec" or key.startswith("torchcodec."):
                del sys.modules[key]
        install_torchcodec_stubs()

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer
