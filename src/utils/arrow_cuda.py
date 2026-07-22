"""Windows CUDA + pyarrow coexistence helpers.

On this host (Windows + CUDA torch wheels), calling any ``torch.cuda.*`` API
*before* pandas/pyarrow has been imported and used for a parquet read can hard
crash the process with ``0xC0000005`` (access violation) on the first
``pd.read_parquet``. Once parquet I/O has been warmed up, CUDA may be
initialised safely and further parquet reads succeed.

Call :func:`warmup_parquet_io` before the first ``torch.cuda.is_available`` /
``get_device_name`` / ``tensor.to('cuda')``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

_WARMED = False


def warmup_parquet_io(sample: Optional[PathLike] = None) -> None:
    """Import pandas/pyarrow and optionally read one parquet before CUDA."""
    global _WARMED
    if _WARMED:
        return
    import pandas as pd

    # Touch pyarrow's native module even if no sample file exists.
    import pyarrow.parquet as pq  # noqa: F401

    if sample is not None:
        p = Path(sample)
        if p.is_file():
            pd.read_parquet(p)
    _WARMED = True


def read_parquet(path: PathLike, **kwargs):
    """``pd.read_parquet`` after ensuring the CUDA-safe warmup has run."""
    warmup_parquet_io(path)
    import pandas as pd

    return pd.read_parquet(path, **kwargs)
