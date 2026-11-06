import sys
from unittest.mock import Mock

import pytest

from benchmarks.benchmarks import common


@pytest.mark.parametrize("backend", ["cupy", "dask+cupy"])
@pytest.mark.parametrize("cupy_installed", [False, True])
def test_gpu_benchmark_skips_when_cuda_unavailable(
    monkeypatch, backend, cupy_installed
):
    available = Mock(return_value=False)
    monkeypatch.setattr(common, "has_cuda_and_cupy", available)
    cupy = Mock() if cupy_installed else None
    if cupy_installed:
        cupy.asarray.side_effect = AssertionError("must skip GPU conversion")
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    with pytest.raises(NotImplementedError):
        common.get_xr_dataarray((2, 3), backend)

    available.assert_called_once_with()
    if cupy_installed:
        cupy.asarray.assert_not_called()


def test_cupy_benchmark_reaches_conversion_when_cuda_available(monkeypatch):
    available = Mock(return_value=True)
    monkeypatch.setattr(common, "has_cuda_and_cupy", available)
    cupy = Mock()
    cupy.asarray.side_effect = RuntimeError("conversion reached")
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    with pytest.raises(RuntimeError, match="conversion reached"):
        common.get_xr_dataarray((2, 3), "cupy")

    available.assert_called_once_with()
    cupy.asarray.assert_called_once()
