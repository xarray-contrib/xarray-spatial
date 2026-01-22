"""
Diagnostics module for xarray-spatial.

Provides utilities to help users identify common pitfalls and issues
with DataArrays before running xarray-spatial operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import xarray as xr

from xrspatial.utils import (
    _infer_coord_unit_type,
    _infer_vertical_unit_type,
    get_dataarray_resolution,
)


@dataclass
class DiagnosticIssue:
    """Represents a single diagnostic issue found during analysis."""
    code: str
    severity: str  # 'warning' or 'error'
    message: str
    suggestion: str


@dataclass
class DiagnosticReport:
    """Results from diagnosing a DataArray."""
    issues: List[DiagnosticIssue] = field(default_factory=list)
    horizontal_unit_type: Optional[str] = None
    vertical_unit_type: Optional[str] = None
    resolution: Optional[tuple] = None

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == 'warning' for i in self.issues)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == 'error' for i in self.issues)

    def __str__(self) -> str:
        if not self.issues:
            return "No issues detected."

        lines = []
        for issue in self.issues:
            lines.append(f"[{issue.severity.upper()}] {issue.code}: {issue.message}")
            lines.append(f"  Suggestion: {issue.suggestion}")
        return "\n".join(lines)


def _check_unit_mismatch(agg: xr.DataArray, report: DiagnosticReport) -> None:
    """
    Check for horizontal vs vertical unit mismatch.

    Detects the common case of coordinates in degrees (lon/lat) with
    elevation values in meters/feet.
    """
    try:
        cellsize_x, cellsize_y = get_dataarray_resolution(agg)
        report.resolution = (cellsize_x, cellsize_y)
    except Exception:
        return

    if len(agg.dims) < 2:
        return

    dim_y, dim_x = agg.dims[-2], agg.dims[-1]
    coord_x = agg.coords.get(dim_x, None)
    coord_y = agg.coords.get(dim_y, None)

    if coord_x is None or coord_y is None:
        return

    horiz_x = _infer_coord_unit_type(coord_x, cellsize_x)
    horiz_y = _infer_coord_unit_type(coord_y, cellsize_y)
    vert = _infer_vertical_unit_type(agg)

    report.vertical_unit_type = vert

    horiz_types = {horiz_x, horiz_y} - {"unknown"}
    if horiz_types:
        report.horizontal_unit_type = next(iter(horiz_types))

    if not horiz_types or vert == "unknown":
        return

    if "degrees" in horiz_types and vert == "elevation":
        report.issues.append(DiagnosticIssue(
            code="UNIT_MISMATCH",
            severity="warning",
            message=(
                "Input DataArray appears to have coordinates in degrees "
                "but elevation values in a linear unit (e.g. meters/feet)."
            ),
            suggestion=(
                "Slope/aspect/curvature operations expect horizontal distances "
                "in the same units as vertical. Consider reprojecting to a "
                "projected CRS with meter-based coordinates."
            ),
        ))


def diagnose(agg: xr.DataArray, tool: Optional[str] = None) -> DiagnosticReport:
    """
    Diagnose a DataArray for common xarray-spatial pitfalls.

    Runs a series of heuristic checks to identify potential issues
    that could lead to incorrect results when using xarray-spatial
    functions.

    Parameters
    ----------
    agg : xr.DataArray
        The input DataArray to diagnose.
    tool : str, optional
        Name of the xarray-spatial tool you intend to use (e.g., 'slope',
        'aspect', 'curvature'). When specified, only diagnostics relevant
        to that tool are run. If None, all diagnostics are run.

    Returns
    -------
    DiagnosticReport
        A report containing any issues found, along with inferred
        metadata about the DataArray.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from xrspatial.diagnostics import diagnose
    >>> # Create a DataArray with lon/lat coordinates but meter elevations
    >>> data = np.random.rand(100, 100) * 1000 + 500
    >>> da = xr.DataArray(
    ...     data,
    ...     dims=['y', 'x'],
    ...     coords={
    ...         'y': np.linspace(40.0, 41.0, 100),
    ...         'x': np.linspace(-105.0, -104.0, 100),
    ...     }
    ... )
    >>> report = diagnose(da)
    >>> print(report)
    [WARNING] UNIT_MISMATCH: Input DataArray appears to have coordinates...
    >>> if report.has_warnings:
    ...     print("Consider reprojecting your data!")
    """
    report = DiagnosticReport()

    # Tools that are sensitive to unit mismatch
    unit_mismatch_tools = {'slope', 'aspect', 'curvature', 'hillshade'}

    # Run unit mismatch check if tool is relevant or no tool specified
    if tool is None or tool.lower() in unit_mismatch_tools:
        _check_unit_mismatch(agg, report)

    return report
