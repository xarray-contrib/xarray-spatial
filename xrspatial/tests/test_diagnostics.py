import numpy as np
import xarray as xr
import pytest

from xrspatial import diagnostics
from xrspatial.diagnostics import diagnose, DiagnosticReport, DiagnosticIssue


class TestDiagnosticIssue:
    def test_dataclass_fields(self):
        issue = DiagnosticIssue(
            code="TEST_CODE",
            severity="warning",
            message="Test message",
            suggestion="Test suggestion",
        )
        assert issue.code == "TEST_CODE"
        assert issue.severity == "warning"
        assert issue.message == "Test message"
        assert issue.suggestion == "Test suggestion"


class TestDiagnosticReport:
    def test_empty_report(self):
        report = DiagnosticReport()
        assert report.issues == []
        assert report.horizontal_unit_type is None
        assert report.vertical_unit_type is None
        assert report.resolution is None
        assert not report.has_issues
        assert not report.has_warnings
        assert not report.has_errors

    def test_has_issues_with_warning(self):
        report = DiagnosticReport()
        report.issues.append(DiagnosticIssue(
            code="TEST",
            severity="warning",
            message="msg",
            suggestion="sug",
        ))
        assert report.has_issues
        assert report.has_warnings
        assert not report.has_errors

    def test_has_issues_with_error(self):
        report = DiagnosticReport()
        report.issues.append(DiagnosticIssue(
            code="TEST",
            severity="error",
            message="msg",
            suggestion="sug",
        ))
        assert report.has_issues
        assert not report.has_warnings
        assert report.has_errors

    def test_str_no_issues(self):
        report = DiagnosticReport()
        assert str(report) == "No issues detected."

    def test_str_with_issues(self):
        report = DiagnosticReport()
        report.issues.append(DiagnosticIssue(
            code="UNIT_MISMATCH",
            severity="warning",
            message="Test message",
            suggestion="Test suggestion",
        ))
        output = str(report)
        assert "[WARNING] UNIT_MISMATCH: Test message" in output
        assert "Suggestion: Test suggestion" in output


class TestDiagnoseUnitMismatch:
    def test_degrees_horizontal_elevation_vertical(self, monkeypatch):
        """
        If coordinates look like degrees (lon/lat) and values look like elevation
        (e.g., meters), diagnose should report a UNIT_MISMATCH warning.
        """
        data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)

        # Coordinates in degrees (lon/lat-ish)
        y = np.linspace(5.0, 5.0025, 10)
        x = np.linspace(-74.93, -74.9275, 10)

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "m"},  # elevation in meters
        )

        def fake_get_dataarray_resolution(arr):
            return float(x[1] - x[0]), float(y[1] - y[0])

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da)

        assert report.has_issues
        assert report.has_warnings
        assert len(report.issues) == 1
        assert report.issues[0].code == "UNIT_MISMATCH"
        assert report.horizontal_unit_type == "degrees"
        assert report.vertical_unit_type == "elevation"

    def test_no_warning_for_projected_like_grid(self, monkeypatch):
        """
        If coordinates look like projected linear units (e.g., meters) and values
        look like elevation, we should NOT report any issues.
        """
        data = np.linspace(0, 999, 100 * 100, dtype=float).reshape(100, 100)

        # Coordinates in meters (projected-looking) with larger span
        y = 4_500_000.0 + np.arange(100) * 30.0  # UTM-ish northings
        x = 500_000.0 + np.arange(100) * 30.0    # UTM-ish eastings

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "m"},  # elevation in meters
        )

        def fake_get_dataarray_resolution(arr):
            return float(x[1] - x[0]), float(y[1] - y[0])  # 30 m

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da)

        assert not report.has_issues
        assert report.horizontal_unit_type == "linear"
        assert report.vertical_unit_type == "elevation"

    def test_degrees_but_angle_vertical(self, monkeypatch):
        """
        If coordinates are in degrees but the DataArray itself looks like an angle
        (e.g., units='degrees'), we should NOT report issues; slope/aspect outputs
        fall into this category.
        """
        data = np.linspace(0, 90, 10 * 10, dtype=float).reshape(10, 10)

        # Coordinates in degrees
        y = np.linspace(5.0, 5.0025, 10)
        x = np.linspace(-74.93, -74.9275, 10)

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "degrees"},  # angle, not elevation
        )

        def fake_get_dataarray_resolution(arr):
            return float(x[1] - x[0]), float(y[1] - y[0])

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da)

        assert not report.has_issues
        assert report.horizontal_unit_type == "degrees"
        assert report.vertical_unit_type == "angle"


class TestDiagnoseWithToolParameter:
    def test_slope_tool_runs_unit_mismatch_check(self, monkeypatch):
        """Unit mismatch check should run when tool='slope'."""
        data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
        y = np.linspace(5.0, 5.0025, 10)
        x = np.linspace(-74.93, -74.9275, 10)

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "m"},
        )

        def fake_get_dataarray_resolution(arr):
            return float(x[1] - x[0]), float(y[1] - y[0])

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da, tool='slope')
        assert report.has_issues
        assert report.issues[0].code == "UNIT_MISMATCH"

    def test_aspect_tool_runs_unit_mismatch_check(self, monkeypatch):
        """Unit mismatch check should run when tool='aspect'."""
        data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
        y = np.linspace(5.0, 5.0025, 10)
        x = np.linspace(-74.93, -74.9275, 10)

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "m"},
        )

        def fake_get_dataarray_resolution(arr):
            return float(x[1] - x[0]), float(y[1] - y[0])

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da, tool='aspect')
        assert report.has_issues

    def test_curvature_tool_runs_unit_mismatch_check(self, monkeypatch):
        """Unit mismatch check should run when tool='curvature'."""
        data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
        y = np.linspace(5.0, 5.0025, 10)
        x = np.linspace(-74.93, -74.9275, 10)

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "m"},
        )

        def fake_get_dataarray_resolution(arr):
            return float(x[1] - x[0]), float(y[1] - y[0])

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da, tool='curvature')
        assert report.has_issues

    def test_hillshade_tool_runs_unit_mismatch_check(self, monkeypatch):
        """Unit mismatch check should run when tool='hillshade'."""
        data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
        y = np.linspace(5.0, 5.0025, 10)
        x = np.linspace(-74.93, -74.9275, 10)

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "m"},
        )

        def fake_get_dataarray_resolution(arr):
            return float(x[1] - x[0]), float(y[1] - y[0])

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da, tool='hillshade')
        assert report.has_issues

    def test_tool_name_case_insensitive(self, monkeypatch):
        """Tool name matching should be case insensitive."""
        data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
        y = np.linspace(5.0, 5.0025, 10)
        x = np.linspace(-74.93, -74.9275, 10)

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
            attrs={"units": "m"},
        )

        def fake_get_dataarray_resolution(arr):
            return float(x[1] - x[0]), float(y[1] - y[0])

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da, tool='SLOPE')
        assert report.has_issues


class TestDiagnoseEdgeCases:
    def test_1d_array_no_crash(self):
        """diagnose should handle 1D arrays gracefully."""
        da = xr.DataArray(np.arange(10), dims=["x"])
        report = diagnose(da)
        assert not report.has_issues

    def test_missing_coords_no_crash(self):
        """diagnose should handle arrays without coordinates gracefully."""
        da = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
        # No coords assigned
        report = diagnose(da)
        # Should not crash, may or may not have issues depending on heuristics
        assert isinstance(report, DiagnosticReport)

    def test_resolution_stored_in_report(self, monkeypatch):
        """Resolution should be stored in the report."""
        data = np.linspace(0, 999, 10 * 10, dtype=float).reshape(10, 10)
        y = np.arange(10) * 30.0
        x = 500_000.0 + np.arange(10) * 30.0

        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={"y": y, "x": x},
        )

        def fake_get_dataarray_resolution(arr):
            return 30.0, 30.0

        monkeypatch.setattr(diagnostics, "get_dataarray_resolution", fake_get_dataarray_resolution)

        report = diagnose(da)
        assert report.resolution == (30.0, 30.0)


class TestTopLevelImport:
    def test_diagnose_importable_from_xrspatial(self):
        """diagnose should be importable from the top-level xrspatial package."""
        from xrspatial import diagnose as diagnose_top
        assert diagnose_top is diagnose
