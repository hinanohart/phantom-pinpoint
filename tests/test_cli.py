"""Smoke tests for the CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from phantom_pinpoint.cli import app


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0." in result.stdout


def test_baseline_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "baseline.parquet"
    result = runner.invoke(app, ["baseline", "--n-runs", "32", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
