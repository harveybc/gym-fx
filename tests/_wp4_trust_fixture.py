"""C35: the trust-injecting fixture seam lives HERE, under tests/,
NOT in the distributed production module. It drives the private
_resolve_manifest / _build_package with fixture=True, which can only
emit the FIXTURE schema — a fixture package can never masquerade as
production authority. Production code never imports this file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from tools.wp4_session_readiness import (ResolvedTrust,
                                         VerifiedSource, _build_package,
                                         _resolve_manifest,
                                         strict_json_loads)


def resolve_fixture_manifest(path: Any, *,
                             expected_digest: str) -> ResolvedTrust:
    """Load a PROVISIONED fixture manifest for tests only."""
    raw = Path(path).read_bytes()
    return _resolve_manifest(strict_json_loads(raw),
                             expected_digest=expected_digest)


def build_fixture_readiness(source: VerifiedSource,
                            trust: ResolvedTrust, *,
                            bar_hours: float, evaluation_as_of,
                            realized_vol_window_bars: int = 3,
                            calendar_tz: Optional[str] = None,
                            session_export: Optional[Any] = None,
                            activation_receipt: Optional[Any] = None,
                            required_pre_bars: int = 4,
                            required_post_bars: int = 4,
                            operator_exceptions:
                            Optional[Sequence[Any]] = None) -> dict:
    """Build a FIXTURE-schema readiness package under an isolated
    provisioned manifest. Never emits the production schema."""
    return _build_package(
        source, trust, bar_hours=bar_hours,
        evaluation_as_of=evaluation_as_of,
        realized_vol_window_bars=realized_vol_window_bars,
        calendar_tz=calendar_tz, session_export=session_export,
        activation_receipt=activation_receipt,
        required_pre_bars=required_pre_bars,
        required_post_bars=required_post_bars,
        operator_exceptions=operator_exceptions, fixture=True)
