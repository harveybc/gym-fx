"""C35/C37: the trust-injecting fixture seam lives HERE, under
tests/, NOT in the distributed production module. It drives the
private _resolve_manifest / _derive_readiness_body and seals the
result EXCLUSIVELY into the FIXTURE schema — the sealing function
below hardcodes FIXTURE_SCHEMA and fixture_marker=True, so this seam
structurally CANNOT select or emit the production schema. Production
code never imports this file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from tools.wp4_session_readiness import (FIXTURE_SCHEMA,
                                         PINNED_TRUST_MANIFEST_DIGEST,
                                         ReadinessError, ResolvedTrust,
                                         VerifiedSource,
                                         _derive_readiness_body,
                                         _resolve_manifest,
                                         canonical_bytes, sha256_hex,
                                         strict_json_loads)


def resolve_fixture_manifest(path: Any, *,
                             expected_digest: str) -> ResolvedTrust:
    """Load a PROVISIONED fixture manifest for tests only."""
    raw = Path(path).read_bytes()
    return _resolve_manifest(strict_json_loads(raw),
                             expected_digest=expected_digest)


def _seal_fixture(body: dict) -> dict:
    """Pure sealing function: FIXTURE schema only. There is no
    parameter by which a caller could select the production schema."""
    package = {"schema": FIXTURE_SCHEMA, "fixture_marker": True,
               **body}
    package["digest"] = sha256_hex(canonical_bytes(package))
    return package


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
    if trust.manifest_digest == PINNED_TRUST_MANIFEST_DIGEST:
        raise ReadinessError(
            "the fixture seam cannot use the pinned production trust")
    body = _derive_readiness_body(
        source, trust, bar_hours=bar_hours,
        evaluation_as_of=evaluation_as_of,
        realized_vol_window_bars=realized_vol_window_bars,
        calendar_tz=calendar_tz, session_export=session_export,
        activation_receipt=activation_receipt,
        required_pre_bars=required_pre_bars,
        required_post_bars=required_post_bars,
        operator_exceptions=operator_exceptions)
    return _seal_fixture(body)
