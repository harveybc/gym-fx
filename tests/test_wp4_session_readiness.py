"""C18-C22 battery (orders agent-multi@0ca5f7af §5, correction
agent-multi@d198451c). The eight frozen counterexamples refuse or
report the corrected typed state; authority is derived only from
sealed evidence; temporal metrics read the roles they name; the
observed-gap taxonomy uses geometry; the package binds the per-unit
ledger; public code carries no operator topology and Tier-A absence
FAILS rather than skips."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from tools.wp4_session_readiness import (
    ACTIVATION_RECEIPT_SCHEMA, ColumnRoleContract, ETH_SPOT_CONCLUSION,
    EvidenceError, GAP_STAMP, JoinContractError,
    OPERATOR_EXCEPTION_SCHEMA, PROVENANCE_BROKER, PROVENANCE_OBSERVED,
    ReadinessError, SESSION_EXPORT_SCHEMA, UNAVAILABLE,
    WP4_MIN_PAIRED_WEEKS, build_readiness_package,
    canonical_bytes, classify_observed_gap, count_paired_weeks,
    derive_paired_weeks, inventory_observed_gaps,
    load_authoritative_intervals, opening_gap_return,
    post_reopen_realized_vol, quote_continuity, readiness_verdict,
    seal, sha256_hex)

VENUE, ACCT, SYM = "mt5_demo", "fp-1", "ETHUSD"
ROLES = ColumnRoleContract(datetime_col="DATE_TIME", open_col="OPEN",
                           close_col="CLOSE")


def sealed_export(intervals):
    return json.dumps(seal({
        "schema": SESSION_EXPORT_SCHEMA, "venue": VENUE,
        "account_fingerprint": ACCT, "symbol": SYM,
        "acquisition_range": ["2024-01-01", "2024-12-31"],
        "intervals": intervals}))


def sealed_receipt(export_json):
    export = json.loads(export_json)
    return json.dumps(seal({
        "schema": ACTIVATION_RECEIPT_SCHEMA, "venue": VENUE,
        "account_fingerprint": ACCT, "symbol": SYM,
        "activation_identity": "act-1",
        "activated_at": "2024-01-01T00:00:00Z",
        "bound_export_sha256": export["seal_sha256"]}))


def _weekly_intervals(n, *, start="2024-01-05 20:00"):
    """n weekend closures a week apart (Fri 20:00 -> Mon 00:00)."""
    out = []
    base = pd.Timestamp(start, tz="UTC")
    for i in range(n):
        close = base + pd.Timedelta(weeks=i)
        reopen = close + pd.Timedelta(hours=52)
        out.append({"close_at": close.isoformat(),
                    "reopen_at": reopen.isoformat()})
    return out


def _bars_covering(intervals, *, pre=4, post=4, bar_hours=4.0):
    """Continuous 4h bars with pre/post bars around every interval
    and the gap physically absent."""
    stamps = []
    bar = pd.Timedelta(hours=bar_hours)
    for iv in intervals:
        c = pd.Timestamp(iv["close_at"])
        r = pd.Timestamp(iv["reopen_at"])
        for k in range(pre, 0, -1):
            stamps.append(c - k * bar)
        for k in range(post):
            stamps.append(r + k * bar)
    stamps = sorted(set(stamps))
    n = len(stamps)
    return pd.DataFrame({
        "DATE_TIME": [s.tz_convert("UTC").tz_localize(None)
                      for s in stamps],
        "OPEN": 100.0 + np.arange(n),
        "CLOSE": 100.5 + np.arange(n)})


# ================================================================== #
# the eight frozen counterexamples                                   #
# ================================================================== #

class TestFrozenCounterexamples:

    def test_1_no_scalar_can_mint_sufficiency(self):
        """CRITICAL-1 dead: there is no public API accepting
        collector_active or an authoritative integer."""
        import inspect
        import tools.wp4_session_readiness as mod
        sig = inspect.signature(mod.readiness_verdict)
        assert "collector_active" not in sig.parameters
        assert "authoritative_units" not in sig.parameters
        # the only way to a sufficient verdict is a real bundle
        v = readiness_verdict(authoritative=None, paired=None,
                              observed_units=0)
        assert v["state"] == \
            "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"

    def test_2_tuesday_56h_is_never_weekend(self):
        """HIGH-1 dead."""
        pre = pd.Timestamp("2024-01-09 08:00", tz="UTC")   # Tue
        post = pre + pd.Timedelta(hours=56)                # Thu
        assert classify_observed_gap(pre, post) == \
            "midweek_outage_shaped"

    def test_3_opening_gap_reads_open_not_close(self):
        """CRITICAL-2a dead: reopened OPEN=150 over pre-close 100 is
        +0.5, not the close-based +0.2."""
        assert opening_gap_return(150.0, 100.0) == pytest.approx(0.5)

    def test_4_volume_is_never_volatility(self):
        """CRITICAL-2b dead: realized vol comes from close-to-close
        log returns, never a passed VOLUME column — the API has no
        vol_col parameter."""
        import inspect
        import tools.wp4_session_readiness as mod
        assert "vol_col" not in inspect.signature(
            mod.inventory_observed_gaps).parameters
        vol = post_reopen_realized_vol([100.0, 100.0, 100.0, 100.0],
                                       window_bars=3)
        assert vol["value"] == pytest.approx(0.0)
        assert "log returns" in vol["definition"]

    def test_5_quote_continuity_needs_quote_evidence(self):
        """CRITICAL-2c dead: no quote evidence -> UNAVAILABLE."""
        qc = quote_continuity(None, expected_spacing_seconds=None)
        assert qc["value"] == UNAVAILABLE

    def test_6_ledger_is_bound_into_the_package(self):
        """HIGH-2 dead: mutating a unit changes the package digest."""
        a = _weekly_intervals(1)
        frame = _bars_covering(a)
        pkg = build_readiness_package(
            frame, roles=ROLES, bar_hours=4.0,
            source_logical_id="fixture", source_digest="d")
        # perturb the REOPEN bar's OPEN, which feeds the unit's
        # opening_gap_return — a real per-unit metric change
        frame2 = frame.copy()
        reopen_at = pd.Timestamp(a[0]["reopen_at"]).tz_localize(None)
        mask = frame2["DATE_TIME"] == reopen_at
        assert mask.any()
        frame2.loc[mask, "OPEN"] += 5.0
        pkg2 = build_readiness_package(
            frame2, roles=ROLES, bar_hours=4.0,
            source_logical_id="fixture", source_digest="d")
        assert pkg["unit_ledger_digest"] != pkg2["unit_ledger_digest"]
        assert pkg["digest"] != pkg2["digest"]

    def test_7_enum_string_confers_no_authority(self):
        """HIGH-4 dead: authority is only via sealed evidence; there
        is no public SessionEnvelopeInterval to label."""
        import tools.wp4_session_readiness as mod
        assert not hasattr(mod, "SessionEnvelopeInterval")
        # a forged (unsealed) export refuses
        with pytest.raises(EvidenceError, match="sealed"):
            load_authoritative_intervals(
                json.dumps({"schema": SESSION_EXPORT_SCHEMA,
                            "intervals": []}),
                sealed_receipt(sealed_export([])),
                expected_venue=VENUE,
                expected_account_fingerprint=ACCT,
                expected_symbol=SYM)

    def test_8_no_home_paths_or_tier_a_skips_in_public_tests(self):
        source = open(__file__).read()
        # this file's own scan strings excluded
        body = "\n".join(l for l in source.split("\n")
                         if "assert" not in l and "def test_8" not in l)
        assert "/home/" not in body
        assert "pytest.skip" not in body


# ================================================================== #
# C18 authority derivation                                           #
# ================================================================== #

class TestAuthorityFromSealedEvidence:

    def test_a_clean_bundle_derives_intervals(self):
        ivs = _weekly_intervals(3)
        exp = sealed_export(ivs)
        auth = load_authoritative_intervals(
            exp, sealed_receipt(exp), expected_venue=VENUE,
            expected_account_fingerprint=ACCT, expected_symbol=SYM)
        assert len(auth["intervals"]) == 3
        assert auth["collector_active"] is True
        assert all(iv.provenance == PROVENANCE_BROKER
                   for iv in auth["intervals"])

    def test_a_transplanted_receipt_refuses(self):
        exp_a = sealed_export(_weekly_intervals(1))
        exp_b = sealed_export(_weekly_intervals(2,
                              start="2024-06-07 20:00"))
        with pytest.raises(EvidenceError, match="transplanted"):
            load_authoritative_intervals(
                exp_a, sealed_receipt(exp_b), expected_venue=VENUE,
                expected_account_fingerprint=ACCT,
                expected_symbol=SYM)

    def test_foreign_binding_refuses(self):
        exp = sealed_export(_weekly_intervals(1))
        with pytest.raises(EvidenceError, match="bound"):
            load_authoritative_intervals(
                exp, sealed_receipt(exp), expected_venue="other",
                expected_account_fingerprint=ACCT,
                expected_symbol=SYM)

    def test_duplicate_identity_counts_once(self):
        iv = _weekly_intervals(1)
        exp = sealed_export(iv + iv)     # same physical interval twice
        auth = load_authoritative_intervals(
            exp, sealed_receipt(exp), expected_venue=VENUE,
            expected_account_fingerprint=ACCT, expected_symbol=SYM)
        assert len(auth["intervals"]) == 1

    def test_overlapping_intervals_refuse(self):
        ivs = [{"close_at": "2024-01-05 20:00:00+00:00",
                "reopen_at": "2024-01-09 00:00:00+00:00"},
               {"close_at": "2024-01-08 00:00:00+00:00",
                "reopen_at": "2024-01-10 00:00:00+00:00"}]
        exp = sealed_export(ivs)
        with pytest.raises(JoinContractError, match="overlap"):
            load_authoritative_intervals(
                exp, sealed_receipt(exp), expected_venue=VENUE,
                expected_account_fingerprint=ACCT,
                expected_symbol=SYM)

    def test_tampered_seal_refuses(self):
        exp = json.loads(sealed_export(_weekly_intervals(1)))
        exp["intervals"].append({
            "close_at": "2024-02-02 20:00:00+00:00",
            "reopen_at": "2024-02-05 00:00:00+00:00"})
        with pytest.raises(EvidenceError, match="seal digest"):
            load_authoritative_intervals(
                json.dumps(exp), sealed_receipt(sealed_export(
                    _weekly_intervals(1))),
                expected_venue=VENUE,
                expected_account_fingerprint=ACCT,
                expected_symbol=SYM)


# ================================================================== #
# C18 paired-week derivation and the acceptance battery              #
# ================================================================== #

class TestPairedWeeksAndAcceptance:

    def _bundle(self, n, *, pre=4, post=4):
        ivs = _weekly_intervals(n)
        exp = sealed_export(ivs)
        auth = load_authoritative_intervals(
            exp, sealed_receipt(exp), expected_venue=VENUE,
            expected_account_fingerprint=ACCT, expected_symbol=SYM)
        frame = _bars_covering(ivs, pre=pre, post=post)
        paired = derive_paired_weeks(
            auth["intervals"], frame, roles=ROLES, bar_hours=4.0,
            required_pre_bars=4, required_post_bars=4)
        return auth, paired, frame

    def test_29_weeks_is_inconclusive_deficit_1(self):
        auth, paired, _ = self._bundle(29)
        acc = count_paired_weeks(paired)
        assert acc["supported_paired_weeks"] == 29
        assert acc["exact_deficit"] == 1
        assert acc["status"] == "INCONCLUSIVE"
        v = readiness_verdict(authoritative=auth, paired=paired,
                              observed_units=0)
        assert v["state"] == "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING"

    def test_30_weeks_is_sufficient_readiness_not_grid(self):
        auth, paired, _ = self._bundle(30)
        v = readiness_verdict(authoritative=auth, paired=paired,
                              observed_units=0)
        assert v["state"] == \
            "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"
        assert v["economic_grid_authorized"] is False

    def test_removing_one_matched_bar_reduces_support(self):
        auth, paired, frame = self._bundle(30)
        assert count_paired_weeks(paired)["status"] == "SUFFICIENT"
        # strip the post bars of the last interval
        last = auth["intervals"][-1]
        trimmed = frame[pd.to_datetime(frame["DATE_TIME"], utc=True)
                        < last.reopen_at]
        paired2 = derive_paired_weeks(
            auth["intervals"], trimmed, roles=ROLES, bar_hours=4.0,
            required_pre_bars=4, required_post_bars=4)
        assert paired2["supported_paired_weeks"] == 29

    def test_no_receipt_no_authority(self):
        """Removing the activation receipt refuses derivation."""
        exp = sealed_export(_weekly_intervals(3))
        with pytest.raises(EvidenceError):
            load_authoritative_intervals(
                exp, json.dumps({"schema": "x"}),
                expected_venue=VENUE,
                expected_account_fingerprint=ACCT,
                expected_symbol=SYM)


# ================================================================== #
# Tier-A real data: fail-closed, no skip, logical roots              #
# ================================================================== #

def _tier_a_root():
    root = os.environ.get("WP4_TIER_A_ROOT")
    if not root:
        pytest.fail(
            "WP4_TIER_A_ROOT is not set — Tier-A evidence must FAIL "
            "closed, never skip (set it to the data checkout root)")
    return root


class TestTierAEthConclusion:

    def test_eth_h4_is_spot_history_not_mt5_session_authority(self):
        root = _tier_a_root()
        path = os.path.join(
            root, "predictor/examples/data/project3/"
            "ethusdt_4h_tech_stat_full_model_ready.csv")
        if not os.path.isfile(path):
            pytest.fail(f"Tier-A ETH dataset absent under the "
                        f"configured root ({path}) — fail closed")
        frame = pd.read_csv(path, usecols=["DATE_TIME", "OPEN",
                                           "CLOSE"])
        digest = hashlib.sha256(
            open(path, "rb").read()).hexdigest()
        pkg = build_readiness_package(
            frame, roles=ROLES, bar_hours=4.0,
            source_logical_id="predictor:project3/ethusdt_4h",
            source_digest=digest, authoritative=None, paired=None)
        # ZERO weekend-shaped broker closures; no authority
        assert pkg["inventory_summary"]["kind_counts"].get(
            "weekend_shaped_observed_gap", 0) == 0
        assert pkg["verdict"]["state"] == \
            "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
        assert pkg["verdict"]["paired_week_accounting"][
            "exact_deficit"] == WP4_MIN_PAIRED_WEEKS
        assert pkg["verdict"]["economic_grid_authorized"] is False


# ================================================================== #
# C22 strict inputs                                                  #
# ================================================================== #

class TestStrictInputs:

    def _frame(self):
        return pd.DataFrame({
            "DATE_TIME": pd.date_range("2024-01-01", periods=4,
                                       freq="4h"),
            "OPEN": [1.0, 2.0, 3.0, 4.0],
            "CLOSE": [1.0, 2.0, 3.0, 4.0]})

    @pytest.mark.parametrize("bad", [0, -1, 1.0 + 0j if False else
                                     float("nan"), float("inf"),
                                     True])
    def test_bad_bar_hours_refuse(self, bad):
        with pytest.raises(ReadinessError):
            inventory_observed_gaps(self._frame(), roles=ROLES,
                                    bar_hours=bad)

    def test_duplicate_timestamps_refuse(self):
        frame = self._frame()
        frame.loc[frame.index[1], "DATE_TIME"] = frame[
            "DATE_TIME"].iloc[0]
        with pytest.raises(ReadinessError, match="duplicate"):
            inventory_observed_gaps(frame, roles=ROLES,
                                    bar_hours=4.0)

    def test_non_numeric_ohlc_refuses(self):
        frame = self._frame()
        frame["CLOSE"] = ["a", "b", "c", "d"]
        with pytest.raises(ReadinessError, match="not numeric"):
            inventory_observed_gaps(frame, roles=ROLES,
                                    bar_hours=4.0)

    def test_missing_required_column_refuses(self):
        frame = self._frame().drop(columns=["OPEN"])
        with pytest.raises(ReadinessError, match="missing required"):
            inventory_observed_gaps(frame, roles=ROLES,
                                    bar_hours=4.0)


class TestSanitizationScan:

    def test_no_private_topology_in_public_code(self):
        import tools.wp4_session_readiness as mod
        src = open(mod.__file__).read()
        assert "/home/" not in src
        assert "harveybc" not in src
