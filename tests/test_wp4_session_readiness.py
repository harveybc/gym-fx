"""C23-C27 battery: authority is born only from evidence signed under
an order-fixed Ed25519 key (a self-consistent bundle refuses), there
is one derivation path (no caller authority dict/count), pairing is
LOCAL and causal (remote bars never certify, a bar inside a closure
refuses), the package binds trust/activation/intervals/pairing/source,
and schemas are strict. The five audit counterexamples are dead."""
from __future__ import annotations

import binascii
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey)

from tools.wp4_session_readiness import (
    ACTIVATION_RECEIPT_SCHEMA, ColumnRoleContract, ETH_SPOT_CONCLUSION,
    EvidenceError, JoinContractError, OPERATOR_EXCEPTION_SCHEMA,
    ReadinessError, SESSION_EXPORT_SCHEMA, TrustContract, TrustError,
    UNAVAILABLE, WP4_MIN_PAIRED_WEEKS, build_readiness_package,
    canonical_bytes, classify_observed_gap, inventory_observed_gaps,
    opening_gap_return, post_reopen_realized_vol, quote_continuity,
    sha256_hex, strict_json_loads)

VENUE, ACCT, SYM = "mt5_demo", "fp-1", "ETHUSD"
EXPORTER, PARSER, CODE = "exp-1", "par-1", "code-1"
NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)
ROLES = ColumnRoleContract(datetime_col="DATE_TIME", open_col="OPEN",
                           close_col="CLOSE")


@pytest.fixture(scope="module")
def key():
    return Ed25519PrivateKey.generate()


@pytest.fixture(scope="module")
def trust(key):
    pub = key.public_key().public_bytes_raw()
    return TrustContract(
        public_key_hex=binascii.hexlify(pub).decode(),
        venue=VENUE, account_fingerprint=ACCT, symbol=SYM,
        exporter_identity=EXPORTER, parser_identity=PARSER,
        code_identity=CODE)


def _sign(body, key):
    sig = key.sign(canonical_bytes(body))
    return json.dumps({**body, "signature":
                       binascii.hexlify(sig).decode()})


def _weekly_intervals(n, *, start="2024-01-05 20:00"):
    base = pd.Timestamp(start, tz="UTC")
    return [{"close_at": (base + pd.Timedelta(weeks=i)).isoformat(),
             "reopen_at": (base + pd.Timedelta(weeks=i, hours=52))
             .isoformat()} for i in range(n)]


def _export(intervals, key, *, acq=None):
    acq = acq or ["2024-01-01T00:00:00Z", "2024-12-31T00:00:00Z"]
    return _sign({"schema": SESSION_EXPORT_SCHEMA, "venue": VENUE,
                  "account_fingerprint": ACCT, "symbol": SYM,
                  "exporter_identity": EXPORTER,
                  "parser_identity": PARSER, "code_identity": CODE,
                  "acquisition_range": acq,
                  "intervals": intervals}, key)


def _receipt(export_json, key):
    body = {k: v for k, v in json.loads(export_json).items()
            if k != "signature"}
    export_digest = sha256_hex(canonical_bytes(body))
    return _sign({"schema": ACTIVATION_RECEIPT_SCHEMA, "venue": VENUE,
                  "account_fingerprint": ACCT, "symbol": SYM,
                  "exporter_identity": EXPORTER,
                  "parser_identity": PARSER, "code_identity": CODE,
                  "activation_identity": "act-1",
                  "activated_at": "2024-02-01T00:00:00Z",
                  "bound_export_sha256": export_digest}, key)


def _local_bars(intervals, *, pre=4, post=4, bar_hours=4.0,
                remote_only=False):
    """Adjacent local pre/post bars around EVERY interval (or, when
    remote_only, only around the first close and last reopen)."""
    bar = pd.Timedelta(hours=bar_hours)
    stamps = set()
    targets = ([intervals[0]] if remote_only else intervals)
    for iv in targets:
        c = pd.Timestamp(iv["close_at"]); r = pd.Timestamp(
            intervals[-1]["reopen_at"] if remote_only
            else iv["reopen_at"])
        for k in range(pre, 0, -1):
            stamps.add(c - k * bar)
        for k in range(post):
            stamps.add(r + k * bar)
    stamps = sorted(stamps)
    n = len(stamps)
    return pd.DataFrame({
        "DATE_TIME": [s.tz_convert("UTC").tz_localize(None)
                      for s in stamps],
        "OPEN": 100.0 + np.arange(n),
        "CLOSE": 100.5 + np.arange(n)})


def _pkg(frame, key=None, trust_c=None, export=None, receipt=None,
         **kw):
    return build_readiness_package(
        source_bytes=b"src-bytes", source_logical_id="fx:eth",
        roles=ROLES, bar_hours=4.0, frame=frame,
        session_export=export, activation_receipt=receipt,
        trust=trust_c, now=NOW, **kw)


# ================================================================== #
# the five frozen counterexamples                                    #
# ================================================================== #

class TestFrozenCounterexamples:

    def test_1_self_signed_forgery_refuses(self, trust):
        """A bundle 'sealed' by an attacker's own key cannot verify
        under the order-fixed public key."""
        attacker = Ed25519PrivateKey.generate()
        exp = _export(_weekly_intervals(30), attacker)
        with pytest.raises(TrustError, match="signature"):
            _pkg(_local_bars(_weekly_intervals(30)),
                 trust_c=trust, export=exp,
                 receipt=_receipt(exp, attacker))

    def test_2_no_caller_authority_dict_or_count(self):
        """The single path takes evidence bytes + trust, never an
        authoritative dict, paired dict or collector_active flag."""
        import inspect
        import tools.wp4_session_readiness as mod
        params = inspect.signature(
            mod.build_readiness_package).parameters
        for banned in ("authoritative", "paired", "collector_active"):
            assert banned not in params
        # with no evidence, the verdict is non-authoritative
        pkg = _pkg(_local_bars(_weekly_intervals(1)))
        assert pkg["verdict"]["state"] == \
            "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"

    def test_3_remote_bars_never_certify(self, key, trust):
        """Eight remote bars cannot support thirty local windows."""
        ivs = _weekly_intervals(30)
        exp = _export(ivs, key)
        pkg = _pkg(_local_bars(ivs, remote_only=True), trust_c=trust,
                   export=exp, receipt=_receipt(exp, key))
        acc = pkg["verdict"]["paired_week_accounting"]
        assert acc["supported_paired_weeks"] < 30
        assert acc["status"] == "INCONCLUSIVE"

    def test_4_distinct_authority_changes_digest(self, key, trust):
        """Two authoritative populations of equal cardinality over
        the SAME observed frame produce different package digests."""
        frame = _local_bars(_weekly_intervals(2) +
                            _weekly_intervals(2, start="2024-06-07 20:00"))
        a = _weekly_intervals(2)
        b = _weekly_intervals(2, start="2024-06-07 20:00")
        ea, eb = _export(a, key), _export(b, key)
        pa = _pkg(frame, trust_c=trust, export=ea,
                  receipt=_receipt(ea, key), required_pre_bars=1,
                  required_post_bars=1)
        pb = _pkg(frame, trust_c=trust, export=eb,
                  receipt=_receipt(eb, key), required_pre_bars=1,
                  required_post_bars=1)
        assert pa["authoritative"]["authoritative_pairing_digest"] \
            != pb["authoritative"]["authoritative_pairing_digest"]
        assert pa["digest"] != pb["digest"]

    def test_5_absolute_path_and_bad_digest_refuse(self):
        with pytest.raises(ReadinessError, match="logical id"):
            build_readiness_package(
                source_bytes=b"x",
                source_logical_id="/home/secret/eth.csv",
                roles=ROLES, bar_hours=4.0,
                frame=_local_bars(_weekly_intervals(1)))
        with pytest.raises(ReadinessError, match="BYTES"):
            build_readiness_package(
                source_bytes="not-a-digest",
                source_logical_id="fx:eth", roles=ROLES,
                bar_hours=4.0,
                frame=_local_bars(_weekly_intervals(1)))


# ================================================================== #
# C23 trust root                                                     #
# ================================================================== #

class TestTrustRoot:

    def test_a_signed_bundle_activates(self, key, trust):
        ivs = _weekly_intervals(3)
        exp = _export(ivs, key)
        pkg = _pkg(_local_bars(ivs), trust_c=trust, export=exp,
                   receipt=_receipt(exp, key))
        assert pkg["verdict"]["collector_active"] is True

    @pytest.mark.parametrize("swap", ["export", "receipt"])
    def test_substituting_a_signer_refuses(self, key, trust, swap):
        other = Ed25519PrivateKey.generate()
        ivs = _weekly_intervals(3)
        good = _export(ivs, key)
        exp = _export(ivs, other) if swap == "export" else good
        rcpt = (_receipt(good, other) if swap == "receipt"
                else _receipt(good, key))
        with pytest.raises(TrustError):
            _pkg(_local_bars(ivs), trust_c=trust, export=exp,
                 receipt=rcpt)

    def test_wrong_exporter_identity_refuses(self, key):
        bad_trust = TrustContract(
            public_key_hex=binascii.hexlify(
                key.public_key().public_bytes_raw()).decode(),
            venue=VENUE, account_fingerprint=ACCT, symbol=SYM,
            exporter_identity="OTHER", parser_identity=PARSER,
            code_identity=CODE)
        ivs = _weekly_intervals(3)
        exp = _export(ivs, key)
        with pytest.raises(TrustError, match="exporter"):
            _pkg(_local_bars(ivs), trust_c=bad_trust, export=exp,
                 receipt=_receipt(exp, key))

    def test_transplanted_receipt_refuses(self, key, trust):
        a = _export(_weekly_intervals(1), key)
        b = _export(_weekly_intervals(2, start="2024-06-07 20:00"),
                    key)
        with pytest.raises(TrustError, match="transplanted"):
            _pkg(_local_bars(_weekly_intervals(1)), trust_c=trust,
                 export=a, receipt=_receipt(b, key))

    def test_interval_outside_acquisition_refuses(self, key, trust):
        ivs = _weekly_intervals(1, start="2025-06-07 20:00")
        exp = _export(ivs, key)   # acq default is 2024
        with pytest.raises(EvidenceError, match="acquisition"):
            _pkg(_local_bars(ivs), trust_c=trust, export=exp,
                 receipt=_receipt(exp, key))


# ================================================================== #
# C25 local causal pairing + acceptance battery                      #
# ================================================================== #

class TestLocalPairingAndAcceptance:

    def _run(self, n, key, trust, *, pre=4, post=4, frame=None):
        ivs = _weekly_intervals(n)
        exp = _export(ivs, key)
        return _pkg(frame if frame is not None else _local_bars(ivs),
                    trust_c=trust, export=exp,
                    receipt=_receipt(exp, key), required_pre_bars=pre,
                    required_post_bars=post)

    def test_29_local_windows_deficit_1(self, key, trust):
        # 30 intervals but the last has no local post window
        ivs = _weekly_intervals(30)
        frame = _local_bars(ivs)
        last = pd.Timestamp(ivs[-1]["reopen_at"]).tz_localize(None)
        frame = frame[pd.to_datetime(frame["DATE_TIME"]) < last]
        pkg = self._run(30, key, trust, frame=frame)
        acc = pkg["verdict"]["paired_week_accounting"]
        assert acc["supported_paired_weeks"] == 29
        assert acc["exact_deficit"] == 1
        assert pkg["verdict"]["state"] == \
            "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING"

    def test_30_local_windows_sufficient_not_grid(self, key, trust):
        pkg = self._run(30, key, trust)
        assert pkg["verdict"]["state"] == \
            "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"
        assert pkg["verdict"]["economic_grid_authorized"] is False

    def test_removing_a_local_bar_reduces_support(self, key, trust):
        ivs = _weekly_intervals(30)
        frame = _local_bars(ivs)
        # drop one pre-close bar of the last interval
        drop_at = (pd.Timestamp(ivs[-1]["close_at"]) -
                   pd.Timedelta(hours=4)).tz_localize(None)
        frame = frame[pd.to_datetime(frame["DATE_TIME"]) != drop_at]
        pkg = self._run(30, key, trust, frame=frame)
        assert pkg["verdict"]["paired_week_accounting"][
            "supported_paired_weeks"] == 29

    def test_adding_remote_bars_does_not_restore(self, key, trust):
        ivs = _weekly_intervals(30)
        frame = _local_bars(ivs)
        drop_at = (pd.Timestamp(ivs[-1]["close_at"]) -
                   pd.Timedelta(hours=4)).tz_localize(None)
        frame = frame[pd.to_datetime(frame["DATE_TIME"]) != drop_at]
        # add distant bars far from every interval and unique to
        # the grid (2023, before the whole acquisition window)
        extra = pd.DataFrame({
            "DATE_TIME": pd.date_range("2023-06-01", periods=4,
                                       freq="4h"),
            "OPEN": [1.0, 2.0, 3.0, 4.0],
            "CLOSE": [1.0, 2.0, 3.0, 4.0]})
        frame = pd.concat([extra, frame]).drop_duplicates(
            "DATE_TIME").reset_index(drop=True)
        pkg = self._run(30, key, trust, frame=frame)
        assert pkg["verdict"]["paired_week_accounting"][
            "supported_paired_weeks"] == 29

    def test_a_bar_inside_a_closure_refuses(self, key, trust):
        ivs = _weekly_intervals(1)
        frame = _local_bars(ivs)
        inside = (pd.Timestamp(ivs[0]["close_at"]) +
                  pd.Timedelta(hours=8)).tz_localize(None)
        frame = pd.concat([frame, pd.DataFrame({
            "DATE_TIME": [inside], "OPEN": [1.0], "CLOSE": [1.0]})]
            ).reset_index(drop=True)
        with pytest.raises(JoinContractError, match="inside"):
            self._run(1, key, trust, frame=frame)


# ================================================================== #
# C19 metric truth + C27 strict schemas                              #
# ================================================================== #

class TestMetricsAndSchemas:

    def test_opening_gap_reads_open(self):
        assert opening_gap_return(150.0, 100.0) == pytest.approx(0.5)

    def test_volume_is_never_volatility(self):
        import inspect
        import tools.wp4_session_readiness as mod
        assert "vol_col" not in inspect.signature(
            mod.inventory_observed_gaps).parameters

    def test_quote_continuity_needs_ordered_sufficient(self):
        assert quote_continuity(
            None, expected_spacing_seconds=None)["value"] == \
            UNAVAILABLE
        # out-of-order cannot be true
        t = ["2024-01-01T00:00:10Z", "2024-01-01T00:00:00Z"]
        assert quote_continuity(
            t, expected_spacing_seconds=30)["value"] == UNAVAILABLE

    def test_tuesday_56h_is_never_weekend(self):
        pre = pd.Timestamp("2024-01-09 08:00", tz="UTC")
        assert classify_observed_gap(
            pre, pre + pd.Timedelta(hours=56)) == \
            "midweek_outage_shaped"

    def test_duplicate_json_keys_refuse(self):
        with pytest.raises(ReadinessError, match="duplicate JSON"):
            strict_json_loads('{"a": 1, "a": 2}')

    def test_non_finite_json_refuses(self):
        with pytest.raises(ReadinessError, match="non-finite"):
            strict_json_loads('{"a": NaN}')

    @pytest.mark.parametrize("bad", [0, -1, float("nan"),
                                     float("inf"), True])
    def test_bad_bar_hours_refuse(self, bad):
        with pytest.raises(ReadinessError):
            inventory_observed_gaps(
                pd.DataFrame({"DATE_TIME": pd.date_range(
                    "2024-01-01", periods=2, freq="4h"),
                    "OPEN": [1.0, 2.0], "CLOSE": [1.0, 2.0]}),
                roles=ROLES, bar_hours=bad)

    def test_non_positive_price_refuses(self):
        with pytest.raises(ReadinessError, match="non-positive"):
            inventory_observed_gaps(
                pd.DataFrame({"DATE_TIME": pd.date_range(
                    "2024-01-01", periods=2, freq="4h"),
                    "OPEN": [1.0, 2.0], "CLOSE": [0.0, 2.0]}),
                roles=ROLES, bar_hours=4.0)


# ================================================================== #
# Tier-A: fail closed, no skip                                       #
# ================================================================== #

def _tier_a_root():
    root = os.environ.get("WP4_TIER_A_ROOT")
    if not root:
        pytest.fail("WP4_TIER_A_ROOT is not set — Tier-A must FAIL "
                    "closed, never skip")
    return root


class TestTierAEthConclusion:

    def test_eth_h4_is_spot_history_not_mt5_session_authority(self):
        root = _tier_a_root()
        path = os.path.join(
            root, "predictor/examples/data/project3/"
            "ethusdt_4h_tech_stat_full_model_ready.csv")
        if not os.path.isfile(path):
            pytest.fail(f"Tier-A ETH dataset absent ({path})")
        raw = open(path, "rb").read()
        frame = pd.read_csv(path, usecols=["DATE_TIME", "OPEN",
                                           "CLOSE"])
        pkg = build_readiness_package(
            source_bytes=raw,
            source_logical_id="predictor:project3/ethusdt_4h",
            roles=ROLES, bar_hours=4.0, frame=frame, now=NOW)
        assert pkg["inventory_summary"]["kind_counts"].get(
            "weekend_shaped_observed_gap", 0) == 0
        assert pkg["verdict"]["state"] == \
            "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
        assert pkg["verdict"]["paired_week_accounting"][
            "exact_deficit"] == WP4_MIN_PAIRED_WEEKS
        assert pkg["eth_conclusion_when_spot"] == \
            ETH_SPOT_CONCLUSION


class TestSanitization:

    def test_no_private_topology_in_public_code(self):
        import tools.wp4_session_readiness as mod
        src = open(mod.__file__).read()
        assert "/home/" not in src
        assert "harveybc" not in src
