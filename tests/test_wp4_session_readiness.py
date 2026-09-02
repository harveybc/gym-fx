"""C28-C32 battery: the trust root is FIXED BY THE EXECUTING PATH
(the committed manifest ships NOT_PROVISIONED, so the production API
never activates); a caller cannot inject trust into production;
source bytes and frame are one population; the as-of contract refuses
future evidence; the closure is [close_at, reopen_at); and boolean
spacing cannot make continuity true. The five audit counterexamples
are dead. Provisioned-authority tests use an ISOLATED fixture
manifest through the explicit TEST_ONLY door."""
from __future__ import annotations

import binascii
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey)

import tools.wp4_session_readiness as mod
from tools.wp4_session_readiness import (
    ACTIVATION_RECEIPT_SCHEMA, ColumnRoleContract, ETH_SPOT_CONCLUSION,
    EvidenceError, JoinContractError, OPERATOR_EXCEPTION_SCHEMA,
    ReadinessError, SESSION_EXPORT_SCHEMA, STATUS_PROVISIONED,
    TRUST_MANIFEST_SCHEMA, TrustError, UNAVAILABLE,
    WP4_MIN_PAIRED_WEEKS, VerifiedSource, build_readiness_package,
    build_readiness_package_with_trust_TEST_ONLY, canonical_bytes,
    classify_observed_gap, inventory_observed_gaps,
    load_pinned_production_trust, load_trust_manifest_TEST_ONLY,
    opening_gap_return, quote_continuity, sha256_hex,
    strict_json_loads)

VENUE, ACCT, SYM = "mt5_demo", "fp-1", "ETHUSD"
EXP_D = "a" * 64
PAR_D = "b" * 64
COD_D = "c" * 64
AS_OF = datetime(2024, 8, 1, tzinfo=timezone.utc)
OBSERVED_THROUGH = "2024-07-30T00:00:00Z"
EXPORTED_AT = "2024-07-31T00:00:00Z"
ROLES = ColumnRoleContract(datetime_col="DATE_TIME", open_col="OPEN",
                           close_col="CLOSE")


@pytest.fixture(scope="module")
def key():
    return Ed25519PrivateKey.generate()


@pytest.fixture(scope="module")
def provisioned(tmp_path_factory, key):
    """An ISOLATED provisioned manifest fixture and its own pin — the
    production pin is never touched."""
    pub = binascii.hexlify(
        key.public_key().public_bytes_raw()).decode()
    manifest = {"schema": TRUST_MANIFEST_SCHEMA,
                "status": STATUS_PROVISIONED, "public_key_hex": pub,
                "venue": VENUE, "account_fingerprint": ACCT,
                "symbol": SYM, "exporter_code_digest": EXP_D,
                "parser_code_digest": PAR_D,
                "code_identity_digest": COD_D,
                "max_activation_age_days": 3650.0,
                "approving_order_reference": "test-fixture",
                "approving_order_digest": "d" * 64}
    digest = sha256_hex(canonical_bytes(manifest))
    path = tmp_path_factory.mktemp("trust") / "provisioned.json"
    path.write_text(json.dumps({**manifest, "manifest_digest":
                                digest}))
    return load_trust_manifest_TEST_ONLY(path, expected_digest=digest)


def _sign(body, key):
    sig = binascii.hexlify(key.sign(canonical_bytes(body))).decode()
    return json.dumps({**body, "signature": sig})


def _weekly(n, *, start="2024-01-05 20:00"):
    b = pd.Timestamp(start, tz="UTC")
    return [{"close_at": (b + pd.Timedelta(weeks=i)).isoformat(),
             "reopen_at": (b + pd.Timedelta(weeks=i, hours=52))
             .isoformat()} for i in range(n)]


def _export(intervals, key, *, acq=None, observed=OBSERVED_THROUGH,
            exported=EXPORTED_AT):
    return _sign({"schema": SESSION_EXPORT_SCHEMA, "venue": VENUE,
                  "account_fingerprint": ACCT, "symbol": SYM,
                  "exporter_identity": EXP_D, "parser_identity": PAR_D,
                  "code_identity": COD_D,
                  "acquisition_range": acq or
                  ["2024-01-01T00:00:00Z", "2024-07-01T00:00:00Z"],
                  "exported_at": exported,
                  "observed_through": observed,
                  "intervals": intervals}, key)


def _receipt(export_json, key, *, activated="2024-02-01T00:00:00Z"):
    body = {k: v for k, v in json.loads(export_json).items()
            if k != "signature"}
    return _sign({"schema": ACTIVATION_RECEIPT_SCHEMA, "venue": VENUE,
                  "account_fingerprint": ACCT, "symbol": SYM,
                  "exporter_identity": EXP_D, "parser_identity": PAR_D,
                  "code_identity": COD_D, "activation_identity":
                  "act-1", "activated_at": activated,
                  "bound_export_sha256": sha256_hex(
                      canonical_bytes(body))}, key)


def _bars(intervals, *, pre=4, post=4, remote_only=False):
    bar = pd.Timedelta(hours=4)
    stamps = set()
    targets = [intervals[0]] if remote_only else intervals
    for iv in targets:
        c = pd.Timestamp(iv["close_at"])
        r = pd.Timestamp(intervals[-1]["reopen_at"] if remote_only
                         else iv["reopen_at"])
        for k in range(pre, 0, -1):
            stamps.add(c - k * bar)
        for k in range(post):
            stamps.add(r + k * bar)
    stamps = sorted(stamps)
    n = len(stamps)
    frame = pd.DataFrame({
        "DATE_TIME": [s.tz_convert("UTC").tz_localize(None)
                      for s in stamps],
        "OPEN": 100.0 + np.arange(n), "CLOSE": 100.5 + np.arange(n)})
    return frame


def _source(frame, logical="fx:eth"):
    raw = frame.to_csv(index=False).encode()
    return VerifiedSource.from_csv_bytes(raw, roles=ROLES,
                                         source_logical_id=logical)


def _auth_pkg(frame, trust, key, intervals, *, pre=4, post=4,
              as_of=AS_OF, **kw):
    exp = _export(intervals, key, **kw)
    return build_readiness_package_with_trust_TEST_ONLY(
        _source(frame), trust, bar_hours=4.0, evaluation_as_of=as_of,
        session_export=exp, activation_receipt=_receipt(exp, key),
        required_pre_bars=pre, required_post_bars=post)


# ================================================================== #
# the five frozen counterexamples                                    #
# ================================================================== #

class TestFrozenCounterexamples:

    def test_1_production_never_accepts_caller_trust(self, key):
        """CRITICAL-1 dead: production build_readiness_package has no
        trust parameter and loads the pinned NOT_PROVISIONED
        manifest; an attacker key + bundle cannot activate."""
        assert "trust" not in inspect.signature(
            build_readiness_package).parameters
        ivs = _weekly(30)
        exp = _export(ivs, key)
        pkg = build_readiness_package(
            _source(_bars(ivs)), bar_hours=4.0, evaluation_as_of=AS_OF,
            session_export=exp, activation_receipt=_receipt(exp, key))
        assert pkg["verdict"]["state"] == \
            "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
        assert pkg["verdict"]["collector_active"] is False

    def test_2_frame_is_the_hashed_bytes(self):
        """CRITICAL-2 dead: the frame is parsed FROM the bytes, so
        distinct rows give distinct source digests."""
        with pytest.raises(TypeError):
            VerifiedSource(source_bytes=b"x", source_digest="d",
                           source_logical_id="fx", roles=ROLES,
                           frame=pd.DataFrame(), extra=1)
        a = _source(pd.DataFrame({
            "DATE_TIME": pd.date_range("2024-01-01", periods=2,
                                       freq="4h"),
            "OPEN": [1.0, 2.0], "CLOSE": [1.0, 2.0]}))
        b = _source(pd.DataFrame({
            "DATE_TIME": pd.date_range("2024-01-01", periods=2,
                                       freq="4h"),
            "OPEN": [999.0, 998.0], "CLOSE": [999.0, 998.0]}))
        assert a.source_digest != b.source_digest

    def test_3_future_intervals_refuse(self, provisioned, key):
        """CRITICAL-3 dead: an interval reopening past
        observed_through refuses."""
        ivs = _weekly(30)   # runs to late July, past observed 07-30
        with pytest.raises(TrustError, match="future|observed"):
            _auth_pkg(_bars(ivs), provisioned, key, ivs,
                      observed="2024-06-01T00:00:00Z",
                      exported="2024-06-02T00:00:00Z",
                      as_of=datetime(2024, 6, 3, tzinfo=timezone.utc))

    def test_4_bar_at_close_refuses(self, provisioned, key):
        """HIGH-1 dead: a bar exactly at close_at is inside the
        [close_at, reopen_at) closure."""
        ivs = _weekly(1)
        frame = _bars(ivs, pre=1, post=1)
        at_close = pd.Timestamp(ivs[0]["close_at"]).tz_localize(None)
        frame = pd.concat([frame, pd.DataFrame({
            "DATE_TIME": [at_close], "OPEN": [1.0], "CLOSE": [1.0]})]
            ).drop_duplicates("DATE_TIME").reset_index(drop=True)
        with pytest.raises(JoinContractError, match="inside"):
            _auth_pkg(frame, provisioned, key, ivs, pre=1, post=1)

    def test_5_boolean_spacing_is_unavailable(self):
        """HIGH-2 dead."""
        qc = quote_continuity(
            ["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"],
            expected_spacing_seconds=True)
        assert qc["value"] == UNAVAILABLE


# ================================================================== #
# C28 pinned trust                                                   #
# ================================================================== #

class TestPinnedTrust:

    def test_production_manifest_is_not_provisioned(self):
        trust = load_pinned_production_trust()
        assert trust.status == "NOT_PROVISIONED_NON_AUTHORIZING"
        assert trust.authorizing is False

    def test_no_production_param_injects_a_key(self):
        params = inspect.signature(
            build_readiness_package).parameters
        for banned in ("trust", "public_key", "public_key_hex",
                       "trust_manifest"):
            assert banned not in params

    def test_a_redigested_manifest_refuses(self, tmp_path, key):
        pub = binascii.hexlify(
            key.public_key().public_bytes_raw()).decode()
        man = {"schema": TRUST_MANIFEST_SCHEMA,
               "status": STATUS_PROVISIONED, "public_key_hex": pub,
               "venue": VENUE, "account_fingerprint": ACCT,
               "symbol": SYM, "exporter_code_digest": EXP_D,
               "parser_code_digest": PAR_D,
               "code_identity_digest": COD_D,
               "max_activation_age_days": 3650.0,
               "approving_order_reference": "x",
               "approving_order_digest": "d" * 64}
        digest = sha256_hex(canonical_bytes(man))
        path = tmp_path / "m.json"
        path.write_text(json.dumps({**man, "manifest_digest": digest}))
        # asking for a DIFFERENT expected digest refuses
        with pytest.raises(TrustError, match="pinned"):
            load_trust_manifest_TEST_ONLY(path,
                                          expected_digest="e" * 64)

    def test_bool_max_age_refuses(self, tmp_path, key):
        pub = binascii.hexlify(
            key.public_key().public_bytes_raw()).decode()
        man = {"schema": TRUST_MANIFEST_SCHEMA,
               "status": STATUS_PROVISIONED, "public_key_hex": pub,
               "venue": VENUE, "account_fingerprint": ACCT,
               "symbol": SYM, "exporter_code_digest": EXP_D,
               "parser_code_digest": PAR_D,
               "code_identity_digest": COD_D,
               "max_activation_age_days": True,
               "approving_order_reference": "x",
               "approving_order_digest": "d" * 64}
        digest = sha256_hex(canonical_bytes(man))
        path = tmp_path / "m.json"
        path.write_text(json.dumps({**man, "manifest_digest": digest}))
        with pytest.raises(ReadinessError, match="real positive"):
            load_trust_manifest_TEST_ONLY(path,
                                          expected_digest=digest)


# ================================================================== #
# C28/C30 authority and the as-of contract                           #
# ================================================================== #

class TestAuthorityAndAsOf:

    def test_a_provisioned_bundle_activates(self, provisioned, key):
        ivs = _weekly(3)
        pkg = _auth_pkg(_bars(ivs), provisioned, key, ivs)
        assert pkg["verdict"]["collector_active"] is True

    def test_substituting_signer_and_trust_together_refuses(
            self, tmp_path):
        """The real attack: attacker key AND attacker manifest. The
        attacker manifest is not the fixture's pinned digest."""
        attacker = Ed25519PrivateKey.generate()
        pub = binascii.hexlify(
            attacker.public_key().public_bytes_raw()).decode()
        man = {"schema": TRUST_MANIFEST_SCHEMA,
               "status": STATUS_PROVISIONED, "public_key_hex": pub,
               "venue": VENUE, "account_fingerprint": ACCT,
               "symbol": SYM, "exporter_code_digest": EXP_D,
               "parser_code_digest": PAR_D,
               "code_identity_digest": COD_D,
               "max_activation_age_days": 3650.0,
               "approving_order_reference": "attacker",
               "approving_order_digest": "d" * 64}
        digest = sha256_hex(canonical_bytes(man))
        path = tmp_path / "atk.json"
        path.write_text(json.dumps({**man, "manifest_digest": digest}))
        # loading it requires the caller to already know its digest;
        # the PRODUCTION path never loads a caller path — it only
        # loads the code-pinned one. Prove production stays inert:
        ivs = _weekly(30)
        exp = _export(ivs, attacker)
        pkg = build_readiness_package(
            _source(_bars(ivs)), bar_hours=4.0, evaluation_as_of=AS_OF,
            session_export=exp,
            activation_receipt=_receipt(exp, attacker))
        assert pkg["verdict"]["collector_active"] is False

    def test_activated_after_observed_refuses(self, provisioned, key):
        ivs = _weekly(1)
        with pytest.raises(TrustError, match="as-of"):
            _auth_pkg(_bars(ivs), provisioned, key, ivs,
                      observed="2024-01-01T00:00:00Z",
                      exported="2024-07-31T00:00:00Z")

    def test_bar_past_as_of_refuses(self, provisioned, key):
        ivs = _weekly(1)
        frame = _bars(ivs)
        # push a bar past the as-of horizon
        far = pd.Timestamp("2024-09-01").tz_localize(None)
        frame = pd.concat([frame, pd.DataFrame({
            "DATE_TIME": [far], "OPEN": [1.0], "CLOSE": [1.0]})]
            ).reset_index(drop=True)
        with pytest.raises(TrustError, match="as_of|future"):
            _auth_pkg(frame, provisioned, key, ivs, pre=1, post=1)


# ================================================================== #
# C31 interval semantics + C25 local pairing acceptance              #
# ================================================================== #

class TestLocalPairing:

    def test_reopen_bar_is_valid_post(self, provisioned, key):
        ivs = _weekly(1)
        pkg = _auth_pkg(_bars(ivs, pre=1, post=1), provisioned, key,
                        ivs, pre=1, post=1)
        assert pkg["verdict"]["paired_week_accounting"][
            "supported_paired_weeks"] == 1

    def test_29_local_windows_deficit_1(self, provisioned, key):
        """30 authoritative intervals within range, but the last has
        no local post window -> 29 supported, deficit 1."""
        ivs = _weekly(30, start="2024-01-05 20:00")
        frame = _bars(ivs)
        last_reopen = pd.Timestamp(
            ivs[-1]["reopen_at"]).tz_localize(None)
        frame = frame[pd.to_datetime(frame["DATE_TIME"])
                      < last_reopen]
        exp = _export(ivs, key, observed="2024-07-29T00:00:00Z",
                      exported="2024-07-30T00:00:00Z",
                      acq=["2024-01-01T00:00:00Z",
                           "2024-07-29T00:00:00Z"])
        as_of = datetime(2024, 7, 31, tzinfo=timezone.utc)
        pkg = build_readiness_package_with_trust_TEST_ONLY(
            _source(frame), provisioned, bar_hours=4.0,
            evaluation_as_of=as_of, session_export=exp,
            activation_receipt=_receipt(exp, key),
            required_pre_bars=4, required_post_bars=4)
        acc = pkg["verdict"]["paired_week_accounting"]
        assert acc["supported_paired_weeks"] == 29
        assert acc["exact_deficit"] == 1
        assert pkg["verdict"]["state"] == \
            "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING"

    def test_30_local_windows_sufficient(self, provisioned, key):
        ivs = _weekly(30, start="2024-01-05 20:00")
        exp = _export(ivs, key, observed="2024-07-29T00:00:00Z",
                      exported="2024-07-30T00:00:00Z",
                      acq=["2024-01-01T00:00:00Z",
                           "2024-07-29T00:00:00Z"])
        as_of = datetime(2024, 7, 31, tzinfo=timezone.utc)
        pkg = build_readiness_package_with_trust_TEST_ONLY(
            _source(_bars(ivs)), provisioned, bar_hours=4.0,
            evaluation_as_of=as_of, session_export=exp,
            activation_receipt=_receipt(exp, key),
            required_pre_bars=4, required_post_bars=4)
        assert pkg["verdict"]["state"] == \
            "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"
        assert pkg["verdict"]["economic_grid_authorized"] is False

    def test_remote_bars_never_certify(self, provisioned, key):
        ivs = _weekly(12)
        pkg = _auth_pkg(_bars(ivs, remote_only=True), provisioned,
                        key, ivs)
        assert pkg["verdict"]["paired_week_accounting"][
            "supported_paired_weeks"] < 12

    def test_removing_a_local_bar_reduces(self, provisioned, key):
        ivs = _weekly(3)
        frame = _bars(ivs)
        drop = (pd.Timestamp(ivs[-1]["close_at"]) -
                pd.Timedelta(hours=4)).tz_localize(None)
        frame = frame[pd.to_datetime(frame["DATE_TIME"]) != drop]
        pkg = _auth_pkg(frame, provisioned, key, ivs)
        assert pkg["verdict"]["paired_week_accounting"][
            "supported_paired_weeks"] == 2


# ================================================================== #
# C32 strict remaining boundaries + operator exceptions              #
# ================================================================== #

class TestStrictBoundaries:

    def test_opening_gap_reads_open(self):
        assert opening_gap_return(150.0, 100.0) == pytest.approx(0.5)

    def test_tuesday_56h_never_weekend(self):
        pre = pd.Timestamp("2024-01-09 08:00", tz="UTC")
        assert classify_observed_gap(
            pre, pre + pd.Timedelta(hours=56)) == \
            "midweek_outage_shaped"

    @pytest.mark.parametrize("bad", [True, "5", 0, float("nan"),
                                     float("inf")])
    def test_bad_spacing_is_unavailable(self, bad):
        qc = quote_continuity(
            ["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"],
            expected_spacing_seconds=bad)
        assert qc["value"] == UNAVAILABLE

    def test_duplicate_json_keys_refuse(self):
        with pytest.raises(ReadinessError, match="duplicate"):
            strict_json_loads('{"a": 1, "a": 2}')

    def test_inverted_operator_exception_refuses(self, provisioned,
                                                 key):
        ivs = _weekly(1)
        exc = _sign({"schema": OPERATOR_EXCEPTION_SCHEMA,
                     "venue": VENUE, "account_fingerprint": ACCT,
                     "symbol": SYM, "exporter_identity": EXP_D,
                     "parser_identity": PAR_D, "code_identity": COD_D,
                     "named_intervals": [{
                         "close_at": "2024-03-10T00:00:00Z",
                         "reopen_at": "2024-03-09T00:00:00Z"}]}, key)
        exp = _export(ivs, key)
        with pytest.raises(EvidenceError, match="precede"):
            build_readiness_package_with_trust_TEST_ONLY(
                _source(_bars(ivs)), provisioned, bar_hours=4.0,
                evaluation_as_of=AS_OF, session_export=exp,
                activation_receipt=_receipt(exp, key),
                operator_exceptions=[exc])

    def test_absolute_logical_id_refuses(self):
        with pytest.raises(ReadinessError, match="logical id"):
            VerifiedSource.from_csv_bytes(
                b"DATE_TIME,OPEN,CLOSE\n",
                roles=ROLES,
                source_logical_id="/abs/redacted/session.csv")


# ================================================================== #
# Tier-A real data: fail closed, no skip                             #
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
        src = VerifiedSource.from_csv_bytes(
            raw, roles=ROLES,
            source_logical_id="predictor:project3/ethusdt_4h")
        pkg = build_readiness_package(
            src, bar_hours=4.0,
            evaluation_as_of=datetime(2026, 9, 1,
                                      tzinfo=timezone.utc))
        assert pkg["inventory_summary"]["kind_counts"].get(
            "weekend_shaped_observed_gap", 0) == 0
        assert pkg["verdict"]["state"] == \
            "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
        assert pkg["verdict"]["paired_week_accounting"][
            "exact_deficit"] == WP4_MIN_PAIRED_WEEKS
        assert pkg["eth_conclusion_when_spot"] == ETH_SPOT_CONCLUSION


class TestSanitization:

    def test_no_private_topology_in_code_or_manifest(self):
        # needles built without literals so the scan is clean of
        # itself
        home_needle = "/" + "home/"
        operator_needle = "harvey" + "bc"
        for path in (mod.__file__,
                     str(mod.PINNED_TRUST_MANIFEST_PATH)):
            src = open(path).read()
            assert home_needle not in src
            assert operator_needle not in src
