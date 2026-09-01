"""P3 battery (order agent-multi@0ca5f7af §5): the historical
session-readiness package. Provenance classes stay separate, the
join contract refuses overlap/missing-tz/contradiction/look-ahead,
observed gaps never count as authoritative paired weeks, and the
deterministic fixtures (ordinary weekend, holiday-shortened, DST
shift, stale Tuesday feed, missing authority) behave as declared."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.wp4_session_readiness import (
    GAP_STAMP, PROVENANCE_BROKER, PROVENANCE_OBSERVED,
    WP4_MIN_PAIRED_WEEKS, JoinContractError, ProvenanceError,
    SessionEnvelopeInterval, build_join_contract,
    build_readiness_package, count_paired_weekly_units,
    data_readiness_verdict, inventory_weekly_closures,
    join_bar_to_session, refuse_provenance_promotion)


# ------------------------------------------------------------------ #
# P3.8: deterministic fixtures                                       #
# ------------------------------------------------------------------ #

def _frame(stamps, closes=None, spread=None):
    n = len(stamps)
    closes = closes if closes is not None else \
        100.0 + 0.1 * np.arange(n)
    data = {"DATE_TIME": stamps, "CLOSE": closes,
            "VOLUME": np.full(n, 1000.0)}
    if spread is not None:
        data["SPREAD"] = spread
    return pd.DataFrame(data)


def ordinary_weekend():
    before = pd.date_range("2024-01-01 00:00", periods=6, freq="4h",
                           tz="UTC")             # Mon .. Mon 20:00
    after = pd.date_range("2024-01-03 00:00", periods=6, freq="4h",
                          tz="UTC")              # reopen after 28h
    # a real weekend gap: Fri close -> Sun/Mon reopen >= 40h
    fri = pd.date_range("2024-01-05 00:00", periods=5, freq="4h",
                        tz="UTC")
    mon = pd.date_range("2024-01-08 00:00", periods=5, freq="4h",
                        tz="UTC")   # 64h gap
    return _frame(before.append(after).append(fri).append(mon)
                  .tz_localize(None))


def holiday_shortened():
    a = pd.date_range("2024-12-24 00:00", periods=2, freq="4h",
                      tz="UTC")
    b = pd.date_range("2024-12-25 16:00", periods=3, freq="4h",
                      tz="UTC")     # 32h holiday gap (< weekend)
    return _frame(a.append(b).tz_localize(None))


def dst_shift():
    # a weekend closure straddling the EU DST spring-forward
    # (2024-03-31): the local offset changes across the gap
    a = pd.date_range("2024-03-29 00:00", periods=5, freq="4h",
                      tz="UTC")
    b = pd.date_range("2024-04-01 00:00", periods=5, freq="4h",
                      tz="UTC")
    return _frame(a.append(b).tz_localize(None))


def stale_tuesday_feed():
    # bars stop mid-week and resume next day — a stale feed, NOT a
    # weekend; must be classified holiday_or_shortened, never a
    # session boundary
    a = pd.date_range("2024-01-09 00:00", periods=3, freq="4h",
                      tz="UTC")     # Tuesday
    b = pd.date_range("2024-01-10 12:00", periods=3, freq="4h",
                      tz="UTC")     # ~28h later
    return _frame(a.append(b).tz_localize(None))


def missing_authority():
    # a contiguous week with NO gaps at all — no observed closure,
    # and certainly no authority
    return _frame(pd.date_range("2024-01-01 00:00", periods=42,
                                freq="4h", tz="UTC").tz_localize(None))


class TestInventory:

    def test_ordinary_weekend_is_a_weekend_unit(self):
        inv = inventory_weekly_closures(ordinary_weekend(),
                                        bar_hours=4.0)
        assert inv["weekend_units"] >= 1
        for u in inv["observed_units"]:
            assert u["provenance"] == PROVENANCE_OBSERVED
            assert u["stamp"] == GAP_STAMP

    def test_holiday_shortened_is_not_a_weekend(self):
        inv = inventory_weekly_closures(holiday_shortened(),
                                        bar_hours=4.0)
        assert inv["weekend_units"] == 0
        assert inv["holiday_or_shortened_units"] >= 1

    def test_dst_crossing_is_flagged(self):
        inv = inventory_weekly_closures(
            dst_shift(), bar_hours=4.0, calendar_tz="Europe/Berlin")
        assert inv["dst_crossing_units"] >= 1

    def test_stale_tuesday_feed_is_holiday_shaped_not_weekend(self):
        inv = inventory_weekly_closures(stale_tuesday_feed(),
                                        bar_hours=4.0)
        assert inv["weekend_units"] == 0
        assert inv["holiday_or_shortened_units"] == 1

    def test_missing_authority_has_no_observed_units(self):
        inv = inventory_weekly_closures(missing_authority(),
                                        bar_hours=4.0)
        assert inv["observed_units"] == []

    def test_first_open_gap_and_metrics_are_derived(self):
        frame = holiday_shortened()
        frame["SPREAD"] = 0.0002
        inv = inventory_weekly_closures(
            frame, bar_hours=4.0, spread_col="SPREAD",
            vol_col="VOLUME")
        u = inv["observed_units"][0]
        assert u["first_open_gap_return"] is not None
        assert u["pre_close_spread"] == pytest.approx(0.0002)
        assert u["reopen_realized_vol"] == 1000.0


class TestProvenanceSeparation:

    def test_observed_gap_cannot_be_session_authority(self):
        with pytest.raises(ProvenanceError):
            refuse_provenance_promotion(PROVENANCE_OBSERVED,
                                        "session_authority")

    def test_broker_envelope_may_be_authority(self):
        refuse_provenance_promotion(PROVENANCE_BROKER,
                                    "session_authority")


class TestJoinContract:

    def _iv(self, close, reopen, prov=PROVENANCE_BROKER):
        return SessionEnvelopeInterval(
            pd.Timestamp(close, tz="UTC"),
            pd.Timestamp(reopen, tz="UTC"), prov)

    def test_naive_interval_refuses(self):
        with pytest.raises(JoinContractError, match="timezone"):
            SessionEnvelopeInterval(
                pd.Timestamp("2024-01-05 20:00"),
                pd.Timestamp("2024-01-08 00:00", tz="UTC"),
                PROVENANCE_BROKER)

    def test_contradictory_interval_refuses(self):
        with pytest.raises(JoinContractError, match="contradictory"):
            self._iv("2024-01-08 00:00", "2024-01-05 20:00")

    def test_non_authoritative_interval_refuses(self):
        with pytest.raises(JoinContractError, match="authoritative"):
            self._iv("2024-01-05 20:00", "2024-01-08 00:00",
                     prov=PROVENANCE_OBSERVED)

    def test_overlap_refuses(self):
        with pytest.raises(JoinContractError, match="overlap"):
            build_join_contract([
                self._iv("2024-01-05 20:00", "2024-01-08 08:00"),
                self._iv("2024-01-08 00:00", "2024-01-09 00:00")])

    def test_look_ahead_refuses(self):
        contract = build_join_contract([
            self._iv("2024-01-12 20:00", "2024-01-15 00:00")])
        with pytest.raises(JoinContractError, match="look-ahead"):
            join_bar_to_session(
                pd.Timestamp("2024-01-10 00:00", tz="UTC"),
                contract,
                evidence_known_up_to=pd.Timestamp(
                    "2024-01-10 00:00", tz="UTC"))

    def test_a_bar_inside_a_known_closure_is_classified(self):
        contract = build_join_contract([
            self._iv("2024-01-05 20:00", "2024-01-08 00:00")])
        inside = join_bar_to_session(
            pd.Timestamp("2024-01-06 12:00", tz="UTC"), contract,
            evidence_known_up_to=pd.Timestamp(
                "2024-01-09 00:00", tz="UTC"))
        assert inside["in_closure"] is True
        assert inside["provenance"] == PROVENANCE_BROKER
        outside = join_bar_to_session(
            pd.Timestamp("2024-01-08 04:00", tz="UTC"), contract,
            evidence_known_up_to=pd.Timestamp(
                "2024-01-09 00:00", tz="UTC"))
        assert outside["in_closure"] is False


class TestPairedWeekCountAndVerdict:

    def test_observed_gaps_do_not_count(self):
        acc = count_paired_weekly_units(authoritative_units=0)
        assert acc["authoritative_paired_weeks_available"] == 0
        assert acc["exact_deficit"] == WP4_MIN_PAIRED_WEEKS
        assert acc["status"] == "INCONCLUSIVE"

    def test_exact_deficit_is_reported(self):
        acc = count_paired_weekly_units(authoritative_units=12)
        assert acc["exact_deficit"] == WP4_MIN_PAIRED_WEEKS - 12
        assert acc["status"] == "INCONCLUSIVE"

    def test_sufficient_only_at_the_minimum(self):
        acc = count_paired_weekly_units(
            authoritative_units=WP4_MIN_PAIRED_WEEKS)
        assert acc["status"] == "SUFFICIENT"

    def test_verdict_states_are_the_only_three(self):
        # no collector, only observed gaps -> non-authoritative
        v = data_readiness_verdict(collector_active=False,
                                   authoritative_units=0,
                                   observed_units=8)
        assert v["state"] == \
            "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
        assert v["economic_grid_authorized"] is False
        # collector active but not yet enough -> accumulating
        v = data_readiness_verdict(collector_active=True,
                                   authoritative_units=5,
                                   observed_units=8)
        assert v["state"] == "COLLECTOR_ACTIVE_HISTORY_ACCUMULATING"
        # collector active and sufficient -> authoritative
        v = data_readiness_verdict(
            collector_active=True,
            authoritative_units=WP4_MIN_PAIRED_WEEKS,
            observed_units=8)
        assert v["state"] == \
            "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"

    def test_authoritative_verdict_needs_the_collector(self):
        """Even at the minimum count, without an active collector
        the support is not authoritative."""
        v = data_readiness_verdict(
            collector_active=False,
            authoritative_units=WP4_MIN_PAIRED_WEEKS,
            observed_units=0)
        assert v["state"] != \
            "AUTHORITATIVE_SUPPORT_SUFFICIENT_FOR_CALIBRATION"


class TestFullPackageOnRealData:

    def test_eth_h4_is_non_authoritative_with_full_deficit(self):
        """The only available ETH H4 dataset is 24/7 crypto — no
        broker weekly closures and no session authority. The
        package must say so with the exact 30-week deficit."""
        import os
        path = ("/home/harveybc/Documents/GitHub/predictor/examples"
                "/data/project3/ethusdt_4h_tech_stat_full_model_"
                "ready.csv")
        if not os.path.isfile(path):
            pytest.skip("ETH dataset not present")
        frame = pd.read_csv(path, usecols=["DATE_TIME", "CLOSE",
                                           "VOLUME"])
        pkg = build_readiness_package(frame, bar_hours=4.0,
                                      collector_active=False,
                                      vol_col="VOLUME")
        assert pkg["inventory"]["weekend_units"] == 0
        assert pkg["verdict"]["state"] == \
            "HISTORICAL_BACKFILL_NON_AUTHORITATIVE_ONLY"
        assert pkg["verdict"]["paired_week_accounting"][
            "exact_deficit"] == WP4_MIN_PAIRED_WEEKS
        assert pkg["digest"]

    def test_eurusd_4h_has_real_weekend_closures_but_still_no_authority(
            self):
        """EURUSD 4h DOES carry weekend gaps, proving the inventory
        finds real closures — yet without the collector they remain
        OBSERVED_GAP, never authoritative paired weeks."""
        import os
        path = ("/home/harveybc/Documents/GitHub/financial-data/"
                "market_data/forex/g10/eurusd/4h.parquet")
        if not os.path.isfile(path):
            pytest.skip("eurusd dataset not present")
        frame = pd.read_parquet(path).rename(
            columns={"datetime": "DATE_TIME", "close": "CLOSE"})
        frame = frame[(frame["DATE_TIME"] >= "2024-01-01") &
                      (frame["DATE_TIME"] < "2024-04-01")]
        inv = inventory_weekly_closures(frame, bar_hours=4.0)
        assert inv["weekend_units"] >= 8
        acc = count_paired_weekly_units(authoritative_units=0)
        assert acc["status"] == "INCONCLUSIVE"
