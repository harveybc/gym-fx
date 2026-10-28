"""The optimized boundary must preserve the canonical row-clock precision."""

import pandas as pd
import pytest

from test_reopen_evidence_equivalence import reference
from test_session_exposure_env import PRE_BARS, REOPEN_AT, _env


@pytest.mark.parametrize("transition", ["legacy_delayed_v1", "next_open_v2"])
@pytest.mark.parametrize("zone", [None, "UTC", "America/New_York"])
@pytest.mark.filterwarnings("ignore:Discarding nonzero nanoseconds")
def test_submicrosecond_reopen_boundary_matches_row_clock(tmp_path, transition, zone):
    env = _env(tmp_path, execution_transition=transition,
               execution_max_delay_hours=4)
    try:
        stamps = list(env.dataframe.index)
        stamps[PRE_BARS] = REOPEN_AT.tz_localize(None) + pd.Timedelta(nanoseconds=1)
        index = pd.DatetimeIndex(stamps)
        if zone is not None:
            index = index.tz_localize("UTC").tz_convert(zone)
        env.dataframe = env.dataframe.copy()
        env.dataframe.index = index
        step = PRE_BARS + 1
        now = env._session_now(step)
        assert env._session_reopen_evidence(step, now) == reference(env, step, now)
    finally:
        env.close()
