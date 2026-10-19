"""Opt-in completed-H1 clocks; row labels and prices are never shifted."""
from datetime import timedelta, timezone


def completed_bar_mode(config):
    semantics = config.get('timestamp_semantics', 'legacy')
    if semantics not in ('legacy', 'completed_bar_available_at'):
        raise ValueError('UNKNOWN_TIMESTAMP_SEMANTICS')
    enabled = semantics == 'completed_bar_available_at'
    if enabled:
        if config.get('timeframe') != '1h':
            raise ValueError('COMPLETED_BAR_TIMEFRAME_1H_REQUIRED')
        if config.get('execution_transition') != 'next_open_v2':
            raise ValueError('COMPLETED_BAR_REQUIRES_NEXT_OPEN_V2')
    return enabled


def entry_fill_clock(label, config):
    """Entry's containing interval starts here; Market fills exactly here.

    H1 is UTC-hour aligned, so intrabar limit/stop fills also belong to this
    NY business date, including both DST transitions. No intrabar instant is
    inferred for those orders.
    """
    if not completed_bar_mode(config):
        return label
    if label.tzinfo is None:
        label = label.replace(tzinfo=timezone.utc)
    return label - timedelta(hours=1)
