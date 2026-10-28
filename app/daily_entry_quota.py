"""Conservative per-business-day entry budget, separate from exit permission."""
from datetime import timezone
from zoneinfo import ZoneInfo


class DailyEntryQuota:
    def __init__(self, limit=None, timezone_name="America/New_York"):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("DAILY_ENTRY_LIMIT_POSITIVE_INTEGER_REQUIRED")
        self.limit = limit
        self.timezone = ZoneInfo(timezone_name)
        self.submissions = {}
        self.fills = {}
        self.filled_refs = set()
        self.entry_refs = set()

    def day(self, timestamp):
        # Backtrader returns naive UTC datetimes by default.
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(self.timezone).date().isoformat()

    def allows(self, timestamp):
        if self.limit is None:
            return True
        day = self.day(timestamp)
        return max(self.submissions.get(day, 0), self.fills.get(day, 0)) < self.limit

    def submitted(self, timestamp, order_ref):
        if not self.allows(timestamp):
            raise ValueError("DAILY_ENTRY_LIMIT_EXCEEDED")
        day = self.day(timestamp)
        self.submissions[day] = self.submissions.get(day, 0) + 1
        self.entry_refs.add(order_ref)

    def filled(self, timestamp, order_ref):
        if order_ref not in self.entry_refs or order_ref in self.filled_refs:
            return
        day = self.day(timestamp)
        self.fills[day] = self.fills.get(day, 0) + 1
        self.filled_refs.add(order_ref)
        if self.limit is not None and self.fills[day] > self.limit:
            raise ValueError("DAILY_ENTRY_FILL_LIMIT_EXCEEDED")
