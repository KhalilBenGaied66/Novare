"""SLA deadline computation in business hours.

Business hours are Monday to Friday, 08:00 to 18:00, read on the wall clock of the
datetime given by the caller (local time). Public holidays are not handled: a holiday
counts as a normal business day.
"""

from datetime import datetime, time, timedelta

OPENING = time(8, 0)
CLOSING = time(18, 0)
_SATURDAY = 5  # datetime.weekday(): Monday is 0


def sla_deadline(start: datetime, hours: int, *, around_the_clock: bool) -> datetime:
    """Latest intervention time for a request received at `start` with an SLA of `hours`.

    With `around_the_clock` (24/7 on-call contracts) the clock never stops. Otherwise
    only business hours count: Friday 17:00 + 4 h gives Monday 11:00, and a request
    received during the weekend starts counting on Monday at 08:00.
    """
    if hours < 0:
        raise ValueError("hours must be positive or zero")
    if around_the_clock:
        return start + timedelta(hours=hours)

    remaining = timedelta(hours=hours)
    current = _next_business_time(start)
    while True:
        closing = datetime.combine(current.date(), CLOSING, tzinfo=current.tzinfo)
        left_today = closing - current
        if remaining <= left_today:
            return current + remaining
        remaining -= left_today
        current = _next_business_time(closing)


def _next_business_time(moment: datetime) -> datetime:
    """`moment` if it falls within business hours, else the next opening time."""
    if moment.weekday() < _SATURDAY and OPENING <= moment.time() < CLOSING:
        return moment
    day = moment.date()
    if moment.weekday() >= _SATURDAY or moment.time() >= CLOSING:
        day += timedelta(days=1)
        while day.weekday() >= _SATURDAY:
            day += timedelta(days=1)
    return datetime.combine(day, OPENING, tzinfo=moment.tzinfo)
