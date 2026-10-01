"""SLA deadlines: business hours (Mon-Fri 08:00-18:00) versus around-the-clock contracts."""

from datetime import UTC, datetime, time, timedelta

import pytest

from app.domain.sla import sla_deadline

# October 2026: the 5th is a Monday, the 9th a Friday, the 10th a Saturday.
MON, TUE, WED, THU, FRI, SAT, SUN = (datetime(2026, 10, day) for day in range(5, 12))
NEXT_MON, NEXT_WED, NEXT_THU = MON + timedelta(days=7), WED + timedelta(days=7), THU + timedelta(7)


def at(day: datetime, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute, second=second)


BUSINESS_HOURS_CASES = [
    # (label, start, SLA hours, expected deadline)
    ("same day", at(MON, 9), 4, at(MON, 13)),
    ("ends exactly at closing time", at(MON, 14), 4, at(MON, 18)),
    ("overflows to the next morning", at(MON, 15), 4, at(TUE, 9)),
    ("before opening starts at 08:00", at(MON, 6, 30), 2, at(MON, 10)),
    ("minutes and seconds are kept", at(MON, 9, 17, 23), 4, at(MON, 13, 17, 23)),
    ("Friday 17:00 + 4 h", at(FRI, 17), 4, at(NEXT_MON, 11)),
    ("Friday at closing time", at(FRI, 18), 2, at(NEXT_MON, 10)),
    ("Friday evening", at(FRI, 21, 30), 4, at(NEXT_MON, 12)),
    ("Friday 17:45:30 + 1 h", at(FRI, 17, 45, 30), 1, at(NEXT_MON, 8, 45, 30)),
    ("Saturday morning", at(SAT, 10), 4, at(NEXT_MON, 12)),
    ("Sunday night", at(SUN, 23, 59), 8, at(NEXT_MON, 16)),
    ("Essentiel P2: 24 h over three days", at(MON, 9), 24, at(WED, 13)),
    ("Confort P3: 48 h across a weekend", at(THU, 16), 48, at(NEXT_THU, 14)),
    ("Essentiel P3: 72 h across a weekend", at(MON, 9), 72, at(NEXT_WED, 11)),
    ("a full business day ends at closing time", at(MON, 8), 10, at(MON, 18)),
    ("zero hours during business hours", at(MON, 10), 0, at(MON, 10)),
    ("zero hours during the weekend", at(SAT, 10), 0, at(NEXT_MON, 8)),
]


@pytest.mark.parametrize(
    ("start", "hours", "expected"),
    [case[1:] for case in BUSINESS_HOURS_CASES],
    ids=[case[0] for case in BUSINESS_HOURS_CASES],
)
def test_business_hours_deadline(start, hours, expected):
    assert sla_deadline(start, hours, around_the_clock=False) == expected


AROUND_THE_CLOCK_CASES = [
    ("Friday evening stays on Friday", at(FRI, 17), 4, at(FRI, 21)),
    ("Saturday night crosses midnight", at(SAT, 23), 2, at(SUN, 1)),
    ("Premium P3: 24 h over the weekend", at(SAT, 10, 30), 24, at(SUN, 10, 30)),
    ("72 h from Friday", at(FRI, 17), 72, at(NEXT_MON, 17)),
    ("night request", at(TUE, 3, 15), 2, at(TUE, 5, 15)),
]


@pytest.mark.parametrize(
    ("start", "hours", "expected"),
    [case[1:] for case in AROUND_THE_CLOCK_CASES],
    ids=[case[0] for case in AROUND_THE_CLOCK_CASES],
)
def test_around_the_clock_deadline(start, hours, expected):
    assert sla_deadline(start, hours, around_the_clock=True) == expected


def is_business_minute(moment: datetime) -> bool:
    return moment.weekday() < 5 and time(8, 0) <= moment.time() < time(18, 0)


def count_minute_by_minute(start: datetime, hours: int) -> datetime:
    """Independent reference: walk the calendar one minute at a time."""
    moment, counted = start, 0
    while counted < hours * 60:
        if is_business_minute(moment):
            counted += 1
        moment += timedelta(minutes=1)
    return moment


@pytest.mark.parametrize("hours", [1, 4, 8, 24])
def test_matches_a_minute_by_minute_count(hours):
    # Starts every 5 h 13 min over nine days: every weekday, night, weekend and
    # opening/closing boundaries are crossed.
    start = at(MON, 0)
    while start < MON + timedelta(days=9):
        expected = count_minute_by_minute(start, hours)
        assert sla_deadline(start, hours, around_the_clock=False) == expected, start
        start += timedelta(hours=5, minutes=13)


def test_business_deadline_always_falls_within_business_hours():
    start = at(MON, 0)
    while start < MON + timedelta(days=7):
        deadline = sla_deadline(start, 4, around_the_clock=False)
        assert deadline.weekday() < 5, start
        assert time(8, 0) < deadline.time() <= time(18, 0), start
        assert deadline > start
        start += timedelta(hours=1)


def test_timezone_of_the_start_is_kept():
    start = at(FRI, 17).replace(tzinfo=UTC)
    assert sla_deadline(start, 4, around_the_clock=False) == at(NEXT_MON, 11).replace(tzinfo=UTC)
    assert sla_deadline(start, 4, around_the_clock=True) == at(FRI, 21).replace(tzinfo=UTC)


@pytest.mark.parametrize("around_the_clock", [True, False])
def test_negative_hours_are_rejected(around_the_clock):
    with pytest.raises(ValueError, match="hours"):
        sla_deadline(at(MON, 9), -1, around_the_clock=around_the_clock)
