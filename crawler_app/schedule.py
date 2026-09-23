"""Read and change the schedule that fires the SKU checker Lambda.

AWS has two services that can do this and an account may use either:

* **EventBridge Rules** (`events`) - the classic one. Cron is always UTC.
* **EventBridge Scheduler** (`scheduler`) - the newer one, which supports
  a timezone per schedule.

Both are supported and auto-detected from the Lambda's ARN, so the app
works whichever was used to set up the Monday/Friday run.

No Streamlit here, so it can be tested with fakes.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

DAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
DAY_LABELS = {
    "MON": "Monday", "TUE": "Tuesday", "WED": "Wednesday", "THU": "Thursday",
    "FRI": "Friday", "SAT": "Saturday", "SUN": "Sunday",
}
# EventBridge accepts day names or 1-7 with 1 = Sunday.
_NUMERIC_DAYS = {"1": "SUN", "2": "MON", "3": "TUE", "4": "WED", "5": "THU", "6": "FRI", "7": "SAT"}

WEEKLY_CRON = re.compile(
    r"^cron\(\s*(?P<minute>\d{1,2})\s+(?P<hour>\d{1,2})\s+(?P<dom>\S+)\s+(?P<month>\S+)\s+(?P<dow>\S+)\s+(?P<year>\S+)\s*\)$",
    re.I,
)


@dataclass
class Schedule:
    """A schedule as it exists in AWS."""
    service: str                      # "rule" (events) or "scheduler"
    name: str
    expression: str = ""              # cron(...) / rate(...)
    timezone: str = "UTC"
    enabled: bool = True
    arn: str = ""
    group: str = "default"
    target_arn: str = ""
    description: str = ""
    raw: Dict = field(default_factory=dict)

    @property
    def supports_timezone(self) -> bool:
        # EventBridge Rules are UTC-only; Scheduler takes a timezone.
        return self.service == "scheduler"


# ---------------------------------------------------------------- expressions

def build_weekly_cron(days: Sequence[str], hour: int, minute: int) -> str:
    """cron() for "these weekdays at this time".

    Day-of-month must be `?` when day-of-week is given, and EventBridge
    requires the sixth (year) field, unlike standard cron.
    """
    ordered = [d for d in DAY_NAMES if d in {x.upper() for x in days}]
    if not ordered:
        raise ValueError("Pick at least one day.")
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("Time must be between 00:00 and 23:59.")
    return f"cron({minute} {hour} ? * {','.join(ordered)} *)"


def parse_weekly_cron(expression: str) -> Optional[Tuple[List[str], int, int]]:
    """(days, hour, minute) for a simple weekly/daily cron, else None.

    Anything with step values, ranges or multiple hours is left alone - the
    UI then shows the raw expression rather than pretending to understand it.
    """
    match = WEEKLY_CRON.match((expression or "").strip())
    if not match:
        return None
    minute_field, hour_field = match.group("minute"), match.group("hour")
    if not minute_field.isdigit() or not hour_field.isdigit():
        return None

    dow = match.group("dow").upper()
    if dow in ("*", "?"):
        days = list(DAY_NAMES)
    else:
        if any(c in dow for c in "-/"):
            return None
        days = []
        for token in dow.split(","):
            token = token.strip()
            name = _NUMERIC_DAYS.get(token, token)
            if name not in DAY_NAMES:
                return None
            days.append(name)
        days = [d for d in DAY_NAMES if d in set(days)]

    if match.group("dom") not in ("?", "*"):
        return None
    if match.group("month") not in ("*", "?"):
        return None

    return days, int(hour_field), int(minute_field)


def describe_expression(expression: str, timezone: str = "UTC") -> str:
    """Plain-English summary for the UI."""
    parsed = parse_weekly_cron(expression)
    if not parsed:
        return expression or "(none)"
    days, hour, minute = parsed
    if len(days) == 7:
        when = "every day"
    elif days == ["MON", "TUE", "WED", "THU", "FRI"]:
        when = "every weekday"
    else:
        when = "every " + ", ".join(DAY_LABELS[d] for d in days)
    return f"{when} at {hour:02d}:{minute:02d} {timezone}"


def next_runs(expression: str, timezone: str = "UTC", count: int = 5,
              now: Optional[datetime] = None) -> List[datetime]:
    """Next `count` fire times, in the schedule's own timezone.

    Returns [] for expressions this module does not model (rate(), steps,
    ranges), so the UI can simply not show a preview.
    """
    parsed = parse_weekly_cron(expression)
    if not parsed:
        return []
    days, hour, minute = parsed
    wanted = {DAY_NAMES.index(d) for d in days}

    if now is None:
        try:
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo(timezone))
        except Exception:  # noqa: BLE001 - unknown tz name
            now = datetime.utcnow()

    runs: List[datetime] = []
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    for _ in range(14):
        if candidate.weekday() in wanted:
            runs.append(candidate)
            if len(runs) >= count:
                break
        candidate += timedelta(days=1)
    return runs


# ---------------------------------------------------------------- AWS access

class ScheduleClient:
    """Finds and edits the schedule that targets a given Lambda."""

    def __init__(self, region: str, lambda_arn: str, access_key_id: str = "",
                 secret_access_key: str = "", events=None, scheduler=None) -> None:
        self.region = region
        self.lambda_arn = lambda_arn
        self._key = access_key_id
        self._secret = secret_access_key
        self._events = events
        self._scheduler = scheduler

    def _client(self, service: str):
        import boto3

        kwargs = {"region_name": self.region}
        if self._key and self._secret:
            kwargs["aws_access_key_id"] = self._key
            kwargs["aws_secret_access_key"] = self._secret
        return boto3.client(service, **kwargs)

    @property
    def events(self):
        if self._events is None:
            self._events = self._client("events")
        return self._events

    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = self._client("scheduler")
        return self._scheduler

    # -- discovery -----------------------------------------------------
    def find_schedules(self) -> List[Schedule]:
        """Every rule/schedule pointing at the Lambda. Both services are
        tried; whichever the account uses answers."""
        found: List[Schedule] = []
        found.extend(self._find_event_rules())
        found.extend(self._find_scheduler_schedules())
        return found

    def _find_event_rules(self) -> List[Schedule]:
        out: List[Schedule] = []
        try:
            names = self.events.list_rule_names_by_target(TargetArn=self.lambda_arn).get("RuleNames", [])
        except Exception as e:  # noqa: BLE001 - no permission / not used
            log.info("EventBridge rules not readable: %s", e)
            return out
        for name in names:
            try:
                rule = self.events.describe_rule(Name=name)
            except Exception as e:  # noqa: BLE001
                log.warning("Could not describe rule %s: %s", name, e)
                continue
            if not rule.get("ScheduleExpression"):
                continue          # event-pattern rule, not a schedule
            out.append(Schedule(
                service="rule",
                name=name,
                expression=rule.get("ScheduleExpression", ""),
                timezone="UTC",
                enabled=rule.get("State") == "ENABLED",
                arn=rule.get("Arn", ""),
                target_arn=self.lambda_arn,
                description=rule.get("Description", "") or "",
                raw=rule,
            ))
        return out

    def _find_scheduler_schedules(self) -> List[Schedule]:
        out: List[Schedule] = []
        try:
            paginator = self.scheduler.get_paginator("list_schedules")
            pages = paginator.paginate()
        except Exception as e:  # noqa: BLE001 - service unused / no permission
            log.info("EventBridge Scheduler not readable: %s", e)
            return out
        try:
            for page in pages:
                for item in page.get("Schedules", []):
                    target = (item.get("Target") or {}).get("Arn", "")
                    name, group = item["Name"], item.get("GroupName", "default")
                    if target and not self._same_function(target):
                        continue
                    try:
                        detail = self.scheduler.get_schedule(Name=name, GroupName=group)
                    except Exception as e:  # noqa: BLE001
                        log.warning("Could not read schedule %s: %s", name, e)
                        continue
                    detail_target = (detail.get("Target") or {}).get("Arn", "")
                    if not self._same_function(detail_target):
                        continue
                    out.append(Schedule(
                        service="scheduler",
                        name=name,
                        expression=detail.get("ScheduleExpression", ""),
                        timezone=detail.get("ScheduleExpressionTimezone", "UTC") or "UTC",
                        enabled=detail.get("State") == "ENABLED",
                        arn=detail.get("Arn", ""),
                        group=group,
                        target_arn=detail_target,
                        description=detail.get("Description", "") or "",
                        raw=detail,
                    ))
        except Exception as e:  # noqa: BLE001
            log.info("Listing schedules failed: %s", e)
        return out

    def _same_function(self, arn: str) -> bool:
        """Compare ignoring a version/alias suffix on either side."""
        def base(value: str) -> str:
            parts = (value or "").split(":")
            return ":".join(parts[:7]) if len(parts) > 7 else (value or "")
        return bool(arn) and base(arn) == base(self.lambda_arn)

    # -- changes -------------------------------------------------------
    def update_expression(self, schedule: Schedule, expression: str,
                          timezone: Optional[str] = None) -> Schedule:
        """Change when it runs, leaving the target and everything else as is."""
        if schedule.service == "rule":
            kwargs = {
                "Name": schedule.name,
                "ScheduleExpression": expression,
                "State": "ENABLED" if schedule.enabled else "DISABLED",
            }
            if schedule.description:
                kwargs["Description"] = schedule.description
            self.events.put_rule(**kwargs)
            return self._reread(schedule)

        detail = dict(schedule.raw)
        payload = {
            "Name": schedule.name,
            "GroupName": schedule.group,
            "ScheduleExpression": expression,
            "ScheduleExpressionTimezone": timezone or schedule.timezone,
            "State": "ENABLED" if schedule.enabled else "DISABLED",
            "FlexibleTimeWindow": detail.get("FlexibleTimeWindow", {"Mode": "OFF"}),
            "Target": detail.get("Target", {}),
        }
        for key in ("Description", "StartDate", "EndDate", "KmsKeyArn"):
            if detail.get(key):
                payload[key] = detail[key]
        self.scheduler.update_schedule(**payload)
        return self._reread(schedule)

    def set_enabled(self, schedule: Schedule, enabled: bool) -> Schedule:
        if schedule.service == "rule":
            if enabled:
                self.events.enable_rule(Name=schedule.name)
            else:
                self.events.disable_rule(Name=schedule.name)
            return self._reread(schedule)

        updated = Schedule(**{**schedule.__dict__, "enabled": enabled})
        return self.update_expression(updated, schedule.expression, schedule.timezone)

    def _reread(self, schedule: Schedule) -> Schedule:
        """Read the schedule back so the UI shows what AWS actually stored."""
        for found in self.find_schedules():
            if found.name == schedule.name and found.service == schedule.service:
                return found
        return schedule
