"""
The gate: gives each day a status (OK, WARNING or CRITICAL) from its failed checks, and writes the
WARNING and CRITICAL days to an alerts file, one JSON object per line, for people or tools to pick up.
"""
import json

STATUSES = ["OK", "WARNING", "CRITICAL"]  # from least to most serious


def decide(failures, severity):
    """
    The day's status: the most serious status among its failed checks, where severity maps
    each check to a status (from config/monitoring.yaml). A day with no failed checks is OK.
    """
    return max((severity[check] for check, _ in failures), key=STATUSES.index, default="OK")


def make_alert(day, date, status, failures):
    """The alert for one day: when it happened, its status, and what failed."""
    return {
        "day": day,
        "date": str(date),
        "status": status,
        "failed_checks": [{"check": check, "message": message} for check, message in failures],
    }


def write_alert(path, alert):
    """Adds one alert to the end of the alerts file."""
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(alert) + "\n")
