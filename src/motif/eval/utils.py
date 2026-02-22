from pandas import Timedelta


def format_tdelta(tdelta: Timedelta) -> str:
    """Formats a timedelta object in the form of floating hours
    (e.g. 1.5 for 1 hour and 30 minutes, -2.25 for minus 2 hours and 15 minutes).
    """
    total_seconds = tdelta.total_seconds()
    hours = total_seconds / 3600
    return f"{hours:.2f}h"
