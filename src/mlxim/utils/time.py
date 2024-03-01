import datetime

STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"


def now() -> str:
    """Return current time in the format: %Y-%m-%d-%H-%M-%S.

    Returns:
        str: current time
    """
    STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
    return datetime.datetime.now().strftime(STRFTIME_FORMAT)
