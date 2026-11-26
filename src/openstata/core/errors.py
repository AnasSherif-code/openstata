"""Structured errors carrying Stata-compatible return codes."""


class StataError(Exception):
    """Base error raised during parsing or execution of Stata commands."""

    def __init__(self, message: str, rc: int = 198) -> None:
        super().__init__(message)
        self.message = message
        self.rc = rc

    def __str__(self) -> str:
        return f"{self.message}\nr({self.rc});"


class BreakSignal(Exception):
    """Internal control-flow signal for `continue, break`."""


class ContinueSignal(Exception):
    """Internal control-flow signal for `continue`."""


class ExitSignal(Exception):
    """Internal control-flow signal for `exit`."""

    def __init__(self, rc: int = 0) -> None:
        super().__init__(f"exit {rc}")
        self.rc = rc
