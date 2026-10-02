"""Shared, dependency-free build/runtime Boolean contract."""


def parse_flag(value, name="flag"):
    """Missing/empty means unspecified; whitespace is not silently stripped."""
    if value is None or value == "":
        return None
    value = value.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be empty or one of 1/0, true/false, yes/no, on/off")
