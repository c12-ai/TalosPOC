import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


def _jsonable(value: Any) -> Any:
    result: Any = value
    if isinstance(value, BaseModel):
        result = value.model_dump()
    elif isinstance(value, Enum):
        result = value.value
    elif isinstance(value, datetime):
        result = value.isoformat()
    elif is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = {k: _jsonable(v) for k, v in value.items()}
    elif isinstance(value, (set, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        result = [_jsonable(item) for item in value]
    elif type(value).__name__.endswith("Message"):
        result = {"type": type(value).__name__, "content": getattr(value, "content", None)}
    return result


def _pretty(value: Any) -> str:
    """Pretty-print arbitrary objects as JSON (best-effort), for demos/debugging."""
    return json.dumps(_jsonable(value), indent=2, ensure_ascii=False, default=str)
