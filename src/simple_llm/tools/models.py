from pydantic import BaseModel
from typing import Any, Callable, Dict, Optional

class Tool(BaseModel):
    name: str
    description: str
    function: Callable
    auto_execute: Optional[bool] = True
    return_output: Optional[bool] = False
    settings: Optional[Dict] = {}

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]

class ToolOutput(BaseModel):
    name: str
    output: str