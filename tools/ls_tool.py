import os

from .utils import is_ignored, verify_path_exists

ls_tool_schema = {
    "name": "ls_tool",
    "description": "Lists the files and folders inside a directory.",
    "args": {
        "dir": {
            "type": "str",
            "required": True,
            "description": "Absolute or relative path to the directory to list.",
        }
    },
}


def ls_tool(dir: str) -> list[str] | str:
    """List files and folders in a directory."""
    if not verify_path_exists(dir):
        return f"Error: path does not exist: {dir}"
    return [name for name in os.listdir(dir) if not is_ignored(os.path.join(dir, name))]


__all__ = ["ls_tool", "ls_tool_schema"]
