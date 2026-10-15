from .utils import is_ignored, verify_path_exists

MAX_LINES = 300  # keeps one read well inside the model's 32k context
MAX_CHARS = 16_000  # ~4k tokens; lockfiles hit this before MAX_LINES


read_tool_schema = {
    "name": "read_tool",
    "description": (
        f"Reads a text file and returns up to {MAX_LINES} lines, each prefixed with its line number. "
        "Says where to continue if the file is longer."
    ),
    "args": {
        "file_path": {
            "type": "str",
            "required": True,
            "description": "Absolute or relative path to the file to read.",
        },
        "start_line": {
            "type": "int",
            "required": False,
            "description": "1-based line to start from. Defaults to 1.",
        },
    },
}


def read_tool(file_path: str, start_line: int = 1) -> str:
    """Read up to MAX_LINES lines of a file, starting at start_line."""
    if not verify_path_exists(file_path):
        return f"Error: file does not exist: {file_path}"
    if is_ignored(file_path):
        return f"Error: {file_path} is hidden from this exploration; read source files instead"
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if not lines:
        return "File is empty"

    start = max(int(start_line), 1)
    out_lines, size = [], 0
    for i, line in enumerate(lines[start - 1 : start - 1 + MAX_LINES], start):
        numbered = f"{i}: {line}"
        if out_lines and size + len(numbered) > MAX_CHARS:
            break
        out_lines.append(numbered)
        size += len(numbered) + 1
    out = "\n".join(out_lines)
    end = start - 1 + len(out_lines)
    if end < len(lines):
        out += f"\n[showing lines {start}-{end} of {len(lines)}; call again with start_line={end + 1} for more]"
    return out


__all__ = ["read_tool", "read_tool_schema"]
