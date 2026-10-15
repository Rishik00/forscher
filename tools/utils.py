import fnmatch
import os

# Hidden from ls_tool and refused by read_tool.
IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", "target"}
IGNORED_FILES = [
    "*.lock", "package-lock.json", "pnpm-lock.yaml",  # noise the agent never needs
    "README*", "readme*", "Readme*",  # they hand over the answer; make the agent read code
]


def verify_path_exists(path: str) -> bool:
    """Return True if the given path exists on disk."""
    return os.path.exists(path)


def is_ignored(path: str) -> bool:
    """True if the path matches IGNORED_FILES or sits inside an ignored directory."""
    parts = os.path.normpath(path).split(os.sep)
    if any(p in IGNORED_DIRS for p in parts):
        return True
    return any(fnmatch.fnmatch(parts[-1], pat) for pat in IGNORED_FILES)
