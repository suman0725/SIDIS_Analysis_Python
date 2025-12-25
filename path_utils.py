"""
path_utils.py

Helpers for:
  - enabling TAB-completion for filesystem paths
  - asking the user for a directory + file pattern
"""

import os
import glob

try:
    import readline
except ImportError:
    readline = None


def _path_completer(text, state):
    """
    Simple filesystem tab-completer for input().

    text  : current word being completed
    state : 0,1,2,... which candidate to return
    """
    if text is None:
        text = ""

    # Expand ~ and env vars
    text_expanded = os.path.expanduser(os.path.expandvars(text))

    matches = glob.glob(text_expanded + "*")

    # Add '/' for directories
    matches = [
        m + ("/" if os.path.isdir(m) else "")
        for m in matches
    ]

    # readline protocol: return None when out of matches
    matches.append(None)
    return matches[state]


def enable_path_completion():
    """
    Enable TAB completion for paths in input(), if readline is available.
    """
    if readline is None:
        return

    # Don't split on '/' so we can complete whole paths
    readline.set_completer_delims(" \t\n;")

    readline.set_completer(_path_completer)

    # macOS usually uses libedit, needs different binding syntax
    if readline.__doc__ and "libedit" in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def ask_files_for_target(label: str):
    """
    Ask the user for a directory and a file pattern, then return
    a sorted list of matching files.

    Parameters
    ----------
    label : str
        For printing (e.g. 'LD2', 'CxC', 'CuSn').

    Returns
    -------
    files : list[str]
        Sorted list of matching paths (may be empty).
    """
    print(f"\nConfigure paths for target {label}:")

    # turn on TAB completion once we’re about to read a path
    enable_path_completion()

    base_dir = input("  Directory with ROOT files: ").strip()
    pattern  = input("  File pattern (e.g. 'rec_clas_*.root' or 'mat_simu_ld2_*.root'): ").strip()

    # Expand ~ and env vars just in case
    base_dir = os.path.expanduser(os.path.expandvars(base_dir))

    full_pattern = os.path.join(base_dir, pattern)
    files = sorted(glob.glob(full_pattern))

    if not files:
        print(f"  WARNING: no files matched {full_pattern}")
    else:
        print(f"  Found {len(files)} files for {label}")

    return files
