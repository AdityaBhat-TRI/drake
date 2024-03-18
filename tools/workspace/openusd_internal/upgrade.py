"""
Upgrades the lockfile for OpenUSD.
"""

import argparse
from pathlib import Path
import re

from bazel_tools.tools.python.runfiles import runfiles

# This is the full list of OpenUSD libraries that Drake cares about.
SUBDIRS = [
    "pxr/base/arch",
    "pxr/base/gf",
    "pxr/base/js",
    "pxr/base/plug",
    "pxr/base/tf",
    "pxr/base/trace",
    "pxr/base/vt",
    "pxr/base/work",
    "pxr/usd/ar",
    "pxr/usd/kind",
    "pxr/usd/ndr",
    "pxr/usd/pcp",
    "pxr/usd/sdf",
    "pxr/usd/sdr",
    "pxr/usd/usd",
    "pxr/usd/usdGeom",
    "pxr/usd/usdShade",
    "pxr/usd/usdUtils",
]


def _scrape(subdir: str) -> dict[str, str | list[str]]:
    """Scrapes the pxr_library() call from the given subdir's CMakeLists.txt
    and returns its arguments as more Pythonic data structure. (Refer to the
    `result` dict inline below, for details.)
    """

    # Slurp the file
    manifest = runfiles.Create()
    found = manifest.Rlocation(f"openusd_internal/{subdir}/CMakeLists.txt")
    cmake = Path(found).read_text(encoding="utf-8")

    # Find the arguments and split into tokens.
    match = re.search(r"pxr_library\((.*?)\)", cmake, flags=re.DOTALL)
    assert match, subdir
    text, = match.groups()
    text = re.sub(r'#.*', "", text)
    text = text.replace("\n", " ")
    tokens = text.split()

    # Set up a skeleton result with the args we want.
    name = tokens.pop(0)
    result = dict(
        NAME=name,
        LIBRARIES=[],
        PUBLIC_CLASSES=[],
        PUBLIC_HEADERS=[],
        PRIVATE_CLASSES=[],
        PRIVATE_HEADERS=[],
        CPPFILES=[],
    )

    # File the tokens into collections. They are currently in a list, e.g.,
    #   [FOO, bar, baz, QUUX, alpha, bravo]
    # and we want them in a dict, e.g.,
    #   {FOO: [bar, baz], QUUX: [alpha, bravo]}
    current_collection = None
    for token in tokens:
        if token.isupper():
            current_collection = token
            continue
        assert current_collection is not None, subdir
        if "$" in token:
            # Skip over CMake substitutions.
            continue
        if current_collection in result:
            result[current_collection].append(token)

    return result


def _generate() -> str:
    """Returns the expected contents of the lockfile (lock/files.bzl)."""
    lines = [
        "# This file is automatically generated by upgrade.py.",
    ]
    lines.append("FILES = {")
    for subdir in SUBDIRS:
        lines.append(f'    "{subdir}": {{')
        scraped = _scrape(subdir)
        for name, value in scraped.items():
            if isinstance(value, str):
                lines.append(f'        "{name}": "{value}",')
                continue
            assert isinstance(value, list)
            if len(value) == 0:
                lines.append(f'        "{name}": [],')
                continue
            lines.append(f'        "{name}": [')
            for item in value:
                lines.append(f'            "{item}",')
            lines.append(f'        ],')
        lines.append(f'    }},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        prog="upgrade", description=__doc__)
    parser.add_argument(
        "--relock", action="store_true",
        help="Overwrite the lockfile in the source tree.")
    parser.add_argument(
        "--output", type=Path,
        help="Write the lockfile to the given path.")
    args = parser.parse_args()
    assert args.relock ^ (args.output is not None)
    if args.relock:
        output = Path(__file__).resolve().parent / "lock/files.bzl"
    else:
        output = args.output
    content = _generate()
    output.write_text(content, encoding="utf-8")


assert __name__ == "__main__"
main()
