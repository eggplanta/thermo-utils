"""User settings file (TOML): display units for every calculation and per calculation."""

import json
import os
import re
import sys
import tomllib
from pathlib import Path

from thermo_utils.units import display, label, units_of, _norm

WARNINGS = []
_PER_CALC = {}
_FILE_KEYS = {"typed_units_win": "follow_typed"}
_UNIT_KEYS = [k for k in display.__dataclass_fields__ if k not in ("min_decimals", "follow_typed")]
_KEYS = _UNIT_KEYS + ["min_decimals", "typed_units_win"]

HEADER = """\
# thermo-utils settings
#
# Read every time thermo-utils starts, in the menu and on the command line.
# Format: TOML. Lines starting with # are comments; text goes in quotes,
# numbers and true/false do not. Check this file with: thermo-utils config
#
# The menu option "u" (Display units) rewrites only the values in [display];
# comments and the other sections are kept.
#
# Priority, highest first:
#   1. command-line options (--T-unit, --decimals, ...), for that run only
#   2. units typed with a value (25C, 10atm), when typed_units_win = true
#   3. [display.<calculation>], for that calculation only
#   4. [display], for every calculation
#
# Energy units set the family (J or cal) and the basis (per mol or per kg);
# the k prefix (J or kJ) is picked per result from the size of the values.
"""

PER_CALC_EXAMPLE = """\
# Display units for one calculation. They override [display] for that
# calculation only; keys left out keep the [display] value. The name after
# "display." is the calculation id shown by `thermo-utils list`
# (refrigeration, rankine, brayton, delta, vle, device, ...).
# Remove the # to use the examples:
#
# [display.refrigeration]
# temperature = "°C"
# pressure = "kPa"
#
# [display.rankine]
# pressure = "MPa"
# min_decimals = 2
"""


def path():
    if os.environ.get("THERMO_UTILS_CONFIG"):
        return Path(os.environ["THERMO_UTILS_CONFIG"])
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "thermo-utils" / "settings.toml"


# Reading

def load(calc_ids=()):
    """Apply [display] as the saved units and keep the per-calculation overrides."""
    WARNINGS.clear()
    _PER_CALC.clear()
    p = path()
    if not p.exists():
        return
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        WARNINGS.append(f"{p} ignored: {exc}")
        return
    for key in data:
        if key != "display":
            WARNINGS.append(f"Unknown section [{key}]")
    section = data.get("display", {})
    if not isinstance(section, dict):
        WARNINGS.append("[display] must be a section")
        return
    for attr, value in _check({k: v for k, v in section.items() if not isinstance(v, dict)}, "[display]").items():
        setattr(display, attr, value)
    display.save()
    for calc, values in section.items():
        if not isinstance(values, dict):
            continue
        if calc_ids and calc not in calc_ids:
            WARNINGS.append(f"[display.{calc}]: unknown calculation (see thermo-utils list)")
            continue
        _PER_CALC[calc] = _check(values, f"[display.{calc}]")


def apply_calc(calc_id):
    """Per-calculation units on top of the saved ones; call after display.reset()."""
    for attr, value in _PER_CALC.get(calc_id, {}).items():
        setattr(display, attr, value)


def _check(values, where):
    out = {}
    for key, value in values.items():
        attr = _FILE_KEYS.get(key, key)
        if key not in _KEYS:
            WARNINGS.append(f"{where} {key}: unknown setting")
        elif key == "typed_units_win":
            if isinstance(value, bool):
                out[attr] = value
            else:
                WARNINGS.append(f"{where} {key}: use true or false")
        elif key == "min_decimals":
            if isinstance(value, int) and not isinstance(value, bool) and 2 <= value <= 8:
                out[attr] = value
            else:
                WARNINGS.append(f"{where} {key}: use a whole number from 2 to 8")
        elif isinstance(value, str) and _norm(value) in {_norm(u) for u in units_of(key)}:
            out[attr] = label(value)
        else:
            WARNINGS.append(f"{where} {key}: unknown unit {value!r}; options: {', '.join(_options(key))}")
    return out


def _options(key):
    return list(dict.fromkeys(label(u) for u in units_of(key)))


# Writing

def _value(key):
    v = display.saved(_FILE_KEYS.get(key, key))
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v) if isinstance(v, int) else json.dumps(v, ensure_ascii=False)


def _comment(key):
    if key == "min_decimals":
        return "2 to 8; results keep 6 significant figures"
    if key == "typed_units_win":
        return "units you type are kept in the results"
    return ", ".join(_options(key))


def template():
    width = max(len(f"{k} = {_value(k)}") for k in _KEYS)
    lines = [HEADER, "# Display units for every calculation", "[display]"]
    lines += [f"{f'{k} = {_value(k)}':<{width}}  # {_comment(k)}" for k in _KEYS]
    return "\n".join(lines) + "\n\n" + PER_CALC_EXAMPLE


def create():
    """Write the commented template if the file does not exist; True if created."""
    p = path()
    if p.exists():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(template(), encoding="utf-8")
    return True


_SECTION = re.compile(r"^\s*\[")
_DISPLAY = re.compile(r"^\s*\[\s*display\s*\]\s*(#.*)?$")


def store():
    """Save the menu units into [display], keeping comments and other sections."""
    if create():
        return
    p = path()
    text = p.read_text(encoding="utf-8")
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"fix {p} first: {exc}") from None
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _DISPLAY.match(ln)), None)
    if start is None:
        top = next((i for i, ln in enumerate(lines) if ln.strip() and not ln.lstrip().startswith("#")), len(lines))
        lines[top:top] = ["[display]", *(f"{k} = {_value(k)}" for k in _KEYS), ""]
    else:
        end = next((i for i in range(start + 1, len(lines)) if _SECTION.match(lines[i])), len(lines))
        for key in _KEYS:
            pattern = re.compile(rf'^(\s*{key}\s*=\s*)("(?:[^"\\]|\\.)*"|[^\s#]+)(.*)$')
            hit = next((i for i in range(start + 1, end) if pattern.match(lines[i])), None)
            if hit is None:
                last = end
                while last > start + 1 and not lines[last - 1].strip():
                    last -= 1
                lines.insert(last, f"{key} = {_value(key)}")
                end += 1
            else:
                m = pattern.match(lines[hit])
                lines[hit] = m.group(1) + _value(key) + m.group(3)
    new = "\n".join(lines) + "\n"
    tomllib.loads(new)
    p.write_text(new, encoding="utf-8")
