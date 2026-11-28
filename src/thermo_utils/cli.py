import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from dataclasses import replace
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape
from rich.text import Text

from thermo_utils import (
    __version__, calcs, cycles, drawing, equations, plots, properties, psychro, settings, steplog, substances,
)
from thermo_utils.result import harmonize
from thermo_utils.units import num
from thermo_utils.units import DEFAULT_INPUT_UNITS, display, units_of
from thermo_utils.units import _norm as _norm_unit, label as unit_label

warnings.filterwarnings("ignore")
console = Console(highlight=False)
app = typer.Typer(add_completion=False, help=(
    "Thermodynamics calculator for chemical engineering.\n\n"
    "Run without arguments for the menu, or: thermo-utils <calc> <substances> key=value ...\n\n"
    "Example: thermo-utils delta water T1=25C P1=1atm T2=300C P2=20bar"))

LOGO = r"""
              (        )  (        )      (          )
     )  (     )\    ( /(  )\    ( /(      )\ )     (/(   )
   )/(_))(  ( (()   )\())((_)  ))\(_)  ( ((_)(   )\ )\  (()  )
  / /_/ /) )\/_ )\((_)\_(_ )\_(_)\_()\  )\ (_(-))(/ /(_)/ /\(\_
 / __/ __ \/ _ \/ ___/ __ `__ \/ __ \_)-/(_/ / / / __/ / / ___/
/ /_/ / / /  __/ /  / / / / / / /_/ /_____/ /_/ / /_/ / (__  )
\__/_/ /_/\___/_/  /_/ /_/ /_/\____/      \__,_/\__/_/_/____/
"""

COLORS = [(255, 220, 80), (255, 150, 40), (230, 70, 30), (180, 40, 50)]
ACCENT = "rgb(255,150,40)"


# Logo

def _gradient(t):
    g = len(COLORS) - 1
    i = min(int(t * g), g - 1)
    a, b = COLORS[i], COLORS[i + 1]
    f = t * g - i
    return tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))


def print_logo():
    width = shutil.get_terminal_size().columns
    lines = LOGO.strip("\n").splitlines()
    pad = max(0, (width - max(len(l) for l in lines)) // 2)
    for i, line in enumerate(lines):
        r, g, b = _gradient(i / max(1, len(lines) - 1))
        console.print(" " * pad + line, style=f"rgb({r},{g},{b})", markup=False)
    console.print(f"version {__version__}".center(width), style="dim")


# Output

def show_equations(lines):
    if lines:
        body = "\n".join(lines)
        console.print(Panel(Text(body), title="[b]Equations[/b]", title_align="left", border_style="dim"))


def _box(title, rows, width):
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()
    table.add_column(style=f"bold {ACCENT}", justify="center", min_width=3)
    for label, value, mark in rows:
        table.add_row(label, value, mark)
    return Panel(table, title=f"[b]{title}[/b]", title_align="left", border_style=ACCENT, width=width)


def show_pair(pair):
    rows1, label, rows2 = pair
    rows = rows1 + rows2
    width = max(len(r[0]) for r in rows) + max(len(r[1]) for r in rows) + 13
    b1, b2 = _box("State 1", rows1, width), _box("State 2", rows2, width)
    line = max(len(label), 8)
    if console.width >= 2 * width + line + 5:
        arrow = Text(justify="center")
        if label:
            arrow.append(label + "\n", style="bold")
        arrow.append("━" * line + "▶", style=ACCENT)
        grid = Table.grid(padding=(0, 2))
        grid.add_column()
        grid.add_column(vertical="middle")
        grid.add_column()
        grid.add_row(b1, arrow, b2)
        console.print(grid)
    else:
        console.print(b1)
        console.print(Text(f"{'':>{width // 2 - 1}}▼  {label}", style=f"bold {ACCENT}"))
        console.print(b2)


_LEAD = ("State", "Process", "")


def _grid(headers, rows, cols):
    table = Table(box=None, padding=(0, 1), header_style="bold")
    for i in cols:
        table.add_column(headers[i], justify="left" if headers[i] in _LEAD else "right")
    for cells in rows:
        table.add_row(*(Text(cells[i][0], style=ACCENT if cells[i][1] else None) for i in cols))
    return table


def show_table(heading, headers, rows):
    """Split the columns over stacked tables when they do not fit."""
    lead = [i for i, h in enumerate(headers) if h in _LEAD]
    rest = [i for i in range(len(headers)) if i not in lead]
    for parts in range(1, len(rest) + 1):
        size = -(-len(rest) // parts)
        chunks = [rest[i:i + size] for i in range(0, len(rest), size)]
        grids = [_grid(headers, rows, lead + c) for c in chunks]
        if all(console.measure(g).minimum + 4 <= console.width for g in grids):
            break
    for k, g in enumerate(grids, 1):
        title = f"[b]{heading}[/b]" + (f" [dim]({k}/{len(grids)})[/dim]" if len(grids) > 1 else "")
        console.print(Panel(g, title=title, title_align="left", border_style=ACCENT, expand=False))


def show_result(res, calc=None):
    res.notes += list(substances.NOTICES.values())
    substances.NOTICES.clear()
    harmonize(res)
    if calc is not None and not display.hide_equations:
        show_equations(equations.for_calc(calc.id, res.model) + res.equations)
    if res.pair:
        show_pair(res.pair)
    for heading, headers, rows in res.tables:
        show_table(heading, headers, rows)
    for heading, rows in res.sections:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        for label, value, *given in rows:
            table.add_row(label, Text(value, style=ACCENT if given and given[0] else None))
        console.print(Panel(table, title=f"[b]{heading}[/b]", title_align="left", border_style=ACCENT))
    if res.diagram:
        show_diagram(res.diagram)
    for note in res.notes:
        console.print(f"[yellow]• {note}[/yellow]")


def show_diagram(d):
    left = drawing.plot(d, accent=ACCENT)
    right = drawing.equipment(d["equipment"], ACCENT) if d.get("equipment") else []
    lw = max(len(t) for t in left)
    rw = max((len(t) for t in right), default=0)
    console.print()
    if right and console.width >= lw + rw + 6:
        off = max(0, (len(left) - len(right)) // 2)
        for i in range(max(len(left), len(right) + off)):
            t = Text("  ")
            t.append_text(left[i] if i < len(left) else Text(""))
            t.pad_right(lw + 6 - len(t))
            j = i - off
            if 0 <= j < len(right):
                t.append_text(right[j])
            console.print(t, no_wrap=True)
    else:
        for t in left + ([Text("")] + right if right else []):
            console.print(Text("  ").append_text(t), no_wrap=True)
    console.print()


def _plot_factory(res):
    if res.plot is not None:
        return res.plot
    if res.diagram:
        return lambda: plots.figure([res.diagram], res.title)
    return None


def _footer(res):
    return f"model: {res.model}" if res.model else ""


def save_plot(res, open_it=True):
    factory = _plot_factory(res)
    if factory is None:
        console.print("[yellow]No plot for this result.[/yellow]")
        return
    name = res.plot_name if res.plot is not None and res.plot_name != "plot" else res.title
    with console.status("Plotting..."):
        png, data = plots.save(factory, name, _footer(res))
    console.print(f"[green]Plot saved:[/green] {png}\n[green]Plot data:[/green] {data}", soft_wrap=True)
    if open_it:
        plots.open_file(png)


def save_csv(res, open_it=True):
    path = plots.export_csv(res, res.title)
    console.print(f"[green]Table saved:[/green] {path}", soft_wrap=True)
    if open_it:
        plots.open_file(path)


def _has_table(res):
    return res is not None and bool(res.sections or res.tables or res.pair)


# Keyboard input

class Escape(Exception):
    pass


def _read_key_posix():
    import select
    fd = sys.stdin.fileno()
    b = os.read(fd, 1)
    if b == b"\x1b":
        if not select.select([fd], [], [], 0.05)[0]:
            return "esc"
        if os.read(fd, 1) not in (b"[", b"O"):
            return ""
        seq = c = os.read(fd, 1)
        while c and not (c.isalpha() or c == b"~"):
            c = os.read(fd, 1)
            seq += c
        return {b"A": "up", b"B": "down", b"C": "right", b"D": "left", b"H": "home", b"F": "end",
                b"5~": "pgup", b"6~": "pgdn", b"1~": "home", b"4~": "end", b"7~": "home", b"8~": "end"}.get(
            seq if seq.endswith(b"~") else c, "")
    if b and b[0] >= 0xC0:
        b += os.read(fd, 3 if b[0] >= 0xF0 else 2 if b[0] >= 0xE0 else 1)
    return b.decode(errors="ignore")


@contextmanager
def _cbreak():
    if os.name == "nt" or not sys.stdin.isatty():
        yield
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd, termios.TCSANOW)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_windows():
    import msvcrt
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        return {"H": "up", "P": "down", "M": "right", "K": "left", "I": "pgup", "Q": "pgdn",
                "G": "home", "O": "end"}.get(msvcrt.getwch(), "")
    return "esc" if ch == "\x1b" else ch


def ask_text(prompt):
    """Line input where ESC exits the program."""
    if not sys.stdin.isatty():
        console.print(prompt + " ", end="")
        return input().strip()
    with _cbreak():
        console.print(prompt + " ", end="")
        return _read_line(_read_key_windows if os.name == "nt" else _read_key_posix)


def select(label, options, default=None):
    """Arrow-key choice; options is a list of (key, description)."""
    keys = [k for k, _ in options]
    if not sys.stdin.isatty():
        while True:
            text = ask_text(f"  {label} [dim]({' / '.join(keys)})[/dim]").lower() or default or keys[0]
            if text in keys:
                return text
            console.print(f"  [red]Options: {', '.join(keys)}[/red]")
    i = keys.index(default) if default in keys else 0
    width = max(len(k) for k in keys)

    def draw(first):
        if not first:
            sys.stdout.write(f"\x1b[{len(options)}A")
        for j, (k, desc) in enumerate(options):
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
            pointer = f"[b {ACCENT}]❯ {k:<{width}}[/]" if j == i else f"  {k:<{width}}"
            text = f"    {pointer}  [dim]{desc}[/dim]" if desc != k else f"    {pointer}"
            console.print(text, no_wrap=True, overflow="ellipsis")

    console.print(f"  {label} [dim](use ↑ ↓ to select, Enter to confirm)[/dim]")
    sys.stdout.write("\x1b[?25l")
    read = _read_key_windows if os.name == "nt" else _read_key_posix
    try:
        with _cbreak():
            draw(True)
            while True:
                key = read()
                if key == "up":
                    i = (i - 1) % len(options)
                elif key == "down":
                    i = (i + 1) % len(options)
                elif key in ("\r", "\n"):
                    break
                elif key == "esc":
                    raise Escape
                elif key == "\x03":
                    raise KeyboardInterrupt
                else:
                    continue
                draw(False)
    finally:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()
    sys.stdout.write(f"\x1b[{len(options) + 1}A\r\x1b[J")
    sys.stdout.flush()
    desc = options[i][1]
    console.print(f"  {label}  {desc if len(desc) <= 30 else keys[i]}")
    return keys[i]


def buttons(options, default=0):
    """Horizontal button row; None when there is no terminal."""
    if not sys.stdin.isatty():
        return None
    i = default
    read = _read_key_windows if os.name == "nt" else _read_key_posix

    def draw():
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()
        row = "   ".join(f"[b reverse {ACCENT}] {o} [/]" if j == i else f"[dim] {o} [/dim]"
                         for j, o in enumerate(options))
        console.print(f"  {row}    [dim](← → to select, Enter to confirm, Esc to exit)[/dim]", end="", no_wrap=True)

    sys.stdout.write("\x1b[?25l")
    try:
        with _cbreak():
            draw()
            while True:
                key = read()
                if key in ("left", "up"):
                    i = (i - 1) % len(options)
                elif key in ("right", "down", "\t"):
                    i = (i + 1) % len(options)
                elif key in ("\r", "\n"):
                    break
                elif key == "esc":
                    raise Escape
                elif key == "\x03":
                    raise KeyboardInterrupt
                else:
                    continue
                draw()
    finally:
        sys.stdout.write("\r\x1b[2K\x1b[?25h")
        sys.stdout.flush()
    return options[i]


def _read_line(read):
    buf = ""
    while True:
        key = read()
        if key == "esc":
            console.print()
            raise Escape
        if key == "\x03":
            console.print()
            raise KeyboardInterrupt
        if key in ("\r", "\n"):
            console.print()
            return buf.strip()
        if key in ("\x7f", "\x08"):
            if buf:
                buf = buf[:-1]
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if key.isprintable():
            buf += key
            sys.stdout.write(key)
            sys.stdout.flush()


# Interactive menu

def _prompt_field(f):
    if f.type == calcs.CHOICE:
        return select(f.label, [(k, v) for k, v in f.choices.items()], f.default)
    hint = ""
    if f.type == calcs.QUANTITY:
        hint = f" [dim]({unit_label(DEFAULT_INPUT_UNITS[f.qkind]) or '-'} unless you type a unit)[/dim]"
    elif f.type == calcs.CHOICE:
        hint = f" [dim]({' / '.join(f.choices)})[/dim]"
    if f.default is not None:
        hint += f" [dim]\\[default {f.default}][/dim]"
    elif f.optional:
        hint += " [dim]\\[optional][/dim]"
    while True:
        text = ask_text(f"  {f.label}{hint}")
        try:
            return calcs.parse_value(f, text)
        except Exception as exc:
            console.print(f"  [red]{exc}[/red]")


def _ask_state(f, i):
    while True:
        out = {k: _prompt_field(f[k + i]) for k in ("T", "P")}
        out = {k: v for k, v in out.items() if v is not None}
        if len(out) == 2:
            return out
        if len(out) == 1:
            v = _prompt_field(f["V" + i])
            if v is not None:
                return {**out, "V": v}
            return {**out, "x": _prompt_field(replace(f["x" + i], optional=False))}
        console.print("  [red]Give T or P (or both).[/red]")


def _ask_spec(f, i, limit):
    """Ask T, P, V of one state, stopping once it has `limit` values."""
    while True:
        out = {}
        for k in ("T", "P", "V"):
            if len(out) == limit:
                break
            v = _prompt_field(f[k + i])
            if v is not None:
                out[k] = v
        if len(out) == limit or (limit == 2 and len(out) == 1):
            return out
        console.print(f"  [red]State {i} needs {'one value' if limit == 1 else 'one or two values'}.[/red]")


def _ask_delta(calc):
    f = {x.name: x for x in calc.fields}
    kwargs = {"sub": _prompt_field(f["sub"])}
    mode = select("Calculate", [("delta", "Δ from two states"), ("state", "a state from a given Δ")])
    if mode == "delta":
        for i in "12":
            kwargs.update({k + i: v for k, v in _ask_state(f, i).items()})
    else:
        key = select("Given property", [(k, f"Δ{p}") for k, p in properties.DELTAS.items()])
        kwargs[key] = _prompt_field(replace(f[key], optional=False))
        console.print("  [dim]Leave blank what you don't know: two values for one state, one for the other.[/dim]")
        specs = [_ask_spec(f, "1", 2), None]
        specs[1] = _ask_spec(f, "2", 3 - len(specs[0]))
        for i, sp in zip("12", specs):
            kwargs.update({k + i: v for k, v in sp.items()})
    kwargs["model"] = _prompt_field(f["model"])
    return kwargs


def _ask_pair(calc, third):
    """Ask T and P; ask the third input only if one is missing."""
    kwargs = {}
    for f in calc.fields:
        if f.name == third and kwargs.get("T_") is not None and kwargs.get("P_") is not None:
            continue
        kwargs[f.name] = _prompt_field(f)
    return kwargs


def _ask_cycle(calc):
    f = {x.name: x for x in calc.fields}
    kwargs = {}
    if calc.id == "step":
        kwargs["kind"] = _prompt_field(f["kind"])
        model = cycles.STEPS[kwargs["kind"]]
    else:
        model = cycles.MODELS[calc.id]
    gas = model.fluid == "gas"
    default = cycles.default_sub(model)
    while True:
        text = ask_text(f"  Substance [dim]\\[default {default}][/dim]") or default
        try:
            kwargs["sub"] = calcs.parse_value(f["sub"], text)
            break
        except Exception as exc:
            console.print(f"  [red]{exc}[/red]")
    if gas and select("Gas properties", [("cp(T)", "variable Cp(T), as in the air tables"),
                                         ("cold", "constant Cp (cold air-standard)")]) == "cold":
        text = ask_text("  k = cp/cv [dim]\\[default 1.4][/dim]") or "1.4"
        kwargs["k"] = calcs.parse_value(f["k"], text)
    rows = [(key, desc, f[key]) if key else (None, desc, None) for key, desc in cycles.FORMS[model.id]]
    return kwargs | form(rows)


def _ask_humid(calc):
    f = {x.name: x for x in calc.fields}
    kwargs = {}
    if "kind" in f:
        kwargs["kind"] = _prompt_field(f["kind"])
    rows = psychro.FORMS[kwargs.get("kind", "state")]
    return kwargs | form([(key, desc, f[key]) if key else (None, desc, None) for key, desc in rows])


def _unit_hint(field):
    if field.type != calcs.QUANTITY:
        return ""
    return unit_label(DEFAULT_INPUT_UNITS[field.qkind])


def form(rows):
    """Fill-in form: rows are (key, description, Field) or (None, header, None)."""
    if not sys.stdin.isatty():
        return _form_lines(rows)
    fields = [i for i, r in enumerate(rows) if r[0]]
    values = {r[0]: "" for r in rows if r[0]}
    kw = max(len(r[0]) for r in rows if r[0])
    dw = max(len(r[1]) for r in rows if r[0])
    pos, msg, top = 0, "", 0
    height = max(5, min(len(rows), shutil.get_terminal_size().lines - 9))
    scroll = height < len(rows)
    n_lines = height + 3 + (2 if scroll else 0)
    console.print("  [dim]Type what you know and leave the rest blank. Units are optional (27C, 100kPa, "
                  "800kJ/kg).[/dim]")
    console.print("  [dim]↑ ↓ move   Enter next   Esc quit[/dim]")

    def line(text):
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()
        console.print(text, no_wrap=True, overflow="ellipsis")

    def draw(first):
        nonlocal top
        if not first:
            sys.stdout.write(f"\x1b[{n_lines}A")
        cur = fields[pos] if pos < len(fields) else None
        if cur is None:
            top = len(rows) - height
        elif cur < top:
            top = cur - 1 if cur > 0 and rows[cur - 1][0] is None else cur
        elif cur >= top + height:
            top = cur - height + 1
        top = max(0, min(top, len(rows) - height))
        if scroll:
            line(f"  [dim]{'▲ more' if top > 0 else ''}[/dim]")
        for i in range(top, top + height):
            key, desc, field = rows[i]
            if key is None:
                line(f"  [b {ACCENT}]{desc}[/]")
                continue
            here = i == cur
            pointer = f"[b {ACCENT}]❯[/]" if here else " "
            value = escape(values[key])
            if here:
                value += "[reverse] [/reverse]"
            elif not value:
                value = f"[dim]{_unit_hint(field)}[/dim]"
            line(f"  {pointer} [b]{key:<{kw}}[/b]  [dim]{desc:<{dw}}[/dim]  {value}")
        if scroll:
            line(f"  [dim]{'▼ more' if top + height < len(rows) else ''}[/dim]")
        line("")
        line(f"  [b {ACCENT}]❯ Calculate[/]" if cur is None else "[dim]    Calculate[/dim]")
        line(f"  [red]{escape(msg)}[/red]" if msg else "")

    read = _read_key_windows if os.name == "nt" else _read_key_posix
    sys.stdout.write("\x1b[?25l")
    try:
        with _cbreak():
            draw(True)
            while True:
                key = read()
                name = rows[fields[pos]][0] if pos < len(fields) else None
                msg = ""
                if key == "up":
                    pos = (pos - 1) % (len(fields) + 1)
                elif key == "down":
                    pos = (pos + 1) % (len(fields) + 1)
                elif key in ("\r", "\n"):
                    if name is None:
                        try:
                            out = _parse_form(rows, values)
                            break
                        except ValueError as exc:
                            bad, msg = exc.args if len(exc.args) == 2 else (None, str(exc))
                            if bad:
                                pos = next(j for j, i in enumerate(fields) if rows[i][0] == bad)
                    else:
                        pos += 1
                elif key == "esc":
                    raise Escape
                elif key == "\x03":
                    raise KeyboardInterrupt
                elif key in ("\x7f", "\x08"):
                    if name:
                        values[name] = values[name][:-1]
                elif name and len(key) == 1 and key.isprintable():
                    values[name] += key
                else:
                    continue
                draw(False)
    finally:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()
    sys.stdout.write(f"\x1b[{n_lines + 2}A\r\x1b[J")
    sys.stdout.flush()
    console.print("  " + "   ".join(f"[b]{k}[/b] = {escape(v)}" for k, v in values.items() if v.strip()))
    return out


def _parse_form(rows, values):
    out = {}
    for key, _, field in rows:
        if key and values[key].strip():
            try:
                out[field.name] = calcs.parse_value(field, values[key])
            except Exception as exc:
                raise ValueError(key, f"{key}: {exc}")
    if not out:
        raise ValueError(None, "Type at least one value.")
    return out


def _form_lines(rows):
    while True:
        values = {}
        for key, desc, field in rows:
            if key:
                values[key] = ask_text(f"  {key} [dim]{desc} ({_unit_hint(field) or '-'})[/dim]")
        try:
            return _parse_form(rows, values)
        except ValueError as exc:
            console.print(f"  [red]{exc.args[-1]}[/red]")


CUSTOM = {
    "delta": _ask_delta,
    "pvt": lambda calc: _ask_pair(calc, "V_"),
    "state": lambda calc: _ask_pair(calc, "x"),
}
CUSTOM.update({c.id: _ask_cycle for c in calcs.CALCS if c.group == calcs.CYCLES})
CUSTOM.update({c.id: _ask_humid for c in calcs.CALCS if c.group == calcs.HUMID})


def run_interactive(calc):
    display.reset()
    settings.apply_calc(calc.id)
    substances.NOTICES.clear()
    console.rule(f"[b]{calc.title}[/b]", style=ACCENT)
    if calc.help and calc.id != "delta":
        console.print(f"[dim]{calc.help}[/dim]")
    if calc.id in CUSTOM:
        kwargs = CUSTOM[calc.id](calc)
    else:
        kwargs = {f.name: _prompt_field(f) for f in calc.fields}
    console.print()
    with console.status("Calculating..."):
        res = compute(calc, kwargs)
    show_result(res, calc)
    return res


def compute(calc, kwargs):
    """Run a calculation and keep its step log."""
    steplog.start()
    try:
        _log_inputs(calc, kwargs)
        res = calc.fn(**kwargs)
    finally:
        entries = steplog.take()
    res.log = entries
    return res


def _log_inputs(calc, kwargs):
    steplog.step("Inputs: {}", calc.title)
    for f in calc.fields:
        v = kwargs.get(f.name)
        if v is None or v == "":
            continue
        name = f.name.rstrip("_")
        if f.type == calcs.SUBSTANCE:
            steplog.line(f"{name} = {{}} ({{}}, CAS {{}}, M = {{}} g/mol)", v.name, v.formula, v.cas, f"{v.MW:.5g}")
        elif f.type == calcs.SUBSTANCES:
            steplog.line(f"{name} = {{}}", ", ".join(s.name for s in v))
        elif f.type == calcs.QUANTITY:
            steplog.line(f"{name} = {{}}", steplog.q(v, f.qkind))
        elif f.type == calcs.FLOW:
            steplog.line(f"{name} = {{}} {{}}", num(v[1]), "kg/s" if v[0] == "mass" else "mol/s")
        elif f.type == calcs.CHOICE and f.choices and v in f.choices:
            steplog.line(f"{name} = {{}} ({{}})", v, f.choices[v])
        elif isinstance(v, (list, tuple)):
            steplog.line(f"{name} = {{}}", ", ".join(map(str, v)))
        else:
            steplog.line(f"{name} = {{}}", v)


PLOT, TABLE, DETAILS = "Plot", "Table (CSV)", "Show details"


def after_result(res):
    """Menu / Show details / Plot / Table buttons; False to exit."""
    options = ["Menu"]
    if res is not None and res.log:
        options.append(DETAILS)
    if res is not None and _plot_factory(res) is not None:
        options.append(PLOT)
    if _has_table(res):
        options.append(TABLE)
    default = 0
    while (pick := buttons(options, default)) not in ("Menu", None):
        default = options.index(pick)
        try:
            if pick == DETAILS:
                show_log(res)
            elif pick == PLOT:
                save_plot(res)
            else:
                save_csv(res)
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
    return pick is not None


# Details pager

def show_log(res):
    lines = steplog.render(res.log)
    title = f"{res.title}: calculation details"
    if not sys.stdin.isatty():
        console.print(title, markup=False)
        for ln in lines:
            console.print(ln, markup=False)
        return
    import textwrap
    read = _read_key_windows if os.name == "nt" else _read_key_posix
    help_ = " ↑/↓ scroll   PgUp/PgDn page   Home/End   q close (back to the result) "
    top, width, wrapped = 0, None, []
    with console.screen(hide_cursor=True), _cbreak():
        while True:
            cols, rows = shutil.get_terminal_size()
            if cols != width:
                width, wrapped = cols, []
                for ln in lines:
                    pad = len(ln) - len(ln.lstrip()) + 2
                    wrapped += textwrap.wrap(ln, max(20, cols - 1), subsequent_indent=" " * pad) or [""]
            body = max(1, rows - 2)
            top = max(0, min(top, len(wrapped) - body))
            shown = wrapped[top:top + body]
            pos = f" lines {top + 1}-{top + len(shown)} of {len(wrapped)} "
            head = f" {title}"[: max(0, cols - len(pos))]
            out = ["\x1b[H\x1b[7m" + head + " " * max(0, cols - len(head) - len(pos)) + pos + "\x1b[0m"]
            for ln in shown:
                style = "\x1b[1m" if ln.startswith("[") else ""
                out.append("\x1b[2K" + style + ln + ("\x1b[0m" if style else ""))
            out += ["\x1b[2K"] * (body - len(shown))
            out.append("\x1b[2K\x1b[7m" + help_[:cols].ljust(cols - 1) + "\x1b[0m")
            sys.stdout.write("\r\n".join(out))
            sys.stdout.flush()
            key = read()
            if key in ("q", "Q", "esc"):
                return
            if key == "\x03":
                raise KeyboardInterrupt
            step = {"up": -1, "k": -1, "down": 1, "j": 1, "\r": 1, "\n": 1, "pgup": -body, "b": -body,
                    "pgdn": body, " ": body, "home": -len(wrapped), "g": -len(wrapped),
                    "end": len(wrapped), "G": len(wrapped)}.get(key)
            if step:
                top += step


def units_menu():
    console.rule("[b]Display units[/b]", style=ACCENT)
    follow = select("Show results in the units you type?", [
        ("yes", "Yes, typed units win over the ones below"),
        ("no", "No, always use the units below"),
    ], "yes" if display.follow_typed else "no")
    display.follow_typed = follow == "yes"
    for kind in ("temperature", "pressure", "molar_volume", "power"):
        options = [unit_label(o) for o in units_of(kind)]
        current = display.unit_for(kind)
        text = ask_text(f"  {kind.replace('_', ' ')} [dim]({', '.join(options)}) [{current}][/dim]") or current
        if _norm_unit(text) in [_norm_unit(o) for o in options]:
            display.adopt(kind, text, force=True)
        else:
            console.print(f"  [red]Unknown unit, keeping {current}[/red]")
    family = select("energy", [
        ("J", "J/mol, J/(mol·K)   (kJ when most values are large)"),
        ("cal", "cal/mol, cal/(mol·K)   (kcal when most values are large)"),
    ], "cal" if "cal" in display.molar_energy else "J")
    display.adopt("molar_energy", f"{family}/mol", force=True)
    display.adopt("molar_entropy", f"{family}/(mol·K)", force=True)
    text = ask_text(f"  minimum decimal places [dim](2-8) [{display.min_decimals}][/dim]")
    try:
        _set_decimals(text or display.min_decimals)
    except ValueError as exc:
        console.print(f"  [red]{exc}, keeping {display.min_decimals}[/red]")
    display.save()
    try:
        settings.store()
        console.print(f"  [dim]Saved in {settings.path()}[/dim]")
    except (OSError, ValueError) as exc:
        console.print(f"  [red]Could not save the settings: {exc}[/red]")


def _set_decimals(n):
    n = int(n) if str(n).strip().isdigit() else -1
    if not 2 <= n <= 8:
        raise ValueError("Minimum decimal places must be a whole number from 2 to 8")
    display.min_decimals = n


def menu():
    if not display.hide_logo:
        print_logo()
    console.print(
        "\nInputs accept units: [b]25 C[/b], [b]10 atm[/b], [b]300[/b] (default unit). "
        "Substances by name, formula or CAS.\n"
    )
    while True:
        items = []
        table = Table(show_header=False, box=None, padding=(0, 1))
        for group, entries in calcs.groups().items():
            table.add_row("", f"[b {ACCENT}]{group}[/]")
            for c in entries:
                items.append(c)
                table.add_row(f"[b]{len(items)}[/b]", c.title)
        table.add_row("", "")
        s = display.saved
        table.add_row("[b]u[/b]", f"Display units [dim](T {s('temperature')}, P {s('pressure')}, "
                                   f"E {s('molar_energy')}, ≥{s('min_decimals')} decimals, "
                                   f"{'typed units win' if s('follow_typed') else 'fixed'})[/dim]")
        table.add_row("[b]Esc[/b]", "Exit")
        console.print(table)
        choice = ask_text("\n=>").lower()
        console.print()
        if choice == "u":
            units_menu()
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(items):
            console.print("[red]Invalid option.[/red]\n")
            continue
        res = None
        try:
            res = run_interactive(items[int(choice) - 1])
        except (Escape, KeyboardInterrupt):
            raise
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
        console.print()
        if not after_result(res):
            return
        console.rule(style="dim")
        console.print()


# Command line

def _apply_units(t_unit, p_unit, e_unit, v_unit):
    for kind, unit in (("temperature", t_unit), ("pressure", p_unit), ("molar_energy", e_unit),
                       ("molar_volume", v_unit)):
        if unit:
            display.adopt(kind, unit, force=True)
    for kind, unit in (("specific_energy", e_unit), ("specific_volume", v_unit)):
        if unit:
            display.adopt(kind, unit, force=True)


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        try:
            with _cbreak():
                menu()
        except (Escape, KeyboardInterrupt, EOFError):
            pass


def _usage(f):
    name, opt = f.name.rstrip("_"), f.optional or f.default is not None
    if f.type in (calcs.SUBSTANCE, calcs.SUBSTANCES):
        return f"\\[{name}]" if opt else f"<{name}>"
    return f"{name}=" + ("?" if opt else "")


@app.command("list")
def list_calcs():
    """List every calculation and its inputs."""
    for group, entries in calcs.groups().items():
        console.print(f"\n[b {ACCENT}]{group}[/]")
        for c in entries:
            if c.group == calcs.CYCLES:
                args = _cycle_usage(c)
            else:
                args = " ".join(_usage(f) for f in c.fields)
            console.print(f"  [b]{c.id:<15}[/b] {c.title}\n  {'':<15} [dim]{args}[/dim]")
            if c.id == "device":
                console.print(f"  {'':<15} [dim]or: {' / '.join(DEVICES)} <sub> T1= P1= P2= ...[/dim]")


def _cycle_usage(c):
    if c.id == "step":
        return f"<{'|'.join(cycles.STEPS)}> \\[sub] key=value ...   (see the keys in the menu)"
    model = cycles.MODELS[c.id]
    others = [v.key for v in model.primaries + model.outputs if not v.key[-1].isdigit() and v.key != "strokes"]
    gas = "[k=] " if model.fluid == "gas" else ""
    return (f"\\[sub] {gas}any of: {cycles.STATE_KEYS[model.fluid]} (1-4), " + " ".join(others))


DEVICES = ("compressor", "pump", "turbine", "valve")


@app.command(hidden=True)
def run(
    calc_id: str = typer.Argument(..., help="Calculation id (see 'thermo-utils list')"),
    args: Optional[List[str]] = typer.Argument(None, help="Substances, then inputs as key=value"),
    plot: bool = typer.Option(False, "--plot", help="Save the plot (PNG + data CSV) and open it"),
    table: bool = typer.Option(False, "--csv", help="Save the result as a CSV table and open it"),
    no_open: bool = typer.Option(False, "--no-open", help="With --plot/--csv: save without opening"),
    t_unit: str = typer.Option(None, "--T-unit", help="Display unit for temperature"),
    p_unit: str = typer.Option(None, "--P-unit", help="Display unit for pressure"),
    e_unit: str = typer.Option(None, "--E-unit", help="Display unit for molar energy"),
    v_unit: str = typer.Option(None, "--V-unit", help="Display unit for molar volume"),
    decimals: int = typer.Option(None, "--decimals", help="Minimum decimal places shown (2-8, default 4)"),
    details: bool = typer.Option(False, "--details", help="Print the step-by-step calculation log"),
):
    """Run one calculation without the menu."""
    given = {}
    if calc_id in DEVICES:
        calc_id, given["kind"] = "device", calc_id
    try:
        calc = calcs.by_id(calc_id)
    except KeyError:
        console.print(f"[red]Unknown calculation '{calc_id}'. See 'thermo-utils list'.[/red]")
        raise typer.Exit(1)
    settings.apply_calc(calc.id)
    _apply_units(t_unit, p_unit, e_unit, v_unit)
    if decimals is not None:
        try:
            _set_decimals(decimals)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1)
    loose = []
    for item in args or []:
        if "=" in item:
            k, v = item.split("=", 1)
            given[k.strip().lower()] = v
        else:
            loose.append(item)
    for f in calc.fields:
        key = f.name.rstrip("_").lower()
        if loose and f.type == calcs.CHOICE and key not in given and loose[0].lower() in f.choices:
            given[key] = loose.pop(0)
        if loose and f.type in (calcs.SUBSTANCE, calcs.SUBSTANCES) and key not in given:
            given[key] = loose.pop(0)
    if loose:
        console.print(f"[red]Unexpected input '{loose[0]}'. Use key=value.[/red]")
        raise typer.Exit(1)
    kwargs = {}
    try:
        for f in calc.fields:
            key = f.name.rstrip("_").lower()
            kwargs[f.name] = calcs.parse_value(f, given.pop(key, ""))
        if given:
            raise ValueError(f"Unknown inputs: {', '.join(given)}")
        res = compute(calc, kwargs)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    show_result(res, calc)
    if details:
        console.print()
        console.rule("[b]Calculation details[/b]", style="dim")
        for ln in steplog.render(res.log):
            console.print(ln, markup=False, highlight=False, soft_wrap=True)
    if plot:
        save_plot(res, not no_open)
    if table:
        save_csv(res, not no_open)


@app.command("config")
def config():
    """Create the settings file if missing, check it and show where it is."""
    created = settings.create()
    settings.load([c.id for c in calcs.CALCS])
    console.print(f"Settings file: {settings.path()}" + ("  [dim](created with the current values)[/dim]" if created else ""))
    if not settings.WARNINGS:
        console.print("[green]No problems found.[/green]")


def main():
    settings.load([c.id for c in calcs.CALCS])
    for w in settings.WARNINGS:
        console.print(f"[yellow]Settings: {escape(w)}[/yellow]")
    argv, i = sys.argv[1:], 0
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i].endswith("-unit") or argv[i] == "--decimals" else 1
    if i < len(argv) and argv[i] not in ("list", "run", "config"):
        sys.argv[1:] = ["run", argv[i]] + argv[:i] + argv[i + 1:]
    app()


if __name__ == "__main__":
    main()
