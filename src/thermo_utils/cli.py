import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from thermo_utils import __version__, calcs, equations, plots
from thermo_utils.units import DEFAULT_INPUT_UNITS, display, units_of

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
        console.print(Panel(body, title="[b]Equations[/b]", title_align="left", border_style="dim"))


def show_result(res, calc=None):
    if calc is not None:
        show_equations(equations.for_calc(calc.id, res.model))
    for heading, rows in res.sections:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        for label, value in rows:
            table.add_row(label, value)
        console.print(Panel(table, title=f"[b]{heading}[/b]", title_align="left", border_style=ACCENT))
    for note in res.notes:
        console.print(f"[yellow]• {note}[/yellow]")


def handle_plot(res, ask=True, save=False, open_it=False):
    if res.plot is None:
        return
    if ask:
        save = ask_text("Save a plot of this result? [dim](y/N)[/dim]").lower().startswith("y")
        if not save:
            return
        open_it = not ask_text("Open it now? [dim](Y/n)[/dim]").lower().startswith("n")
    if not save:
        return
    with console.status("Plotting..."):
        path = plots.save(res.plot, res.plot_name)
    console.print(f"[green]Plot saved:[/green] {path}")
    if open_it:
        plots.open_file(path)


# Keyboard input

class Escape(Exception):
    pass


def _read_key_posix():
    import select
    fd = sys.stdin.fileno()
    b = os.read(fd, 1)
    if b == b"\x1b":
        if select.select([fd], [], [], 0.05)[0]:
            os.read(fd, 16)
            return ""
        return "esc"
    if b and b[0] >= 0xC0:
        b += os.read(fd, 3 if b[0] >= 0xF0 else 2 if b[0] >= 0xE0 else 1)
    return b.decode(errors="ignore")


@contextmanager
def _cbreak():
    if os.name == "nt":
        yield
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_windows():
    import msvcrt
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        msvcrt.getwch()
        return ""
    return "esc" if ch == "\x1b" else ch


def ask_text(prompt):
    """Line input where ESC exits the program."""
    if not sys.stdin.isatty():
        console.print(prompt + " ", end="")
        return input().strip()
    with _cbreak():
        console.print(prompt + " ", end="")
        return _read_line(_read_key_windows if os.name == "nt" else _read_key_posix)


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
    hint = ""
    if f.type == calcs.QUANTITY:
        hint = f" [dim]({DEFAULT_INPUT_UNITS[f.qkind] or '-'} unless you type a unit)[/dim]"
    elif f.type == calcs.CHOICE:
        hint = f" [dim]({' / '.join(f.choices)})[/dim]"
    if f.default is not None:
        hint += f" [dim][default {f.default}][/dim]"
    elif f.optional:
        hint += " [dim][optional][/dim]"
    while True:
        text = ask_text(f"  {f.label}{hint}")
        try:
            return calcs.parse_value(f, text)
        except Exception as exc:
            console.print(f"  [red]{exc}[/red]")


def run_interactive(calc):
    console.rule(f"[b]{calc.title}[/b]", style=ACCENT)
    if calc.help:
        console.print(f"[dim]{calc.help}[/dim]")
    kwargs = {f.name: _prompt_field(f) for f in calc.fields}
    console.print()
    with console.status("Calculating..."):
        res = calc.fn(**kwargs)
    show_result(res, calc)
    handle_plot(res)


def units_menu():
    console.rule("[b]Display units[/b]", style=ACCENT)
    for kind in ("temperature", "pressure", "molar_volume", "molar_energy", "molar_entropy", "power"):
        options = units_of(kind)
        current = display.unit_for(kind)
        text = ask_text(f"  {kind.replace('_', ' ')} [dim]({', '.join(options)}) [{current}][/dim]") or current
        if text.lower() in [o.lower() for o in options]:
            setattr(display, kind, text)
        else:
            console.print(f"  [red]Unknown unit, keeping {current}[/red]")


def menu():
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
        table.add_row("[b]u[/b]", f"Display units [dim](T {display.temperature}, P {display.pressure}, "
                                   f"E {display.molar_energy})[/dim]")
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
        try:
            run_interactive(items[int(choice) - 1])
        except (Escape, KeyboardInterrupt):
            raise
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
        return


# Command line

def _apply_units(t_unit, p_unit, e_unit, v_unit):
    for kind, unit in (("temperature", t_unit), ("pressure", p_unit), ("molar_energy", e_unit),
                       ("molar_volume", v_unit)):
        if unit:
            setattr(display, kind, unit)


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        try:
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
            args = " ".join(_usage(f) for f in c.fields)
            console.print(f"  [b]{c.id:<15}[/b] {c.title}\n  {'':<15} [dim]{args}[/dim]")
            if c.id == "device":
                console.print(f"  {'':<15} [dim]or: {' / '.join(DEVICES)} <sub> T1= P1= P2= ...[/dim]")


DEVICES = ("compressor", "pump", "turbine", "valve")


@app.command(hidden=True)
def run(
    calc_id: str = typer.Argument(..., help="Calculation id (see 'thermo-utils list')"),
    args: Optional[List[str]] = typer.Argument(None, help="Substances, then inputs as key=value"),
    plot: bool = typer.Option(False, "--plot", help="Save a plot when available"),
    open_plot: bool = typer.Option(False, "--open", help="Open the saved plot"),
    t_unit: str = typer.Option(None, "--T-unit", help="Display unit for temperature"),
    p_unit: str = typer.Option(None, "--P-unit", help="Display unit for pressure"),
    e_unit: str = typer.Option(None, "--E-unit", help="Display unit for molar energy"),
    v_unit: str = typer.Option(None, "--V-unit", help="Display unit for molar volume"),
):
    """Run one calculation without the menu."""
    _apply_units(t_unit, p_unit, e_unit, v_unit)
    given = {}
    if calc_id in DEVICES:
        calc_id, given["kind"] = "device", calc_id
    try:
        calc = calcs.by_id(calc_id)
    except KeyError:
        console.print(f"[red]Unknown calculation '{calc_id}'. See 'thermo-utils list'.[/red]")
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
        res = calc.fn(**kwargs)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)
    show_result(res, calc)
    handle_plot(res, ask=False, save=plot or open_plot, open_it=open_plot)


def main():
    argv, i = sys.argv[1:], 0
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i].endswith("-unit") else 1
    if i < len(argv) and argv[i] not in ("list", "run"):
        sys.argv[1:] = ["run", argv[i]] + argv[:i] + argv[i + 1:]
    app()


if __name__ == "__main__":
    main()
