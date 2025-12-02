import shutil
import sys
from colorama import init
init()


LOGO = r"""
              (        )  (        )      (          )    
   )   (      )\    ( /(  )\    ( /(      )\ )     (/(   )
  (\   ))\   ( ((   )\())((_)  ))\(_\  ( ((_)(   )\ )\  ((
   (-)_) )\ )\/_ )\((_)\ (_ )\)(_)\(()/  )\ _((_))((_)| \_)-)\
   / /_/ /(_|))_))((_)_((_))_((_) (_)\ \) ) ))((_))/ /(_)/ /\__)
  / __/ __ \/ _ \/ ___/ __ `__ \/ __ \_(\/__/ / / / __/ / / ___/
/ /_/ / / /  __/ /  / / / / / / /_/ /_____/ /_/ / /_/ / (__  )
\__/_/ /_/\___/_/  /_/ /_/ /_/\____/      \__,_/\__/_/_/____/
"""
VERSION = "version 0.1.0"
MESSAGE = "This program uses bar, cm³, mol and K.\nPlease type your input values using these units.\n\n"
COMMANDS = "Type a number:\n\n1 - PVT calculus (ideal gases equation)\n2 - PVT calculus (virial equation)\n3 - PVT calculus (Peng-Robinson equation)\n4 - Fugacity\n5 - Activity coefficient\n6 - Margules parameters using experimental data (T constant)\n0 - Exit"

CSI = "\033["
RESET = "\033[0m"

COLORS = [
    (255, 220, 80),   # yellow
    (255, 150, 40),   # orange
    (230, 70, 30),    # red-orange
    (180, 40, 50),    # dark red
]

def lerp(a, b, t):
    return int(a + (b - a) * t)

def get_color(t):
    # t = 0..1
    g = len(COLORS) - 1
    i = int(t * g)
    if i >= g:
        return COLORS[-1]
    a = COLORS[i]
    b = COLORS[i + 1]
    f = t * g - i
    return tuple(lerp(a[k], b[k], f) for k in range(3))

def colorize(art):
    lines = art.splitlines()
    h = len(lines)
    out = []
    for i, line in enumerate(lines):
        t = i / max(1, h - 1)
        r, g, b = get_color(t)
        color = f"{CSI}38;2;{r};{g};{b}m"
        new_line = "".join(color + ch + RESET if ch != " " else " " for ch in line)
        out.append(new_line)
    return "\n".join(out)

def center_text(text: str) -> str:
    width = shutil.get_terminal_size().columns
    lines = text.split("\n")
    return "\n".join(line.center(width) for line in lines)

#===========================================================#

def read_float(prompt):
    value = input(prompt)
    if value.strip() == "":    
        return None
    return float(value)


def cli():
   
    centered_logo = center_text(LOGO)
    centered_colorized_logo = colorize(centered_logo)
    print(centered_colorized_logo)
    print(center_text(VERSION))
    print("\n")
    print((MESSAGE))
    
    while True:
        
        print(COMMANDS)
        print("\n")
        
        a = int(input("=> "))
        print("\n")
        if a == 1:
            print("Leave empty the one you want to calculate:")
            print("(V in cm³/mol)\n")
            P=read_float("P: ")
            V=read_float("v: ")
            T=read_float("T: ")
            PVTideal(P,V,T)
        elif a == 2:
            print("Leave empty the one you want to calculate:")
            print("(V in cm³/mol)\n")
            P=read_float("P: ")
            V=read_float("v: ")
            T=read_float("T: ")
            Pc=read_float("Pc: ")
            Tc=read_float("Tc: ")
            Omega=read_float("Acentric factor: ")
        elif a == 3:
            print("Leave empty the one you want to calculate:")
            print("(V in cm³/mol)\n")
            P=read_float("P: ")
            V=read_float("v: ")
            T=read_float("T: ")
            Pc=read_float("Pc: ")
            Tc=read_float("Tc: ")
            Omega=read_float("Acentric factor: ")
        elif a == 4:
            print("Fugacity selected.\n")
        elif a == 5:
            print("Activity coefficient selected.\n")
        elif a == 6:
            print("Margules parameters selected.\n")
        elif a == 0:
            break
        else:
            print("Invalid input.\n")

if __name__ == "__main__":
    cli()