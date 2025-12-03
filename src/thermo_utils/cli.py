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
COMMANDS = "Type a number:\n\n1 - PVT calculus (ideal gases equation)\n2 - PVT calculus (virial equation)\n3 - PVT calculus (Peng-Robinson equation)\n4 - Fugacity (pure substance)\n5 - Fugacity (mix of two components)\n6 - Activity coefficient (pure substance)\n7 - Activity coefficient (mix of two components)\n8 - Margules parameters using experimental data (T constant)\n0 - Exit"

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
            V=read_float("V: ")
            T=read_float("T: ")
            pvtIdeal(P,V,T)
        elif a == 2:
            print("Leave empty the one you want to calculate:")
            print("(V in cm³/mol)\n")
            P=read_float("P: ")
            V=read_float("V: ")
            T=read_float("T: ")
            Pc=read_float("Pc: ")
            Tc=read_float("Tc: ")
            Omega=read_float("Acentric factor: ")
            pvtVirial(P,V,T,Pc,Tc,Omega)
        elif a == 3:
            print("Leave empty the one you want to calculate:")
            print("(V in cm³/mol)\n")
            P=read_float("P: ")
            V=read_float("V: ")
            T=read_float("T: ")
            Pc=read_float("Pc: ")
            Tc=read_float("Tc: ")
            Omega=read_float("Acentric factor: ")
            pvtPengRobinson(P,V,T,Pc,Tc,Omega)
        elif a == 4:
            P=read_float("P: ")
            T=read_float("T: ")
            Pc=read_float("Pc: ")
            Tc=read_float("Tc: ")
            Omega=read_float("Acentric factor: ")
            fugacity(P,T,Pc,Tc,Omega)
        elif a == 5:
            P=read_float("P: ")
            T=read_float("T: ")
            Pc1=read_float("Pc of the component A: ")
            Tc1=read_float("Tc of the component A: ")
            Omega1=read_float("Acentric factor of the component A: ")
            Y1=read_float("Gas fraction of the component A (0-1): ")
            Pc2=read_float("Pc of the component B: ")
            Tc2=read_float("Tc of the component B: ")
            Omega2=read_float("Acentric factor of the component B: ")
            fugacityMix(P,T,Pc1,Tc1,Omega1,Y1,Pc1,Tc1,Omega2)
        elif a == 6:
            P=read_float("P: ")
            T=read_float("T: ")
            Pc1=read_float("Pc of the component A: ")
            Tc1=read_float("Tc of the component A: ")
            Omega1=read_float("Acentric factor of the component A: ")
            Y1=read_float("Gas fraction of the component A (0-1): ")
            Pc2=read_float("Pc of the component B: ")
            Tc2=read_float("Tc of the component B: ")
            Omega2=read_float("Acentric factor of the component B: ")
            activity(P,T,Pc1,Tc1,Omega1,Y1,Pc1,Tc1,Omega2)
        elif a == 7:
            P=read_float("P: ")
            T=read_float("T: ")
            Pc1=read_float("Pc of the component A: ")
            Tc1=read_float("Tc of the component A: ")
            Omega1=read_float("Acentric factor of the component A: ")
            Y1=read_float("Gas fraction of the component A (0-1): ")
            Pc2=read_float("Pc of the component B: ")
            Tc2=read_float("Tc of the component B: ")
            Omega2=read_float("Acentric factor of the component B: ")
            activityMix(P,T,Pc1,Tc1,Omega1,Y1,Pc1,Tc1,Omega2)
        elif a == 8:
            T=read_float("T: ")
            Pc1=read_float("Pc of the component A: ")
            Tc1=read_float("Tc of the component A: ")
            Omega1=read_float("Acentric factor of the component A: ")
            Pc2=read_float("Pc of the component B: ")
            Tc2=read_float("Tc of the component B: ")
            Omega2=read_float("Acentric factor of the component B: ")
            print("Import a CSV file with the experimental data (P,x1,y1) sending its path. Example: C:/Users/Daniel/Documents/experimental_data.csv\n")
            csv_path = input("Path: ")
            margules(T,Pc1,Tc1,Omega1,Pc2,Tc2,Omega2,csv_path)
        elif a == 0:
            break
        else:
            print("Invalid input.\n")

if __name__ == "__main__":
    cli()