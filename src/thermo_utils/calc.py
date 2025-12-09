import numpy as np
from scipy.optimize import fsolve

R = 83.144598

def pvtIdeal(P,V,T):

    if P is None:

        P = (R*T) / V
        return P

    elif V is None:

        V = (R*T) / P
        return V

    elif T is None:

        T = (P*V) / R
        return T

    return None

def calcB(T,Tc,Pc,omega):

    B0 = 0.083 - ((0.422) / (T/Tc)**1.6)
    B1 = 0.139 - ((0.172) / (T/Tc)**4.2)
    B = (R*Tc/Pc) * (B0 + (omega*B1))

    return B

def pvtVirial(P,V,T,Pc,Tc,omega):

    if P is None:

        B = calcB(T,Tc,Pc,omega)
        P = (R*T) / (V-B)
        return P

    elif V is None:

        B = calcB(T,Tc,Pc,omega)
        V = ((R*T) / P) + B
        return V

    elif T is None:

        def f(Tguess): 
            
            B = calcB(Tguess,Tc,Pc,omega)
            zero = P*(V - B)/R - Tguess   # T = P*(V - B)/R
            return zero

        T = fsolve(f, Tc)[0]
        return T

    return None

def pvtPengRobinson(P,V,T,Pc,Tc,omega):
    
    if P is None:

        alpha = ( 1 + (0.37464 + 1.54226*omega - 0.26992*(omega**2)) * (1-(T/Tc)**(1/2)) )**2
        a = (0.457236*alpha*(R**2)*(Tc**2))/Pc
        b = (0.0777961*R*Tc)/Pc

        P = ((R*T)/(V-b)) - (a/((V**2) + 2*V*b - (b**2)))
        return P
        
    elif V is None:

        def f(Vguess):
            
            alpha = ( 1 + (0.37464 + 1.54226*omega - 0.26992*(omega**2)) * (1-(T/Tc)**(1/2)) )**2
            a = (0.457236*alpha*(R**2)*(Tc**2))/Pc
            b = (0.0777961*R*Tc)/Pc
            zero = ((R*T)/(Vguess-b)) - (a/((Vguess**2) + 2*Vguess*b - (b**2))) - P
            return zero
        
        Vgas_guess = R*T/P
        Vliq_guess = 1.1*((0.0777961*R*Tc)/Pc)
        Vliq = fsolve(f,Vliq_guess)[0]
        Vgas = fsolve(f,Vgas_guess)[0]
        return [Vliq, Vgas]
        
    elif T is None:

        def f(Tguess):

            alpha = ( 1 + (0.37464 + 1.54226*omega - 0.26992*(omega**2)) * (1-(Tguess/Tc)**(1/2)) )**2
            a = (0.457236*alpha*(R**2)*(Tc**2))/Pc
            b = (0.0777961*R*Tc)/Pc
            zero = ((R*Tguess)/(V-b)) - (a/((V**2) + 2*V*b - (b**2))) - P
            return zero

        T = fsolve(f,Tc)[0]
        return T

    return None

