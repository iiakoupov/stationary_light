# Copyright (c) 2017 Ivan Iakoupov
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from mpmath import findroot, sqrt, eye, cos, pi, matrix, exp, acos
import numpy as np

def q_eit(delta, g1d, Deltac, Omega):
    gprime = 1-g1d
    Delta = Deltac+delta
    return -0.5*g1d*delta/((Delta+0.5j*gprime)*delta-Omega**2)

def eq_secular(q, delta, g1d, Deltac, Omega):
    gprime = 1-g1d
    Delta = Deltac+delta
    eta = Omega**2/(4*delta*(Delta+0.5j*gprime))
    return q**2-(g1d/(2*Delta+1j*gprime))**2/(1-2*eta)

def q_secular(delta, g1d, Deltac, Omega):
    gprime = 1-g1d
    Delta = Deltac+delta
    eta = Omega**2/(4*delta*(Delta+0.5j*gprime))
    return -g1d/(2*Delta+1j*gprime)/sqrt(1-2*eta)

def delta_secular(q, g1d, Deltac, Omega, delta_starting):
    return findroot(lambda x: eq_secular(q, x, g1d, Deltac, Omega), delta_starting, solver='muller')

def eq_Lambda_cold(q, delta, g1d, Deltac, Omega):
    gprime = 1-g1d
    Delta = Deltac+delta
    x = Omega**2/(delta*(Delta+0.5j*gprime))
    return q**2-(g1d/(Delta+0.5j*gprime))**2*(2-x-2*sqrt(1-x))/(x**2*sqrt(1-x))

def q_Lambda_cold(delta, g1d, Deltac, Omega):
    if (delta == 0):
        return 0
    gprime = 1-g1d
    Delta = Deltac+delta
    x = Omega**2/(delta*(Delta+0.5j*gprime))
    return -g1d/(2*(Delta+0.5j*gprime))*sqrt((4*(-1+sqrt(1-x))**2)/(sqrt(1-x)*x**2))
    #return g1d/(2*(Delta+0.5j*gprime))*(2*(-1+sqrt(1-x)))/((1-x)**(0.25)*x)

def delta_Lambda_cold(q, g1d, Deltac, Omega, delta_starting):
    return findroot(lambda x: eq_Lambda_cold(q, x, g1d, Deltac, Omega), delta_starting, solver='muller')

def q_Lambda_cold_numeric(delta, g1d, Deltac, Omega, n):
    if (delta == 0):
        return 0
    total_n = 2*n+1
    Delta = Deltac+delta
    gprime = 1-g1d
    Delta_tilde = Delta+0.5j*(1-g1d)
    M = np.zeros((total_n, total_n), dtype=np.complex128)
    M[n,n] = delta # the middle element
    for i in range(1, n+1):
        if i%2 == 1:
            # We are now adding two rows of \sigma_{ab}^{(\pm i)}
            # (i odd) plus the coupling terms on the lower order
            # rows
            M[n-i,n-i] = Delta_tilde
            M[n-i,n-(i-1)] = 0.5*Omega
            M[n-(i-1), n-i] = 0.5*Omega.conjugate()
            M[n+i, n+i] = Delta_tilde
            M[n+i, n+(i-1)] = 0.5*Omega
            M[n+(i-1), n+i] = 0.5*Omega.conjugate()
        else:
            # We are now adding two rows of \sigma_{ac}^{(\pm i)}
            # (i even) plus the coupling terms on the lower order
            # rows
            M[n-i, n-i] = delta
            M[n-i, n-(i-1)] = 0.5*Omega.conjugate()
            M[n-(i-1), n-i] = 0.5*Omega
            M[n+i, n+i] = delta
            M[n+i, n+(i-1)] = 0.5*Omega.conjugate()
            M[n+(i-1), n+i] = 0.5*Omega
    V = np.zeros((total_n, 2), dtype=np.complex128)
    V[n-1, 0] = 1
    V[n+1, 1] = 1
    M2 = np.dot(np.transpose(V),np.linalg.solve(M,V))
    if delta > 0:
        return 0.5*g1d*np.sqrt(M2[0,0]*M2[1,1]-M2[1,0]*M2[0,1])
    else:
        return -0.5*g1d*np.sqrt(M2[0,0]*M2[1,1]-M2[1,0]*M2[0,1])

def mat_dual_color_numeric(delta, g1d, Deltac, Deltad, Omega, n):
    total_n = 2*n+1
    Delta = Deltac+delta
    gprime = 1-g1d
    Delta_tilde = Delta+0.5j*(1-g1d)
    M = np.zeros((total_n, total_n), dtype=np.complex128)
    M[n,n] = delta # the middle element
    for i in range(1, n+1):
        if i%2 == 1:
            # We are now adding two rows of \sigma_{ab}^{(\pm i)}
            # (i odd) plus the coupling terms on the lower order
            # rows
            M[n-i,n-i] = Delta_tilde-i*Deltad
            M[n-i,n-(i-1)] = 0.5*Omega
            M[n-(i-1), n-i] = 0.5*Omega.conjugate()
            M[n+i, n+i] = Delta_tilde+i*Deltad
            M[n+i, n+(i-1)] = 0.5*Omega
            M[n+(i-1), n+i] = 0.5*Omega.conjugate()
        else:
            # We are now adding two rows of \sigma_{ac}^{(\pm i)}
            # (i even) plus the coupling terms on the lower order
            # rows
            M[n-i, n-i] = delta-i*Deltad
            M[n-i, n-(i-1)] = 0.5*Omega.conjugate()
            M[n-(i-1), n-i] = 0.5*Omega
            M[n+i, n+i] = delta+i*Deltad
            M[n+i, n+(i-1)] = 0.5*Omega.conjugate()
            M[n+(i-1), n+i] = 0.5*Omega
    V = np.zeros((total_n, 2), dtype=np.complex128)
    V[n-1, 0] = 1
    V[n+1, 1] = 1
    M2 = np.dot(np.transpose(V),np.linalg.solve(M,V))
    return M2

def q_dual_color_numeric(delta, g1d, Deltac, Deltad, Omega, n):
    M2 = mat_dual_color_numeric(delta, g1d, Deltac, Deltad, Omega, n)
    #print("M[1,0]-M[0,1]={}".format(M2[1,0]-M2[0,1]))
    a = 1
    b = -0.5*g1d*(-M2[0,0]+M2[1,1])
    c = -(0.5*g1d)**2*(M2[0,0]*M2[1,1]-M2[1,0]*M2[0,1])
    return [(-b+np.sqrt(b**2-4*a*c))/(2*a), (-b-np.sqrt(b**2-4*a*c))/(2*a)]

def eq_dual_color_numeric(q, delta, g1d, Deltac, Deltad, Omega, n):
    M2 = mat_dual_color_numeric(delta, g1d, Deltac, Deltad, Omega, n)
    a = 1
    b = -0.5*g1d*(-M2[0,0]+M2[1,1])
    c = -(0.5*g1d)**2*(M2[0,0]*M2[1,1]-M2[1,0]*M2[0,1])
    return q**2+b*q+c

def delta_dual_color_numeric(q, g1d, Deltac, Deltad, Omega, n, delta_starting):
    return findroot(lambda x: eq_dual_color_numeric(q, x, g1d, Deltac, Deltad, Omega, n), delta_starting, solver='muller', tol=1e-10, maxsteps=10000)

def q_eit_standing_wave(Delta, Deltac, Omega, g1d, periodLength, phaseShift):
    if Delta == Deltac:
        return 0
    gprime = 1-g1d
    kd = pi/periodLength
    Mcell = eye(2)
    for i in range(periodLength):
        OmegaAtThisSite = Omega*cos(kd*i+pi*phaseShift)
        beta3 = (g1d*(Delta-Deltac))/((-2.0j*Delta+gprime)*(Delta-Deltac)+2.0j*OmegaAtThisSite**2)
        M3 = matrix([[1-beta3, -beta3], [beta3, 1+beta3]])
        Mf = matrix([[exp(1j*kd), 0], [0, exp(-1j*kd)]])
        Mcell = Mf*M3*Mcell
    ret = (1.0/periodLength)*acos(-0.5*(Mcell[0,0]+Mcell[1,1]))
    return ret
