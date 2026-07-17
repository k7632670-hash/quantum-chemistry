# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:08:54 2026

@author: kaz2255pq
"""
import numpy as np
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

import scipy.special
import time
#from scipy.special import erf
from math import erf
from numba import jit, f8, i8

import cProfile
from collections import Counter

###Load modules
from basis import GaussianBasis
from properties import MolecularProperties
from integrals import GaussianIntegrals 
from scfdriver import SCFDriver


covalent_radius = {
     "H": 0.31,   # H
     "He": 0.28,   # He
     "Li": 1.28,   # Li
     "Be": 0.96,   # Be
     "B": 0.84,   # B
     "C": 0.76,   # C
     "N": 0.71,   # N
     "O": 0.66,   # O
     "F": 0.57,   # F
    "Ne": 0.58,   # Ne
    }


def primitive_gto(x,y,z,A,dA,aL):
    l,m,n = aL
    dx = x - A[0]
    dy = y - A[1]
    dz = z - A[2]
    
    r2 = dx**2 + dy**2 + dz**2
    poly = dx**l * dy**m * dz**n
    val = poly * np.exp(-dA*r2)
    #print(val.shape)
    return val

def contracted_gto(x,y,z,A,dA,coef, aL):
    val = 0 
    for beta,alpha in zip(dA,coef):
        
        N = primitive_norm(beta,aL)
        val += alpha * N *primitive_gto(x,y,z,A,beta,aL)
    
    return val

def molecular_orbital(x,y,z,C,A,dA,coef,aL):
    psi = np.zeros_like(x)
    for mu in range(C.size):
        chi = contracted_gto(x,y,z,A,dA,coef,aL)
        psi += C[mu] * chi
        
    return psi

def fact2(n):
    if n <= 0:
        return 1
    return scipy.special.factorial2(n)

@jit
def boys(n, x):
    # analytic limit
    if x < 1e-8:
        return  1 / (2*n + 1)
    
    else:
        # upward recurrence from F0
        F = 0.5*np.sqrt(np.pi/x) * erf(np.sqrt(x))
        for m in range(n):
            F = ( (2*m + 1)*F - np.exp(-x))/(2*x )
    
        return F


def sto(x,y,z,zeta,n,l,m):
    #zeta = Z/n
    r = (x**2+y**2+z**2)**0.5
    if n==1:
        orb = 1/(np.pi)**0.5*zeta**1.5*np.exp(-zeta*r)
    elif n==2 and l==0:
        orb = 1/(3*np.pi)**0.5*zeta**2.5*r*np.exp(-zeta*r)
    elif n==2 and l==1 and m==0:
        orb = 1/(np.pi)**0.5*zeta**2.5*x*np.exp(-zeta*r)
    elif n==2 and l==1 and m==-1:
        orb = 1/(np.pi)**0.5*zeta**2.5*y*np.exp(-zeta*r)
    elif n==2 and l==1 and m==1:
        orb = 1/(np.pi)**0.5*zeta**2.5*z*np.exp(-zeta*r)
    elif n==3 and l==0:
        orb = 2/3/(10*np.pi)**0.5*zeta**2.7*r**2*np.exp(-zeta*r)
    elif n==3 and l==1 and m==0:
        orb = 1/(15*np.pi)**0.5*zeta**2.7*r*x*np.exp(-zeta*r)
    elif n==3 and l==1 and m==-1:
        orb = 1/(15*np.pi)**0.5*zeta**2.7*r*y*np.exp(-zeta*r)
    elif n==3 and l==1 and m==1:
        orb = 1/(15*np.pi)**0.5*zeta**2.7*r*z*np.exp(-zeta*r)
    elif n==3 and l==2 and m==0:
        orb = 1/(6*np.pi)**0.5*zeta**2.7*(3*z**2-r**2)/(3**0.5)*np.exp(-zeta*r)  
    elif n==3 and l==2 and m==-1:
        orb = 1/(6*np.pi)**0.5*zeta**2.7*(2*x*z)*np.exp(-zeta*r) 
    elif n==3 and l==2 and m==1:
        orb = 1/(6*np.pi)**0.5*zeta**2.7*(2*y*z)*np.exp(-zeta*r) 
    elif n==3 and l==2 and m==-2:
        orb = 1/(6*np.pi)**0.5*zeta**2.7*(x**2-y**2)*np.exp(-zeta*r)
    elif n==3 and l==2 and m==2:
        orb = 1/(6*np.pi)**0.5*zeta**2.7*(2*x*y)*np.exp(-zeta*r)
        
    return orb




def angm_to_nlm(l,m):
    if l == 0:
        return (0,0,0)
    
    if l == 1:
        if m == -1:
            return (1,0,0) #px
        elif m == 1:
            return (0,1,0) #py 
        elif m == 0:
            return (0,0,1) #pz
        
def primitive_norm(alpha,nL):
    l = nL[0]
    m = nL[1]
    n = nL[2]
    L = l+m+n

    num = (4*alpha)**L
    den = fact2(2*l-1)*fact2(2*m-1)*fact2(2*n-1)  
        
    return  (2*alpha/np.pi)**0.75 * np.sqrt(num/den)

def contraction_norm(coeff, d, aL,keyA,keyB):
    S = 0.0
    A = np.zeros(3)
    for i in range(len(d)):
        for j in range(len(d)):

            Ni = primitive_norm(d[i],aL)
            Nj = primitive_norm(d[j],aL)

            Sij = overlap_os( aL,aL,A,A,d[i],d[j],keyA,keyA )

            S += ( coeff[i] * coeff[j] * Ni * Nj * Sij )

    return 1/np.sqrt(S)

def overlap_ss(aL,bL,A,B,da,db):
    gamma = da + db
    mu = da*db/gamma
    Rab2 = np.dot(A-B,A-B)
    return (np.pi/gamma)**1.5*np.exp(-mu*Rab2)

cache_ol = {}
def overlap_os(aL,bL,A,B,da,db,keyA,keyB):
    key = (
            aL,
            bL,
            keyA,
            keyB,
            da,
            db
        )
    
    if key in cache_ol:
        return cache_ol[key]
    
    if aL==(0,0,0) and bL==(0,0,0):
        val = overlap_ss(aL,bL,A,B,da,db)
        cache_ol[key] = val
        return val
    
    gamma = da+db
    P = (da*A + db*B)/gamma

    for i in range(3):
        if aL[i] > 0:

            a_down = list(aL)
            a_down[i] -= 1
            a_down = tuple(a_down)

            term1 = ( (P[i]-A[i]) * overlap_os(a_down,bL,A,B,da,db,keyA,keyB) )

            term2 = 0.0
            if a_down[i] >= 0:

                a_down2 = list(a_down)
                a_down2[i] -= 1

                if a_down2[i] >= 0:

                    a_down2 = tuple(a_down2)
                    term2 =  a_down[i]/(2*gamma)  * overlap_os( a_down2,bL,A,B,da,db,keyA,keyB )

            term3 = 0.0
            if bL[i] > 0:

                b_down = list(bL)
                b_down[i] -= 1
                b_down = tuple(b_down)

                term3 =  bL[i]/(2*gamma) * overlap_os( a_down,b_down,A,B,da,db,keyA,keyB )
                
            val = (term1 + term2 + term3)
            #if  abs(val) < 1e-8:
            #    val = 0
            cache_ol[key] = val
            
            return val
        
    for i in range(3):
        if bL[i] > 0:

            b_down = list(bL)
            b_down[i] -= 1
            b_down = tuple(b_down)

            term1 = (P[i]-B[i]) * overlap_os(aL,b_down,A,B,da,db,keyA,keyB)
            
            term2 = 0.0
            if b_down[i] >= 0:

                b_down2 = list(b_down)
                b_down2[i] -= 1

                if b_down2[i] >= 0:

                    b_down2 = tuple(b_down2)
                    term2 = b_down[i]/(2*gamma)* overlap_os(aL,b_down2,A,B,da,db,keyA,keyB)
                    
            term3 = 0.0
            if aL[i] > 0:

                a_down = list(aL)
                a_down[i] -= 1
                a_down = tuple(a_down)

                term3 =  aL[i]/(2*gamma)* overlap_os(a_down,b_down,A,B,da,db,keyA,keyB)
                
            val = (term1 + term2 + term3)
            #if  abs(val) < 1e-8:
            #    val = 0
            cache_ol[key] = val
            return val

cache_k = {}
def kinetic_os(aL,bL,A,B,da,db,keyA,keyB):

    bx,by,bz = bL
    Lb = bx+by+bz
    key = (
            aL,
            bL,
            keyA,
            keyB,
            da,
            db
        )
    
    if key in cache_k:
        return cache_k[key]

    term1 = db*(2*Lb+3) * overlap_os(aL,bL,A,B,da,db,keyA,keyB)
    term2 = 0.0

    for i in range(3):
        b_up = list(bL)
        b_up[i] += 2
        b_up = tuple(b_up)
        term2 += overlap_os(aL,b_up,A,B,da,db,keyA,keyB)

    term2 *= -2*db*db
    term3 = 0.0

    for i in range(3):
        if bL[i] >= 2:
            b_down = list(bL)
            b_down[i] -= 2
            b_down = tuple(b_down)

            term3 += bL[i]*(bL[i]-1)  * overlap_os(aL,b_down,A,B,da,db,keyA,keyB)  

    term3 *= -0.5
    val = term1 + term2 + term3
    #if abs(val) < 1e-8:
    #    val = 0
    cache_k[key] = val
    return val

def ne_ss(aL,b,m,A,B,C,da,db,Zc):
    gamma = da + db
    mu = da*db/gamma
    P = (da*A + db*B)/gamma
    Rab2 = np.dot(A-B,A-B)
    Rpc2= np.dot(P-C,P-C)
    T = gamma*Rpc2
    Vss = -2*np.pi*Zc/gamma*np.exp(-mu*Rab2)*boys(m,T)
    #print(T, boys(m,T))
    return Vss

cache_ne = {}
def ne_os(aL,bL,m,A,B,C,da,db,Zc,keyA,keyB,keyC):
    key = (
            aL,
            bL,
            int(m),
            keyA,
            keyB,
            keyC,
            da,
            db,
            Zc
        )
    
    if key in cache_ne:
        return cache_ne[key]
    
    if (aL==(0,0,0)and bL==(0,0,0)):
        val = ne_ss(aL,bL,m,A,B,C,da,db,Zc)
        cache_ne[key] = val
        return val

    gamma = da + db
    P = (da*A + db*B)/gamma
    
    for i in range(3):
        if aL[i] > 0:
            a_down = list(aL)
            a_down[i] -= 1
            a_down = tuple(a_down)
            
            term1 = ( (P[i]-A[i])* ne_os(a_down,bL,m,A,B,C,da,db,Zc,keyA,keyB,keyC) )     
            term2 = - (P[i]-C[i])* ne_os(a_down,bL,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) 
            
            term3 = 0.0
            if a_down[i] > 0:
                a_down2 = list(a_down)
                a_down2[i] -= 1
                a_down2 = tuple(a_down2)
                term3 = ( a_down[i]/(2*gamma)) * (ne_os(a_down2,bL,m,A,B,C,da,db,Zc,keyA,keyB,keyC) -ne_os(a_down2,bL,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) )
                    
            term4 = 0.0
            if bL[i] > 0:
                b_down = list(bL)
                b_down[i] -= 1
                b_down = tuple(b_down)
                term4 = (bL[i]/(2*gamma)) * (ne_os(a_down,b_down,m,A,B,C,da,db,Zc,keyA,keyB,keyC) -ne_os(a_down,b_down,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) )
                    
            val = (term1 + term2 + term3 + term4)
            #if  abs(val) < 1e-8:
            #    val = 0
            cache_ne[key] = val

            return val
        
    for i in range(3): 
        if bL[i] > 0:
            b_down = list(bL)
            b_down[i] -= 1
            b_down = tuple(b_down)
            
            term1 = ( (P[i]-B[i])* ne_os(aL,b_down,m,A,B,C,da,db,Zc,keyA,keyB,keyC) )     
            term2 = - (P[i]-C[i])* ne_os(aL,b_down,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) 
            
            term3 = 0.0
            if b_down[i] > 0:
                b_down2 = list(b_down)
                b_down2[i] -= 1
                b_down2 = tuple(b_down2)
                term3 = ( b_down[i]/(2*gamma)) * (ne_os(aL,b_down2,m,A,B,C,da,db,Zc,keyA,keyB,keyC) -ne_os(aL,b_down2,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) )
                    
            term4 = 0.0
            if aL[i] > 0:
                a_down = list(aL)
                a_down[i] -= 1
                a_down = tuple(a_down)
                term4 = (aL[i]/(2*gamma)) * (ne_os(a_down,b_down,m,A,B,C,da,db,Zc,keyA,keyB,keyC) -ne_os(a_down,b_down,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) )
                    

                    
            val = (term1 + term2 + term3 + term4)
            #if  abs(val) < 1e-8:
            #    val = 0
            cache_ne[key] = val

            return val

def eri_ssss(m, A,B,C,D,da,db,dc,dd):
    gamma = da + db
    delta = dc + dd

    P = (da*A + db*B)/gamma
    Q = (dc*C + dd*D)/delta

    Rab2 = np.dot(A-B,A-B)
    Rcd2 = np.dot(C-D,C-D)
    Rpq2 = np.dot(P-Q,P-Q)

    T = gamma*delta/(gamma+delta) * Rpq2

    pref = ( 2*np.pi**2.5 /(gamma*delta*np.sqrt(gamma+delta)) )

    pref *= np.exp(-da*db/gamma * Rab2 -dc*dd/delta * Rcd2 )

    return pref * boys(m,T)

cache_eri = {}
def eri_os(aL,bL,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD):
    key = (
            aL,bL,cL,dL,int(m),
            keyA,keyB,keyC,keyD,
            da,
            db,
            dc,
            dd
        )

    if key in cache_eri:
        return cache_eri[key]
    
    if (aL==(0,0,0)and bL==(0,0,0)and cL==(0,0,0) and dL==(0,0,0)):
        val = eri_ssss(m,A,B,C,D,da,db,dc,dd)
        cache_eri[key] = val
        return val

    gamma = da + db
    delta = dc + dd

    P = (da*A + db*B)/gamma
    Q = (dc*C + dd*D)/delta

    W = (gamma*P + delta*Q)/(gamma + delta)
    
    for i in range(3):
        if aL[i] > 0:
            a_down = list(aL)
            a_down[i] -= 1
            a_down = tuple(a_down)
            
            term1 = ( (P[i]-A[i])* eri_os(a_down,bL,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )     
            term2 = ( (W[i]-P[i])* eri_os(a_down,bL,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
            
            term3 = 0.0
            if a_down[i] > 0:
                a_down2 = list(a_down)
                a_down2[i] -= 1
                a_down2 = tuple(a_down2)
                term3 = ( a_down[i]/(2*gamma)) * (eri_os(a_down2,bL,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - delta/(gamma+delta)  * eri_os(a_down2,bL,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                    
            term4 = 0.0
            if bL[i] > 0:
                b_down = list(bL)
                b_down[i] -= 1
                b_down = tuple(b_down)
                term4 = (bL[i]/(2*gamma)) * ( eri_os(a_down,b_down,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - delta/(gamma+delta) * eri_os(a_down,b_down,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                    
            term5 = 0.0
            if cL[i] > 0:
                c_down = list(cL)
                c_down[i] -= 1
                c_down = tuple(c_down)
                term5 = ( cL[i]/(2*(gamma+delta))) * eri_os(a_down,bL,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD )
                    
            term6 = 0.0
            if dL[i] > 0:
                d_down = list(dL)
                d_down[i] -= 1
                d_down = tuple(d_down)
                term6 = (dL[i]/(2*(gamma+delta))) * eri_os(a_down,bL,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD)
                    
            val = (term1 + term2 + term3 + term4 + term5 + term6 )
            #if  abs(val) < 1e-8:
            #    val = 0
            cache_eri[key] = val

            return val

    for i in range(3):
        if bL[i] > 0:
            b_down = list(bL)
            b_down[i] -= 1
            b_down = tuple(b_down)
            
            term1 = ( (P[i]-B[i])* eri_os(aL,b_down,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )     
            term2 = ( (W[i]-P[i])* eri_os(aL,b_down,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
            
            term3 = 0.0
            if b_down[i] > 0:
                b_down2 = list(b_down)
                b_down2[i] -= 1
                b_down2 = tuple(b_down2)
                term3 = ( b_down[i]/(2*gamma)) * (eri_os(aL,b_down2,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - delta/(gamma+delta)  * eri_os(aL,b_down2,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                    
            term4 = 0.0
            if aL[i] > 0:
                a_down = list(aL)
                a_down[i] -= 1
                a_down = tuple(a_down)
                term4 = (aL[i]/(2*gamma)) * ( eri_os(a_down,b_down,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - delta/(gamma+delta) * eri_os(a_down,b_down,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                    
            term5 = 0.0
            if cL[i] > 0:
                c_down = list(cL)
                c_down[i] -= 1
                c_down = tuple(c_down)
                term5 = ( cL[i]/(2*(gamma+delta))) * eri_os(aL,b_down,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD )
                    
            term6 = 0.0
            if dL[i] > 0:
                d_down = list(dL)
                d_down[i] -= 1
                d_down = tuple(d_down)

                term6 = (dL[i]/(2*(gamma+delta))) * eri_os(aL,b_down,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD)
                    
            val = (term1 + term2 + term3 + term4 + term5 + term6 )
            #if  abs(val) < 1e-8:
            #    val = 0
            cache_eri[key] = val

            return val
        
    for i in range(3):
        if cL[i] > 0:
            c_down = list(cL)
            c_down[i] -= 1
            c_down = tuple(c_down)
            
            term1 = ( (Q[i]-C[i])* eri_os(aL,bL,c_down,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )     
            term2 = ( (W[i]-Q[i])* eri_os(aL,bL,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
            
            term3 = 0.0
            if c_down[i] > 0:
                c_down2 = list(c_down)
                c_down2[i] -= 1
                c_down2 = tuple(c_down2)
                term3 = ( c_down[i]/(2*delta)) * (eri_os(aL,bL,c_down2,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - gamma/(gamma+delta)  * eri_os(aL,bL,c_down2,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                    
            term4 = 0.0
            if dL[i] > 0:
                d_down = list(dL)
                d_down[i] -= 1
                d_down = tuple(d_down)
                term4 = (dL[i]/(2*delta)) * ( eri_os(aL,bL,c_down,d_down,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - gamma/(gamma+delta) * eri_os(aL,bL,c_down,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                    
            term5 = 0.0
            if aL[i] > 0:
                a_down = list(aL)
                a_down[i] -= 1
                a_down = tuple(a_down)
                term5 = ( aL[i]/(2*(gamma+delta))) * eri_os(a_down,bL,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD )
                    
            term6 = 0.0
            if bL[i] > 0:
                b_down = list(bL)
                b_down[i] -= 1
                b_down = tuple(b_down)

                term6 = (bL[i]/(2*(gamma+delta))) * eri_os(aL,b_down,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD)
                    
            val = (term1 + term2 + term3 + term4 + term5 + term6 )
            #if  abs(val) < 1e-8:
            #    val = 0
            cache_eri[key] = val

            return val
        
    for i in range(3):
        if dL[i] > 0:
            d_down = list(dL)
            d_down[i] -= 1
            d_down = tuple(d_down)
            
            term1 = ( (Q[i]-D[i])* eri_os(aL,bL,cL,d_down,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )     
            term2 = ( (W[i]-Q[i])* eri_os(aL,bL,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
            
            term3 = 0.0
            if d_down[i] > 0:
                d_down2 = list(d_down)
                d_down2[i] -= 1
                d_down2 = tuple(d_down2)
                term3 = ( d_down[i]/(2*delta)) * (eri_os(aL,bL,cL,d_down2,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - gamma/(gamma+delta)  * eri_os(aL,bL,cL,d_down2,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                    
            term4 = 0.0
            if cL[i] > 0:
                c_down = list(cL)
                c_down[i] -= 1
                c_down = tuple(c_down)
                term4 = (cL[i]/(2*delta)) * ( eri_os(aL,bL,c_down,d_down,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - gamma/(gamma+delta) * eri_os(aL,bL,c_down,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                    
            term5 = 0.0
            if aL[i] > 0:
                a_down = list(aL)
                a_down[i] -= 1
                a_down = tuple(a_down)
                term5 = ( aL[i]/(2*(gamma+delta))) * eri_os(a_down,bL,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD )
                    
            term6 = 0.0
            if bL[i] > 0:
                b_down = list(bL)
                b_down[i] -= 1
                b_down = tuple(b_down)

                term6 = (bL[i]/(2*(gamma+delta))) * eri_os(aL,b_down,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD)
                    
            val = (term1 + term2 + term3 + term4 + term5 + term6 )
            #if  abs(val) < 1e-8:
            #    val = 0
            cache_eri[key] = val

            return val
        

def Overlap(p,q,ncoord):
    AOA = AO_data[p]
    AOB = AO_data[q]

    normA = AOA["norm"]
    normB = AOB["norm"]
    
    CNA = AOA["CN"]
    CNB = AOB["CN"]
    
    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]

    dA = AOA["expo"]
    dB = AOB["expo"]

    aL = AOA["aL"]
    bL = AOB["aL"]

    coeffA = coeffA * CNA
    coeffB = coeffB * CNB
    
    
    A = ncoord[3*p:3*p+3]
    B = ncoord[3*q:3*q+3]
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    
    f = 1.
    S = 0
    for i in range(3):
        for j in range(3):
            S += coeffA[i]*coeffB[j] * normA[i] * normB[j] * overlap_os(aL,bL,A,B,dA[i],dB[j],keyA,keyB)
    return S

def Kinetic_Energy(p,q,ncoord_for_ao):
    AOA = AO_data[p]
    AOB = AO_data[q]

    normA = AOA["norm"]
    normB = AOB["norm"]

    CNA = AOA["CN"]
    CNB = AOB["CN"]

    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]

    dA = AOA["expo"]
    dB = AOB["expo"]

    aL = AOA["aL"]
    bL = AOB["aL"]
    
    A = ncoord_for_ao[3*p:3*p+3]
    B = ncoord_for_ao[3*q:3*q+3]
    
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    
    f = 1.
    coeffA = coeffA * CNA
    coeffB = coeffB * CNB

    I = 0
    for i in range(3):
        for j in range(3):
            I += coeffA[i]*coeffB[j] *normA[i] * normB[j] * kinetic_os(aL,bL,A,B,dA[i],dB[j],keyA,keyB)

    return I

def Neuclear_Atraction(p,q,ncoord,ncoord_for_ao,Nl,natom):
    AOA = AO_data[p]
    AOB = AO_data[q]

    normA = AOA["norm"]
    normB = AOB["norm"]

    CNA = AOA["CN"]
    CNB = AOB["CN"]

    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]

    dA = AOA["expo"]
    dB = AOB["expo"]

    aL = AOA["aL"]
    bL = AOB["aL"]
    
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    
    A = ncoord_for_ao[3*p:3*p+3]
    B = ncoord_for_ao[3*q:3*q+3]
    
    f = 1.
    
    coeffA = coeffA * CNA
    coeffB = coeffB * CNB
    
    I = 0
    m = 0
    for n in range(natom):
        AtC = Atom_data[n]
        keyC = AtC["atom_coord_key"]
        C = ncoord[3*n:3*n+3]
        Zc = Nl[n]
        for i in range(3):
            for j in range(3):
                I += coeffA[i]*coeffB[j] *normA[i] * normB[j] *ne_os(aL,bL,m,A,B,C,dA[i],dB[j],Zc,keyA,keyB,keyC)

    return I

def Electron_Repulsive_Integral(p,q,r,s, ncoord_for_ao):
    AOA = AO_data[p]
    AOB = AO_data[q]
    AOC = AO_data[r]
    AOD = AO_data[s]
    
    normA = AOA["norm"]
    normB = AOB["norm"]
    normC = AOC["norm"]
    normD = AOD["norm"]
    
    CNA = AOA["CN"]
    CNB = AOB["CN"]
    CNC = AOC["CN"]
    CND = AOD["CN"]
    
    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]
    coeffC = AOC["coeff"]
    coeffD = AOD["coeff"]
    
    dA = AOA["expo"]
    dB = AOB["expo"]
    dC = AOC["expo"]
    dD = AOD["expo"]
    
    aL = AOA["aL"]
    bL = AOB["aL"]
    cL = AOC["aL"]
    dL = AOD["aL"]
    
    coeffA = coeffA * CNA
    coeffB = coeffB * CNB
    coeffC = coeffC * CNC
    coeffD = coeffD * CND
    
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    keyC =  AOC["coord_key"]
    keyD =  AOD["coord_key"]
    
    A = ncoord_for_ao[3*p:3*p+3]
    B = ncoord_for_ao[3*q:3*q+3]
    C = ncoord_for_ao[3*r:3*r+3]
    D = ncoord_for_ao[3*s:3*s+3]
    
    m = 0
    eri = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    ERI = eri_os(aL,bL,cL,dL,m,A,B,C,D,dA[i],dB[j],dC[k],dD[l],keyA,keyB,keyC,keyD)
                    eri += coeffA[i]*coeffB[j]*coeffC[k]*coeffD[l]* normA[i] * normB[j] * normC[k] * normD[l] * ERI 
    
        
    return eri


def eri_key(p,q,r,s):
    pair1 = tuple(sorted((p,q)))
    pair2 = tuple(sorted((r,s)))
    if pair2 < pair1:
        pair1, pair2 = pair2, pair1

    return (pair1, pair2)

cache_ERIs = {}
def check_ERI(p,q,r,s, ncoord_for_ao):
    key = eri_key(p,q,r,s)
    if key in cache_ERIs:
        return cache_ERIs[key] 
    else:
        eri = Electron_Repulsive_Integral(p,q,r,s, ncoord_for_ao)
        cache_ERIs[key] = eri
        return eri
    
def build_schwarz(ncoord_for_ao,nao):
    sch = np.zeros((nao,nao))
    
    for p in range(nao):
        for q in range(nao):
            val = check_ERI(p,q,p,q,ncoord_for_ao)
            
            sch[p,q] = np.sqrt(abs(val))
    return sch

def build_G(P, ERI):
    J = np.einsum('rs,pqrs->pq', P, ERI, optimize=True )
    K = np.einsum('rs,prqs->pq', P, ERI, optimize=True )
    return J - 0.5*K


AO_data = []
Atom_data = []
def update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao):
    AO_data.clear()
    Atom_data.clear()
    for p in range(nao):
        qnlA = qnl[p]
        basisA = basis_set.get_basis(Nl_for_ao[p],tuple(qnlA[:-1].tolist()))
        coeff = basisA[:3]
        expo = basisA[3:]
        aL = angm_to_nlm(qnlA[1],qnlA[2])
        keyA = tuple(ncoord_for_ao[3*p:3*p+3])

        AO_data.append(
                    {
                    "coeff":np.array(coeff),
                    "expo":np.array(expo),
                    "aL":aL,
                    "CN":contraction_norm(coeff,expo,aL,keyA,keyA),
                    "norm":[primitive_norm(expo[i],aL) for i in range(3) ],
                    "coord_key":tuple(ncoord_for_ao[3*p:3*p+3]),
                    }
                            )
    
    for A in range(natom):
        Atom_data.append(
                    {
                    "atom_coord_key":tuple(ncoord[3*A:3*A+3])
                    }
                    )


def scf_driver(Cinit,alpha,tole,max_ite,log=True):
    ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl,natom, nao, nele, spin,atomic_mass = MPs.get_properties()

    gi = GaussianIntegrals()
    gi.set_properties(MPs)
    
    S = gi.overlap_matrix()
    T = gi.kinetic_matrix()
    V = gi.nuclear_attraction_matrix()
    ERI = gi.electron_repulsive_matrix()
    
    scf = SCFDriver()
    E, eps, C, P, nocc = scf.scfdriver(S,T,V,ERI,Cinit,nele,alpha,tole,max_ite,log)

    ###Energy
    En = gi.nuclear_repulsion()
    Etotal = E + En
    
    ###Clear Caches
    gi.clear_cache()

    return Etotal, E, En, eps, C, P, nocc

### Analytical Gradient
cache_overlap_grad={}
def overlap_deriv_os(aL, bL,axis,center, A, B, dA, dB,keyA,keyB):
    term1 = 0.0
    term2 = 0.0
    
    key = (
            aL,
            bL,
            axis,
            center,
            keyA,
            keyB,
            dA,
            dB
            )
    
    if key in cache_overlap_grad:
        return cache_overlap_grad[key]
    
    
    if center == 0:
        lp = list(aL)
        lp[axis] += 1 
        lp = tuple(lp)
        term1 = 2*dA * overlap_os(lp, bL, A, B, dA, dB,keyA,keyB)
    
        if aL[axis] > 0:
            lm = list(aL)
            lm[axis] -= 1
            lm = tuple(lm)
            term2 = aL[axis] * overlap_os( lm, bL, A, B, dA,dB,keyA,keyB )
            
        cache_overlap_grad[key] = term1 - term2
            
    if center == 1:
        lp = list(bL)
        lp[axis] += 1 
        lp = tuple(lp)
        term1 = 2*dB * overlap_os(aL, lp, A, B, dA, dB,keyA,keyB)
    
        if bL[axis] > 0:
            lm = list(bL)
            lm[axis] -= 1
            lm = tuple(lm)
            term2 = bL[axis] * overlap_os( aL, lm, A, B, dA, dB,keyA,keyB )
        
        cache_overlap_grad[key] = term1 - term2

    return term1 - term2

cache_kinetic_grad={}
def kinetic_deriv_os(aL,bL,axis,center,A,B,dA,dB,keyA,keyB):
    term1 = 0.0
    term2 = 0.0
    
    key = (
        aL,bL,axis,center        , 
        keyA,
        keyB,
        dA,dB
        )
    
    if key in cache_kinetic_grad:
        return cache_kinetic_grad[key]
    
    if center == 0:
        a_plus = list(aL)
        a_plus[axis] += 1
        a_plus = tuple(a_plus)
        term1 = 2*dA * kinetic_os(a_plus,bL,A,B,dA,dB,keyA,keyB)
    
        if aL[axis] > 0:
            a_minus =  list(aL)
            a_minus[axis] -= 1
            a_minus = tuple(a_minus)
            term2 = aL[axis] * kinetic_os(a_minus,bL,A,B,dA,dB,keyA,keyB)
        cache_kinetic_grad[key] = term1 - term2
            
    if center == 1:
        b_plus = list(bL)
        b_plus[axis] += 1
        b_plus = tuple(b_plus)
        term1 = 2*dB * kinetic_os(aL,b_plus,A,B,dA,dB,keyA,keyB)
    
        if bL[axis] > 0:
            b_minus =  list(bL)
            b_minus[axis] -= 1
            b_minus = tuple(b_minus)
            term2 = bL[axis] * kinetic_os(aL,b_minus,A,B,dA,dB,keyA,keyB)
        cache_kinetic_grad[key] = term1 - term2

    return term1 - term2

cache_na_grad={}
def nuclear_deriv_os(aL,bL,axis,center,A,B,C,dA,dB,Z,keyA,keyB,keyC):
    term1 = 0.0
    term2 = 0.0
    
    key = (
        aL,bL,axis,center,
        keyA,
        keyB,
        keyC,
        dA,dB,Z
        )
    
    if key in cache_na_grad:
        return cache_na_grad[key]
    
    if center == 0:
        a_plus = list(aL)
        a_plus[axis] += 1
        a_plus = tuple(a_plus)
        term1 = 2*dA * ne_os(a_plus,bL,0,A,B,C,dA,dB,Z,keyA,keyB,keyC) 
    
        if aL[axis] > 0:
            a_minus =  list(aL)
            a_minus[axis] -= 1
            a_minus = tuple(a_minus)
            term2 = aL[axis] * ne_os(a_minus,bL,0,A,B,C,dA,dB,Z,keyA,keyB,keyC) 
            
        cache_na_grad[key] = term1 - term2
        
        return term1 - term2

    if center == 1:
        b_plus = list(bL)
        b_plus[axis] += 1
        b_plus = tuple(b_plus)
        term1 = 2*dB * ne_os(aL,b_plus,0,A,B,C,dA,dB,Z,keyA,keyB,keyC) 
    
        if bL[axis] > 0:
            b_minus =  list(bL)
            b_minus[axis] -= 1
            b_minus = tuple(b_minus)
            term2 = bL[axis] * ne_os(aL,b_minus,0,A,B,C,dA,dB,Z,keyA,keyB,keyC) 
            
        cache_na_grad[key] = term1 - term2
        
        return term1 - term2
            
    if center == 2:
        keyA = ( aL,bL,axis,"A",keyA,keyB,keyC,dA,dB,Z )
        keyB = ( aL,bL,axis,"B",keyA,keyB,keyC,dA,dB,Z )
        dVA = cache_na_grad[keyA] 
        dVB = cache_na_grad[keyB] 

        return -(dVA + dVB)


def eri_grad_os(aL,bL,cL,dL,axis,center,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD):
    term1 = 0.0
    term2 = 0.0

    if center == 0: #A
        a_plus = list(aL)
        a_plus[axis] += 1
        a_plus = tuple(a_plus)
        term1 =2*dA* eri_os(a_plus,bL,cL,dL,0,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD) 
    
        if aL[axis] > 0:
            a_minus =  list(aL)
            a_minus[axis] -= 1
            a_minus = tuple(a_minus)
            term2 = aL[axis] * eri_os(a_minus,bL,cL,dL,0,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD ) 
            

    elif center == 1: #B
        b_plus = list(bL)
        b_plus[axis] += 1
        b_plus = tuple(b_plus)
        term1 =2*dB* eri_os(aL,b_plus,cL,dL,0,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD) 
    
        if bL[axis] > 0:
            b_minus =  list(bL)
            b_minus[axis] -= 1
            b_minus = tuple(b_minus)
            term2 = bL[axis] * eri_os(aL,b_minus,cL,dL,0,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD ) 
            
        
    elif center == 2: #C
        c_plus = list(cL)
        c_plus[axis] += 1
        c_plus = tuple(c_plus)
        term1 =2*dC* eri_os(aL,bL,c_plus,dL,0,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD) 
    
        if cL[axis] > 0:
            c_minus =  list(cL)
            c_minus[axis] -= 1
            c_minus = tuple(c_minus)
            term2 = cL[axis] * eri_os(aL,bL,c_minus,dL,0,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD ) 
            
            
    elif center == 3: #D
        d_plus = list(dL)
        d_plus[axis] += 1
        d_plus = tuple(d_plus)
        term1 =2*dD* eri_os(aL,bL,cL,d_plus,0,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD) 
    
        if dL[axis] > 0:
            d_minus =  list(dL)
            d_minus[axis] -= 1
            d_minus = tuple(d_minus)
            term2 = dL[axis] * eri_os(aL,bL,cL,d_minus,0,A,B,C,D,dA,dB,dC,dD,keyA,keyB,keyC,keyD ) 
            
        #cache_eri_grad[key] = term1 - term2
            
    return term1 - term2

def Overlap_grad(p,q,ncoord_for_ao,axis,center):
    AOA = AO_data[p]
    AOB = AO_data[q]
    
    normA = AOA["norm"]
    normB = AOB["norm"]
    
    CNA = AOA["CN"]
    CNB = AOB["CN"]
    
    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]
    
    dA = AOA["expo"]
    dB = AOB["expo"]
    
    aL = AOA["aL"]
    bL = AOB["aL"]
    
    coeffA = coeffA * CNA
    coeffB = coeffB * CNB
    
    A = ncoord_for_ao[3*p:3*p+3]
    B = ncoord_for_ao[3*q:3*q+3]
    
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    f = 1.

    dS = 0
    for i in range(3):
        for j in range(3):
            dS += coeffA[i]*coeffB[j] * normA[i] * normB[j] * overlap_deriv_os(aL,bL, axis,center,A,B,dA[i],dB[j],keyA,keyB)
    return dS

def Kinetic_Energy_grad(p,q,ncoord_for_ao,axis,center):
    AOA = AO_data[p]
    AOB = AO_data[q]
    
    normA = AOA["norm"]
    normB = AOB["norm"]
    
    CNA = AOA["CN"]
    CNB = AOB["CN"]
    
    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]
    
    dA = AOA["expo"]
    dB = AOB["expo"]
    
    aL = AOA["aL"]
    bL = AOB["aL"]
    
    coeffA = coeffA * CNA
    coeffB = coeffB * CNB
    
    #C = np.aray(])
    A = ncoord_for_ao[3*p:3*p+3]
    B = ncoord_for_ao[3*q:3*q+3]
    
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    
    f = 1.
    dI = 0
    for i in range(3):
        for j in range(3):
            dI += coeffA[i]*coeffB[j] *normA[i] * normB[j] *  kinetic_deriv_os(aL,bL,axis,center,A,B,dA[i],dB[j],keyA,keyB)

    return dI

def Neuclear_Atraction_grad(p,q,ncoord,ncoord_for_ao,Nl,axis,atom_p,atom_q,target_atom):
    AOA = AO_data[p]
    AOB = AO_data[q]
    
    normA = AOA["norm"]
    normB = AOB["norm"]
    
    CNA = AOA["CN"]
    CNB = AOB["CN"]
    
    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]
    
    dA = AOA["expo"]
    dB = AOB["expo"]
    
    aL = AOA["aL"]
    bL = AOB["aL"]

    coeffA = coeffA * CNA
    coeffB = coeffB * CNB
    
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    
    A = ncoord_for_ao[3*p:3*p+3]
    B = ncoord_for_ao[3*q:3*q+3]
    natom = int(ncoord.size/3)
    
    dI = 0
    for n in range(natom):
        AOC = Atom_data[n]
        keyC = AOC["atom_coord_key"]
        C = ncoord[3*n:3*n+3]
        Zc = Nl[n]
        for i in range(3):
            for j in range(3):
                if atom_p == target_atom:
                    dVA_I = nuclear_deriv_os(aL,bL,axis,0,A,B,C,dA[i],dB[j],Zc,keyA,keyB,keyC)
                    dI += coeffA[i]*coeffB[j] *normA[i] * normB[j] *dVA_I
                
                if atom_q == target_atom:
                    dVB_I = nuclear_deriv_os(aL,bL,axis,1,A,B,C,dA[i],dB[j],Zc,keyA,keyB,keyC)
                    dI += coeffA[i]*coeffB[j] *normA[i] * normB[j] *dVB_I
                
                if n == target_atom:
                    dVA_I = nuclear_deriv_os(aL,bL,axis,0,A,B,C,dA[i],dB[j],Zc,keyA,keyB,keyC)
                    dVB_I = nuclear_deriv_os(aL,bL,axis,1,A,B,C,dA[i],dB[j],Zc,keyA,keyB,keyC)
                    dI += -coeffA[i]*coeffB[j] *normA[i] * normB[j] *(dVA_I + dVB_I)
                #I +=  nuclear_deriv_os(aL,bL,axis,center,A,B,C,dA[i],dB[j],Zc)
                        
    return dI

#cache_eri_grad_ao = {}
def Electron_Repulsive_Integral_grad(p,q,r,s, ncoord_for_ao,axis,target_atom,center):
    AOA = AO_data[p]
    AOB = AO_data[q]
    AOC = AO_data[r]
    AOD = AO_data[s]
    
    normA = AOA["norm"]
    normB = AOB["norm"]
    normC = AOC["norm"]
    normD = AOD["norm"]
    
    CNA = AOA["CN"]
    CNB = AOB["CN"]
    CNC = AOC["CN"]
    CND = AOD["CN"]
    
    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]
    coeffC = AOC["coeff"]
    coeffD = AOD["coeff"]
    
    dA = AOA["expo"]
    dB = AOB["expo"]
    dC = AOC["expo"]
    dD = AOD["expo"]
    
    aL = AOA["aL"]
    bL = AOB["aL"]
    cL = AOC["aL"]
    dL = AOD["aL"]
    
    coeffA = coeffA * CNA
    coeffB = coeffB * CNB
    coeffC = coeffC * CNC
    coeffD = coeffD * CND
    
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    keyC =  AOC["coord_key"]
    keyD =  AOD["coord_key"]
    
    A = ncoord_for_ao[3*p:3*p+3]
    B = ncoord_for_ao[3*q:3*q+3]
    C = ncoord_for_ao[3*r:3*r+3]
    D = ncoord_for_ao[3*s:3*s+3]
    
    deri = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    dERI = eri_grad_os(aL,bL,cL,dL,axis,center,A,B,C,D,dA[i],dB[j],dC[k],dD[l],keyA,keyB,keyC,keyD)
                    deri += coeffA[i]*coeffB[j]*coeffC[k]*coeffD[l]* normA[i] * normB[j] * normC[k] * normD[l] * dERI 
    
    return deri

def Neculear_Repulsion_deriv(ncoord,Nl):
    natom = int(len(ncoord)/3)
    dEnuc = np.zeros((natom,3))
    for i in range(natom):
        for j in range(natom):
            if i == j:
                continue
            
            RAB = ncoord[3*i:3*i+3] - ncoord[3*j:3*j+3]
            r = np.linalg.norm(RAB)
            
            dEnuc[i] += (-Nl[i]*Nl[j] * RAB / r**3)

    return dEnuc

def deri_key(p,q,r,s):
    pair1 = tuple(sorted((p,q)))
    pair2 = tuple(sorted((r,s)))

    if pair2 < pair1:
        pair1, pair2 = pair2, pair1

    return (pair1, pair2)

cache_dERI = {}
def dERI_store(p,q,r,s,axis):
    key, perm = canonical_quartet(p,q,r,s)
    d = cache_dERI[(key,axis)]
    
    dA = d[perm[0]]
    dB = d[perm[1]]
    dC = d[perm[2]]
    dD = d[perm[3]]
   
    return dA, dB, dC, dD
    
def pair_index(i,j):
    if j > i:
        i,j = j,i
    return i*(i+1)//2 + j

def canonical_quartet(p, q, r, s):
    labels = [0,1,2,3]

    # -----------------------------
    # ① p>=q にする
    # -----------------------------
    if p < q:
        p, q = q, p
        labels[0], labels[1] = labels[1], labels[0]

    # -----------------------------
    # ② r>=s にする
    # -----------------------------
    if r < s:
        r, s = s, r
        labels[2], labels[3] = labels[3], labels[2]

    # -----------------------------
    # ③ pair index を比較
    # -----------------------------
    pq = pair_index(p, q)
    rs = pair_index(r, s)

    if pq < rs:
        p, q, r, s = r, s, p, q
        labels[0], labels[2] = labels[2], labels[0]
        labels[1], labels[3] = labels[3], labels[1]
        
    # inverse permutation
    perm = tuple(np.argsort(labels))

    key = (p,q,r,s)
    #perm = tuple(labels)

    return key, perm


#cache_dERI = {}
def build_dERI(dERI,ncoord_for_ao):
    #tole_sch = 1e-6
    count = 0
    for axis in range(3):
        for p in range(nao):
            for q in range(nao):
                pq = pair_index(p,q)
                for r in range(nao):
                    for s in range(nao):
                        rs = pair_index(r,s)
                        if rs > pq:
                            continue
                        key, _ = canonical_quartet(p,q,r,s)
                        cp,cq,cr,cs = key
                        atom_p = ao_to_atom[cp]
                        atom_q = ao_to_atom[cq]
                        atom_r = ao_to_atom[cr]
                        atom_s = ao_to_atom[cs]
                        
                        #bound = sch[cp,cq]*sch[cr,cs]
                        #if bound < tole_sch:
                        #    count += 1
                        #    continue
                        
                        dA = Electron_Repulsive_Integral_grad(cp,cq,cr,cs, ncoord_for_ao,axis,atom_p,0)
                        dB = Electron_Repulsive_Integral_grad(cp,cq,cr,cs, ncoord_for_ao,axis,atom_q,1)
                        dC = Electron_Repulsive_Integral_grad(cp,cq,cr,cs, ncoord_for_ao,axis,atom_r,2)
                        dD = Electron_Repulsive_Integral_grad(cp,cq,cr,cs, ncoord_for_ao,axis,atom_s,3)
                        
                        cache_dERI[(key,axis)] = [dA,dB,dC,dD]

    for axis in range(3):
        for p in range(nao):
            for q in range(nao):
                for r in range(nao):
                    for s in range(nao):
                        atom_p = ao_to_atom[p]
                        atom_q = ao_to_atom[q]
                        atom_r = ao_to_atom[r]
                        atom_s = ao_to_atom[s]
                        
                        dA, dB, dC, dD = dERI_store(p,q,r,s,axis)
                        dERI[int(atom_p),axis,p,q,r,s] += dA
                        dERI[int(atom_q),axis,p,q,r,s] += dB
                        dERI[int(atom_r),axis,p,q,r,s] += dC
                        dERI[int(atom_s),axis,p,q,r,s] += dD
              
    print(count)
    return dERI
    
def analytical_gradient(mole_ar,P,Cocc,eps):
    #0:A 1:B 2:C 3:D
    
    ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl,natom, nao, nele, ao_to_atom,atomic_mass = MPs.get_properties()
    grad = np.zeros((natom,3))
    
    dS = np.zeros((natom,3,nao,nao))
    dT = np.zeros((natom,3,nao,nao))
    dV = np.zeros((natom,3,nao,nao))
    dH = np.zeros((natom,3,nao,nao))
    dERI = np.zeros((natom,3,nao,nao,nao,nao))
    
    for i, target_atom in enumerate(range(natom)):
        for j, axis in enumerate(range(3)):
            
            for p in range(nao):
                for q in range(nao):
                    atom_p = ao_to_atom[p]
                    atom_q = ao_to_atom[q]
                    
                    dV[i,j,p,q] += Neuclear_Atraction_grad(p,q,ncoord,ncoord_for_ao,Nl,axis,atom_p,atom_q,target_atom)
                    if atom_p == target_atom:
                        dS[i,j,p,q] += Overlap_grad(p,q,ncoord_for_ao,axis,0)
                        dT[i,j,p,q] += Kinetic_Energy_grad(p,q,ncoord_for_ao,axis,0)
                        
                    if atom_q == target_atom:
                        dS[i,j,p,q] += Overlap_grad(p,q,ncoord_for_ao,axis,1)
                        dT[i,j,p,q] += Kinetic_Energy_grad(p,q,ncoord_for_ao,axis,1)

                    
    dERI = build_dERI(dERI,ncoord_for_ao)
    dH = dT + dV
    W = 2 * Cocc @ np.diag(eps) @ Cocc.T 
    PdH = np.einsum('pq,abpq->ab', P, dH)
    WdS = np.einsum('pq,abpq->ab',W,dS)
    Jgrad = 0.5*np.einsum('pq,rs,abpqrs->ab',P,P,dERI)
    Kgrad = -0.25*np.einsum('pq,rs,abprqs->ab',P,P,dERI)
    PPdERI = Jgrad + Kgrad
    dEnuc = Neculear_Repulsion_deriv(ncoord,Nl)
    
    grad = PdH + PPdERI - WdS + dEnuc
    
    ###Clear Caches
    cache_ol.clear()
    cache_k.clear()
    cache_ne.clear()
    cache_eri.clear()
    cache_ERIs.clear()
    cache_dERI.clear()
    cache_overlap_grad.clear()
    cache_kinetic_grad.clear()
    cache_na_grad.clear()
    return grad

def numerical_gradient(mole_ar,Cinit,h=1e-4):
    ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl,natom, nao, nele, ao_to_atom,atomic_mass = MPs.get_properties()
    grad = np.zeros((natom,3))
    update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
    #print(grad.shape)
    for A in range(natom):
        for xyz in range(3):
            #print(mole_ar[A][1][xyz])
            mole_ar[A][1][xyz] += h
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            Ep ,_,_,_,_,_,_ = scf_driver(Cinit,alpha,tole,max_ite,log=False)

            mole_ar[A][1][xyz] -= 2*h
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            Em ,_,_,_,_,_,_ = scf_driver(Cinit,alpha,tole,max_ite,log=False)
            
            grad[A,xyz] = (Ep - Em)/(2*h)

            mole_ar[A][1][xyz] += h
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            
    return grad

def build_Hessian(mole_ar,Cinit,h=1e-4):
    ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl,natom, nao, nele, ao_to_atom,atomic_mass = MPs.get_properties()
    hessian = np.zeros((3*natom,3*natom))
    #print(grad.shape)
    col = 0
    for A in range(natom):
        for xyz in range(3):
            mole_ar[A][1][xyz] += h
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
           
            #grad_f = numerical_gradient(mole_ar,C,h=1e-4).reshape(-1)
            _, _, _, eps,  C, P, nocc  = scf_driver(Cinit,alpha,tole,max_ite,log=False)
            Cocc = C[:,:nocc]
            grad_f = analytical_gradient(mole_ar,P,Cocc,eps[:nocc]).reshape(-1)
            
            mole_ar[A][1][xyz] -= 2*h
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            #grad_b = numerical_gradient(mole_ar,C,h=1e-4).reshape(-1)
            _, _, _, eps,  C, P, nocc  = scf_driver(Cinit,alpha,tole,max_ite,log=False)
            Cocc = C[:,:nocc]
            grad_b = analytical_gradient(mole_ar,P,Cocc,eps[:nocc]).reshape(-1)
            
            hessian[:,col] = (grad_f - grad_b)/(2*h)

            mole_ar[A][1][xyz] += h
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            col += 1
            
    hessian = 0.5 * (hessian + hessian.T)
    return hessian

def vibration_analysis(mole_ar,hessian):
    ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl,natom, nao, nele, ao_to_atom,atomic_mass = MPs.get_properties()
    
    mass = np.repeat(atomic_mass,3)
    F = hessian / np.sqrt(np.outer(mass,mass))
    eigval, eigvec = np.linalg.eigh(F)
    return eigval, eigvec

def dipole_integral(aL,bL,A,B,dA,dB,keyA,keyB,axis):
    p = dA + dB
    P =(dA*A[axis] + dB*B[axis])/p
    term1 = P * overlap_os(aL,bL,A,B,dA,dB,keyA,keyB)
    
    term2 = 0

    if aL[axis] > 0:
        a_down = list(aL)
        a_down[axis] -= 1
        a_down = tuple(a_down)
        term2 = aL[axis]/(2*p) * overlap_os(a_down,bL,A,B,dA,dB,keyA,keyB)
    
    term3 = 0
    if bL[axis] > 0:
        b_down = list(bL)
        b_down[axis] -= 1
        b_down = tuple(b_down)
        term3 = bL[axis]/(2*p) * overlap_os(aL,b_down,A,B,dA,dB,keyA,keyB)

    val = term1 + term2 + term3
    
    return val

def build_dipole_matrix_element(p,q,ncoord_for_ao,axis):
    AOA = AO_data[p]
    AOB = AO_data[q]
    
    normA = AOA["norm"]
    normB = AOB["norm"]
    
    CNA = AOA["CN"]
    CNB = AOB["CN"]
    
    coeffA = AOA["coeff"]
    coeffB = AOB["coeff"]
    
    dA = AOA["expo"]
    dB = AOB["expo"]
    
    aL = AOA["aL"]
    bL = AOB["aL"]
    
    coeffA = coeffA * CNA
    coeffB = coeffB * CNB
    
    A = ncoord_for_ao[3*p:3*p+3]
    B = ncoord_for_ao[3*q:3*q+3]
    
    keyA =  AOA["coord_key"]
    keyB =  AOB["coord_key"]
    dipole = 0 
    for i in range(3):
        for j in range(3):
            dipole += coeffA[i] * coeffB[j] * normA[i] * normB[j] *dipole_integral(aL,bL,A,B,dA[i],dB[j],keyA,keyB,axis)

    return dipole

def Plot_MOorbs(mole_ar,C,n_homo):
    ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl,natom, nao, nele, spin,atomic_mass = MPs.get_properties()
    ### Plot MO orbitals
    ncoord = ncoord.reshape((ncoord.size//3,3))
    Xc = ncoord[:,0].mean()
    Yc = ncoord[:,1].mean()
    Zc = ncoord[:,2].mean()
    L = 8
    Res = 100
    mid = Res//2
    x = np.linspace(-L+Xc,L+Xc,Res)
    y = np.linspace(-L+Yc,L+Yc,Res)
    z = np.linspace(-L+Zc,L+Zc,Res)
    
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')

    
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    MO_orbs = np.zeros((nao,Res,Res,Res))
    for i in range(nao):
        psi = np.zeros((Res,Res,Res))
        for j in range(nao):
            qnlA = qnl[j]
            basisA = basis_set.get_basis(Nl_for_ao[j],tuple(qnlA[:-1].tolist()))
            A = ncoord_for_ao[3*j:3*j+3]
            #Zn = Nl_for_ao[j]
            aL = angm_to_nlm(qnlA[1],qnlA[2])
            dA = basisA[3:]
            coef = basisA[:3]
            chi = contracted_gto(X,Y,Z,A,dA,coef,aL)
            psi += C[j,i] * chi
        MO_orbs[i,:,:,:] = psi
    
    fig_size = 4
    row_size = int(fig_size*nao)
    col_size = int(fig_size*3)
    fig, ax = plt.subplots(nao,3,figsize=(col_size,row_size))
    threshold = 0.02
    
    for i in range(nao):#(num_bands-1):#num_bands-1):
        MO_data = MO_orbs[i,:,:,:]
        threshold = 0.05*np.max(np.abs(MO_data))
        MO_data = np.where(abs(MO_data) > threshold, MO_data, 0 )
        # XY
        ax[i,0].imshow(MO_data[:,:,mid].T, origin='lower', extent=[-L,L,-L,L],vmin=-1,vmax=1, cmap='seismic')
        #XZ
        ax[i,1].imshow(MO_data[:,mid,:].T, origin='lower', extent=[-L,L,-L,L],vmin=-1,vmax=1, cmap='seismic')
        #YZ
        ax[i,2].imshow(MO_data[mid,:,:].T, origin='lower', extent=[-L,L,-L,L],vmin=-1,vmax=1, cmap='seismic')
        ax[0,0].set_title("Alpha XY Plane")
        ax[0,1].set_title("Alpha XZ Plane")
        ax[0,2].set_title("Alpha YZ Plane")
        
        if i == (n_homo - 1):
            ax[i,0].set_title("HOMO")
            ax[i,1].set_title("HOMO")
            ax[i,2].set_title("HOMO")
        elif i == (n_homo):
            ax[i,0].set_title("LUMO")
            ax[i,1].set_title("LUMO")
            ax[i,2].set_title("LUMO")

        #plt.title("XY plane")
        #plt.xlabel("x")
        #plt.ylabel("y")
        #plt.colorbar()
        
    plt.show()
    plt.close(fig)
    
def drow_mole(mole_ar):
    ax = plt.figure().add_subplot(projection='3d')
    lim_draw_L = 5
    ax.set_xlim([-lim_draw_L,lim_draw_L])
    ax.set_ylim([-lim_draw_L,lim_draw_L])
    ax.set_zlim([-lim_draw_L,lim_draw_L])
    ax.view_init(elev=20., azim=30, roll=0)
    for i in range(natom):
        atom_size = covalent_radius[ mole_ar[i][0] ]
        color = color_set[mole_ar[i][0]]
        x = mole_ar[i][1][0]
        y = mole_ar[i][1][1]
        z = mole_ar[i][1][2]
        ax.scatter(x,y,z,c=color, edgecolors="black" ,s=atom_size*1000)
    plt.show()
    plt.close
    
def get_number_of_vib_mode(mole_ar):
    cnt = 0 
    Rc = np.zeros(3)
    for atom_ar in mole_ar:
        cnt += 1
        Rc[0] += atom_ar[1][0]
        Rc[1] += atom_ar[1][1]
        Rc[2] += atom_ar[1][2]
    Rc = Rc/cnt
    
    Xc = np.zeros((cnt,3))
    for i in range(cnt):
        Xc[i,0] = mole_ar[i][1][0] - Rc[0]
        Xc[i,1] = mole_ar[i][1][1] - Rc[1]
        Xc[i,2] = mole_ar[i][1][2] - Rc[2]
        
    _, Ss, _ = np.linalg.svd(Xc)
    idx = np.where(Ss > 1e-5) 
    is_linear = Ss[idx].size
    
    if is_linear == 1:
        num_vib_mode = 3*cnt - 5
    else:
        num_vib_mode = 3*cnt - 6 
    return num_vib_mode
   
if __name__ == '__main__':
    print("Start")
    ### Set Parameters ###
    GEO_OPT = False
    VIB_ANA = False
    DRAW_VIB = False
    DRAW_MOLE = True
    
    tole = 1e-8
    max_ite = 200
    alpha = 0.5
    geo_thr = 1e-8
    opt_maxiter = 300
    mole_ar = [
                ["H",[ 0, 1.430523, 1.107379]],
                ["H",[ 0,-1.430523, 1.107379]],
                ["O",[ 0,    0, 0 ]],
              ]
    
    color_set = {
                    "H":"blue",
                    "O":"red",
                    "C":"grey"
                }
    
    basis_set = GaussianBasis("sto-3g")
    MPs = MolecularProperties(basis_set, mole_ar)
    mo_properties =  MPs.get_properties()
    AO_data = MPs.get_AO_data()
    Atom_data = MPs.get_Atom_data()
    ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl, natom, nao, nele, ao_to_atom, atomic_mass = mo_properties
    
    start = time.time()
    nocc = nele//2
    Cocc = np.zeros((nao,nocc))
    R = 0
    
    num_vib_mode = get_number_of_vib_mode(mole_ar)
    update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
    ### Compute Constants ###
    #me  = 9.1093837e-31 #kg
    #ep0 = 8.854187e-12  #F/m
    #e   = 1.60217663e-19 #C
    #h   = 6.62607015e-34 #m2kg/s
    #Eh = me*e**4/(2*np.pi*ep0*h)**2
    Eh = 4.3597454783513e-18
    omega_const = 4.1341380505336E+16
    
    if DRAW_MOLE:
        drow_mole(mole_ar)
    
    if GEO_OPT:
        Ropt = []
        Eopt = []
        thresh = geo_thr
        alp = 0.6
        for i, step in enumerate(range(opt_maxiter)):
            if i < 5:
                for A in range(natom):
                    for xyz in range(3):
                        er = random.uniform(-0.01,0.01)
                        mole_ar[A][1][xyz] -= er
                        
            _, E, _, eps,  C, P, nocc  = scf_driver(Cocc,alpha,tole,max_ite,log=False)
            Cocc = C[:,:nocc]
            
            #grad = numerical_gradient(mole_ar,Cocc,h=1e-4)
            grad = analytical_gradient(mole_ar,P,Cocc,eps[:nocc])
            gnorm = np.linalg.norm(grad)
            print(f"{step}: Energy {E.round(7)} ,grad {gnorm.round(8)}")
        
            if gnorm < thresh:
                update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
                break
            
            for A in range(natom):
                for xyz in range(3):
                    mole_ar[A][1][xyz] -= alp*grad[A,xyz]
                    
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            
            if DRAW_MOLE:
                drow_mole(mole_ar)
                
    Etotal, E, En, eps,  C, P, nocc = scf_driver(Cocc,alpha,tole,max_ite,log=True)
    Cocc = C[:, :nocc]
    tdiff = time.time() - start
    
    print("Final Results")
    print("Molecler Arangement:")
    print(mole_ar)
    print(f"eg: {eps[:nocc].round(3)} hartree")
    print(f"eu: {eps[nocc:].round(3)} hartree")
    print(f"Eelectron: {E} hartree")
    print(f"Etotal: {Etotal.round(3)} hartree")
    print(f"End:{tdiff} s")
    n_homo = eps[:nocc].size
    Plot_MOorbs(mole_ar,C,n_homo)
    
    if VIB_ANA:
        start_t = time.time()
        hessian = build_Hessian(mole_ar,Cocc,h=1e-4)
        eigval, eigvec = vibration_analysis(mole_ar, hessian)
        end_t = time.time() - start_t
        print("Process time:",end_t," s")
        wave_num = 219474.63137*np.sqrt(np.clip(eigval[-num_vib_mode:],0,None))
        print(f"Vibration: {wave_num} cm-1")
        
        delta = 0.01
        mass = np.repeat(atomic_mass,3)
        myu_plus = np.zeros((int(3*natom),3))
        myu_minus = np.zeros((int(3*natom),3))
        delta_myu = np.zeros((int(3*natom),3))
        for k in range(int(3*natom)):
            ##dipole μ+
            disp = delta * eigvec[:,k] / np.sqrt(mass)
            disp = disp.reshape(len(disp)//3,3)
            for A in range(natom):
                mole_ar[A][1] += disp[A]
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            
            #build P matrix
            Etotal, E, En, eps,  C, P, nocc = scf_driver(Cocc,alpha,tole,max_ite,log=False)
            
            #build Dipole matrix
            Dipole_x = np.zeros((nao,nao))
            Dipole_y = np.zeros((nao,nao))
            Dipole_z = np.zeros((nao,nao))
            ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl, natom, nao, nele, ao_to_atom, atomic_mass = MPs.get_properties()
            for p in range(int(nao)):
                for q in range( p + 1):
                    Dipole_x[p,q] = build_dipole_matrix_element(p,q,ncoord_for_ao,0)
                    Dipole_y[p,q] = build_dipole_matrix_element(p,q,ncoord_for_ao,1)
                    Dipole_z[p,q] = build_dipole_matrix_element(p,q,ncoord_for_ao,2)
                    
                    Dipole_x[q,p] = Dipole_x[p,q]
                    Dipole_y[q,p] = Dipole_y[p,q]
                    Dipole_z[q,p] = Dipole_z[p,q]
            
            #Nuclear contribution
            myu_nu_x = 0
            myu_nu_y = 0
            myu_nu_z = 0
            for A in range(natom):
                myu_nu_x += Nl_for_ao[A]*mole_ar[A][1][0]
                myu_nu_y += Nl_for_ao[A]*mole_ar[A][1][1]
                myu_nu_z += Nl_for_ao[A]*mole_ar[A][1][2]
        
            myu_x = -np.sum(P*Dipole_x) + myu_nu_x
            myu_y = -np.sum(P*Dipole_y) + myu_nu_y
            myu_z = -np.sum(P*Dipole_z) + myu_nu_z
            
            dipole_plus = np.array([myu_x,myu_y,myu_z])
            myu_plus[k] = dipole_plus
            #Restore mole arangement
            for A in range(natom):
                disp = delta * eigvec[:,k] / np.sqrt(mass)
                disp = disp.reshape(len(disp)//3,3)
                mole_ar[A][1] -= disp[A]
            
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            
            ##dipole μ-
            for A in range(natom):
                mole_ar[A][1] -= disp[A]
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            
            ###build P matrix
            Etotal, E, En, eps,  C, P, nocc = scf_driver(Cocc,alpha,tole,max_ite,log=False)
            
            #build Dipole matrix
            Dipole_x = np.zeros((nao,nao))
            Dipole_y = np.zeros((nao,nao))
            Dipole_z = np.zeros((nao,nao))
            ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl, natom, nao, nele, ao_to_atom, atomic_mass = MPs.get_properties()
            for p in range(int(nao)):
                for q in range( p + 1):
                    Dipole_x[p,q] = build_dipole_matrix_element(p,q,ncoord_for_ao,0)
                    Dipole_y[p,q] = build_dipole_matrix_element(p,q,ncoord_for_ao,1)
                    Dipole_z[p,q] = build_dipole_matrix_element(p,q,ncoord_for_ao,2)
                    
                    Dipole_x[q,p] = Dipole_x[p,q]
                    Dipole_y[q,p] = Dipole_y[p,q]
                    Dipole_z[q,p] = Dipole_z[p,q]
            
            #Nuclear contribution
            myu_nu_x = 0
            myu_nu_y = 0
            myu_nu_z = 0
            for A in range(natom):
                myu_nu_x += Nl_for_ao[A]*mole_ar[A][1][0]
                myu_nu_y += Nl_for_ao[A]*mole_ar[A][1][1]
                myu_nu_z += Nl_for_ao[A]*mole_ar[A][1][2]
        
            myu_x = -np.sum(P*Dipole_x) + myu_nu_x
            myu_y = -np.sum(P*Dipole_y) + myu_nu_y
            myu_z = -np.sum(P*Dipole_z) + myu_nu_z
            
            dipole_minus = np.array([myu_x,myu_y,myu_z])
            myu_minus[k] = dipole_minus
            
            #Restore mole arangement
            for A in range(natom):
                disp = delta * eigvec[:,k] / np.sqrt(mass)
                disp = disp.reshape(len(disp)//3,3)
                mole_ar[A][1] += disp[A]
            
            update_Atom_AO_data(qnl, Nl_for_ao,natom, ncoord, ncoord_for_ao)
            
            ###Compute delta dipole moment
            delta_myu[k] = (myu_plus[k] - myu_minus[k]) / (2*delta)
        print(delta_myu[-num_vib_mode:])

    
    if VIB_ANA and DRAW_VIB:
        ite_num_vib = 3
        eigval = eigval[-num_vib_mode:]
        eigvec = eigvec[:,-num_vib_mode:]
        mass = np.repeat(atomic_mass,3)
        Amp = 20
        mole_ar_vib = mole_ar.copy()
        R0 = np.array([atom[1].copy() for atom in mole_ar])
        for k in range(num_vib_mode):
            print(f"Mode: {k+1}")
            eigvec_k = eigvec[:,k]/np.sqrt(mass)
            eigvec_k = eigvec_k.reshape(eigvec_k.size//3,3)
            omega = np.sqrt(eigval[k])*omega_const
            t = np.linspace(0,2*np.pi/omega,10)
            for rep in range(ite_num_vib):
                #mole_ar_vib = mole_ar.copy()
                for t_i in t:
                    Q = Amp*np.cos(omega*t_i)
                    #mole_ar_vib = mole_ar.copy()
                    for A in range(natom):
                        delta_x = eigvec_k[A,0]*Q
                        delta_y = eigvec_k[A,1]*Q 
                        delta_z = eigvec_k[A,2]*Q
                        disp = np.array([delta_x,delta_y,delta_z])
                        mole_ar_vib[A][1] = R0[A] + disp
                        #print(mole_ar_vib[A][1])
                    drow_mole(mole_ar_vib)
    
    
    #cProfile.run( "analytical_gradient(mole_ar,P,Cocc,eps[:nocc])" )
    
