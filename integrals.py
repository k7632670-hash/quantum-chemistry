# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 22:55:30 2026
@author: kaz2255pq
"""

import numpy as np
from math import erf
from numba import jit

class GaussianIntegrals():
    def __init__(self):
        self.cache_ol = {}
        self.cache_k = {}
        self.cache_ne = {}
        self.cache_eri = {}
        self.cache_ERIs = {}
        
    def set_properties(self,MPs):
        self.AO_data = MPs.get_AO_data()
        self.Atom_data = MPs.get_Atom_data()
        mo_properties = MPs.get_properties()
        self.ncoord, self.ncoord_for_ao, self.qnl, self.Nl_for_ao, self.Nl, self.natom, self.nao, self.nele, self.ao_to_atom, self.atomic_mass = mo_properties
        
    def clear_cache(self):
        self.cache_ol.clear()
        self.cache_k.clear()
        self.cache_ne.clear()
        self.cache_eri.clear()
        self.cache_ERIs.clear()
        
    @staticmethod
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
    
    def overlap_ss(self,aL,bL,A,B,da,db):
        gamma = da + db
        mu = da*db/gamma
        Rab2 = np.dot(A-B,A-B)
        return (np.pi/gamma)**1.5*np.exp(-mu*Rab2)

    
    def overlap_os(self,aL,bL,A,B,da,db,keyA,keyB):
        key = (
                aL,
                bL,
                keyA,
                keyB,
                da,
                db
            )
        
        if key in self.cache_ol:
            return self.cache_ol[key]
        
        if aL==(0,0,0) and bL==(0,0,0):
            val = self.overlap_ss(aL,bL,A,B,da,db)
            self.cache_ol[key] = val
            return val
        
        gamma = da+db
        P = (da*A + db*B)/gamma

        for i in range(3):
            if aL[i] > 0:

                a_down = list(aL)
                a_down[i] -= 1
                a_down = tuple(a_down)

                term1 = ( (P[i]-A[i]) * self.overlap_os(a_down,bL,A,B,da,db,keyA,keyB) )

                term2 = 0.0
                if a_down[i] >= 0:

                    a_down2 = list(a_down)
                    a_down2[i] -= 1

                    if a_down2[i] >= 0:

                        a_down2 = tuple(a_down2)
                        term2 =  a_down[i]/(2*gamma)  * self.overlap_os( a_down2,bL,A,B,da,db,keyA,keyB )

                term3 = 0.0
                if bL[i] > 0:

                    b_down = list(bL)
                    b_down[i] -= 1
                    b_down = tuple(b_down)

                    term3 =  bL[i]/(2*gamma) * self.overlap_os( a_down,b_down,A,B,da,db,keyA,keyB )
                    
                val = (term1 + term2 + term3)
                #if  abs(val) < 1e-8:
                #    val = 0
                self.cache_ol[key] = val
                
                return val
            
        for i in range(3):
            if bL[i] > 0:

                b_down = list(bL)
                b_down[i] -= 1
                b_down = tuple(b_down)

                term1 = (P[i]-B[i]) * self.overlap_os(aL,b_down,A,B,da,db,keyA,keyB)
                
                term2 = 0.0
                if b_down[i] >= 0:

                    b_down2 = list(b_down)
                    b_down2[i] -= 1

                    if b_down2[i] >= 0:

                        b_down2 = tuple(b_down2)
                        term2 = b_down[i]/(2*gamma)* self.overlap_os(aL,b_down2,A,B,da,db,keyA,keyB)
                        
                term3 = 0.0
                if aL[i] > 0:

                    a_down = list(aL)
                    a_down[i] -= 1
                    a_down = tuple(a_down)

                    term3 =  aL[i]/(2*gamma)* self.overlap_os(a_down,b_down,A,B,da,db,keyA,keyB)
                    
                val = (term1 + term2 + term3)
                #if  abs(val) < 1e-8:
                #    val = 0
                self.cache_ol[key] = val
                return val
    
    def kinetic_os(self,aL,bL,A,B,da,db,keyA,keyB):

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
        
        if key in self.cache_k:
            return self.cache_k[key]

        term1 = db*(2*Lb+3) * self.overlap_os(aL,bL,A,B,da,db,keyA,keyB)
        term2 = 0.0

        for i in range(3):
            b_up = list(bL)
            b_up[i] += 2
            b_up = tuple(b_up)
            term2 += self.overlap_os(aL,b_up,A,B,da,db,keyA,keyB)

        term2 *= -2*db*db
        term3 = 0.0

        for i in range(3):
            if bL[i] >= 2:
                b_down = list(bL)
                b_down[i] -= 2
                b_down = tuple(b_down)

                term3 += bL[i]*(bL[i]-1)  * self.overlap_os(aL,b_down,A,B,da,db,keyA,keyB)  

        term3 *= -0.5
        val = term1 + term2 + term3
        #if abs(val) < 1e-8:
        #    val = 0
        self.cache_k[key] = val
        return val
    
    def na_ss(self, aL,b,m,A,B,C,da,db,Zc):
        gamma = da + db
        mu = da*db/gamma
        P = (da*A + db*B)/gamma
        Rab2 = np.dot(A-B,A-B)
        Rpc2= np.dot(P-C,P-C)
        T = gamma*Rpc2
        Vss = -2*np.pi*Zc/gamma*np.exp(-mu*Rab2)*self.boys(m,T)
        #print(T, boys(m,T))
        return Vss
    
    def na_os(self,aL,bL,m,A,B,C,da,db,Zc,keyA,keyB,keyC):
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
        
        if key in self.cache_ne:
            return self.cache_ne[key]
        
        if (aL==(0,0,0)and bL==(0,0,0)):
            val = self.na_ss(aL,bL,m,A,B,C,da,db,Zc)
            self.cache_ne[key] = val
            return val

        gamma = da + db
        P = (da*A + db*B)/gamma
        
        for i in range(3):
            if aL[i] > 0:
                a_down = list(aL)
                a_down[i] -= 1
                a_down = tuple(a_down)
                
                term1 = ( (P[i]-A[i])* self.na_os(a_down,bL,m,A,B,C,da,db,Zc,keyA,keyB,keyC) )     
                term2 = - (P[i]-C[i])* self.na_os(a_down,bL,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) 
                
                term3 = 0.0
                if a_down[i] > 0:
                    a_down2 = list(a_down)
                    a_down2[i] -= 1
                    a_down2 = tuple(a_down2)
                    term3 = ( a_down[i]/(2*gamma)) * (self.na_os(a_down2,bL,m,A,B,C,da,db,Zc,keyA,keyB,keyC) - self.na_os(a_down2,bL,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) )
                        
                term4 = 0.0
                if bL[i] > 0:
                    b_down = list(bL)
                    b_down[i] -= 1
                    b_down = tuple(b_down)
                    term4 = (bL[i]/(2*gamma)) * (self.na_os(a_down,b_down,m,A,B,C,da,db,Zc,keyA,keyB,keyC) - self.na_os(a_down,b_down,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) )
                        
                val = (term1 + term2 + term3 + term4)
                #if  abs(val) < 1e-8:
                #    val = 0
                self.cache_ne[key] = val

                return val
            
        for i in range(3): 
            if bL[i] > 0:
                b_down = list(bL)
                b_down[i] -= 1
                b_down = tuple(b_down)
                
                term1 = (P[i]-B[i])* self.na_os(aL,b_down,m,A,B,C,da,db,Zc,keyA,keyB,keyC)     
                term2 = - (P[i]-C[i])* self.na_os(aL,b_down,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) 
                
                term3 = 0.0
                if b_down[i] > 0:
                    b_down2 = list(b_down)
                    b_down2[i] -= 1
                    b_down2 = tuple(b_down2)
                    term3 = ( b_down[i]/(2*gamma)) * (self.na_os(aL,b_down2,m,A,B,C,da,db,Zc,keyA,keyB,keyC) - self.na_os(aL,b_down2,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) )
                        
                term4 = 0.0
                if aL[i] > 0:
                    a_down = list(aL)
                    a_down[i] -= 1
                    a_down = tuple(a_down)
                    term4 = (aL[i]/(2*gamma)) * (self.na_os(a_down,b_down,m,A,B,C,da,db,Zc,keyA,keyB,keyC) - self.na_os(a_down,b_down,m+1,A,B,C,da,db,Zc,keyA,keyB,keyC) )
                        

                        
                val = (term1 + term2 + term3 + term4)
                #if  abs(val) < 1e-8:
                #    val = 0
                self.cache_ne[key] = val

                return val
            
    def eri_ssss(self,m, A,B,C,D,da,db,dc,dd):
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

        return pref * self.boys(m,T)

    
    def eri_os(self,aL,bL,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD):
        key = (
                aL,bL,cL,dL,int(m),
                keyA,keyB,keyC,keyD,
                da,
                db,
                dc,
                dd
            )

        if key in self.cache_eri:
            return self.cache_eri[key]
        
        if (aL==(0,0,0)and bL==(0,0,0)and cL==(0,0,0) and dL==(0,0,0)):
            val = self.eri_ssss(m,A,B,C,D,da,db,dc,dd)
            self.cache_eri[key] = val
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
                
                term1 = ( (P[i]-A[i])* self.eri_os(a_down,bL,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )     
                term2 = ( (W[i]-P[i])* self.eri_os(a_down,bL,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                
                term3 = 0.0
                if a_down[i] > 0:
                    a_down2 = list(a_down)
                    a_down2[i] -= 1
                    a_down2 = tuple(a_down2)
                    term3 = ( a_down[i]/(2*gamma)) * (self.eri_os(a_down2,bL,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - delta/(gamma+delta)  * self.eri_os(a_down2,bL,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                        
                term4 = 0.0
                if bL[i] > 0:
                    b_down = list(bL)
                    b_down[i] -= 1
                    b_down = tuple(b_down)
                    term4 = (bL[i]/(2*gamma)) * ( self.eri_os(a_down,b_down,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - delta/(gamma+delta) * self.eri_os(a_down,b_down,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                        
                term5 = 0.0
                if cL[i] > 0:
                    c_down = list(cL)
                    c_down[i] -= 1
                    c_down = tuple(c_down)
                    term5 = ( cL[i]/(2*(gamma+delta))) * self.eri_os(a_down,bL,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD )
                        
                term6 = 0.0
                if dL[i] > 0:
                    d_down = list(dL)
                    d_down[i] -= 1
                    d_down = tuple(d_down)
                    term6 = (dL[i]/(2*(gamma+delta))) * self.eri_os(a_down,bL,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD)
                        
                val = (term1 + term2 + term3 + term4 + term5 + term6 )
                #if  abs(val) < 1e-8:
                #    val = 0
                self.cache_eri[key] = val

                return val

        for i in range(3):
            if bL[i] > 0:
                b_down = list(bL)
                b_down[i] -= 1
                b_down = tuple(b_down)
                
                term1 = ( (P[i]-B[i])* self.eri_os(aL,b_down,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )     
                term2 = ( (W[i]-P[i])* self.eri_os(aL,b_down,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                
                term3 = 0.0
                if b_down[i] > 0:
                    b_down2 = list(b_down)
                    b_down2[i] -= 1
                    b_down2 = tuple(b_down2)
                    term3 = ( b_down[i]/(2*gamma)) * (self.eri_os(aL,b_down2,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - delta/(gamma+delta)  * self.eri_os(aL,b_down2,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                        
                term4 = 0.0
                if aL[i] > 0:
                    a_down = list(aL)
                    a_down[i] -= 1
                    a_down = tuple(a_down)
                    term4 = (aL[i]/(2*gamma)) * ( self.eri_os(a_down,b_down,cL,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - delta/(gamma+delta) * self.eri_os(a_down,b_down,cL,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                        
                term5 = 0.0
                if cL[i] > 0:
                    c_down = list(cL)
                    c_down[i] -= 1
                    c_down = tuple(c_down)
                    term5 = ( cL[i]/(2*(gamma+delta))) * self.eri_os(aL,b_down,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD )
                        
                term6 = 0.0
                if dL[i] > 0:
                    d_down = list(dL)
                    d_down[i] -= 1
                    d_down = tuple(d_down)

                    term6 = (dL[i]/(2*(gamma+delta))) * self.eri_os(aL,b_down,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD)
                        
                val = (term1 + term2 + term3 + term4 + term5 + term6 )
                #if  abs(val) < 1e-8:
                #    val = 0
                self.cache_eri[key] = val

                return val
            
        for i in range(3):
            if cL[i] > 0:
                c_down = list(cL)
                c_down[i] -= 1
                c_down = tuple(c_down)
                
                term1 = ( (Q[i]-C[i])* self.eri_os(aL,bL,c_down,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )     
                term2 = ( (W[i]-Q[i])* self.eri_os(aL,bL,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                
                term3 = 0.0
                if c_down[i] > 0:
                    c_down2 = list(c_down)
                    c_down2[i] -= 1
                    c_down2 = tuple(c_down2)
                    term3 = ( c_down[i]/(2*delta)) * (self.eri_os(aL,bL,c_down2,dL,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - gamma/(gamma+delta)  * self.eri_os(aL,bL,c_down2,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                        
                term4 = 0.0
                if dL[i] > 0:
                    d_down = list(dL)
                    d_down[i] -= 1
                    d_down = tuple(d_down)
                    term4 = (dL[i]/(2*delta)) * ( self.eri_os(aL,bL,c_down,d_down,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - gamma/(gamma+delta) * self.eri_os(aL,bL,c_down,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                        
                term5 = 0.0
                if aL[i] > 0:
                    a_down = list(aL)
                    a_down[i] -= 1
                    a_down = tuple(a_down)
                    term5 = ( aL[i]/(2*(gamma+delta))) * self.eri_os(a_down,bL,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD )
                        
                term6 = 0.0
                if bL[i] > 0:
                    b_down = list(bL)
                    b_down[i] -= 1
                    b_down = tuple(b_down)

                    term6 = (bL[i]/(2*(gamma+delta))) * self.eri_os(aL,b_down,c_down,dL,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD)
                        
                val = (term1 + term2 + term3 + term4 + term5 + term6 )
                #if  abs(val) < 1e-8:
                #    val = 0
                self.cache_eri[key] = val

                return val
            
        for i in range(3):
            if dL[i] > 0:
                d_down = list(dL)
                d_down[i] -= 1
                d_down = tuple(d_down)
                
                term1 = ( (Q[i]-D[i])* self.eri_os(aL,bL,cL,d_down,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )     
                term2 = ( (W[i]-Q[i])* self.eri_os(aL,bL,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                
                term3 = 0.0
                if d_down[i] > 0:
                    d_down2 = list(d_down)
                    d_down2[i] -= 1
                    d_down2 = tuple(d_down2)
                    term3 = ( d_down[i]/(2*delta)) * (self.eri_os(aL,bL,cL,d_down2,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - gamma/(gamma+delta)  * self.eri_os(aL,bL,cL,d_down2,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                        
                term4 = 0.0
                if cL[i] > 0:
                    c_down = list(cL)
                    c_down[i] -= 1
                    c_down = tuple(c_down)
                    term4 = (cL[i]/(2*delta)) * ( self.eri_os(aL,bL,c_down,d_down,m,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) - gamma/(gamma+delta) * self.eri_os(aL,bL,c_down,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD) )
                        
                term5 = 0.0
                if aL[i] > 0:
                    a_down = list(aL)
                    a_down[i] -= 1
                    a_down = tuple(a_down)
                    term5 = ( aL[i]/(2*(gamma+delta))) * self.eri_os(a_down,bL,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD )
                        
                term6 = 0.0
                if bL[i] > 0:
                    b_down = list(bL)
                    b_down[i] -= 1
                    b_down = tuple(b_down)

                    term6 = (bL[i]/(2*(gamma+delta))) * self.eri_os(aL,b_down,cL,d_down,m+1,A,B,C,D,da,db,dc,dd,keyA,keyB,keyC,keyD)
                        
                val = (term1 + term2 + term3 + term4 + term5 + term6 )
                #if  abs(val) < 1e-8:
                #    val = 0
                self.cache_eri[key] = val

                return val
        
    def overlap_integral(self,p,q):
        AOA = self.AO_data[p]
        AOB = self.AO_data[q]

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
        
        
        A = self.ncoord_for_ao[3*p:3*p+3]
        B = self.ncoord_for_ao[3*q:3*q+3]
        keyA =  AOA["coord_key"]
        keyB =  AOB["coord_key"]
        
        f = 1.
        S = 0
        for i in range(3):
            for j in range(3):
                S += coeffA[i]*coeffB[j] * normA[i] * normB[j] * self.overlap_os(aL,bL,A,B,dA[i],dB[j],keyA,keyB)
        return S
    
    def kinetic_integral(self,p,q):
        AOA = self.AO_data[p]
        AOB = self.AO_data[q]

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
        
        A = self.ncoord_for_ao[3*p:3*p+3]
        B = self.ncoord_for_ao[3*q:3*q+3]
        
        keyA =  AOA["coord_key"]
        keyB =  AOB["coord_key"]
        
        f = 1.
        coeffA = coeffA * CNA
        coeffB = coeffB * CNB

        I = 0
        for i in range(3):
            for j in range(3):
                I += coeffA[i]*coeffB[j] *normA[i] * normB[j] * self.kinetic_os(aL,bL,A,B,dA[i],dB[j],keyA,keyB)

        return I
    
    def nuclear_attraction_integral(self,p,q):
        AOA = self.AO_data[p]
        AOB = self.AO_data[q]

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
        
        A = self.ncoord_for_ao[3*p:3*p+3]
        B = self.ncoord_for_ao[3*q:3*q+3]
        
        f = 1.
        
        coeffA = coeffA * CNA
        coeffB = coeffB * CNB
        
        I = 0
        m = 0
        for n in range(self.natom):
            AtC = self.Atom_data[n]
            keyC = AtC["atom_coord_key"]
            C = self.ncoord[3*n:3*n+3]
            Zc = self.Nl[n]
            for i in range(3):
                for j in range(3):
                    I += coeffA[i]*coeffB[j] *normA[i] * normB[j] *self.na_os(aL,bL,m,A,B,C,dA[i],dB[j],Zc,keyA,keyB,keyC)

        return I
    
    def electron_repulsive_integral(self,p,q,r,s):
        AOA = self.AO_data[p]
        AOB = self.AO_data[q]
        AOC = self.AO_data[r]
        AOD = self.AO_data[s]
        
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
        
        A = self.ncoord_for_ao[3*p:3*p+3]
        B = self.ncoord_for_ao[3*q:3*q+3]
        C = self.ncoord_for_ao[3*r:3*r+3]
        D = self.ncoord_for_ao[3*s:3*s+3]
        
        m = 0
        eri = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        ERI = self.eri_os(aL,bL,cL,dL,m,A,B,C,D,dA[i],dB[j],dC[k],dD[l],keyA,keyB,keyC,keyD)
                        eri += coeffA[i]*coeffB[j]*coeffC[k]*coeffD[l]* normA[i] * normB[j] * normC[k] * normD[l] * ERI 
        return eri
    
    def overlap_matrix(self):
        S_matrix = np.zeros((self.nao,self.nao))
        for p in range(self.nao):
            for q in range(self.nao):
                S_matrix[p,q] = self.overlap_integral(p,q)
        
        return S_matrix
    
    def kinetic_matrix(self):
        T_matrix = np.zeros((self.nao,self.nao))
        for p in range(self.nao):
            for q in range(self.nao):
                T_matrix[p,q] = self.kinetic_integral(p,q)
        
        return T_matrix
    
    def nuclear_attraction_matrix(self):
        V_matrix = np.zeros((self.nao,self.nao))
        for p in range(self.nao):
            for q in range(self.nao):
                V_matrix[p,q] = self.nuclear_attraction_integral(p,q)
        
        return V_matrix
    
    def eri_key(self,p,q,r,s):
        pair1 = tuple(sorted((p,q)))
        pair2 = tuple(sorted((r,s)))
        if pair2 < pair1:
            pair1, pair2 = pair2, pair1

        return (pair1, pair2)

    def check_eri(self,p,q,r,s):
        key = self.eri_key(p,q,r,s)
        if key in self.cache_ERIs:
            return self.cache_ERIs[key] 
        else:
            eri = self.electron_repulsive_integral(p,q,r,s)
            self.cache_ERIs[key] = eri
            return eri
        
    def electron_repulsive_matrix(self):
        ERI_matrix = np.zeros((self.nao,self.nao,self.nao,self.nao))
        for p in range(self.nao):
            for q in range(self.nao):
                for r in range(self.nao):
                    for s in range(self.nao):
                        ERI_matrix[p,q,r,s] = self.check_eri(p,q,r,s)
        
        return ERI_matrix
    
    def nuclear_repulsion(self):
        inv_r12 = 0
        for i in range(self.natom):
            for j in range(self.natom):
                if j > i:
                    r12 =  (((self.ncoord[3*i:3*i+3] - self.ncoord[3*j:3*j+3])**2).sum()) **0.5
                    if r12 > 1e-8:
                        inv_r12 = inv_r12 + (self.Nl[i]*self.Nl[j]) / r12

        return inv_r12
        