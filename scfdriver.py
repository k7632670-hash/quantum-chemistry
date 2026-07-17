# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:22:37 2026
@author: kaz2255pq
"""

import numpy as np

class SCFDriver():
    def __init__(self):
        '''
        self.Cocc = Cinit
        self.S = S
        self.T = T
        self.V = V
        self.ERI = ERI
        self.tole = tole
        self.max_ite = max_ite
        self.nocc = nele//2
        '''
        
    def build_G(self, P, ERI):
        J = np.einsum('rs,pqrs->pq', P, ERI, optimize=True )
        K = np.einsum('rs,prqs->pq', P, ERI, optimize=True )
        return J - 0.5*K
        
    def scfdriver(self,S,T,V,ERI,Cinit,nele,alpha=0.5,tole=1e-8,max_ite=100,log=True):
        nocc = nele//2
        P = 2 * Cinit @ Cinit.T
        H = T + V
        diff = 1
        cnt = 0
        while (diff > tole):
            if cnt >= max_ite:
                break
            if log:
                print(f"SCF: {cnt+1}, diff: {diff} ")
            G = self.build_G(P,ERI)   
            F = H + G
            S = 0.5 * (S + S.T)
            F = 0.5 * (F + F.T)
            
            eigS, U = np.linalg.eigh(S)
            thresh = 1e-8
            idx = eigS > thresh
            eigS2 = eigS[idx]
            U2 = U[:,idx]
            X = U2 @ np.diag(eigS2**(-0.5)) @ U2.T
            Fp = X.T @ F @ X
            eps, Cp = np.linalg.eigh(Fp)
            idx = np.argsort(eps)
            eps = eps[idx]
            Cp = Cp[:,idx]
            C = X @ Cp
            Cocc = C[:,:nocc]

            P_new =  2 * Cocc @ Cocc.T
            diff = np.linalg.norm(P_new - P)
            P_new = alpha * P_new + (1 - alpha) * P
            P_new = 0.5*(P_new + P_new.T)
            P = P_new
            cnt = cnt + 1
            
        G = self.build_G(P,ERI)
        F = H + G
        E = 0.5*np.sum(P*(H+F))
        
        return E, eps, C, P, nocc
    
        ###Energy
        #En = gi.nuclear_repulsion()
        #Etotal = E + En
        