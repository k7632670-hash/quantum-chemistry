# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 23:27:45 2026
@author: kaz2255pq
"""

import numpy as np
import scipy.special
from integrals import GaussianIntegrals

class MolecularProperties():
    def __init__(self,basis_set,mole_ar):
        self.AO_data = []
        self.Atom_data = []
        self.gi = GaussianIntegrals()
        self.update_Atom_AO_data(self.gi, basis_set,mole_ar)
        self.mo_properties = self.molecular_arrangement(mole_ar)
        
        
    def get_AO_data(self):
        return self.AO_data
    
    def get_Atom_data(self):
        return self.Atom_data
    
    def get_properties(self):
        return self.mo_properties
        
    def molecular_arrangement(self, mole_ar):
        ncoord_for_ao = np.zeros(0)
        ncoord = np.zeros(0)
        qnl = np.zeros(0)
        Nl_for_ao = np.zeros(0)
        Nl = np.zeros(0)
        ao_to_atom = np.zeros(0)
        natom = 0
        nao = 0
        nele = 0
        atomic_mass = np.zeros(0)
        for k, mole in enumerate(mole_ar):
            key = mole[0]
            #print(key)
            if key == "H":
                Z = 1
                mcoord = mole[1]
                qn = [ [1,0,0] ]
                ncoord = np.append(ncoord,mcoord)
                ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                qnl = np.append(qnl,qn)
                Nl_for_ao = np.append(Nl_for_ao,Z)
                Nl = np.append(Nl,Z)
                mass = 1837.15264585
                atomic_mass = np.append(atomic_mass,mass)

                ao_to_atom = np.append(ao_to_atom,k)
                natom += 1
                nele += Z
                nao += 1
                
            elif key == "He":
                Z = 2
                mcoord = mole[1]
                qn = [ [1,0,0] ]
                ncoord = np.append(ncoord,mcoord)
                ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                qnl = np.append(qnl,qn)
                Nl_for_ao = np.append(Nl_for_ao,Z)
                Nl = np.append(Nl,Z)
                mass = 7296.29938752
                atomic_mass = np.append(atomic_mass,mass)
                ao_to_atom = np.append(ao_to_atom,k)
                natom += 1
                nele += Z
                nao += 1
                
            elif key == "Li":
                Z = 3
                mcoord = mole[1]
                qn = [ [1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1] ]
                ncoord = np.append(ncoord,mcoord)
                Nl = np.append(Nl,Z)
                mass = 12786.392858
                atomic_mass = np.append(atomic_mass,mass)
                for i in range(len(qn)):
                    ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                    Nl_for_ao = np.append(Nl_for_ao,Z)
                    ao_to_atom = np.append(ao_to_atom,k)
                qnl = np.append(qnl,qn)
                natom += 1
                nele += Z
                nao += len(qn)
                
            elif key == "Be":
                Z = 4
                mcoord = mole[1]
                qn = [ [1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1]]
                ncoord = np.append(ncoord,mcoord)
                Nl = np.append(Nl,Z)
                mass = 16428.204949
                atomic_mass = np.append(atomic_mass,mass)
                for i in range(len(qn)):
                    ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                    Nl_for_ao = np.append(Nl_for_ao,Z)
                    ao_to_atom = np.append(ao_to_atom,k)
                qnl = np.append(qnl,qn)
                natom += 1
                nele += Z
                nao += len(qn)
                
            elif key == "B":
                Z = 5
                mcoord = mole[1]
                qn = [ [1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1]]
                ncoord = np.append(ncoord,mcoord)
                Nl = np.append(Nl,Z)
                mass = 20068.735312
                atomic_mass = np.append(atomic_mass,mass)
                for i in range(len(qn)):
                    ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                    Nl_for_ao = np.append(Nl_for_ao,Z)
                    ao_to_atom = np.append(ao_to_atom,k)
                qnl = np.append(qnl,qn)
                natom += 1
                nele += Z
                nao += len(qn)
                
            elif key == "C":
                Z = 6
                mcoord = mole[1]
                qn = [ [1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1]]
                ncoord = np.append(ncoord,mcoord)
                Nl = np.append(Nl,Z)
                mass = 21874.661834
                atomic_mass = np.append(atomic_mass,mass)
                for i in range(len(qn)):
                    ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                    Nl_for_ao = np.append(Nl_for_ao,Z)
                    ao_to_atom = np.append(ao_to_atom,k)
                qnl = np.append(qnl,qn)
                natom += 1
                nele += Z
                nao += len(qn)
                
            elif key == "N":
                Z = 7
                mcoord = mole[1]
                qn = [ [1,0,0],[2,0,0],[2,1,0],[2,1,-1],[2,1,1]]
                ncoord = np.append(ncoord,mcoord)
                Nl = np.append(Nl,Z)
                mass = 25526.042349
                atomic_mass = np.append(atomic_mass,mass)
                for i in range(len(qn)):
                    ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                    Nl_for_ao = np.append(Nl_for_ao,Z)
                    ao_to_atom = np.append(ao_to_atom,k)
                qnl = np.append(qnl,qn)
                natom += 1
                nele += Z
                nao += len(qn)
                
            elif key == "O":
                Z = 8
                mcoord = mole[1]
                qn = [ [1,0,0],[2,0,0],[2,1,0],[2,1,-1],[2,1,1]]
                ncoord = np.append(ncoord,mcoord)
                Nl = np.append(Nl,Z)
                mass = 29156.945697
                atomic_mass = np.append(atomic_mass,mass)
                for i in range(len(qn)):
                    ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                    Nl_for_ao = np.append(Nl_for_ao,Z)
                    ao_to_atom = np.append(ao_to_atom,k)
                qnl = np.append(qnl,qn)
                natom += 1
                nele += Z
                nao += len(qn)
                
            elif key == "F":
                Z = 9
                mcoord = mole[1]
                qn = [ [1,0,0],[2,0,0],[2,1,0],[2,1,-1],[2,1,1]]
                ncoord = np.append(ncoord,mcoord)
                Nl = np.append(Nl,Z)
                mass = 34631.970382
                atomic_mass = np.append(atomic_mass,mass)
                for i in range(len(qn)):
                    ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                    Nl_for_ao = np.append(Nl_for_ao,Z)
                    ao_to_atom = np.append(ao_to_atom,k)
                qnl = np.append(qnl,qn)
                natom += 1
                nele += Z
                nao += len(qn)
                
            elif key == "Ne":
                Z = 10
                mcoord = mole[1]
                qn = [ [1,0,0],[2,0,0],[2,1,0],[2,1,-1],[2,1,1]]
                ncoord = np.append(ncoord,mcoord)
                Nl = np.append(Nl,Z)
                mass = 36444.492412
                atomic_mass = np.append(atomic_mass,mass)
                for i in range(len(qn)):
                    ncoord_for_ao = np.append(ncoord_for_ao,mcoord)
                    Nl_for_ao = np.append(Nl_for_ao,Z)
                    ao_to_atom = np.append(ao_to_atom,k)
                qnl = np.append(qnl,qn)
                natom += 1
                nele += Z
                nao += len(qn)

        qnl = qnl.reshape(int(len(qnl)/3),3)
        return [ncoord, ncoord_for_ao, qnl.astype(int), Nl_for_ao, Nl, natom, nao, nele, ao_to_atom, atomic_mass]
    
    def fact2(self,n):
        if n <= 0:
            return 1
        return scipy.special.factorial2(n)    

    def primitive_gto(self, x,y,z,A,dA,aL):
        l,m,n = aL
        dx = x - A[0]
        dy = y - A[1]
        dz = z - A[2]
        
        r2 = dx**2 + dy**2 + dz**2
        poly = dx**l * dy**m * dz**n
        val = poly * np.exp(-dA*r2)
        #print(val.shape)
        return val
    
    def contracted_gto(self, x,y,z,A,dA,coef, aL):
        val = 0 
        for beta,alpha in zip(dA,coef):
            
            N = self.primitive_norm(beta,aL)
            val += alpha * N * self.primitive_gto(x,y,z,A,beta,aL)
        
        return val
    
    def angm_to_nlm(self,l,m):
        if l == 0:
            return (0,0,0)
        
        if l == 1:
            if m == -1:
                return (1,0,0) #px
            elif m == 1:
                return (0,1,0) #py 
            elif m == 0:
                return (0,0,1) #pz
            
    def primitive_norm(self,alpha,nL):
        l = nL[0]
        m = nL[1]
        n = nL[2]
        L = l+m+n
    
        num = (4*alpha)**L
        den = self.fact2(2*l-1)*self.fact2(2*m-1)*self.fact2(2*n-1)  
            
        return  (2*alpha/np.pi)**0.75 * np.sqrt(num/den)
    
    def contraction_norm(self,coeff, d, aL,keyA,keyB,gi):
        S = 0.0
        A = np.zeros(3)
        for i in range(len(d)):
            for j in range(len(d)):
    
                Ni = self.primitive_norm(d[i],aL)
                Nj = self.primitive_norm(d[j],aL)
    
                Sij = gi.overlap_os( aL,aL,A,A,d[i],d[j],keyA,keyA )
    
                S += ( coeff[i] * coeff[j] * Ni * Nj * Sij )
    
        return 1/np.sqrt(S)
    
    def update_Atom_AO_data(self,gi, basis_set, mole_ar):
        self.ncoord, self.ncoord_for_ao, self.qnl, self.Nl_for_ao, self.Nl, self.natom, self.nao, self.nele, self.ao_to_atom, self.atomic_mass = self.molecular_arrangement(mole_ar)
        self.AO_data.clear()
        self.Atom_data.clear()
        for p in range(self.nao):
            qnlA = self.qnl[p]
            basisA = basis_set.get_basis(self.Nl_for_ao[p],tuple(qnlA[:-1].tolist()))
            coeff = basisA[:3]
            expo = basisA[3:]
            aL = self.angm_to_nlm(qnlA[1],qnlA[2])
            keyA = tuple(self.ncoord_for_ao[3*p:3*p+3])
    
            self.AO_data.append(
                        {
                        "coeff":np.array(coeff),
                        "expo":np.array(expo),
                        "aL":aL,
                        "CN":self.contraction_norm(coeff,expo,aL,keyA,keyA,gi),
                        "norm":[self.primitive_norm(expo[i],aL) for i in range(3) ],
                        "coord_key":tuple(self.ncoord_for_ao[3*p:3*p+3]),
                        }
                                )
        
        for A in range(self.natom):
            self.Atom_data.append(
                        {
                        "atom_coord_key":tuple(self.ncoord[3*A:3*A+3])
                        }
                        )        
        