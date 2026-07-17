# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:04:04 2026
@author: kaz2255pq
"""

import numpy as np
import time

###Load modules
from basis import GaussianBasis
from properties import MolecularProperties
from integrals import GaussianIntegrals 
from scfdriver import SCFDriver
from drawing import Drawing


# Define SCF CYCLE Procedure
def scf_driver(Cinit,tole,max_ite,alpha=0.6,log=True):
    ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl,natom, nao, nele, ao_to_atom, atomic_mass = MPs.get_properties()

    #Call methods to compute integrals
    gi = GaussianIntegrals()
    gi.set_properties(MPs)
    
    #Compute integrals
    S = gi.overlap_matrix()
    T = gi.kinetic_matrix()
    V = gi.nuclear_attraction_matrix()
    ERI = gi.electron_repulsive_matrix()
    
    #Drive SCF cycle
    scf = SCFDriver()
    Eele, eps, C, P, nocc = scf.scfdriver(S,T,V,ERI,Cinit,nele,alpha,tole,max_ite,log)

    #Nuclear Energy and Total Energy
    En = gi.nuclear_repulsion()
    Etotal = Eele + En
    
    ###Clear Caches of recursions in integral methods
    gi.clear_cache()

    return Etotal, Eele, En, eps, C, P


#----------#
### Main ###
#----------#

### Set Parameters ###
DRAW_MOLE = True       #True: Enable drawing molecular
DRAW_MOLE_ORBs = True  #True: Enalbe drawing molecular orbitals

tole = 1e-8     #SCF Convergence tolerance
max_ite = 200   #SCF Max Iteration
alpha = 0.7     #Arbitraly:default 0.5, SCF Mixing parameter for updating P

#Set molecular arrangement. unit:Bohr
mole_ar = [
            ["H",[ 0, 1.430523, 1.107379]],
            ["H",[ 0,-1.430523, 1.107379]],
            ["O",[ 0,    0, 0 ]],
          ]

#Set colors to draw molecular
color_set = {
                "H":"blue",
                "O":"red",
                "C":"grey"
            }

#Select basis set. Currently sto-3g is only available
basis_set = GaussianBasis("sto-3g")

#Get atomic orbitals information
MPs = MolecularProperties(basis_set, mole_ar)
AO_data = MPs.get_AO_data()
Atom_data = MPs.get_Atom_data()

#Get properties of atomic orbitals
ncoord, ncoord_for_ao, qnl, Nl_for_ao, Nl, natom, nao, nele, ao_to_atom, atomic_mass = MPs.get_properties()

#Calculate the number of electrons in occupyed oribitals
nocc = nele//2

#Set initial C matrix for SCF 
Cocc = np.zeros((nao,nocc))

if DRAW_MOLE:
    #Draow molecular
    dr = Drawing(MPs)
    dr.drow_mole(mole_ar,color_set)
    
print("Start")
start = time.time()

#Implement SCF Cycle for RHF/STO-3G
Etotal, E, En, eps,  C, P = scf_driver(Cocc,tole,max_ite,alpha,log=True)
tdiff = time.time() - start

print("### Final Results ###")
print("Molecler Arangement:")
print(mole_ar)
print(f"eg: {eps[:nocc].round(3)} hartree")
print(f"eu: {eps[nocc:].round(3)} hartree")
print(f"Eelectron: {E} hartree")
print(f"Etotal: {Etotal.round(3)} hartree")
print(f"End:{tdiff} s")

if DRAW_MOLE_ORBs:
    #Plot molecular orbitals
    dr = Drawing(MPs)
    dr.Plot_MOorbs(basis_set, C, nocc)

    
