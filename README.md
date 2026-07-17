##Quantum Chemistry in Python
Molecular energy can be computed by solving roothaan equations for Ristricted Hartee-Fock with STO-3G (RHF/STO-3G). 
Available atoms are from H to Ne because I don't prepare basis set of other atoms and atomic properties.
However, Overlap, Kinetic, Nuclear Attraction, Electron Replusive Integrals are available to other atoms with d, f oribitals.

##How to Use
You need to edit "main.py" to adjust what molecular you want to calculate.
Basically, you can implemet the code by editting only two parts shown below:
"mole_ar" and "color_set"

For example of H2O case,

1.Set molecular arrangement. unit:Bohr
  Like
  mole_ar = [
              ["H",[ 0, 1.430523, 1.107379]],
              ["H",[ 0,-1.430523, 1.107379]],
              ["O",[ 0,    0, 0 ]],
            ]

  mole_ar is dict type. [ Atom symble, cartesian coordinate(x,y,z) ]
  
2.Set colors to draw molecular
  If DRAW_MOLE is False, you don't need to edit color_set.
  Like
  color_set = {
                  "H":"blue",
                  "O":"red",
                  "C":"grey"
              }
  
  If you need, you edit it.
  color_set is also dict type. [ Atom symbole, color ]
  color name is based on matplotlib.
  You can chose colors what you like.

***
Roothaan Equations_RHF sto3g.py is a test code.
This is not used in main.py and regarding programs. 
***

##Library requried
import numpy as np
from math import erf
from numba import jit
import scipy.special
import matplotlib.pyplot as plt

##Features
✓ Hartree-Fock

##Coming soon 
□ Analytical Gradient
□ Geometry Optimization
□ Hessian
□ IR Spectrum
Above functios are already completed in "Roothaan Equations_RHF sto3g.py"
But they are not separeted to class. It's inconvenient to review programs.
So, I'm making them modules now.

##Future Plans
□ DFT
□ MP2



## Basis Set Data
The basis set data included in this repository were obtained from the
Basis Set Exchange (BSE).

Basis Set Exchange:
https://www.basissetexchange.org/

If you use these basis sets in scientific work, please cite the
appropriate Basis Set Exchange publications and the original basis set
references.
