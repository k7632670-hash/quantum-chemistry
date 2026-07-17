## Quantum Chemistry in Python
Molecular energy can be computed by solving roothaan equations for Ristricted Hartee-Fock with STO-3G (RHF/STO-3G). <br>
Available atoms are from H to Ne because I don't prepare basis set of other atoms and atomic properties.<br>
However, Overlap, Kinetic, Nuclear Attraction, Electron Replusive Integrals are available to other atoms with d, f oribitals.<br>

## How to Use
You need to edit "main.py" to adjust what molecular you want to calculate.<br>
Basically, you can implemet the code by editting only two parts shown below:<br>
"mole_ar" and "color_set"<br>

For example of H2O case,

1. Set molecular arrangement. unit:Bohr<br>
   Like<br>
   mole_ar = [<br>
             ["H",[ 0, 1.430523, 1.107379]],<br>
             ["H",[ 0,-1.430523, 1.107379]],<br>
             ["O",[ 0,    0, 0 ]],<br>
            ]<br>
   mole_ar is dict type. [ Atom symble, cartesian coordinate(x,y,z) ]<br>
  
3. Set colors to draw molecular
   If DRAW_MOLE is False, you don't need to edit color_set.<br>
   Like<br>
   color_set = {<br>
                  "H":"blue",<br>
                  "O":"red",<br>
                  "C":"grey"<br>
              }<br>
  
  If you need, you edit it.<br>
  color_set is also dict type. [ Atom symbole, color ]<br>
  color name is based on matplotlib.<br>
  You can chose colors what you like.<br>

***
Roothaan Equations_RHF sto3g.py is a test code.<br>
This is not used in main.py and regarding programs. <br>
***

## Library requried
- import numpy as np
- from math import erf
- from numba import jit
- import scipy.special
- import matplotlib.pyplot as plt

## Features
✓ Hartree-Fock

## Coming soon 
□ Analytical Gradient <br>
□ Geometry Optimization <br>
□ Hessian <br>
□ IR Spectrum <br>

Above functios are already completed in "Roothaan Equations_RHF sto3g.py" <br>
But they are not separeted to class. It's inconvenient to review programs. <br>
So, I'm making them modules now. <br>

## Future Plans
□ DFT <br>
□ MP2 <br>



## Basis Set Data
The basis set data included in this repository were obtained from the <br>
Basis Set Exchange (BSE). <br>

Basis Set Exchange: <br>
https://www.basissetexchange.org/ <br>

If you use these basis sets in scientific work, please cite the <br>
appropriate Basis Set Exchange publications and the original basis set <br>
references. <br>
