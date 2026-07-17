# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 23:50:01 2026

@author: kazuk
"""

from basis import GaussianBasis

basis_set = GaussianBasis("sto-3g")
oxygen = basis_set.get_basis(8,(1,0))

print(oxygen)


