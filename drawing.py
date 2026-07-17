# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:02:00 2026
@author: kaz2255pq
"""
import numpy as np
import matplotlib.pyplot as plt

class Drawing():
    def __init__(self,MPs):
        self.ncoord, self.ncoord_for_ao, self.qnl, self.Nl_for_ao, _, self.natom, self.nao, _, _,_ = MPs.get_properties()
        self.MPs = MPs
        self.covalent_radius = {
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
    
    def Plot_MOorbs(self, basis_set, C, n_homo):
        
        ### Plot MO orbitals
        ncoord = self.ncoord.reshape((self.ncoord.size//3,3))
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
        MO_orbs = np.zeros((self.nao,Res,Res,Res))
        for i in range(self.nao):
            psi = np.zeros((Res,Res,Res))
            for j in range(self.nao):
                qnlA = self.qnl[j]
                basisA = basis_set.get_basis(self.Nl_for_ao[j],tuple(qnlA[:-1].tolist()))
                A = self.ncoord_for_ao[3*j:3*j+3]
                aL = self.MPs.angm_to_nlm(qnlA[1],qnlA[2])
                dA = basisA[3:]
                coef = basisA[:3]
                chi = self.MPs.contracted_gto(X,Y,Z,A,dA,coef,aL)
                psi += C[j,i] * chi
            MO_orbs[i,:,:,:] = psi
        
        fig_size = 4
        row_size = int(fig_size*self.nao)
        col_size = int(fig_size*3)
        fig, ax = plt.subplots(self.nao,3,figsize=(col_size,row_size))
        threshold = 0.02
        
        for i in range(self.nao):#(num_bands-1):#num_bands-1):
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
        
    def drow_mole(self,mole_ar,color_set):
        ax = plt.figure().add_subplot(projection='3d')
        lim_draw_L = 5
        ax.set_xlim([-lim_draw_L,lim_draw_L])
        ax.set_ylim([-lim_draw_L,lim_draw_L])
        ax.set_zlim([-lim_draw_L,lim_draw_L])
        ax.view_init(elev=20., azim=30, roll=0)
        for i in range(self.natom):
            atom_size = self.covalent_radius[ mole_ar[i][0] ]
            color = color_set[mole_ar[i][0]]
            x = mole_ar[i][1][0]
            y = mole_ar[i][1][1]
            z = mole_ar[i][1][2]
            ax.scatter(x,y,z,c=color, edgecolors="black" ,s=atom_size*1000)
        plt.show()
        plt.close