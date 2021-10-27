#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:56:42 2020

@author: jcayuso
"""

import numpy as np
import kszpsz_config as conf
from scipy.interpolate import interp1d
import cosmology 



class binning(object):

    def __init__(self, basic_conf_obj = None, sigma_photo_z = None) :
        if not basic_conf_obj:
            basic_conf_obj = conf
            
        
        self._basic_conf_obj = basic_conf_obj
        self.sigma_0 = self._basic_conf_obj.sigma_photo_z
        #----- get cosmology 
        self.csm = cosmology.cosmology(basic_conf_obj = basic_conf_obj)
        
        # parameters for biining
        self.zmin = self._basic_conf_obj.z_min
        self.zmax = self._basic_conf_obj.z_max
        self.zbins_nr = self._basic_conf_obj.N_bins
        self.zbins_chi = self.Chi_bin_boundaries(self.zmin, self.zmax, self.zbins_nr)
        self.zbins_chicentral = self.Chi_bin_centers(self.zmin, self.zmax, self.zbins_nr)
        self.zbins_z = self.csm.z_from_chi(self.zbins_chi)
        self.zbins_zcentral = self.csm.z_from_chi(self.zbins_chicentral)
        self.deltachi = self.zbins_chi[1]-self.zbins_chi[0]
        
        #holders
        self.windows = {}
        self.windows_max = {}

        
        
    def Chi_bin_boundaries(self, z_min, z_max, N) :
        Chi_min = self.csm.chi_from_z(z_min)
        Chi_max = self.csm.chi_from_z(z_max)
        Chi_boundaries = np.linspace(Chi_min, Chi_max, N+1)    
        return Chi_boundaries
    
    #Get comoving distances at center of of bins from Chi_bin_boundaries()
        
    def Chi_bin_centers(self, z_min, z_max, N) :
        Chi_boundaries = self.Chi_bin_boundaries(z_min, z_max, N)
        Chis = ( Chi_boundaries[:-1] + Chi_boundaries[1:] ) / 2.0
        return Chis
      
        
    def photoz_prob(self,zs,z_a,z_b):
        
        def Int(zp,zr):   
            return np.exp(-(zp-zr)**2/2.0/(self.sigma_0*(1.0+zr))**2)
            
        zp_1 = np.logspace(np.log10(z_a),np.log10(z_b),3000)
        H = self.csm.H_z(zp_1)[:,np.newaxis]
        zp_2 = np.logspace(np.log10(0.001),np.log10(self.zmax+2),6000)
        
        I1 = np.trapz(Int(zp_1[:,None],zs[None,:])/H ,zp_1, axis = 0 )
        I2 = np.trapz(Int(zp_2[:,None],zs[None,:])   ,zp_2, axis = 0 )
        
        
        return I1/I2
        
        
    def get_window(self,i, dndz=None):
        
        if str(i) in self.windows :
            return self.windows[str(i)]
        else:
            
            z_a = self.zbins_z[i]
            z_b = self.zbins_z[i+1]            
            chis_int = np.linspace(0,self.csm.chi_from_z(self.zmax+1.1),1000)
            zs_int   = self.csm.z_from_chi(chis_int)
                    
            #underlying galaxy redshift distribution for this photo-z bin.
            #modelling as in Eq.(13.11) in arXiv: 0912.0201
            #n_i = (zs_int**2)*np.exp(-(zs_int/0.5))*self.photoz_prob(zs_int,z_a,z_b)
            if not dndz:
            	w = self.photoz_prob(zs_int,z_a,z_b)*self.csm.H_z(zs_int)  # Default if no custom dndz specified
            elif dndz == 'unwise':
            	if conf.unwisecolour == 'blue':
            		with open('data/unWISE/blue.txt', 'r') as FILE:
            			x = FILE.readlines()
            	elif conf.unwisecolour == 'green':
            		with open('data/unWISE/green.txt', 'r') as FILE:
            			x = FILE.readlines()
            	elif conf.unwisecolour == 'red':
            		with open('data/unWISE/red_16.2.txt', 'r') as FILE:
            			x = FILE.readlines()
            	z = np.array([float(l.split(' ')[0]) for l in x])
            	dndz = np.array([float(l.split(' ')[1]) for l in x])
            	dndz_mod = dndz / 1.0 # conf.N_bins # code uses N_bin to divide all spectra including galaxy. This will factor out and give correct spectrum.
            	w = interp1d(z,dndz_mod, kind= 'linear',bounds_error=False,fill_value=0)(zs_int)*self.csm.H_z(zs_int) 
            
            self.windows[str(i)] = interp1d(zs_int,w, kind= 'linear',bounds_error=False,fill_value=0)
            self.windows_max[str(i)] = np.max(w)
            
         
            return self.windows[str(i)]

    #### Haar matrix for our choice of normalization
    
    def haar(self,kmax,k,t):

        p=0.
        if k != 0:
            p=int(np.log2(k))
        q=k-2**p+1
        twop=2**p
        
        haarout = 0.0
        
        if (q-1)/twop <= t < (q-0.5)/twop:
            haarout = np.sqrt(twop)
        if (q-0.5)/twop <= t < q/twop:
            haarout = -np.sqrt(twop)
    
        if k==0:
            haarout = 1
        
        return haarout/np.sqrt(kmax)


    def haar_wavelet(self,kmax,k,chis):
    
        if k != 0:
            hv = np.vectorize(self.haar, excluded=['kmax','k'])
            chis_bounds = self.Chi_bin_boundaries(self.zmin, self.zmax, kmax)
            dchi = chis_bounds[1]-chis_bounds[0]
            return hv(kmax,k,(chis-chis_bounds[0])/(chis_bounds[-1]-chis_bounds[0]))/np.sqrt(dchi)
        else:
            chis_bounds = self.Chi_bin_boundaries(self.zmin, self.zmax, kmax)
            dchi = chis_bounds[1]-chis_bounds[0]
            theta_b  = np.where( chis <= chis_bounds[0],0,1)*np.where( chis > chis_bounds[-1] , 0,1)
            return theta_b/ np.sqrt(dchi*kmax)
    
    def bin2haar_brute(self,kmax):
        
        chis_int = np.linspace(self.csm.chi_from_z(1e-2), self.csm.chi_from_z(self.zmax+1), 6000)
        chis_bounds = self.Chi_bin_boundaries(self.zmin, self.zmax, kmax)
        H = np.zeros((kmax,kmax))
        
        for k in np.arange(kmax):
            for i in np.arange(kmax):
                theta_i  = np.where( chis_int <= chis_bounds[i],0,1)*np.where( chis_int >= chis_bounds[i+1],0,1)
                H[k,i] = np.trapz(self.haar_wavelet(kmax,k,chis_int)*theta_i,chis_int)
                
        return H
    
    def bin2haar(self, kmax):

        haarmatrixout = np.zeros((kmax,kmax))
        
        chis_bounds = self.Chi_bin_boundaries(self.zmin, self.zmax, kmax)
        dchi = chis_bounds[1]-chis_bounds[0]
    
        for i in range(kmax):
            for j in range(kmax):
                haarmatrixout[i,j] = self.haar(kmax,i,j/kmax)
    
        return haarmatrixout*np.sqrt(dchi)
    
    def haar2bin(self, kmax):
    
        chis_bounds = self.Chi_bin_boundaries(self.zmin, self.zmax, kmax)
        dchi = chis_bounds[1]-chis_bounds[0]
        return np.transpose(self.bin2haar(kmax)/dchi)
        
        
    def binmatrix(self,nbinfine,nbincoarse):

        # By convention we have nfinebin = 2^n = 2,4,8,16,32,64 etc and so 
        # nbinfine must be ncoarsebin <= nfinebin and equal to 2,4,8,etc. 

        binmatrix = np.zeros((nbincoarse,nbinfine))

        len = nbinfine/nbincoarse

        for i in range(nbincoarse):
            for j in range(nbinfine):
                if i*len <= j < (i+1)*len:
                    binmatrix[i,j] = 1

        return binmatrix/len
    
    def coarse_matrix(self,N_fine,N_coarse, M):
        
        Window = self.binmatrix(N_fine,N_coarse)
            
        return np.dot(np.dot(Window,M),np.transpose(Window))

    def coarse_vector(self,N_fine,N_coarse, M):

        Window = self.binmatrix(N_fine,N_coarse)

        return np.dot(Window,M)