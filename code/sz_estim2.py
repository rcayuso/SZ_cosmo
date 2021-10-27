#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:56:42 2020

@author: jcayuso
"""

import numpy as np
from scipy.interpolate import interp1d
import time
import redshifts
from math import lgamma
import loginterp
import common as c
import copy
from scipy.linalg import cholesky
from scipy import linalg
#import healpy


class estimator(object):
    
    #####################################################################
    ############ CLASS SET UP AND CL TOOLS
    #####################################################################
    
    def __init__(self, data_lmax = None, hashstr = None, conf_module = None) :
        
        if hashstr != None :
            self.basic_conf_dir = c.load_basic_conf(hashstr)
            self.conf = c.dict_to_obj(self.basic_conf_dir)
        elif conf_module != None:
            self.conf = conf_module
            self.basic_conf_dir = c.get_basic_conf(self.conf)
        else:
            raise Exception ("You have to provide a hash string or configuration module to locate precomputed data") 
        
        if data_lmax == None:
            raise Exception ("Please provide data_lmax (This is not necessarily equal to the estimator lmax)") 
        else:
            self.data_lmax = data_lmax
                    
        self.Cls = {}
        self.sims = {}
        self.haar = {}
        self.cs = {}
        
        
        self.zb = redshifts.binning(basic_conf_obj = self.conf)
        self.csm = self.zb.csm        
        self.deltachi = self.zb.deltachi
        self.nbin = self.conf.N_bins
        
        self.N_fine_modes = self.conf.N_bins  
        self.lss = 'g'
        
        self.realnum = 20
    
        print("Default lss = 'g' . Modify with set_lss method.")
        print('Default N_fine_modes = '+str(self.N_fine_modes)+'. Modify with set_Nfine method.')
        
        #Some temporary parameters for frequency dependent noise from SO
        
        self.SO_FREQS = np.array([27, 39, 93, 145, 225, 280])
        self.dTs   = np.array([71, 36, 8.0, 10, 22, 54])
        self.beams = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
        
        
    def set_Nfine(self, Nfine):    
        self.N_fine_modes= Nfine
        print('N_fine_modes reset to '+str(self.N_fine_modes))
        
    def set_lss(self, lss_string):      
        self.lss= lss_string
        print('lss reset to '+str(self.lss))
        
    def load_theory_Cl(self,tag1,tag2):   
        if c.exists(self.basic_conf_dir,'Cl_'+tag1+'_'+tag2+'_lmax='+str(self.data_lmax), dir_base = 'Cls'):
            return c.load(self.basic_conf_dir,'Cl_'+tag1+'_'+tag2+'_lmax='+str(self.data_lmax), dir_base = 'Cls')
        else:
            C = c.load(self.basic_conf_dir,'Cl_'+tag2+'_'+tag1+'_lmax='+str(self.data_lmax), dir_base = 'Cls')          
            return np.transpose(C, axes =[0,2,1])

    def load_L(self,):
        return c.load(self.basic_conf_dir,'L_sample_lmax='+str(self.data_lmax), dir_base = 'Cls')
    
    def set_theory_Cls(self, add_ksz = True, add_ml = True, use_cleaned = False, frequency = None, get_haar = False):

        #frequency is only used if use_cleaned = False. If None, you have primary CMB + a simple gaussian white noise with beam. If you
        #use a frequency, at the moment it should be a SO one. 
        
        self.Cls['lss-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl(self.lss,self.lss),self.load_L())
        self.Cls['taud-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('taud',self.lss),self.load_L())
        self.Cls['taud-taud'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('taud','taud'),self.load_L())
        self.Cls['v-v'] =  loginterp.log_interpolate_matrix(self.load_theory_Cl('v','v'), self.load_L())
        self.Cls['taud-v'] =  loginterp.log_interpolate_matrix(self.load_theory_Cl('taud','v'), self.load_L())
        self.Cls['vt-vt'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('vt', 'vt'), self.load_L())
        self.Cls['ml-ml'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('ml', 'ml'), self.load_L())
        #self.Cls['ml-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('ml', self.lss), self.load_L())
        #self.Cls['lensing-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('lensing',self.lss), self.load_L())
        
        self.Cls['pCMB-pCMB'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('pCMB','pCMB'), c.load(self.basic_conf_dir,'L_pCMB_lmax='+str(self.data_lmax), dir_base = 'Cls'))
        self.Cls['lensing-lensing'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('lensing','lensing'), self.load_L())
        self.Cls['kSZ-kSZ'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('kSZ','Nfine_'+str(self.N_fine_modes)),self.load_L())
        self.Cls['ML-ML'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('ML','Nfine_'+str(self.N_fine_modes)),self.load_L())
                
        if use_cleaned == True:       
            print("Using clean TT")       
            self.Ttag = 'Tc'
            self.Cls['T-T']   =  loginterp.log_interpolate_matrix(self.load_theory_Cl('Tc','Tc'),self.load_L())
            self.Cls['T-lss'] =  loginterp.log_interpolate_matrix(self.load_theory_Cl('Tc',self.lss),self.load_L())
            self.Cls['lss-T'] =  np.transpose(self.Cls['T-lss'], axes =[0,2,1])  
            
        else:  

            if frequency is None:   
            
                self.Ttag = 'T0'
                self.beam = self.conf.beamArcmin_T*np.pi/180./60.
                self.dT   = self.conf.noiseTuKArcmin_T*np.pi/180./60./self.conf.T_CMB

                self.Cls['T-T'] = self.Cls['pCMB-pCMB']\
                                    +self.CMB_noise(np.arange(self.data_lmax+1),self.beam, self.dT)          
                self.Cls['T-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('isw_rs',self.lss), self.load_L())
                self.Cls['lss-T'] = loginterp.log_interpolate_matrix(self.load_theory_Cl(self.lss,'isw_rs'), self.load_L())        
            
            else:
                try:
                    idx = np.where(self.SO_FREQS ==frequency)[0][0]
                except:
                    raise Exception("Please use one of the S0 frequencies")
                
               
                self.Ttag = 'T('+str(frequency)+')'
                self.beam = self.beams[idx]*np.pi/180./60.
                self.dT   = self.dTs[idx]*np.pi/180./60./self.conf.T_CMB

                self.Cls['T-T'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('T('+str(frequency)+')','T('+str(frequency)+')'),self.load_L())\
                                    +self.CMB_noise(np.arange(self.data_lmax+1),self.beam, self.dT)          
                self.Cls['T-lss'] = loginterp.log_interpolate_matrix(self.load_theory_Cl('T('+str(frequency)+')',self.lss), self.load_L())
                self.Cls['lss-T'] = loginterp.log_interpolate_matrix(self.load_theory_Cl(self.lss,'T('+str(frequency)+')'), self.load_L())
            

        if add_ksz:
            self.Cls['T-T'] += self.Cls['kSZ-kSZ']
        if add_ml:
            self.Cls['T-T'] += self.Cls['ML-ML']
            
        if get_haar:
            
            print("Getting fine mode spectra")
            
            kmax =  self.N_fine_modes    
            H = self.get_haar(kmax)
            Bc = self.zb.binmatrix(self.N_fine_modes, self.conf.N_bins)
            
            basic_conf_high = copy.deepcopy(self.basic_conf_dir)
            basic_conf_high['N_bins'] = self.N_fine_modes
            
            Ctaudlss_sample = c.load(basic_conf_high,'Cl_taud_'+self.lss+'_lmax='+str(self.data_lmax), dir_base = 'Cls')  
            #Cmllss_sample   = c.load(basic_conf_high,'Cl_ml_'+self.lss+'_lmax='+str(self.data_lmax), dir_base = 'Cls')  
            
            Chaartaudlss_sample = np.swapaxes(np.dot(np.dot(H,Ctaudlss_sample), np.transpose(Bc)),0,1)
            #Chaarmllss_sample   = np.swapaxes(np.dot(np.dot(H,Cmllss_sample), np.transpose(Bc)),0,1)
            
            
            self.Cls['haartaud-'+self.lss] = Chaartaudlss_sample
            #self.Cls['haarml-'+self.lss]   = Chaarmllss_sample
        

            idx_cut = np.where(self.load_L()  <100)[0][-1]    
            Cvvfine =  loginterp.log_interpolate_matrix(c.load(basic_conf_high,'Cl_v_v_lmax='+str(self.data_lmax), dir_base = 'Cls')[:idx_cut,:,:] ,self.load_L()[:idx_cut])
            #Cvtvtfine =  loginterp.log_interpolate_matrix(c.load(basic_conf_high,'Cl_vt_vt_lmax='+str(self.data_lmax), dir_base = 'Cls')[:idx_cut,:,:] ,self.load_L()[:idx_cut])
            self.Cls['v_fine-v_fine'] = Cvvfine
            #self.Cls['vt_fine-vt_fine'] = Cvtvtfine
            
        

    def set_Cl(self, tag1, tag2, CL, extra =''): 
        self.data_lmax = CL.shape[0]-1
        if CL.shape[0] != self.data_lmax+1:
            raise Exception (" Input CL has to have first axis of length lmax+1")       
        self.Cls[tag1+'-'+tag2+'-'+extra] = CL[self.load_L()]
        
    def load_tau(self,):
        return c.load(self.basic_conf_dir,'tau_binned', dir_base = '')
    
    
    
    #####################################################################
    ############ NOISE AND BIAS CALCULATION
    #####################################################################   
    
    def get_haar(self,kmax):
        
        if 'bin2haar' in self.haar:
            return self.haar['bin2haar']
        else:
            H = self.zb.bin2haar(kmax)
            self.haar['bin2haar'] = H
            return self.haar['bin2haar']

    def wigner_symbol(self, ell, ell_1,ell_2):
         
        if not ((np.abs(ell_1-ell_2) <= ell) and (ell <= ell_1+ell_2)):  
            return 0 
     
        J = ell +ell_1 +ell_2
        if J % 2 != 0:
            return 0
        else:
            g = int(J/2)*1.0
            w = (-1)**(g)*np.exp((lgamma(2.0*g-2.0*ell+1.0)+lgamma(2.0*g-2.0*ell_1+1.0)+lgamma(2.0*g-2.0*ell_2+1.0)\
                                  -lgamma(2.0*g+1.0+1.0))/2.0 +lgamma(g+1.0)-lgamma(g-ell+1.0)-lgamma(g-ell_1+1.0)-lgamma(g-ell_2+1.0))
            
            return w
    
    def CMB_noise(self,Ls, BEAM, DT):
        Nl_CMB_T = np.zeros((len(Ls),1,1))
        with np.errstate(over='ignore'):
            Nl_CMB_T[:,0,0] = (DT**2.)*np.exp(Ls*(Ls+1.)*(BEAM**2.)/8./np.log(2.))

        return Nl_CMB_T
    
    def Cl_cal(self,Ls):
        ls = np.arange(self.data_lmax)
        Ncal_0 = np.exp(-(ls/10)**2)
        A = self.conf.sigma_cal/(np.sum((2.0*ls+np.ones(len(ls)))/(4*np.pi)*Ncal_0))
        Ncal = A*np.exp(-(ls/10)**2)
            
        return Ncal[Ls]
    
       
    def f(self, tag, alpha, gamma, l, l1, l2, Ae = None):
        
        if tag == 'v':
            
            if Ae is None:
                C = self.Cls['taud-lss'][:,gamma,alpha]
            else:
                if 'taud-lss-'+str(Ae) in self.Cls:
                    C = self.Cls['taud-lss-'+str(Ae)][:,gamma,alpha]
                else:
                    basic_conf_Ae= copy.deepcopy(self.basic_conf_dir)
                    basic_conf_Ae['A_electron'] = Ae
                    Ctaudlss_sample = c.load(basic_conf_Ae,'Cl_taud_'+self.lss+'_lmax='+str(self.data_lmax), dir_base = 'Cls')  
                    self.Cls['taud-lss-'+str(Ae)] = loginterp.log_interpolate_matrix(Ctaudlss_sample,self.load_L())
                    C = self.Cls['taud-lss-'+str(Ae)][:,gamma,alpha]

            factor = np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*self.zb.deltachi*C[l2]*self.wigner_symbol(l, l1, l2)
            
        elif tag == 'v_fine':
            C = loginterp.log_interpolate(self.Cls['haartaud-'+self.lss][:,gamma,alpha],self.load_L())
            factor = np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*C[l2]*self.wigner_symbol(l, l1, l2)
            
        elif tag == 'vt':
            C = self.Cls['ml-lss'][:, gamma, alpha]
            factor = 0.5 * (l * (l + 1.0) + l2 * (l2 + 1.0) - l1 * (l1 + 1.0)) * np.sqrt((2 * l + 1.0) * (2 * l1 + 1.0) * (2 * l2 + 1.0) / 4.0 / np.pi)*self.zb.deltachi * C[l2] * self.wigner_symbol(l, l1, l2)
           
        elif tag == 'vt_fine':
                        
            C = loginterp.log_interpolate(self.Cls['haarml-'+self.lss][:,gamma,alpha],self.load_L())
            factor = 0.5 * (l * (l + 1.0) + l2 * (l2 + 1.0) - l1 * (l1 + 1.0)) * np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*C[l2]*self.wigner_symbol(l, l1, l2)
               
        elif tag =='cal':
            C = self.Cls['T-lss'][:,0,alpha]
            factor = np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*C[l2]*self.wigner_symbol(l, l1, l2)
        elif tag =='pCMB':
            C = self.Cls['lensing-lss'][:,0,alpha]
            factor =  0.5 * (l * (l + 1.0) + l2 * (l2 + 1.0) - l1 * (l1 + 1.0)) *np.sqrt((2*l+1.0)*(2*l1+1.0)*(2*l2+1.0)/4.0/np.pi)*C[l2]*self.wigner_symbol(l, l1, l2)
        
        else:
            raise Exception("Weight f not supported for "+tag)
        
        return factor


    def g(self, tag, alpha, l, l1, l2, dndz = None, Ae = None):
        
        if dndz == None:
        
            CTT     = self.Cls['T-T'][:,0,0]
            CTlss   = self.Cls['T-lss'][:,0,alpha]    
            Clsslss = self.Cls['lss-lss'][:,alpha,alpha]
    
            G =  (CTT[l2]*Clsslss[l1]*self.f(tag, alpha, alpha, l, l1, l2, Ae = Ae)-((-1)**(l+l1+l2))*CTlss[l1]*CTlss[l2]*self.f(tag, alpha, alpha, l, l2, l1, Ae = Ae))\
             /(CTT[l1]*CTT[l2]*Clsslss[l1]*Clsslss[l2] - (CTlss[l1]**2)*(CTlss[l2]**2))
                        
            return G
        
        elif dndz == 'unwise':
        
            CTT     = self.Cls['T-T'][:,0,0]
            CTlss   = self.Cls['T-lss'][:,0,0]    
            Clsslss = self.Cls['lss-lss'][:,0,0]
    
            G =  (CTT[l2]*Clsslss[l1]*self.f(tag, 0, alpha, l, l1, l2, Ae = Ae)-((-1)**(l+l1+l2))*CTlss[l1]*CTlss[l2]*self.f(tag, 0, alpha, l, l2, l1, Ae = Ae))\
             /(CTT[l1]*CTT[l2]*Clsslss[l1]*Clsslss[l2] - (CTlss[l1]**2)*(CTlss[l2]**2))
                        
            return G
        
        else:
            raise Exception("dndz not valid")
    
    def cs1_alpha_gamma(self, lmax, tag_g, tag_f, alpha, gamma, ell, dndz = None, Ae = None): 
        
        if str(lmax)+'-'+tag_g+'-'+tag_f+'-'+str(alpha)+'-'+str(gamma)+'-'+str(ell)+'-'+str(Ae) in self.cs:
            return self.cs[str(lmax)+'-'+tag_g+'-'+tag_f+'-'+str(alpha)+'-'+str(gamma)+'-'+str(ell)+'-'+str(Ae) ]
        else:
        
            L = np.unique(np.append(np.geomspace(2,lmax,300).astype(int),lmax))
            
            #First, let's avoid calculating sums that are identically zero
            
            if tag_f == 'v_fine':
                Lnz =np.where(loginterp.log_interpolate(self.Cls['haartaud-'+self.lss][:,gamma,alpha],self.load_L()) != 0.0)
            else:
                Lnz = L
    
            L_int = L[np.in1d(L, Lnz)]
            
            if len(L_int) == 0:
                c = 0
            else:
                a = []
                
                for l1_id, ell_2 in enumerate(L_int):    
                    
                    terms = 0
                    
                    for ell_1 in np.arange(np.abs(ell_2-ell),ell_2+ell+1):
                        if ell_1 > lmax or ell_1 <2:   #triangle rule
                            continue
                        
                        if dndz == None:
                            terms += self.f(tag_f, alpha, gamma, ell, ell_1, ell_2, Ae = Ae)*self.g(tag_g, alpha, ell, ell_1, ell_2, Ae = Ae)
                        elif dndz == 'unwise':
                            terms += self.f(tag_f, 0, gamma, ell, ell_1, ell_2, Ae = Ae)*self.g(tag_g, alpha, ell, ell_1, ell_2, dndz = 'unwise', Ae = Ae)
                        else:
                            raise Exception("dndz not valid")
                      
                    a.append(terms)
                
                if len(L_int) == 1:
                    c = np.asarray(a)[0]
                elif (len(L_int) == 2 or len(L_int) == 3):
                    I = interp1d(L_int ,np.asarray(a), kind = 'linear',bounds_error=False,fill_value=0)(np.arange(lmax+1))
                    c =   np.sum(I)
                else:
                    #Ignore last couple ell cause they can be problematic, regardless of lmax
                    I = interp1d(L_int[:-2] ,np.asarray(a)[:-2], kind = 'linear',bounds_error=False,fill_value=0)(np.arange(lmax+1))
                    c =   np.sum(I)
                    
            
            self.cs[str(lmax)+'-'+tag_g+'-'+tag_f+'-'+str(alpha)+'-'+str(gamma)+'-'+str(ell)+'-'+str(Ae) ] = c
        
        return c
            
        
    
    def cs2_alpha_gamma(self, lmax, tag_g, alpha, gamma, ell, dndz = None, Ae = None):    # This is used for the noise that comes from non-statistically asymmetric contributions
       
        L = np.unique(np.append(np.geomspace(2,lmax,300).astype(int),lmax))   
        a = []
        
    
        if dndz == None:
    
            Clsslss_alpha_gamma  = self.Cls['lss-lss'][:,alpha,gamma]
            CTT                  = self.Cls['T-T'][:,0,0]
            CTlss_alpha          = self.Cls['T-lss'][:,0,alpha]
            CTlss_gamma          = self.Cls['T-lss'][:,0,gamma]             
    
            for l1_id, ell_1 in enumerate(L):   
                
                terms = 0
                
                for ell_2 in np.arange(np.abs(ell_1-ell),ell_1+ell+1):
                    if ell_2 > lmax or ell_2 <2:   #triangle rule
                        continue   
                    g_alpha = self.g(tag_g, alpha, ell, ell_1, ell_2, Ae = Ae)
                    g_gamma_1 = self.g(tag_g, gamma, ell, ell_1, ell_2, Ae = Ae)
                    g_gamma_2 = self.g(tag_g, gamma, ell, ell_2, ell_1, Ae = Ae)
                    terms  += g_alpha*(g_gamma_1*CTT[ell_1]*Clsslss_alpha_gamma[ell_2]+((-1)**(ell+ell_1+ell_2))*g_gamma_2*CTlss_alpha[ell_2]*CTlss_gamma[ell_1] )
                
                a.append(terms)
        
        elif dndz == 'unwise':
            
            Clsslss_alpha_gamma  = self.Cls['lss-lss'][:,0,0]
            CTT                  = self.Cls['T-T'][:,0,0]
            CTlss_alpha          = self.Cls['T-lss'][:,0,0]
            CTlss_gamma          = self.Cls['T-lss'][:,0,0]             
    
            for l1_id, ell_1 in enumerate(L):   
                
                terms = 0
                
                for ell_2 in np.arange(np.abs(ell_1-ell),ell_1+ell+1):
                    if ell_2 > lmax or ell_2 <2:   #triangle rule
                        continue   
                    g_alpha = self.g(tag_g, alpha, ell, ell_1, ell_2, dndz = dndz, Ae = Ae)
                    g_gamma_1 = self.g(tag_g, gamma, ell, ell_1, ell_2, dndz = dndz, Ae = Ae)
                    g_gamma_2 = self.g(tag_g, gamma, ell, ell_2, ell_1, dndz = dndz, Ae = Ae)
                    terms  += g_alpha*(g_gamma_1*CTT[ell_1]*Clsslss_alpha_gamma[ell_2]+((-1)**(ell+ell_1+ell_2))*g_gamma_2*CTlss_alpha[ell_2]*CTlss_gamma[ell_1] )
                
                a.append(terms)
                
        else:
            raise Exception("dndz not valid")
                        
                    
        I = interp1d(L[:-2] ,np.asarray(a)[:-2], kind = 'linear',bounds_error=False,fill_value=0)(np.arange(lmax+1))
        
        
        return np.sum(I) 
    
    
    def Noise_iso(self, lmax, tag_g, alpha, gamma, ell, dndz = None, Ae = None):  
        Nalpha  = (2*ell+1)/self.cs1_alpha_gamma(lmax, tag_g, tag_g, alpha, alpha, ell, dndz = dndz, Ae = Ae)
        Ngamma  = (2*ell+1)/self.cs1_alpha_gamma(lmax, tag_g, tag_g, gamma, gamma, ell, dndz = dndz, Ae = Ae)
        return self.cs2_alpha_gamma(lmax,tag_g, alpha, gamma, ell, dndz = dndz, Ae = Ae)*Nalpha*Ngamma/(2*ell+1)
    
    def R(self, lmax, tag_g, tag_f, alpha,gamma,ell, dndz = None, Ae = None):   
        
        if tag_f in ['v_fine','vt_fine']:
            if gamma < self.nbin:
                return 0

        num = self.cs1_alpha_gamma(lmax, tag_g , tag_f, alpha, gamma, ell, dndz = dndz, Ae = Ae)
        den = self.cs1_alpha_gamma(lmax, tag_g , tag_g, alpha, alpha, ell, dndz = dndz, Ae = Ae)       
        return  num/den
    
    def Gamma(self, lmax, tag_g, tag_f, alpha,gamma,ell, dndz = None, Ae = None):   
        
        if tag_f in ['v_fine','vt_fine']:
            if gamma < self.nbin:
                return 0
        
        A = self.cs1_alpha_gamma(lmax, tag_g , tag_f, alpha, alpha, ell, dndz = dndz)
        num = self.cs1_alpha_gamma(lmax, tag_g , tag_f, alpha, gamma, ell, dndz = dndz)
        den = self.cs1_alpha_gamma(lmax, tag_g , tag_g, alpha, gamma, ell, dndz = dndz, Ae = Ae)    
        
        if num < A*1e-5:
            return 0
        else:
            return  num/den

    
    def Noise_a_from_b(self, lmax, tag_g, tag_f1, tag_f2, alpha, gamma, ell, dndz = None):
        
        if tag_g in ['v_fine','vt_fine']:
            raise Exception ('tag not supported as primary tag. Select est_tag among v or vt') #Update this message when new estimators are considered
            
        if tag_f1 not in ['v_fine','vt_fine'] and tag_f2 not in ['v_fine','vt_fine']:
            
            if tag_f1 == 'cal' or tag_f2 == 'cal':
                if tag_f1 != tag_f2:
                    raise Exception ('Calibration bias only supported when b1_tag = b2_tag = cal')
                else:
                    C = self.Cl_cal(ell)*np.ones((1,1))
            else:      
                if tag_f1 == 'pCMB':
                    C = self.Cls['pCMB-pCMB'][ell]
                else:
                    C = loginterp.log_interpolate_matrix(self.load_theory_Cl(tag_f1, tag_f2), self.load_L())[ell]
                
            N1,N2 = C.shape 
            Noise = 0
            R1 = np.zeros(N1)
            R2 = np.zeros(N2)
            
            for i in np.arange(N1):
                R1[i] = self.R(lmax, tag_g, tag_f1, alpha,i,ell, dndz = dndz)
            for j in np.arange(N1):
                R2[j] = self.R(lmax, tag_g, tag_f2, gamma,j,ell, dndz = dndz)
                    
            Noise = np.dot(np.dot(R1,C),R2)
        
            return Noise
        
        else:
                         
            C = self.Cls[tag_f1+'-'+tag_f2][ell,:,:]
                
            N1,N2 = C.shape    
            Noise = 0
            R1 = np.zeros(N1)
            R2 = np.zeros(N2)
            
            for i in np.arange(N1):
                R1[i] = self.R(lmax, tag_g, tag_f1, alpha,i,ell, dndz = dndz)
            for j in np.arange(N1):
                R2[j] = self.R(lmax, tag_g, tag_f2, gamma,j,ell, dndz = dndz)
            
            H = self.zb.bin2haar(self.N_fine_modes)
            
            Noise = np.dot(np.dot(   np.dot(R1,H)  ,C),   np.dot(R2,H)  )
                    
            return Noise
        

    #####################################################################
    ############ NOISE wrapper functions
    #####################################################################
            
    def Noise_iso_alpha(self,lmax ,est_tag, alpha,L, dndz = None):       
        N = np.zeros(len(L))
        for lid, l in enumerate(L):
            N[lid] = self.Noise_iso(lmax, est_tag, alpha, alpha, l, dndz = dndz)
        return N
    
    def Noise_iso_ell(self,lmax ,est_tag, ell, dndz = None):       
        N = np.zeros((self.nbin,self.nbin))
        for alpha in np.arange(self.nbin):
            for gamma in np.arange(self.nbin):
                N[alpha,gamma] = self.Noise_iso(lmax, est_tag, alpha, gamma, ell, dndz = dndz)
        return N

    
    def R_ell(self,lmax ,est_tag, bias_tag, ell, dndz = None):       
        R = np.zeros((self.nbin,self.nbin))
        for alpha in np.arange(self.nbin):
            for gamma in np.arange(self.nbin):
                R[alpha,gamma] =  self.R(lmax, est_tag, bias_tag,alpha,gamma,ell, dndz = dndz)
        return R
           
    def Noise_a_from_b_alpha(self,lmax, est_tag, b1_tag, b2_tag, alpha, L, dndz = None):       
        N = np.zeros(len(L))
        for lid, l in enumerate(L):
            N[lid] = self.Noise_a_from_b(lmax, est_tag, b1_tag, b2_tag, alpha, alpha, l, dndz = dndz)
        return N
    
    def Noise_a_from_b_ell(self,lmax, tag_g, tag_f1, tag_f2, ell, dndz = None):       
        N = np.zeros((self.nbin,self.nbin))
        for alpha in np.arange(self.nbin):
            for gamma in np.arange(self.nbin):
                N[alpha,gamma] = self.Noise_a_from_b(lmax, tag_g, tag_f1, tag_f2, alpha, gamma, ell, dndz = dndz)
        return N
    
    def Noise_a_from_b_matrix(self,lmax, tag_g, tag_f1, tag_f2, L, dndz = None):       
        for l in L:
            N = np.zeros((self.nbin,self.nbin))
            for alpha in np.arange(self.nbin):
                for gamma in np.arange(self.nbin):
                    N[alpha,gamma] = self.Noise_a_from_b(lmax, tag_g, tag_f1, tag_f2, alpha, gamma, l, dndz = dndz)
            c.dump(self.basic_conf_dir, N,'N_'+str(tag_g)+'_'+str(tag_f1)+'_'+str(tag_f2)+'_Nfine'+str(self.N_fine_modes)+'_l='+str(l)+'_lmax='+str(lmax), dir_base = 'analysis')
            print('Done l = '+str(l))
        return 
    

    def pmode_vv(self, lmax, L, fine = True, cal = True, dndz = None):
                
        SN = np.zeros((len(L),self.nbin))
        
        #Approximation for R and Cn, practically ell independent
        R  = self.R_ell(lmax ,'v', 'v', 2, dndz = dndz)
        Cn0 = self.Noise_iso_ell(lmax ,'v', 2, dndz = dndz)
              
        for lid, ell in enumerate(L):
            
            Cn = 0
            Cn += Cn0
            
            if fine:
                Cn += c.load(self.basic_conf_dir,'N_v_v_fine_v_fine_Nfine512_l='+str(ell)+'_lmax='+str(lmax), dir_base = 'analysis')
            if cal:
                Cn += c.load(self.basic_conf_dir,'N_v_cal_cal_Nfine512_l='+str(ell)+'_lmax='+str(lmax), dir_base = 'analysis')
                                    
            #signal
            C = self.Cls['v-v'][ell]
            Cs = np.dot(np.dot(R,C),np.transpose(R))
        
            #First diagonalization
            
            w1,v1 = np.linalg.eigh(Cn)
            
            R1 = np.transpose(v1)
            R2 = np.zeros_like(Cn)
            
            for i in np.arange(self.nbin):
                R2[i,i] = 1.0/np.sqrt(w1[i])
                
            #second diagonalization
                
            R21 = np.dot(R2,R1)
            Cs_p = np.dot(np.dot(R21,Cs), np.transpose(R21))
            w3,v3 = np.linalg.eigh(Cs_p)
            R3 = np.transpose(v3)
            
            Cs_pp = np.dot(np.dot(R3,Cs_p), np.transpose(R3))
                
            for j in np.arange(self.nbin):
                SN[lid,j] = Cs_pp[j,j]
            
        return L, SN

    def pmode_vtvt(self, lmax, L, fine = True, cal = True, dndz = None):
                
        SN = np.zeros((len(L),self.nbin))
        
        for lid, ell in enumerate(L):
        
            #Approximation for R and Cn, practically ell independent
            R  = self.R_ell(lmax ,'vt', 'vt', ell, dndz = dndz)
            Cn = self.Noise_iso_ell(lmax ,'vt', ell, dndz = dndz)
            
            if fine:
                Cn += c.load(self.basic_conf_dir,'N_vt_vt_fine_vt_fine_Nfine512_l='+str(ell)+'_lmax='+str(lmax), dir_base = 'analysis')
            if cal:
                Cn += c.load(self.basic_conf_dir,'N_vt_cal_cal_Nfine512_l='+str(ell)+'_lmax='+str(lmax), dir_base = 'analysis')
                                    
            #signal
            C = self.Cls['vt-vt'][ell]
            Cs = np.dot(np.dot(R,C),np.transpose(R))
        
            #First diagonalization
            
            w1,v1 = np.linalg.eigh(Cn)
            
            R1 = np.transpose(v1)
            R2 = np.zeros_like(Cn)
            
            for i in np.arange(self.nbin):
                R2[i,i] = 1.0/np.sqrt(w1[i])
                
            #second diagonalization
                
            R21 = np.dot(R2,R1)
            Cs_p = np.dot(np.dot(R21,Cs), np.transpose(R21))
            w3,v3 = np.linalg.eigh(Cs_p)
            R3 = np.transpose(v3)
            
            Cs_pp = np.dot(np.dot(R3,Cs_p), np.transpose(R3))
                
            for j in np.arange(self.nbin):
                SN[lid,j] = Cs_pp[j,j]
            
        return L, SN

    
    def pmode_gg(self,Ns,lmax, dndz = None):
                
        SN = []
        
        for N in Ns:
            
            SN_N = 0
            
            basic_conf_N = copy.deepcopy(self.basic_conf_dir)
            basic_conf_N['N_bins'] = N
            
        
            Nshot = loginterp.log_interpolate_matrix(c.load(basic_conf_N,'Nlshot_g_g_lmax='+str(self.data_lmax), dir_base = 'Cls'),self.load_L())
            Cgg = loginterp.log_interpolate_matrix(c.load(basic_conf_N,'Cl_g_g_noshot_lmax='+str(self.data_lmax), dir_base = 'Cls'),self.load_L())
            
            for ell in np.arange(1,lmax+1):
                            
                #signal and noise for ell
                Cs = Cgg[ell,:,:]
                Cn = Nshot[ell,:,:]
            
                #First diagonalization
                
                w1,v1 = np.linalg.eigh(Cn)
                
                R1 = np.transpose(v1)
                R2 = np.zeros_like(Cn)
                
                for i in np.arange(N):
                    R2[i,i] = 1.0/np.sqrt(w1[i])
                    
                #second diagonalization
                    
                R21 = np.dot(R2,R1)
                Cs_p = np.dot(np.dot(R21,Cs), np.transpose(R21))
                w3,v3 = np.linalg.eigh(Cs_p)
                R3 = np.transpose(v3)
                #R321 = np.dot(R3,R21)
                
                Cs_pp = np.dot(np.dot(R3,Cs_p), np.transpose(R3))
                    
                for j in np.arange(N):
                    SN_N += Cs_pp[j,j]*(2*ell+1)
                    
            SN.append(SN_N)
            
            
        return (Ns,SN)
    
    

    
    #####################################################################
    ############ GAUSSIAN SIMULATIONS
    #####################################################################
    
    
    def covmat_sample_ell(self, labels, lid):
        
        ns = []
              
        for label1 in labels:          
            n1 = []       
            for label2 in labels:           
                n1.append(self.load_theory_Cl(label2,label1)[lid,:,:]) # order of labels is okay, dont change
            ns.append(np.vstack(n1))
                     
        return np.hstack(ns)
                

    def covmat_healpy(self, labels, lswitch):
        
        Cls = []
        num = len(labels)
        obs = []
        
        
        for idx in np.arange(num):
            dim_l = self.load_theory_Cl(labels[idx],labels[idx]).shape[1]
            for d in np.arange(dim_l):
                obs.append((labels[idx],d))
        
              
        for i in np.arange(len(obs)):
            for j in np.arange(i,len(obs)):
                Cls.append(loginterp.log_interpolate(self.load_theory_Cl(obs[i][0],obs[j][0])[:,obs[i][1],obs[j][1]],self.load_L())[:lswitch])
                                                     
        return Cls 
        
    
    def check_symmetric(self, a, tol=1e-8):
        return np.all(np.abs(a-a.T) < tol)
    
       
    def check_correlationmatrix(self,cl):
        
        
        for lid, l in enumerate(self.load_L(self.data_lmax)):
    
            symmetric = self.check_symmetric(cl[l,:,:])
            
            if symmetric == False:
                return print("matrix not symmetric at l="+str(l))
        
            eigenvalues = linalg.eigvals(cl[l,:,:])
        
            if np.any(eigenvalues<1e-25)==True:
                return print("matrix has negative eigenvalues at l="+str(l))
        
            cholesky(cl[l,:,:])

        return print("covmat is symmetric, is positive definite, and cholseky decomposition defined at all l")    
    
    
    def alm_maker_cholesky(self,labels,lmax):
        
        # Note, inputs are the un-interpolated spectra.
        
        # Add possibility of correlated tracers and flag to output map instead of alms.
        # Add sequential writing to file in order to handle large number of bins and nside.  

        dim_list = []
        
        for lab in labels:
            dim_list.append(self.load_theory_Cl(lab,lab).shape[1])
        tot_dim = np.sum(dim_list)
            
        start = time.time()
            
        print("cholesky")
        lsparse = self.load_L()
        lmax_in = len(lsparse)
        almsize = healpy.Alm.getsize(lmax)
        alms = np.zeros((tot_dim,almsize),dtype=complex)
        L = np.zeros((lmax_in,tot_dim,tot_dim))
            
        for lid in range(lmax_in):
            
            if lsparse[lid] ==1:
                continue                         
            L[lid,:,:] = cholesky(self.covmat_sample_ell(labels, lid), lower = True)
            
        L_out = loginterp.log_interpolate_matrix(L, lsparse)
            
        print("generating realization")
            
        for l in range(lmax):
                
            for m in range(l):
                vals = L_out[l,:,:]@np.random.normal(size=tot_dim) + (1j)*L_out[l,:,:]@np.random.normal(size=tot_dim)
                ind=healpy.Alm.getidx(lmax, l, m)
                alms[:,ind]=vals/np.sqrt(2)
    
        end = time.time()
            
        print("sims done in t="+str(end-start))
        
        results = []
        base = 0
        for i in np.arange(len(labels)):     
            results.append(alms[base:base+dim_list[i],:])
            base += dim_list[i]
            
        return results
    
    def alm_maker_healpix(self,labels,lswitch):
    
        # The number of maps nbin should be 2,4,8,16,32,64,128 etc.
        
        # Format the covariance matrix for synalm
        
        dim_list = []
        
        for lab in labels:
            dim_list.append(self.load_theory_Cl(lab,lab).shape[1])
    
        print("formatting covmat")
        
        cltot = self.covmat_healpy(labels, lswitch)
           
        print("generating realization")            
                
        synalms = healpy.synalm(cltot,lswitch,new=False)
        
        results = []
        base = 0
        for i in np.arange(len(labels)):     
            results.append(synalms[base:base+dim_list[i],:])
            base += dim_list[i]
            
        return results
    
    
    
    def alm_maker_hybrid(self, labels, lswitch, lmax):
                
        alms = self.alm_maker_cholesky(labels,lmax)
        alms_low = self.alm_maker_healpix(labels,lswitch)
        
        for l in range(0,lswitch):
            for m in range(0,l+1):
                lmind = healpy.Alm.getidx(lmax, l, m)
                lmind_low = healpy.Alm.getidx(lswitch, l, m)
                
                for idx in np.arange(len(labels)):              
                    alms[idx][:,lmind] = alms_low[idx][:,lmind_low]
                    alms[idx][:,lmind] = alms_low[idx][:,lmind_low] 
            
        return alms
    
    def get_maps_and_alms(self,labels,nside,lmax):
        
        npix = healpy.nside2npix(nside)
        dim_list = []
        map_list = []
        
        for lab in labels:
            dim_list.append(self.load_theory_Cl(lab,lab).shape[1])
            map_list.append(np.zeros((self.load_theory_Cl(lab,lab).shape[1],npix)) )
                     
        lswitch = 40
        
        alms = self.alm_maker_hybrid(labels, lswitch, lmax)
                        
        for idx in range(len(labels)):          
            for b in range(dim_list[idx]):
                map_list[idx][b,:] = healpy.alm2map(alms[idx][b,:],nside)

        return map_list, alms
        
    
    def bare_quad_est(self,nside, n_level, nbin, bin_width, Tfield, lssfield, beam_window, cllsslssbinned, ClTT, clTlssbinned, cltaudlssbinned):

        lcut = 3*nside-500
        ones =  np.ones(3*nside)
        cut = np.where(np.arange(3*nside)<lcut, 1, 1e-30)
        
        cllsslssbinned[0,:,:] = 1e15
        xizeta_n = np.zeros((nbin,healpy.nside2npix(nside)))
        xizetabar_n = np.zeros((nbin,healpy.nside2npix(nside)))    
            
        dTlm = healpy.map2alm(Tfield)
        dTlm_beamed = healpy.almxfl(dTlm,(1./beam_window)*cut)
        
        
        for i in range(nbin):
            
            Cltaudd = cltaudlssbinned[:,i,i]*bin_width
            Cldd = cllsslssbinned[:,i,i]
            ClTd = clTlssbinned[:,0,i]
    
            dlm_in = healpy.almxfl(healpy.map2alm(lssfield[i]),cut)
            
            dTlm_xi = healpy.almxfl(dTlm_beamed,np.divide(ones, ClTT, out=np.zeros_like(ones), where=ClTT!=0))
            dlm_zeta = healpy.almxfl(dlm_in, np.divide(Cltaudd, Cldd, out=np.zeros_like(Cltaudd), where=Cldd!=0))
             
            
            if n_level!=0:
                ffactor1 = ClTd**(2*n_level)
                ffactor2 = (ClTT * Cldd)**(n_level)
                filterf = np.divide(ffactor1, ffactor2, out=np.zeros_like(ffactor1), where=ffactor2!=0)
                dTlm_xi_f = healpy.almxfl(dTlm_xi,filterf)
                dlm_zeta_f = healpy.almxfl(dlm_zeta, filterf)
            else:
                dTlm_xi_f = dTlm_xi
                dlm_zeta_f = dlm_zeta
                
    
            xizeta_n[i] = healpy.alm2map(dTlm_xi_f, nside,verbose=False)*healpy.alm2map(dlm_zeta_f,nside,verbose=False)
            
            
            dTlm_xibar = healpy.almxfl(dTlm_beamed, np.divide(Cltaudd, ClTd, out=np.zeros_like(Cltaudd), where=ClTd!=0) )  
            dlm_zetabar = healpy.almxfl(dlm_in, np.divide(ones, ClTd, out=np.zeros_like(ones), where=ClTd!=0) )
            
            
            ffactor1 = ClTd**(2*n_level+2)
            ffactor2 = (ClTT * Cldd)**(n_level+1)
            filterf = np.divide(ffactor1, ffactor2, out=np.zeros_like(ffactor1), where=ffactor2!=0)

            dTlm_xibar_f = healpy.almxfl(dTlm_xibar, filterf)
            dlm_zetabar_f = healpy.almxfl(dlm_zetabar, filterf)
            
            
            xizetabar_n[i] = healpy.alm2map(dTlm_xibar_f, nside,verbose=False)*healpy.alm2map(dlm_zetabar_f,nside,verbose=False)
            
        return xizeta_n, xizetabar_n
    
    def bare_quad_est_ml(self,nside, n_level, nbin, bin_width, Tfield, lssfield, beam_window, 
                         cllsslssbinned, ClTT, clTlssbinned, clmllssbinned,realization):

        ones =  np.ones(3*nside)
        
        cllsslssbinned[0,:,:] = 1e15
        
        xizeta_n = np.zeros((nbin,healpy.nside2npix(nside)))
        xizetabar_n = np.zeros((nbin,healpy.nside2npix(nside)))
            
        xizetal_n = np.zeros((nbin,healpy.nside2npix(nside)))
        xilzeta_n = np.zeros((nbin,healpy.nside2npix(nside)))

        xizetalbar_n = np.zeros((nbin,healpy.nside2npix(nside)))
        xilzetabar_n = np.zeros((nbin,healpy.nside2npix(nside)))

        dTlm = healpy.map2alm(Tfield)
        dTlm_beamed = healpy.almxfl(dTlm,1./beam_window)
        
        lvec = np.arange(0,3*nside)
        lfac = lvec*(1.+lvec)

        for i in range(nbin):
            
            Clmld = clmllssbinned[:,i,i]
            Cldd = cllsslssbinned[:,i,i]
            ClTd = clTlssbinned[:,0,i]
    
            dlm_in = healpy.map2alm(lssfield[i])
            
            dTlm_xi = healpy.almxfl(dTlm_beamed,np.divide(ones, ClTT, 
                                                          out=np.zeros_like(ones), where=ClTT!=0))
            dlm_zeta = healpy.almxfl(dlm_in, np.divide(Clmld, Cldd,
                                                       out=np.zeros_like(Clmld), where=Cldd!=0))
             
            dTlm_xi_l = healpy.almxfl(dTlm_beamed,np.divide(lfac, ClTT, 
                                                            out=np.zeros_like(ones), where=ClTT!=0))
            dlm_zeta_l = healpy.almxfl(dlm_in, np.divide(lfac*Clmld, Cldd, 
                                                         out=np.zeros_like(Clmld), where=Cldd!=0))

            if n_level!=0:
                ffactor1 = ClTd**(2*n_level)
                ffactor2 = (ClTT * Cldd)**(n_level)
                filterf = np.divide(ffactor1, ffactor2, out=np.zeros_like(ffactor1), where=ffactor2!=0)
                
                dTlm_xi_f = healpy.almxfl(dTlm_xi,filterf)
                dlm_zeta_f = healpy.almxfl(dlm_zeta, filterf)
                
                dTlm_xi_f_l = healpy.almxfl(dTlm_xi_l,filterf)
                dlm_zeta_f_l = healpy.almxfl(dlm_zeta_l, filterf)

            else:
                dTlm_xi_f = dTlm_xi
                dlm_zeta_f = dlm_zeta
                    
                dTlm_xi_f_l = dTlm_xi_l
                dlm_zeta_f_l = dlm_zeta_l

            xi_f = healpy.alm2map(dTlm_xi_f, nside,verbose=False)
            zeta_f = healpy.alm2map(dlm_zeta_f,nside,verbose=False)
            xi_f_l = healpy.alm2map(dTlm_xi_f_l, nside,verbose=False)
            zeta_f_l = healpy.alm2map(dlm_zeta_f_l,nside,verbose=False)
            
            xizeta_n[i]  = xi_f*zeta_f
            xizetal_n[i] = xi_f*zeta_f_l
            xilzeta_n[i] = xi_f_l*zeta_f
            
            ## now the barred quantites

            dTlm_xibar = healpy.almxfl(dTlm_beamed, np.divide(Clmld, ClTd, 
                                                              out=np.zeros_like(Clmld), where=ClTd!=0) )
            dlm_zetabar = healpy.almxfl(dlm_in, np.divide(ones, ClTd, 
                                                          out=np.zeros_like(ones), where=ClTd!=0) )

            dTlm_xibar_l = healpy.almxfl(dTlm_beamed, np.divide(lfac*Clmld, ClTd, 
                                                                out=np.zeros_like(Clmld), where=ClTd!=0) )
            dlm_zetabar_l = healpy.almxfl(dlm_in, np.divide(lfac, ClTd, 
                                                            out=np.zeros_like(ones), where=ClTd!=0) )
            
            ffactor1 = ClTd**(2*n_level+2)
            ffactor2 = (ClTT * Cldd)**(n_level+1)
            filterf = np.divide(ffactor1, ffactor2, out=np.zeros_like(ffactor1), where=ffactor2!=0)

            dTlm_xibar_f = healpy.almxfl(dTlm_xibar, filterf)
            dlm_zetabar_f = healpy.almxfl(dlm_zetabar, filterf)
            
            dTlm_xibar_f_l = healpy.almxfl(dTlm_xibar_l, filterf)
            dlm_zetabar_f_l = healpy.almxfl(dlm_zetabar_l, filterf)

            xizetabar_n[i] = (healpy.alm2map(dTlm_xibar_f, nside,verbose=False)
                              *healpy.alm2map(dlm_zetabar_f,nside,verbose=False))
            
            xilzetabar_n[i] = (healpy.alm2map(dTlm_xibar_f_l, nside,verbose=False)
                               *healpy.alm2map(dlm_zetabar_f,  nside,verbose=False))
            xizetalbar_n[i] = (healpy.alm2map(dTlm_xibar_f,   nside,verbose=False)
                               *healpy.alm2map(dlm_zetabar_f_l,nside,verbose=False))
            
        
        return xizeta_n-xizetabar_n, (xizetal_n-xilzeta_n
                                      -xilzetabar_n+xizetalbar_n)
    

    
    def reconstruct(self, nside, nsideout, n_level, nbin , bin_width, Tfield, gmaps, beam_window, cllsslss, clTT, clTlss, cltaudlss, Noise):
        

        xizeta = np.zeros((nbin,healpy.nside2npix(nside)))
        xizetabar = np.zeros((nbin,healpy.nside2npix(nside)))
        binned_qe = np.zeros((nbin,healpy.nside2npix(nsideout)))
          
            
        for n in range(n_level+1):
            xizetat, xizetabart = self.bare_quad_est(nside, n, nbin, bin_width, Tfield, gmaps, beam_window, cllsslss, clTT, clTlss,cltaudlss)
            xizeta += xizetat
            xizetabar += xizetabart


        for binn in range(nbin):
            veff_reconstlm = healpy.almxfl(healpy.map2alm(xizeta[binn]-xizetabar[binn],lmax=(3*nsideout-1)),Noise[binn])
            binned_qe[binn] = healpy.alm2map(veff_reconstlm, nsideout)

        return binned_qe

    def reconstruct_ml(self, nside, nsideout, n_level, nbin , 
                       bin_width, Tfield, gmaps, vmaps, beam_window, 
                       cllsslss, clTT, clTlss, clmllss,realization,Noise):

        term1 = np.zeros((nbin,healpy.nside2npix(nside)))
        term2 = np.zeros((nbin,healpy.nside2npix(nside)))

        binned_qe = np.zeros((nbin,healpy.nside2npix(nsideout)))

        for n in range(n_level+1):
            if n==0:
                term1, term2 = self.bare_quad_est_ml(nside, n, nbin, bin_width, Tfield, 
                                                     gmaps, beam_window, cllsslss, clTT, 
                                                     clTlss,clmllss,realization)
            else:
                term1t, term2t = self.bare_quad_est_ml(nside, n, nbin, bin_width, Tfield, 
                                                       gmaps, beam_window, cllsslss, clTT, 
                                                       clTlss,clmllss,realization)
                term1 += term1t
                term2 += term2t

        lvec = np.arange(0,3*nsideout)
        lfac = lvec*(1.+lvec)

        factor = 1e30
        for binn in range(nbin):

            term1[binn] = term1[binn]/factor
            veff_reconstlm1 = healpy.almxfl(healpy.map2alm(term1[binn], lmax=(3*nsideout-1)), lfac*Noise[binn])

            term2[binn] = term2[binn]/factor
            veff_reconstlm2 = healpy.almxfl(healpy.map2alm(term2[binn], lmax=(3*nsideout-1)), Noise[binn])

            binned_qe[binn] = ( healpy.alm2map((veff_reconstlm1)*0.5*factor, nsideout)
                               +healpy.alm2map((veff_reconstlm2)*0.5*factor, nsideout))

        return binned_qe


    
    def get_qe_sims(self,nside, nsideout, n_level, use_cleaned = True, frequency = None, mask = False):
        
        
        self.set_theory_Cls(add_ksz = True, add_ml = False, use_cleaned = use_cleaned, frequency = frequency)
            
        lcut = 3*nside-0
        
        if use_cleaned:
            beam_window = np.ones(3*nside)   # We have to determine how to work the beaming here
        else:        
            ls = np.arange(3*nside)
            beam_window = np.exp(-ls*(ls+1)*(self.beam**2)/(16.*np.log(2)))
        
        clTT      = self.Cls['T-T'][:3*nside,0,0]
        clTlss    = self.Cls['T-lss'][:3*nside]
        cltaudlss = self.Cls['taud-lss'][:3*nside]
        cllsslss  = self.Cls['lss-lss'][:3*nside]
        clksz     = self.Cls['kSZ-kSZ'][:3*nside,0,0]
        
 
        #GET THEORY NOISE . ONE CAN SPEED UP THIS ASSUMING ELL DEPENDENCE
        
        print("Getting theory noise")
        
        if c.exists(self.basic_conf_dir,'Nvv_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2'): 
            Noise = c.load(self.basic_conf_dir,'Nvv_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
        else:
            
            Lsamp = np.unique(np.append(np.geomspace(1,3*nsideout-1,20).astype(int),3*nsideout-1))
            Noise_int = np.zeros((len(Lsamp),self.nbin)) 
             
            for lid, l  in enumerate(Lsamp):
                for i in np.arange(self.nbin):
                    Noise_int[lid,i] = self.Noise_iso(lcut, 'v', i, i, l)
                
            Noise = np.swapaxes(loginterp.log_interpolate_vector(Noise_int, Lsamp),0,1)
            
            c.dump(self.basic_conf_dir,Noise,'Nvv_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
            
        print("Theory noise ready")  
        
        print("Getting rotation matrix ")
        
        if c.exists(self.basic_conf_dir,'Rvv_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2'): 
            R = c.load(self.basic_conf_dir,'Rvv_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
        else:
            
            R = np.zeros((self.nbin,self.nbin)) 
             
            for i in np.arange(self.nbin):
                for j in np.arange(self.nbin):      
                    R[i,j] = self.R(lcut,'v','v',i,j,2)
                    
            c.dump(self.basic_conf_dir,R,'Rvv_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
            
        print("Rotation matrix ready")  
            
        
        real_num = self.realnum
                
        for r in np.arange(real_num):
            
            print('real = '+str(r))
            
            if c.exists(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)
                        +'_full'+'_real='+str(r)+'_mask='+str(mask)
                        +'_nlevel='+str(n_level), dir_base = 'sims2'): 
                print("r = "+str(r)+" already done")
                continue
                     
                
            if c.exists(self.basic_conf_dir,'ksz_'+str(nside)+'_real='+str(r), dir_base = 'sims2'):
                
                print("Loading pre-existing sims")
                
                lssmaps  = np.zeros((self.nbin,healpy.nside2npix(nside)))      
                for b in np.arange(self.nbin):   
                    lssmaps[b,:] = c.load(self.basic_conf_dir,'lss_'+str(nside)+'_real='+str(r)+'_bin='+str(b), dir_base = 'sims2')
   
                Tmap = c.load(self.basic_conf_dir,self.Ttag+'_'+str(nside)+'_real='+str(r)+'_bin='+str(0), dir_base = 'sims2')
                kszmap = c.load(self.basic_conf_dir,'ksz_'+str(nside)+'_real='+str(r), dir_base = 'sims2')
            else:
                
                print("Getting sims")
                
                if use_cleaned == True:
                
                    sims, alms = self.get_maps_and_alms(['v','taud','g'],nside,3*nside)   #self.get_maps_and_alms(['v','taud','g',self.Ttag],nside,3*nside)  
                    vmaps    = sims[0]
                    taudmaps = sims[1]
                    lssmaps  = sims[2]
                    Tmap    = healpy.synfast(loginterp.log_interpolate_matrix(self.load_theory_Cl('Tc', 'Tc'), self.load_L())[:3*nside,0,0],nside)  #sims[3][0,:]    
                    kszmap = self.ksz_map_maker(taudmaps, vmaps, nside)
                    
                else:
                    
                    if self.Ttag != 'T0':

                        sims, alms = self.get_maps_and_alms(['v','taud','g',self.Ttag],nside,3*nside)  
                        vmaps    = sims[0]
                        taudmaps = sims[1]
                        lssmaps  = sims[2]
                        Tmap    = sims[3][0,:]
                        kszmap = self.ksz_map_maker(taudmaps, vmaps, nside)
                
                    else:
                        
                        sims, alms = self.get_maps_and_alms(['v','taud','g'],nside,3*nside)  
                        vmaps    = sims[0]
                        taudmaps = sims[1]
                        lssmaps  = sims[2]
                        Tmap    =  healpy.synfast(self.Cls['pCMB-pCMB'][:3*nside,0,0],nside)
                        kszmap = self.ksz_map_maker(taudmaps, vmaps, nside)
                
                vrot = np.zeros((self.nbin,healpy.nside2npix(nsideout)))
                
                for b in np.arange(self.nbin):
                    
                    for i in np.arange(self.nbin):
                        vrot[b,:] += R[b,i]*healpy.pixelfunc.ud_grade(vmaps[i,:], nside_out = nsideout)  #update R so it also depends on ell
                     
                    c.dump(self.basic_conf_dir,lssmaps[b,:],'lss_'+str(nside)+'_real='+str(r)+'_bin='+str(b), dir_base = 'sims2')
                
                c.dump(self.basic_conf_dir,vrot,'vactualrot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r), dir_base = 'sims2')
                c.dump(self.basic_conf_dir,Tmap,self.Ttag+'_'+str(nside)+'_real='+str(r)+'_bin='+str(0), dir_base = 'sims2')       
                c.dump(self.basic_conf_dir, kszmap,'ksz_'+str(nside)+'_real='+str(r), dir_base = 'sims2')
                

            if use_cleaned:
                Tfield_gauss = healpy.synfast(clksz,nside)+Tmap
                Tfield_full =  kszmap + Tmap
            else:
                Tfield_gauss = healpy.smoothing(healpy.synfast(clksz,nside) + Tmap,fwhm=self.beam) + healpy.synfast(self.dT*self.dT*np.ones(3*nside),nside)  
                Tfield_full =  healpy.smoothing(kszmap + Tmap,fwhm=self.beam) + healpy.synfast(self.dT*self.dT*np.ones(3*nside),nside)  

            
            clTT[0:100]    =0   #Low multipoles dont contribute to the reconstruction . At the moment I put this to filter some low ell noise in the realizations.
            cllsslss[0:100]=0   #Low multipoles dont contribute to the reconstruction . At the moment I put this to filter some low ell noise in the realizations.
            
            if mask:
                Tfield_gauss = Tfield_gauss*np.load('SO_mask_N2048.npy')
                Tfield_full  = Tfield_full*np.load('SO_mask_N2048.npy')
                lssmaps = lssmaps*np.load('SO_mask_N2048.npy')
                                    
            print("Reconstructing velocity")
                
            qe_gauss = self.reconstruct(nside, nsideout, n_level, self.nbin, self.deltachi, Tfield_gauss, lssmaps,  beam_window, cllsslss, clTT, clTlss, cltaudlss,Noise)
            qe_full = self.reconstruct(nside, nsideout, n_level, self.nbin, self.deltachi, Tfield_full, lssmaps,  beam_window, cllsslss, clTT, clTlss, cltaudlss,Noise)
    
            c.dump(self.basic_conf_dir,qe_gauss,'qe_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,qe_full,'qe_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
        
        pass
    
    def get_qe_sims_ml(self,nside, nsideout, n_level, use_cleaned = True, mask = False):
        
        
        self.set_theory_Cls(add_ksz = False, add_ml = True, use_cleaned = use_cleaned)
         
        beam_window = np.ones(3*nside)
        
        clTT      = self.Cls['T-T'][:3*nside,0,0]
        clTlss    = self.Cls['T-lss'][:3*nside]
        clmllss   = self.Cls['ml-lss'][:3*nside]
        cllsslss  = self.Cls['lss-lss'][:3*nside]
        clML      = self.Cls['ML-ML'][:3*nside,0,0]
        
        
        #GET THEORY NOISE . ONE CAN SPEED UP THIS ASSUMING ELL DEPENDENCE
        
        print("Getting theory noise")
        
        if c.exists(self.basic_conf_dir,'Nvtvt_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2'): 
            Noise = c.load(self.basic_conf_dir,'Nvtvt_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
        else:
            
            Lsamp = np.unique(np.append(np.geomspace(1,3*nsideout-1,10).astype(int),3*nsideout-1))
            Noise_int = np.zeros((len(Lsamp),self.nbin)) 
            
            #Hack assuming N goes as 1.0/ell/(ell+1) as a function of ell
            
            N2 = self.Noise_iso_alphas(3*nside,'vt', 2)
            for lid, l  in enumerate(Lsamp):
                Noise_int[lid,:] = N2*6/l/(l+1)
            
            #Actual noise with the exact ell dependence 
            
            # for lid, l  in enumerate(Lsamp):
            #     Noise_int[lid,:] = self.Noise_iso_alphas(3*nside,'vt', l)
                
            Noise = np.swapaxes(loginterp.log_interpolate_vector(Noise_int, Lsamp),0,1)
            
            c.dump(self.basic_conf_dir,Noise,'Nvtvt_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
            
        print("Theory noise ready")  

        real_num = self.realnum
                
        for r in np.arange(real_num):
            
            print('real = '+str(r))
            
            if c.exists(self.basic_conf_dir,'qe_vt_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2'):
                print("r = "+str(r)+" already done")
                continue
            
            if use_cleaned:
                
                print("Getting sims")
                    
                sims, alms = self.get_maps_and_alms(['vt','ml','g','Tc'],nside,3*nside)  
                  
                vtmaps    = sims[0]
                lssmaps  = sims[2]
                Tmap    = sims[3][0,:]
                
            else:
                print("Getting sims")
                    
                sims, alms = self.get_maps_and_alms(['vt','ml','g'],nside,3*nside)  
                  
                vtmaps    = sims[0]
                lssmaps  = sims[2]
                Tmap    = 0*healpy.synfast(  (self.Cls['pCMB-pCMB']+self.CMB_noise(np.arange(self.data_lmax+1)))[:,0,0] ,nside)
            
            
            MLmap  = self.ml_map_maker(alms[0], alms[1],  nside)

            vtrot = np.zeros((self.nbin,healpy.nside2npix(nsideout)))
            
            for b in range(self.nbin):       
                for i in np.arange(self.nbin):
                    vtrot[b,:] += self.R(3*nside,'vt','vt',b,i,2)*healpy.pixelfunc.ud_grade(vtmaps[i,:], nside_out = nsideout)  #update R so it also depends on ell
            c.dump(self.basic_conf_dir,vtrot,'vtactualrot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r), dir_base = 'sims2')
            

            Tfield_gauss = healpy.synfast(clML,nside)+Tmap 
            Tfield_full = MLmap + Tmap   
    
            clTT[0:2]=0
            
            if mask:
                Tfield_gauss = Tfield_gauss*np.load('SO_mask_N2048.npy')
                Tfield_full  = Tfield_full*np.load('SO_mask_N2048.npy')
                    
            print("Reconstructing velocity")
                
            qe_gauss = self.reconstruct_ml(nside, nsideout, n_level, self.nbin, self.deltachi, Tfield_gauss, lssmaps,  vtmaps, beam_window, cllsslss, clTT, clTlss, clmllss,Noise)
            qe_full  = self.reconstruct_ml(nside, nsideout, n_level, self.nbin, self.deltachi, Tfield_full,  lssmaps,  vtmaps, beam_window, cllsslss, clTT, clTlss, clmllss,Noise)
    
            c.dump(self.basic_conf_dir,qe_gauss,'qe_vt_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,qe_full,'qe_vt_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
        
        pass
        
    def get_clqe_sims(self,nside, nsideout, n_level, mask = True):
        
    
        real_num = self.realnum
        
        if mask:                                     
            map_mask = healpy.pixelfunc.ud_grade(np.load('SO_mask_N2048.npy').astype(bool), nside_out = nsideout).astype(float)
            mask_d_1 = self.mask_edge(nsideout,map_mask, edgeval =0)
            mask_d_2 = self.mask_edge(nsideout,mask_d_1, edgeval =0)
            mask_d_3 = self.mask_edge(nsideout,mask_d_2, edgeval =0)
            mask_d_4 = self.mask_edge(nsideout,mask_d_3, edgeval =0)
            mask_d_5 = self.mask_edge(nsideout,mask_d_4, edgeval =0)

        for r in np.arange(real_num):
            
            qe_full    = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            qe_gauss   = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            vrot = c.load(self.basic_conf_dir,'vactualrot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r), dir_base = 'sims2')
            
            print(r)
            
            Cvv_recon       = np.zeros((3*nsideout,self.nbin))
            Cvv_actual_rot  = np.zeros((3*nsideout,self.nbin))
            Cvv_diff        = np.zeros((3*nsideout,self.nbin))
            Cvv_noise       = np.zeros((3*nsideout,self.nbin))
            
            Cvv_recon_5       = np.zeros((3*nsideout,self.nbin))
            Cvv_actual_rot_5  = np.zeros((3*nsideout,self.nbin))
            Cvv_diff_5        = np.zeros((3*nsideout,self.nbin))
            Cvv_noise_5       = np.zeros((3*nsideout,self.nbin))
            

            for b in np.arange(self.nbin):   
                
                if mask:
                                                      
                    Cvv_recon[:,b]       = healpy.sphtfunc.anafast(qe_full[b]*map_mask,qe_full[b]*map_mask)
                    Cvv_actual_rot[:,b]  = healpy.sphtfunc.anafast(vrot[b]*map_mask,vrot[b]*map_mask)
                    Cvv_diff[:,b]        = healpy.sphtfunc.anafast(qe_full[b]*map_mask-vrot[b]*map_mask,qe_full[b]*map_mask-vrot[b]*map_mask)
                    Cvv_noise[:,b]       = healpy.sphtfunc.anafast(qe_gauss[b]*map_mask,qe_gauss[b]*map_mask)
                    
                    Cvv_recon_5[:,b]       = healpy.sphtfunc.anafast(qe_full[b]*mask_d_1,qe_full[b]*mask_d_1)
                    Cvv_actual_rot_5[:,b]  = healpy.sphtfunc.anafast(vrot[b]*mask_d_1,vrot[b]*mask_d_1)
                    Cvv_diff_5[:,b]        = healpy.sphtfunc.anafast(qe_full[b]*mask_d_1-vrot[b]*mask_d_1,qe_full[b]*mask_d_1-vrot[b]*mask_d_1)
                    Cvv_noise_5[:,b]       = healpy.sphtfunc.anafast(qe_gauss[b]*mask_d_1,qe_gauss[b]*mask_d_1)
                
                else:
                
                    Cvv_recon[:,b]       = healpy.sphtfunc.anafast(qe_full[b],qe_full[b])
                    Cvv_actual_rot[:,b]  = healpy.sphtfunc.anafast(vrot[b],vrot[b])
                    Cvv_diff[:,b]        = healpy.sphtfunc.anafast(qe_full[b]-vrot[b],qe_full[b]-vrot[b])
                    Cvv_noise[:,b]       = healpy.sphtfunc.anafast(qe_gauss[b],qe_gauss[b])
                         
            c.dump(self.basic_conf_dir,Cvv_recon,'Cvv_recon_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')   
            c.dump(self.basic_conf_dir,Cvv_actual_rot,'Cvv_actual_rot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(mask), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cvv_diff,'Cvv_diff_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cvv_noise,'Cvv_noise_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            
            if mask:
                
                c.dump(self.basic_conf_dir,Cvv_recon_5,'Cvv_recon_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask2='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')   
                c.dump(self.basic_conf_dir,Cvv_actual_rot_5,'Cvv_actual_rot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask2='+str(mask), dir_base = 'sims2')
                c.dump(self.basic_conf_dir,Cvv_diff_5,'Cvv_diff_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask2='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
                c.dump(self.basic_conf_dir,Cvv_noise_5,'Cvv_noise_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask2='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
                
                
                
        pass
    
    def get_clqe_sims_ml(self,nside, nsideout, n_level, use_cleaned = True, mask = True):
        
        real_num = self.realnum

        for r in np.arange(real_num):
            
            qe_full    = c.load(self.basic_conf_dir,'qe_vt_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            qe_gauss   = c.load(self.basic_conf_dir,'qe_vt_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            vtrot = c.load(self.basic_conf_dir,'vtactualrot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r), dir_base = 'sims2')
            
            print(r)
            
            Cvtvt_recon       = np.zeros((3*nsideout,self.nbin))
            Cvtvt_actual_rot  = np.zeros((3*nsideout,self.nbin))
            Cvtvt_diff        = np.zeros((3*nsideout,self.nbin))
            Cvtvt_noise       = np.zeros((3*nsideout,self.nbin))
            
                        
            for b in np.arange(self.nbin):
                
                Cvtvt_recon[:,b]       = healpy.sphtfunc.anafast(qe_full[b],qe_full[b])
                Cvtvt_actual_rot[:,b]  = healpy.sphtfunc.anafast(vtrot[b],vtrot[b])
                Cvtvt_diff[:,b]        = healpy.sphtfunc.anafast(qe_full[b]-vtrot[b],qe_full[b]-vtrot[b])
                Cvtvt_noise[:,b]       = healpy.sphtfunc.anafast(qe_gauss[b],qe_gauss[b])
                         
            c.dump(self.basic_conf_dir,Cvtvt_recon,'Cvtvt_recon_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cvtvt_actual_rot,'Cvtvt_actual_rot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(mask), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cvtvt_diff,'Cvtvt_diff_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cvtvt_noise,'Cvtvt_noise_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(mask)+'_nlevel='+str(n_level), dir_base = 'sims2')
            
        pass
  

    
    def get_Clp_sims(self,nside, nsideout, n_level):
        
        self.set_theory_Cls(add_ksz = True, add_ml = False, use_cleaned = True, frequency = None)
                                       
        map_mask = healpy.pixelfunc.ud_grade(np.load('SO_mask_N2048.npy').astype(bool), nside_out = nsideout).astype(float)
        mask_d_1 = self.mask_edge(nsideout,map_mask, edgeval =0)
        mask_d_2 = self.mask_edge(nsideout,mask_d_1, edgeval =0)
        mask_d_3 = self.mask_edge(nsideout,mask_d_2, edgeval =0)
        mask_d_4 = self.mask_edge(nsideout,mask_d_3, edgeval =0)
        mask_d_5 = self.mask_edge(nsideout,mask_d_4, edgeval =0)
                
        print("Geeting R and Noise")
        R  = c.load(self.basic_conf_dir,'Rvv_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
        if c.exists(self.basic_conf_dir,'C0_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2'):
            C0 = c.load(self.basic_conf_dir,'C0_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
        else:
            C0 = self.Noise_iso_ell(3*nside ,'v', 2, dndz = None)
            c.dump(self.basic_conf_dir,C0,'C0_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
    
        
        real_num = self.realnum
        
        for r in np.arange(real_num):
            
            print("real = "+str(r))
            
            C_recon     = np.zeros((3*nsideout,self.nbin,self.nbin))
            C_actual    = np.zeros((3*nsideout,self.nbin,self.nbin))
            C_diff      = np.zeros((3*nsideout,self.nbin,self.nbin))
            C_noise     = np.zeros((3*nsideout,self.nbin,self.nbin))
        
            C_recon_m   = np.zeros((3*nsideout,self.nbin,self.nbin))
            C_actual_m  = np.zeros((3*nsideout,self.nbin,self.nbin))
            C_diff_m    = np.zeros((3*nsideout,self.nbin,self.nbin))
            C_noise_m   = np.zeros((3*nsideout,self.nbin,self.nbin))
            
            qe_full    = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')
            qe_gauss   = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')
            vrot = c.load(self.basic_conf_dir,'vactualrot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r), dir_base = 'sims2')
            
            qe_full_m    = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')*mask_d_1[np.newaxis,:]
            qe_gauss_m   = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')*mask_d_1[np.newaxis,:]
            vrot_m = c.load(self.basic_conf_dir,'vactualrot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r), dir_base = 'sims2')*mask_d_1[np.newaxis,:]
            
            for b1 in np.arange(self.nbin):
                for b2 in np.arange(self.nbin):
            
                    C_recon[:,b1,b2]       = healpy.sphtfunc.anafast(qe_full[b1],qe_full[b2])
                    C_actual[:,b1,b2]      = healpy.sphtfunc.anafast(vrot[b1],vrot[b2])
                    C_diff[:,b1,b2]        = healpy.sphtfunc.anafast(qe_full[b1]-vrot[b1],qe_full[b2]-vrot[b2])
                    C_noise[:,b1,b2]       = healpy.sphtfunc.anafast(qe_gauss[b1],qe_gauss[b2])
                
                    C_recon_m[:,b1,b2]     = healpy.sphtfunc.anafast(qe_full_m[b1],qe_full_m[b2])
                    C_actual_m[:,b1,b2]    = healpy.sphtfunc.anafast(vrot_m[b1],vrot_m[b2])
                    C_diff_m[:,b1,b2]      = healpy.sphtfunc.anafast(qe_full_m[b1]-vrot_m[b1],qe_full_m[b2]-vrot_m[b2])
                    C_noise_m[:,b1,b2]     = healpy.sphtfunc.anafast(qe_gauss_m[b1],qe_gauss_m[b2])
            
               
            Cpp_recon     = np.zeros((3*nsideout,self.nbin,self.nbin))
            Cpp_actual    = np.zeros((3*nsideout,self.nbin,self.nbin))
            Cpp_diff      = np.zeros((3*nsideout,self.nbin,self.nbin))
            Cpp_noise     = np.zeros((3*nsideout,self.nbin,self.nbin))
        
            Cpp_recon_m   = np.zeros((3*nsideout,self.nbin,self.nbin))
            Cpp_actual_m  = np.zeros((3*nsideout,self.nbin,self.nbin))
            Cpp_diff_m    = np.zeros((3*nsideout,self.nbin,self.nbin))
            Cpp_noise_m   = np.zeros((3*nsideout,self.nbin,self.nbin))
            
            for ell in np.arange(1,50):
                
                print(ell)
                
                C = self.Cls['v-v'][ell]
                Cs = np.dot(np.dot(R,C),np.transpose(R))
        
                #First diagonalization
    
                w,v = np.linalg.eigh(C0)
    
                R1 = np.transpose(v)
                R2 = np.zeros_like(C0)
        
                for i in np.arange(self.nbin):
                    R2[i,i] = 1.0/np.sqrt(w[i])
                    
                R21 = np.dot(R2,R1)
                Cs_p = np.dot(np.dot(R21,Cs), np.transpose(R21))
                w3,v3 = np.linalg.eigh(Cs_p)
                R3 = np.transpose(v3)
                R321 = np.dot(R3,R21)
                
                #pcs_1 =R321[-1,:]
                #pcs_2 =R321[-2,:]
                
                Cpp_recon[ell]     = np.dot(np.dot(R321,C_recon[ell]), np.transpose(R321))
                Cpp_actual[ell]    = np.dot(np.dot(R321,C_actual[ell]), np.transpose(R321))
                Cpp_diff[ell]      = np.dot(np.dot(R321,C_diff[ell]), np.transpose(R321))
                Cpp_noise[ell]     = np.dot(np.dot(R321,C_noise[ell]), np.transpose(R321))
            
                Cpp_recon_m[ell]     = np.dot(np.dot(R321,C_recon_m[ell]), np.transpose(R321))
                Cpp_actual_m[ell]    = np.dot(np.dot(R321,C_actual_m[ell]), np.transpose(R321))
                Cpp_diff_m[ell]      = np.dot(np.dot(R321,C_diff_m[ell]), np.transpose(R321))
                Cpp_noise_m[ell]     = np.dot(np.dot(R321,C_noise_m[ell]), np.transpose(R321))
                
                                
            c.dump(self.basic_conf_dir,Cpp_recon,'Cpp_recon_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')   
            c.dump(self.basic_conf_dir,Cpp_actual,'Cpp_actual_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(False), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cpp_diff,'Cpp_diff_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cpp_noise,'Cpp_noise_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')
 
            c.dump(self.basic_conf_dir,Cpp_recon_m,'Cpp_recon_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')   
            c.dump(self.basic_conf_dir,Cpp_actual_m,'Cpp_actual_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(True), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cpp_diff_m,'Cpp_diff_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,Cpp_noise_m,'Cpp_noise_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')

            c.dump(self.basic_conf_dir,C_recon,'C_recon_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')   
            c.dump(self.basic_conf_dir,C_actual,'C_actual_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(False), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,C_diff,'C_diff_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,C_noise,'C_noise_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')
 
            c.dump(self.basic_conf_dir,C_recon_m,'C_recon_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')   
            c.dump(self.basic_conf_dir,C_actual_m,'C_actual_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(True), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,C_diff_m,'C_diff_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')
            c.dump(self.basic_conf_dir,C_noise_m,'C_noise_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')

        pass
    
    
    def get_pc_sims(self,nside, nsideout, n_level):
        
        self.set_theory_Cls(add_ksz = True, add_ml = False, use_cleaned = True, frequency = None)
                                       
        map_mask = healpy.pixelfunc.ud_grade(np.load('SO_mask_N2048.npy').astype(bool), nside_out = nsideout).astype(float)
        mask_d_1 = self.mask_edge(nsideout,map_mask, edgeval =0)
        mask_d_2 = self.mask_edge(nsideout,mask_d_1, edgeval =0)
        mask_d_3 = self.mask_edge(nsideout,mask_d_2, edgeval =0)
        mask_d_4 = self.mask_edge(nsideout,mask_d_3, edgeval =0)
        mask_d_5 = self.mask_edge(nsideout,mask_d_4, edgeval =0)
                
        print("Geeting R and Noise")
        R  = c.load(self.basic_conf_dir,'Rvv_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
        if c.exists(self.basic_conf_dir,'C0_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2'):
            C0 = c.load(self.basic_conf_dir,'C0_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
        else:
            C0 = self.Noise_iso_ell(3*nside ,'v', 2, dndz = None)
            c.dump(self.basic_conf_dir,C0,'C0_'+str(nside)+'_'+str(nsideout), dir_base = 'sims2')
    
        
        real_num = self.realnum
        
        for r in np.arange(real_num):
            
            print("real = "+str(r))
        
            
            qe_full    = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')
            qe_gauss   = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real='+str(r)+'_mask='+str(False)+'_nlevel='+str(n_level), dir_base = 'sims2')
            vrot = c.load(self.basic_conf_dir,'vactualrot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r), dir_base = 'sims2')
            
            qe_full_m    = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_full'+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')*mask_d_5[np.newaxis,:]
            qe_gauss_m   = c.load(self.basic_conf_dir,'qe_'+str(nside)+'_'+str(nsideout)+'_gauss'+'_real='+str(r)+'_mask='+str(True)+'_nlevel='+str(n_level), dir_base = 'sims2')*mask_d_5[np.newaxis,:]
            vrot_m = c.load(self.basic_conf_dir,'vactualrot_'+str(nside)+'_'+str(nsideout)+'_real='+str(r), dir_base = 'sims2')*mask_d_5[np.newaxis,:]
            
            
            almsize = len(healpy.map2alm(vrot_m[0,:]))
            alms_p_actual   = np.zeros((2,almsize),dtype=complex)
            alms_p_est      = np.zeros((2,almsize),dtype=complex)
            alms_p_noise      = np.zeros((2,almsize),dtype=complex)
            alms_p_actual_m = np.zeros((2,almsize),dtype=complex)
            alms_p_est_m    = np.zeros((2,almsize),dtype=complex)
            alms_p_noise_m    = np.zeros((2,almsize),dtype=complex)
            
            for ell in np.arange(1,50):
                
                print(ell)
                
                C = self.Cls['v-v'][ell]
                Cs = np.dot(np.dot(R,C),np.transpose(R))
        
                #First diagonalization
    
                w,v = np.linalg.eigh(C0)
    
                R1 = np.transpose(v)
                R2 = np.zeros_like(C0)
        
                for i in np.arange(self.nbin):
                    R2[i,i] = 1.0/np.sqrt(w[i])
                    
                R21 = np.dot(R2,R1)
                Cs_p = np.dot(np.dot(R21,Cs), np.transpose(R21))
                w3,v3 = np.linalg.eigh(Cs_p)
                R3 = np.transpose(v3)
                R321 = np.dot(R3,R21)
                
                pcs_1 =R321[-5,:]
                pcs_2 =R321[-6,:]
                

                
                alms_p_actual[0,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_1,vrot)),np.where(np.arange(3*64) == ell ,1,0))
                alms_p_actual[1,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_2,vrot)),np.where(np.arange(3*64) == ell ,1,0))
                
                alms_p_actual_m[0,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_1,vrot_m)),np.where(np.arange(3*64) == ell ,1,0))
                alms_p_actual_m[1,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_2,vrot_m)),np.where(np.arange(3*64) == ell ,1,0))
                
                alms_p_est[0,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_1,qe_full)),np.where(np.arange(3*64) == ell ,1,0))
                alms_p_est[1,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_2,qe_full)),np.where(np.arange(3*64) == ell ,1,0))
                
                alms_p_est_m[0,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_1,qe_full_m)),np.where(np.arange(3*64) == ell ,1,0))
                alms_p_est_m[1,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_2,qe_full_m)),np.where(np.arange(3*64) == ell ,1,0))
                
                alms_p_noise[0,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_1,qe_gauss)),np.where(np.arange(3*64) == ell ,1,0))
                alms_p_noise[1,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_2,qe_gauss)),np.where(np.arange(3*64) == ell ,1,0))
                
                alms_p_noise_m[0,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_1,qe_gauss_m)),np.where(np.arange(3*64) == ell ,1,0))
                alms_p_noise_m[1,:] += healpy.almxfl(healpy.map2alm(np.dot(pcs_2,qe_gauss_m)),np.where(np.arange(3*64) == ell ,1,0))
                
                    
            
            for j in np.arange(2):
                p_actual   = healpy.alm2map(alms_p_actual[j,:],nsideout)
                p_est      = healpy.alm2map(alms_p_est[j,:],nsideout)
                p_noise    = healpy.alm2map(alms_p_noise[j,:],nsideout)
                p_actual_m = healpy.alm2map(alms_p_actual_m[j,:],nsideout)*mask_d_5
                p_est_m    = healpy.alm2map(alms_p_est_m[j,:],nsideout)*mask_d_5
                p_noise_m  = healpy.alm2map(alms_p_noise_m[j,:],nsideout)*mask_d_5
                
                c.dump(self.basic_conf_dir,p_actual,'p_actual_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_j='+str(j+4), dir_base = 'sims2')
                c.dump(self.basic_conf_dir,p_est,'p_est_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_j='+str(j+4), dir_base = 'sims2')
                c.dump(self.basic_conf_dir,p_noise,'p_noise_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_j='+str(j+4), dir_base = 'sims2')
                c.dump(self.basic_conf_dir,p_actual_m,'p_actual_m_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_j='+str(j+4), dir_base = 'sims2')
                c.dump(self.basic_conf_dir,p_est_m,'p_est_m_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_j='+str(j+4), dir_base = 'sims2')
                c.dump(self.basic_conf_dir,p_noise_m,'p_noise_m_'+str(nside)+'_'+str(nsideout)+'_real='+str(r)+'_j='+str(j+4), dir_base = 'sims2')

        pass
    
    def ksz_map_maker(self, taud_map, v_map, nside):
        
        npix = healpy.nside2npix(nside)

        vmaps =  v_map
        taudmaps = taud_map
        kszmap = np.zeros(npix)

        for i in range(self.nbin):
            kszmap[:] += vmaps[i,:]*taudmaps[i,:]*self.deltachi

        return kszmap
    
    def ml_map_maker(self, alm_vt, alm_ml, nside):

        npix = healpy.nside2npix(nside)
        mlmap = np.zeros(npix)

        for i in range(self.nbin):
            map_ml_i, ml_d_theta, ml_d_phi = healpy.sphtfunc.alm2map_der1(alm_ml[i], nside)
            map_vt_i, vt_d_theta, vt_d_phi = healpy.sphtfunc.alm2map_der1(alm_vt[i], nside)

            mlmap[:] +=  (ml_d_theta * vt_d_theta + ml_d_phi * vt_d_phi)*self.deltachi

        return mlmap
            
        

    def mask_edge(self, nside,M, edgeval =0):
        M2 = np.copy(M)
        zeros = np.where(M==0)[0]
        neig = healpy.get_all_neighbours(nside,zeros)
        neig_1d = np.unique(np.reshape(neig,(8*len(zeros))))
        edge = np.setdiff1d(neig_1d,zeros)
        M2[edge] = edgeval
        return M2
    

    
    
    
    
    
    
    
    

        
        
        
