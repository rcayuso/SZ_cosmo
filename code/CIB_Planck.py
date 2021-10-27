#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:24:21 2019

@author: fionamccarthy
"""

from __future__ import print_function
from __future__ import absolute_import
import scipy
import numpy as np

#cosmology module
import kszpsz_config
import cosmology
csm = cosmology.cosmology(basic_conf_obj = kszpsz_config)


planck=6.626e-34
lightspeed=2.99792458e8
kBoltzmann=1.38e-23
            


T0 = 24.4
alpha = 0.36
beta = 1.75
delta = 3.6
sigmasqlm = 0.5
gamma = 1.7
Meff = 10**12.6
Mmin = 10**10
L0 = 0.004755788078226362*1
zplateau = 7


def TD(z): #Dust temperature of starforming galaxyT
            return T0*(1+z)**alpha


def dlnsed_dnu(nu,T):
        nu=nu*1e9
        return 3+beta+planck*nu/(T*kBoltzmann)*(-1+1/(1-np.exp(planck*nu/(kBoltzmann*T))))+gamma
    
def Planck(nu,T): #Planck law for black body radiation
            return 2*(nu)**3*(1/(np.exp(planck*nu/(kBoltzmann*T))-1))*planck/lightspeed**2


def SED(nu,T,zs):
    #eq 8 of 1611.04517
      
            nu=nu*1e9 #multiply by 10^^9 to go from GHz->Hz
            SED=np.zeros(nu.shape)
            nu0s=np.array([scipy.optimize.brentq(dlnsed_dnu,10,10000,args=TD(z))for z in zs])*1e9
            
            print(nu.shape,nu0s.shape,T.shape)
            
            print(((nu[nu<nu0s]/nu0s[nu<nu0s])**beta*Planck(nu[nu<nu0s],T[nu<nu0s])/Planck(nu0s[nu<nu0s],T[nu<nu0s])).shape)
            print(SED[nu<nu0s].shape)
            
            
            SED[nu<nu0s]=(nu[nu<nu0s]/nu0s[nu<nu0s])**beta*Planck(nu[nu<nu0s],T[nu<nu0s])/Planck(nu0s[nu<nu0s],T[nu<nu0s])

            return SED
def redshiftevolutionofl(z):
            
            answer=np.zeros(len(z))
            
            answer=(1+z)**delta
           # if len(answer[z>=zplateau])>0:
            #    answer[z>=zplateau]=(1+zplateau)**delta
            return answer
def Sigma(M):
    
          
            answer= M*(1/(2*np.pi*sigmasqlm)**(1/2))*np.exp(-(np.log10(M)-np.log10(Meff))**2/(2*sigmasqlm))
            answer[M<Mmin]=0
            return answer
def Lnu(nu,z,M): #spectral luminosity radiance
            Lir=L0*1.35e-5 #total IR luminosity
            sed=SED(nu*(1+z),TD(z),z) #normalised such that the integral over all nu is 1.
                
            return Lir*sed*Sigma(M)*redshiftevolutionofl(z)
        

def Scentral(nu,z,Mhalo):
    
    #flux from luminosity; eg eq 7 of 1611.04517
        chi=csm.chi_from_z(z)
        return Lnu(nu,z,Mhalo)/((4*np.pi)*chi**2*(1+z)) # in units of [Lnu]/Mpc**2=solar_luminosity/Mpc**2/Hz

def Luminosity_from_flux(S,z):
    #gives luminosity in [S] * Mpc**2
    return  4 * np.pi * csm.chi_from_z(z)**2*(1+z)*S

def subhalo_mass_function(Msub,Mhost):
    #equation 10 from 0909.1325. Need to integrate against ln M. (it gives dn/dlnM_sub)
    return 0.3*(Msub/Mhost)**-0.7*np.exp(-9.9*(Msub/Mhost)**2.5)

def satellite_intensity(nu,zs,mhalos):
        satellite_masses=mhalos.copy()[:-1]
    
        dndms=subhalo_mass_function(satellite_masses,mhalos[:,np.newaxis])
        return np.trapz((dndms[:,:,np.newaxis]*Scentral(nu,zs,satellite_masses[:,np.newaxis])[np.newaxis,:,:]),np.log(satellite_masses),axis=1)

def conversion(nu):
    if nu==353:
        return 287.45
    elif nu==545:
        return 58.04
    elif nu==857:
        return 2.27
def sn(nu1,nu2):
        if [nu1,nu2]==[857,857]:
            ans= 5364
        elif [nu1,nu2]==[857,545] or [nu1,nu2]==[545,857]:
            ans= 2702
        elif [nu1,nu2]==[857,353] or [nu1,nu2]==[353,857]:
            ans= 953
        
        elif [nu1,nu2]==[545,545]:
            ans= 1690
        elif [nu1,nu2]==[545,353] or [nu1,nu2]==[353,545]:
            ans= 626
        
        elif [nu1,nu2]==[353,353] :
            ans= 262        
        else:
            ans= 0
        return ans*1/conversion(nu1)*1/conversion(nu2)
experiment="Planck"
def Scut(nu):
       #   experiment="Planck"
          if experiment=="Planck":
              fluxcuts=np.array([400,350,225,315,350,710,1000])*1e-3
              frequencies=[100,143,217,353,545,857,3000] # in gHz!!
        
              if nu in frequencies:
                  return fluxcuts[frequencies.index(nu)]
          elif experiment=="Websky":
                  return 400*1e-3
          elif experiment=="Ccatprime":
              frequencies=[220,280,350,410,850,3000]
              fluxcuts=np.array([225,300,315,350,710,1000])*1e-3
              if nu in frequencies:
                  return fluxcuts[frequencies.index(nu)]
   
def prob(dummys,logexpectation_s,sigma):
        return 1/np.sqrt((2*np.pi*sigma**2))*np.exp(-(dummys[:,np.newaxis,np.newaxis]-logexpectation_s)**2/(2*sigma**2))
              
def dndlns(halomodel,dummys,logexpectation_s,sigma):
        mhalos=np.exp(halomodel.lnms)
        nfn=halomodel.nfn[np.newaxis,mhalos>Mmin]
        p=prob(dummys,logexpectation_s,sigma)
        integrand=nfn*p
      
        return np.trapz(integrand,mhalos[mhalos>Mmin],axis=1) 
    
def shot_noise(nu,sigma,fluxes,zs,halomodel):
        chis=csm.chi_from_z(zs)
        
        fluxes[fluxes==0]=1e-100
        logfluxes=np.log(fluxes)
        dummylogs=np.linspace(np.min(logfluxes[logfluxes>-200])-0.5,min(Scut(nu),100),200)
        dnds=dndlns(halomodel,dummylogs,logfluxes,sigma)
       
         
        
        return np.trapz(chis**2*(np.trapz(dnds*np.exp(dummylogs[:,np.newaxis])**2,dummylogs,axis=0)),chis)
def shot_noise_binned(nu,sigma,fluxes,zs,halomodel,zmin,zmax):
    
        chis=csm.chi_from_z(zs)
        
        fluxes[fluxes==0]=1e-100
        logfluxes=np.log(fluxes)
        dummylogs=np.linspace(np.min(logfluxes[logfluxes>-200])-0.5,min(Scut(nu),100),200)
        dnds=dndlns(halomodel,dummylogs,logfluxes,sigma)
       
        integrand = chis**2*(np.trapz(dnds*np.exp(dummylogs[:,np.newaxis])**2,dummylogs,axis=0))
        integrand[zs<zmin]=0
        integrand[zs<zmax]=0

        return np.trapz(integrand,chis)


