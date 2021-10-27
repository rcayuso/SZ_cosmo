#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:24:21 2019

@author: fionamccarthy
"""

from __future__ import print_function
from __future__ import absolute_import
import numpy as np

#cosmology module
import kszpsz_config
import cosmology
csm = cosmology.cosmology(basic_conf_obj = kszpsz_config)

planck=6.626e-34
lightspeed=2.99792458e8
kBoltzmann=1.38e-23
     


T0=20.7
alpha=0.2
beta=1.6
delta=2.4
sigmasqlm=0.3
Meff=10**12.3
Mmin=10**10.1#1470588235294.1174#10**10.1
#Mmin=1470588235294.1174#10**10.1

t4 = True

if not t4:
    L0=6.725699999999999e-08*1.35e-5#5.2313668860490*1.35e-5*1e-5/10*6
else:
    L0= 0.004755788078226362*1000*1.35e-5#*2#6.015648958508135*1.35e-5#1.35e-5*0.004755788078226362#5.2313668860490*1.35e-5
#1131394909046.434*1.35e-5*0.9
zplateau=2


def TD(z): #Dust temperature of starforming galaxyT
            return T0*(1+z)**alpha



def Planck(nu,T): #Planck law for black body radiation
    return 2*(nu)**3*(1/(np.exp(planck*nu/(kBoltzmann*T))-1))*planck/lightspeed**2

def SED(nu,T,z):
    #eq 8 of 1611.04517
      
            nu=nu*1e9 #multiply by 10^^9 to go from GHz->Hz
            SED=np.zeros(nu.shape)
          #  nu02s=353*1e9*(1+z)
            if t4:
                SED=(nu/T)**beta*Planck(nu,T)*1/T**4
            else:
                SED=(nu**beta)*Planck(nu,T)
            return SED
        
        
def redshiftevolutionofl(z):
            
            answer=np.zeros(len(z))
            
            answer=(1+z)**delta
            if len(answer[z>=zplateau])>0:
                answer[z>=zplateau]=(1+zplateau)**delta
            return answer
def Sigma(M):
    
          
            answer= M*(1/(2*np.pi*sigmasqlm)**(1/2))*np.exp(-(np.log10(M)-np.log10(Meff))**2/(2*sigmasqlm))
            answer[M<Mmin]=0
            return answer
def Lnu(nu,z,M): #spectral luminosity radiance
            Lir=L0#*1.35e-5 #total IR luminosity
            sed=SED(nu*(1+z),TD(z),z) #normalised such that the integral over all nu is 1.
            
            ans =  Lir*sed*Sigma(M)*redshiftevolutionofl(z)
            return ans
        

def Scentral(nu,z,Mhalo):
    
    #flux from luminosity; eg eq 7 of 1611.04517
        chi=csm.chi_from_z(z)
        answer=Lnu(nu,z,Mhalo)/((4*np.pi)*chi**2*(1+z)) # in units of [Lnu]/Mpc**2=solar_luminosity/Mpc**2/Hz
        answer[answer>Scut(nu)]=0
        return answer

def Luminosity_from_flux(S,z):
    #gives luminosity in [S] * Mpc**2
    return  4 * np.pi * csm.chi_from_z(z)**2*(1+z)*S

def subhalo_mass_function(Msub,Mhost):
            jiang_gamma_1  = 0.13
            jiang_alpha_1   = -0.83
            jiang_gamma_2   = 1.33
            jiang_alpha_2 = -0.02
            jiang_beta_2     = 5.67
            jiang_zeta      = 1.19
            return (((jiang_gamma_1*((Msub/Mhost)**jiang_alpha_1))+
             (jiang_gamma_2*((Msub/Mhost)**jiang_alpha_2)))*
             (np.exp(-(jiang_beta_2)*((Msub/Mhost)**jiang_zeta))))
        

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
experiment="Websky" 
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
        nfn=halomodel.nfn[np.newaxis]
        p=prob(dummys,logexpectation_s,sigma)
        integrand=nfn*p
      
        return np.trapz(integrand,mhalos,axis=1) 
    
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











