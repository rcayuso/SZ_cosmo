import numpy as np
import kszpsz_config as conf
import camb
from camb import model
from scipy.interpolate import interp1d


class cosmology(object):

    def __init__(self, basic_conf_obj = None) :
        if not basic_conf_obj:
            basic_conf_obj = conf
            
        self._basic_conf_obj = basic_conf_obj
        
        self.cambpars = camb.CAMBparams()
        self.cambpars.set_cosmology(H0 = self._basic_conf_obj.h*100, ombh2=self._basic_conf_obj.ombh2, \
                                    omch2=self._basic_conf_obj.omch2, mnu=self._basic_conf_obj.mnu , \
                                    omk=self._basic_conf_obj.Omega_K, tau=self._basic_conf_obj.tau,  \
                                    TCMB =conf.T_CMB/1e6 )
        self.cambpars.InitPower.set_params(As =self._basic_conf_obj.As*1e-9 ,ns=self._basic_conf_obj.ns, r=0)
        self.cambpars.NonLinear = model.NonLinear_both
        self.cambpars.max_eta_k = 8000.0*self._basic_conf_obj.k_max
        self.data = camb.get_background(self.cambpars)
        self.derived_params_dict = self.data.get_derived_params()
        self.zdec = self.derived_params_dict['zstar']
        self.interp_f_growth =  self.f_growth_interp()
        self.interp_H_z =  self.H_z_interp()
        
        
    def aeq(self,):
        """Scale factor at matter radiation equality"""
        zeq = self.derived_params_dict['zeq']
        return 1.0/(1.0+zeq)

    def f_growth_interp(self,):
        z_sample = np.linspace(0,10,1000)
        f = self.data.get_redshift_evolution(1e-5,z_sample,['growth'])[...,0]
        return interp1d(z_sample, f, kind = 'linear', bounds_error=False,fill_value=0)

    def f_growth(self, z):
        return self.interp_f_growth(z)
        
    
    def H_z_interp(self):
        """Hubble rate at z"""
        z_sample = np.linspace(0,10,1000)
        H = self.data.get_redshift_evolution(0.0,z_sample,['H'])[...,0]*(1.0+z_sample)    
        return  interp1d(z_sample, H, kind = 'linear',bounds_error=False,fill_value=0)

    def H_z(self, z):
        return self.interp_H_z(z)
    
    def chi_from_z(self, z): 
        return self.data.comoving_radial_distance(z)
    
    def z_from_chi(self,chi):
        return self.data.redshift_at_comoving_radial_distance(chi)
    
    def H0(self,h):
        """Hubble parameter today in Mpc**-1"""
        return 100.0 * h / (3.0 * 1.0e5)
    
    def camb_Pk_nonlin(self,k,z):
        
        self.cambpars.set_matter_power(redshifts=z.tolist(), kmax=k[-1],k_per_logint=20)
        PK_nonlin = camb.get_matter_power_interpolator(self.cambpars, nonlinear=True,hubble_units=False, k_hunit=False, kmax=k[-1], zmax=z[-1])
        return PK_nonlin.P(z, k, grid=True)
    
    def camb_Pk_lin(self,k,z):
        
        self.cambpars.set_matter_power(redshifts=z.tolist(), kmax=k[-1],k_per_logint=20)
        PK_lin = camb.get_matter_power_interpolator(self.cambpars, nonlinear=False,hubble_units=False, k_hunit=False, kmax=k[-1], zmax=z[-1])
        return PK_lin.P(z, k, grid=True)
        
        
        
        
        
        
        
        
        
        