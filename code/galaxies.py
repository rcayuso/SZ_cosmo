#different galaxy distributions for shot noise calculation and comparison to other papers
#run: python3 galaxies.py

import os
import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate 
import matplotlib.pyplot as plt
import cosmology
import halomodel

csm = cosmology.cosmology()

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

allsky_squaredeg = 4*np.pi*(180.**2./np.pi**2.)
allsky_arcmin2 = 4*np.pi*(10800/np.pi)**2
allsky_sterad = 4*np.pi

z_min = 0.001
z_max = 7.
z_nr = 10000
z_sampling = np.linspace(z_min,z_max,z_nr)


def get_bins_lsst_mat():
    zbins_z = np.array( [0.,0.4,1.,1.6,2.2,3] )
    zbins_bias = np.array( [1.0588,1.3718,1.7882,2.2232,2.7408] )
    zbins_ngal= np.array( [0.051572,0.020273,0.005947,0.001522,0.000301] )
    zbins_nr = 5
    return zbins_z,zbins_bias,zbins_ngal,zbins_nr

def get_mthreshHOD_lsst(zs_bins):
    fname = dir_path + "data/mthreshHOD_lsst_new.txt"
    dt = ({ 'names' : ('zs', 'mmin'),'formats' : [np.float, np.float] })
    data = np.loadtxt(fname,dtype=dt)
    zs_matching = data['zs']
    mmins_matching = data['mmin']
    mmins_bins = np.zeros( zs_bins.shape )
    for z_id,z in enumerate(zs_bins):  #just take nearest z, no interpolation needed
        zidx = (np.abs(zs_matching-zs_bins[z_id])).argmin()
        mmins_bins[z_id] = mmins_matching[zidx]
        #print ("z idx", zidx)
    return mmins_bins

def get_mmin_lsst_halogalmatching(zs_bins):
    fname = dir_path + "data/mminhalo_lsst_halogalmatching.txt"
    dt = ({ 'names' : ('zs', 'mmin'),'formats' : [np.float, np.float] })
    data = np.loadtxt(fname,dtype=dt)
    zs_matching = data['zs']
    mmins_matching = data['mmin']
    mmins_bins = np.zeros( zs_bins.shape )
    for z_id,z in enumerate(zs_bins):  #just take nearest z, no interpolation needed
        zidx = (np.abs(zs_matching-zs_bins[z_id])).argmin()
        mmins_bins[z_id] = mmins_matching[zidx]
        #print ("z idx", zidx)
    return mmins_bins

def getmthreshHODstellar(LSSexperiment,zbins_central): #HOD threshold for a given experiment experiment as a function of red shift.
    if LSSexperiment == 'ngalsMpc3Fixed': #NOTE: when you use the halo model you should not use this mode, because ngal and mthresh are not independent
        mthreshHODstellar = 10.5*np.ones(zbins_central.shape[0])   #9.3 is an average value in the ballpark of LSST.
    if LSSexperiment == 'LSST':
        # m = (9.6-8.9)/(3.0-0.5) #rough fit by eye to match to LSST red shift dependence (see functions below). TODO: improve
        # b = 9.6-m*3.0
        # mthreshHODstellar = m*zbins_central+b
        mthreshHODstellar = get_mthreshHOD_lsst(zbins_central)
    return mthreshHODstellar

#convert dN/dV to dN/dOmega/dz. n_arcmin and z must be same length and matching z. 
def convert_n_mpc3_arcmin2(n_mpc3,z):
    dz = 0.01
    zmax = z+dz/2.
    zmin = z-dz/2.
    chimax = csm.chi_from_z(zmax)
    chimin = csm.chi_from_z(zmin)
    dV_shell_comov = 4./3. * np.pi * ((chimax)**3. - (chimin)**3.)
    dV_dz = dV_shell_comov/dz
    dV_dZdOmega = dV_dz/allsky_arcmin2
    n_arcmin2 = n_mpc3 * dV_dZdOmega
    return n_arcmin2


#convert dN/dOmega/dz to dN/dV. n_arcmin and z must be same length and matching z. 
def convert_n_arcmin2_mpc3(n_arcmin2,z):
    dz = 0.01
    zmax = z+dz/2.
    zmin = z-dz/2.
    chimax = csm.chi_from_z(zmax)
    chimin = csm.chi_from_z(zmin)
    dV_shell_comov = 4./3. * np.pi * ((chimax)**3. - (chimin)**3.)
    dV_dz = dV_shell_comov/dz
    dV_dZdOmega = dV_dz/allsky_arcmin2
    n_mpc3 = n_arcmin2 / dV_dZdOmega
    return n_mpc3


#LSST whitepaper 0912.0201 13.10
def n_arcmin2_LSST_Wittman(z):
    n_arcmin2 = z**2. * np.exp(-(z/0.5)**1.)
    #n integrates to 0.25
    norm = 50./0.25
    n_arcmin2 *= norm
    return n_arcmin2


#LSST whitepaper 0912.0201 3.8, and email from simone
def n_arcmin2_LSST_goldsample(z):
    z0 = 0.3
    n_arcmin2 = (1./(2.*z0)) * (z/z0)**2. * np.exp(-z/z0)
    norm = 40
    n_arcmin2 *= norm
    return n_arcmin2


#get n_arcmin2 at red shitfs z
def n_arcmin2_LSST_Schmittfull(z):
    dt = ({ 'names' : ('z', 'n'),'formats' : [np.float, np.float] })
    data = np.loadtxt(fname=dir_path + "data/n_schmittfull.txt",dtype=dt)
    #interpolate to the redshifts we want
    lin_interp = interp1d(data['z'],data['n'],bounds_error=False,fill_value=0)
    n_arcmin2 = lin_interp(z)
     
    #plot it
    # fig=plt.figure(figsize=(6,4))
    # ax1 = fig.add_subplot(111)
    # ax1.plot(z,n_arcmin2,color='blue',ls='solid',label=r'$n$ (LSST, Schmittfull)')
    # ax1.plot(z,n_arcmin2_LSST_Wittman(z),color='red',ls='solid',label=r'$n$ (LSSTScience)')
    # ax1.set_yscale('log')
    # plt.legend(loc=1,frameon=False)
    # fig.tight_layout()
    # plt.show()
    # fig.savefig(dir_path+'data/n_LSST.pdf')

    return n_arcmin2


def bias_LSST_Schmittfull(z):
    return 1+z
    

#integrate number density in an array of z over angle and z
def N_tot_angular(n_arcmin2,z_min,z_max,surveyarea_sqdeg): #n_arcmin2 is an array over z
    #f_interp = interp1d(n_arcmin2,z_sampling,bounds_error=False,fill_value=0) #somehow this one does not seem to work for integration
    #f_interp = lambda x: np.interp(x, z_sampling, n_arcmin2) #seems to work but gives a warning
    #N = scipy.integrate.quad(f_interp,z_min,z_max)[0]

    zmin_id = (np.abs(z_sampling - z_min)).argmin()
    zmax_id = (np.abs(z_sampling - z_max)).argmin()
    N = integrate.simps(n_arcmin2[zmin_id:zmax_id],z_sampling[zmin_id:zmax_id])
    
    N *= surveyarea_sqdeg * allsky_arcmin2/allsky_squaredeg
    print ("N_tot",N)
    return N


def survey_volume_mpc3(surveyarea_sqdeg,zmin,zmax):
    chimax = csm.chi_from_z(zmax)
    chimin = csm.chi_from_z(zmin)
    svolume_mpc3 = 4./3. * np.pi * ((chimax)**3. - (chimin)**3.)
    svolume_mpc3 *= (surveyarea_sqdeg/allsky_squaredeg)
    print ("Survey volume Gpc-3", svolume_mpc3*1e-9)
    return svolume_mpc3
    
    
def N_tot_to_n_mpc3(Ntot,surveyvolume_mpc3):
    n = Ntot/surveyvolume_mpc3
    print ("From Ntot and V: density n Mpc-3", n)
    return n


#integrate number density over volume
def N_tot_volume(n_mpc3,surveyvolume_mpc3): #n_mpc3 is a scalar
    return n_mpc3*surveyvolume_mpc3



def match_ngal_mminhalo_halogalmatch():
    #test red shifts where we match mmin and ngal
    #zs_matching = np.arange(0.1,6.1,0.1) 
    #zs_matching = np.arange(0.4,4.1,0.1) 
    #zs_matching = np.array([0.5,1.5,3.,4.5])
    #zs_matching_nr = zs_matching.shape[0]

    zbins_z,zbins_bias,zbins_ngal,zbins_nr = get_bins_lsst_mat()
    zs_matching_nr = zbins_nr
    zs_matching = zbins_z[:-1] + np.diff(zbins_z)/2.
    
    print("z for matches:", zs_matching)

    #calc ngal_mpc_lsst from ngal_arcmin2_lsst at z_central
    #ngal_arcmin2_lsst_z = n_arcmin2_LSST_Schmittfull(zs_matching)
    #ngal_arcmin2_lsst_z = n_arcmin2_LSST_Wittman(zs_matching)
    #ngal_arcmin2_lsst_z = n_arcmin2_LSST_goldsample(zs_matching)
    #ngal_mpc3_lsst_z = convert_n_arcmin2_mpc3(ngal_arcmin2_lsst_z,zs_matching) #convert arcmin to mpc

    ngal_mpc3_lsst_z = zbins_ngal
    
    #------------- get halo model results (single HOD mass threshold, vary the halo mass cut)
    hmod = halomodel.HaloModel()
    massbins =  np.arange(8.,16.05,0.005) #0.1
    ngalMpc3 = np.zeros((zs_matching_nr,massbins.shape[0]))
    nhaloMpc3 = np.zeros((zs_matching_nr,massbins.shape[0]))
    galaxybias = np.zeros((zs_matching_nr,massbins.shape[0]))
    halobias = np.zeros((zs_matching_nr, massbins.shape[0]))
    #grid calc ngal_mpc_halomodel for different mmin.
    for m_id,m in enumerate(massbins[:-1]):
        log10mlow = massbins[m_id]
        log10mhigh = massbins[-1]
        #ngalMpc3[:,m_id] = hmod.nbar_galaxy(zs_matching,log10mlow,log10mhigh) #gives [z,m]
        #galaxybias[:,m_id] = hmod.bias_galaxy(zs_matching,log10mlow,log10mhigh)
        nhaloMpc3[:,m_id] = hmod.nbar_halo(zs_matching,log10mlow,log10mhigh)
        halobias[:,m_id] = hmod.bias_halo(zs_matching,log10mlow,log10mhigh)

    #for each z plot ngal_mpc3_halo(mmin) and a constant ngal_mpc3_LSST
    # fig=plt.figure(figsize=(7,5))
    # ax1 = fig.add_subplot(111)
    # for z_id,z in enumerate(zs_matching):
    #     #ax1.plot(massbins,ngalMpc3[z_id],ls='solid',label=r'ngal_hmod at z='+str(z),color="C"+str(z_id))
    #     ax1.plot(massbins,nhaloMpc3[z_id],ls='dotted',label=r'nhalo_hmod at z='+str(z),color="C"+str(z_id))
    #     #plot the constant from LSST
    #     ax1.axhline(y=ngal_mpc3_lsst_z[z_id],ls='dashed',label=r'ngal_LSST at z='+str(z),color="C"+str(z_id))
    # ax1.set_yscale('log')
    # plt.ylim([1e-10,1e2])
    # plt.legend(loc='lower left',frameon=False)
    # plt.grid()
    # plt.xlabel(r'$M_{\rm min} \ [M_\bigodot]$')
    # ax1.set_ylabel(r'$\bar{n}$ [Mpc]$^{-3}$')
    # fig.tight_layout()
    # plt.show()
    # fig.savefig(dir_path+'data/ngal_lsst_halogalmatching.pdf') #lsst schmittfull

    
    #----------- find the mmin for which the curves and the constant cross.
    mmin_z = np.zeros(zs_matching_nr)
    mmin_id = np.zeros(zs_matching_nr, dtype = np.int)
    for z_id,z in enumerate(zs_matching): #TODO: should be done better, esp for sparse mass binning. 
        mmin_id[z_id] = (np.abs(nhaloMpc3[z_id]-ngal_mpc3_lsst_z[z_id])).argmin()
        #mmin_id[z_id] = (np.abs(ngalMpc3[z_id]-ngal_mpc3_lsst_z[z_id])).argmin()
        mmin_z[z_id] = massbins[mmin_id[z_id]]
    print("mmin matches:", mmin_z)
    #plot
    fig=plt.figure(figsize=(4.5,3))
    ax1 = fig.add_subplot(111)
    ax1.plot(zs_matching,10**mmin_z,ls='solid',label=r'mmin',marker="o")
    #plt.legend(loc='lower left',frameon=False)
    ax1.set_yscale('log')
    plt.grid()
    plt.xlabel(r'z')
    plt.ylabel(r'$M_{\rm min} \ [M_\bigodot]$')
    fig.tight_layout()
    plt.show()    
    fig.savefig(dir_path+'data/mminhalo_lsst_halogalmatching_z4.pdf') 
    #write to file
    fname = dir_path + 'data/mminhalo_lsst_halogalmatching.txt'
    f = open(fname,"w")
    for i in range(zs_matching.shape[0]):
        f.write(str(zs_matching[i]) + " " + str(mmin_z[i]) + "\n")
    f.close()
    
    #now compare bgal_lsst and bgal_halomodel
    #calc bias at these z and mmin from halomodel and compare to lsst prescription
    #bias_lsst = bias_LSST_Schmittfull(zs_matching)
    bias_lsst = zbins_bias
    bias_halomodel = np.zeros(zs_matching_nr)
    dm = 0.1
    for z_id,z in enumerate(zs_matching):   
        bias_halomodel[z_id] = halobias[z_id,mmin_id[z_id]] #galaxybias[z_id,mmin_id[z_id]] 
    #plot
    fig=plt.figure(figsize=(4.5,3))
    ax1 = fig.add_subplot(111)
    ax1.plot(zs_matching,bias_lsst,ls='solid',label=r'b_lsst',marker="o")
    ax1.plot(zs_matching,bias_halomodel,ls='solid',label=r'b_halomodel',marker="o")
    plt.legend(loc='lower left',frameon=False)
    plt.grid()
    plt.xlabel(r'z')
    ax1.set_ylabel(r'$b_g$')
    fig.tight_layout()
    plt.show()    
    fig.savefig(dir_path+'data/bias_lsst_abundancematching.pdf')    


    



def match_ngal_mthreshHOD():
    
    #FOR pszrcode
    #test red shifts where we match mmin and ngal
    zs_matching = np.arange(0.1,6.1,0.1) 
    #zs_matching = np.arange(0.1,4.2,0.5) 
    #zs_matching = np.array([0.5,1.5,3.,4.5])
    zs_matching_nr = zs_matching.shape[0]
    print("z for matches:", zs_matching)
    #calc ngal_mpc_lsst from ngal_arcmin2_lsst at z_central
    #ngal_arcmin2_lsst_z = n_arcmin2_LSST_Schmittfull(zs_matching)
    #ngal_arcmin2_lsst_z = n_arcmin2_LSST_Wittman(zs_matching)
    ngal_arcmin2_lsst_z = n_arcmin2_LSST_goldsample(zs_matching)
    ngal_mpc3_lsst_z = convert_n_arcmin2_mpc3(ngal_arcmin2_lsst_z,zs_matching)

    #FOR 3d box code
    #zbins_z,zbins_bias,zbins_ngal,zbins_nr = get_bins_lsst_mat()
    #zs_matching_nr = zbins_nr
    #zs_matching = zbins_z[:-1] + np.diff(zbins_z)/2.
    #ngal_mpc3_lsst_z = zbins_ngal
    
    
    #----------------- get halo model results (vary HOD mass threshold, integrate over all halo mass)
    hmod = halomodel.HaloModel()
    massbins =  np.arange(5.,16.05,0.005) #
    ngalMpc3 = np.zeros((zs_matching_nr,massbins.shape[0]))
    nhaloMpc3 = np.zeros((zs_matching_nr,massbins.shape[0]))
    galaxybias = np.zeros((zs_matching_nr,massbins.shape[0]))
    halobias = np.zeros((zs_matching_nr, massbins.shape[0]))
    #grid calc ngal_mpc_halomodel for different mthresh.
    for m_id,m in enumerate(massbins[:-1]):
        log10mlow = 5. #fixed
        log10mhigh = 16. #fixed
        mthreshHOD = m*np.ones(zs_matching_nr) #varied
        ngalMpc3[:,m_id] = hmod.nbar_galaxy(zs_matching,log10mlow,log10mhigh,mthreshHOD) #gives [z,m]
        galaxybias[:,m_id] = hmod.bias_galaxy(zs_matching,log10mlow,log10mhigh,mthreshHOD)
   
    # fig=plt.figure(figsize=(7,5))
    # ax1 = fig.add_subplot(111)
    # for z_id,z in enumerate(zs_matching):
    #     ax1.plot(massbins,ngalMpc3[z_id],ls='solid',label=r'ngal_hmod at z='+str(z),color="C"+str(z_id))
    #     #plot the constant from LSST
    #     ax1.axhline(y=ngal_mpc3_lsst_z[z_id],ls='dashed',label=r'ngal_LSST at z='+str(z),color="C"+str(z_id))
    # ax1.set_yscale('log')
    # plt.ylim([1e-10,1e2])
    # plt.legend(loc='lower left',frameon=False)
    # plt.grid()
    # plt.xlabel(r'$M_{\rm threshHOD} \ [M_\bigodot]$')
    # ax1.set_ylabel(r'$\bar{n}$ [Mpc]$^{-3}$')
    # fig.tight_layout()
    # plt.show()
    # #fig.savefig(dir_path+'data/ngal_lsst_mthreshHOD.pdf')
        

    #----------- find the mmin for which the curves and the constant cross.
    mmin_z = np.zeros(zs_matching_nr)
    mmin_id = np.zeros(zs_matching_nr, dtype = np.int)
    for z_id,z in enumerate(zs_matching): #TODO: shuld be done better, esp for sparse mass binning. 
        #mmin_id[z_id] = (np.abs(nhaloMpc3[z_id]-ngal_mpc3_lsst_z[z_id])).argmin()
        mmin_id[z_id] = (np.abs(ngalMpc3[z_id]-ngal_mpc3_lsst_z[z_id])).argmin()
        mmin_z[z_id] = massbins[mmin_id[z_id]]
    print("mmin matches:", mmin_z)
    #plot
    fig=plt.figure(figsize=(4.5,3))
    ax1 = fig.add_subplot(111)
    ax1.plot(zs_matching,10.**mmin_z,ls='solid',label=r'mmin')
    plt.legend(loc='lower left',frameon=False)
    ax1.set_yscale('log')

    #plot rs ranges
    fname_base = "fnl_zbin_lsst_z4_withphotoz_5bins" #"fnl_zbin_lsst_z4_photoz"
    fname = "/Users/mmunchmeyer/Work/physics/gitpapers/ksz_tomography/python/plots/"+fname_base+".txt"  
    dt = ({ 'names' : ('z', 'zmin', 'zmax', 'bin_V_gpc', 'fnl_gv', 'fnl_g'),'formats' : [np.float, np.float, np.float, np.float, np.float, np.float] })
    data = np.loadtxt(fname,dtype=dt)
    for z_id in [0,2,4]:
        ax1.axvspan(data['zmin'][z_id],data['zmax'][z_id], alpha=0.5)
    
    plt.grid()
    plt.xlabel(r'z')
    plt.ylabel(r'$m^{\rm thresh}_{\star} \ [m_\bigodot]$')
    fig.tight_layout()
    plt.show()    
    #fig.savefig(dir_path+'data/mstarHOD_lsst_matching.pdf')
    #write to file
    #fname = dir_path+'data/mminHOD_lsst_gold.txt' #FOR 3d box code
    fname = dir_path+"data/mthreshHOD_lsst.txt" #FOR pszrcode
    f = open(fname,"w")
    for i in range(zs_matching.shape[0]):
        f.write(str(zs_matching[i]) + " " + str(mmin_z[i]) + "\n")
    f.close()


    
    #now compare bgal_lsst and bgal_halomodel
    #calc bias at these z and mmin from halomodel and compare to lsst prescription
    #bias_lsst = bias_LSST_Schmittfull(zs_matching)
    bias_halomodel = np.zeros(zs_matching_nr)
    bias_lsst = zbins_bias
    for z_id,z in enumerate(zs_matching):   
        bias_halomodel[z_id] = galaxybias[z_id,mmin_id[z_id]] #halobias[z_id,mmin_id[z_id]]
    #plot
    fig=plt.figure(figsize=(4.5,3))
    ax1 = fig.add_subplot(111)
    ax1.plot(zs_matching,bias_lsst,ls='solid',label=r'$b_g$ (LSST)',marker="o")
    ax1.plot(zs_matching,bias_halomodel,ls='solid',label=r'$b_g$ (halo model)',marker="o")
    plt.legend(loc='upper left',frameon=False)


    #plot rs ranges
    fname_base = "fnl_zbin_lsst_z4_withphotoz_5bins" #"fnl_zbin_lsst_z4_photoz"
    fname = "/Users/mmunchmeyer/Work/physics/gitpapers/ksz_tomography/python/plots/"+fname_base+".txt"  
    dt = ({ 'names' : ('z', 'zmin', 'zmax', 'bin_V_gpc', 'fnl_gv', 'fnl_g'),'formats' : [np.float, np.float, np.float, np.float, np.float, np.float] })
    data = np.loadtxt(fname,dtype=dt)
    for z_id in [0,2,4]:
        ax1.axvspan(data['zmin'][z_id],data['zmax'][z_id], alpha=0.5)
        
    plt.grid()
    plt.xlabel(r'z')
    ax1.set_ylabel(r'$b_g$')
    fig.tight_layout()
    plt.show()    
    #fig.savefig(dir_path+'data/bias_lsst_mthreshHODmatching.pdf')    



    

def main():
    #get number density differential in z and angle as function of z
    #n_arcmin2_zsampling = n_arcmin2_LSST_goldsample(z_sampling) #n_arcmin2_LSST_Schmittfull(z_sampling)

    #convert number density
    # zs = np.array([1.])
    # ngal_arcmin2_lsst_z = n_arcmin2_LSST_goldsample(zs)
    # ngal_mpc3_lsst_z = convert_n_arcmin2_mpc3(ngal_arcmin2_lsst_z,zs)
    # print ("ngal_mpc3",ngal_mpc3_lsst_z)
    
    #check the total gal number compared to schmittfull paper
    # z_min = 0.5
    # z_max = 1.0
    # surveyarea_sqdeg = 18000. #allsky_squaredeg
    # N_tot = N_tot_angular(n_arcmin2_zsampling,z_min,z_max,surveyarea_sqdeg)

    #compare lsst and halo model
    match_ngal_mthreshHOD()
    #match_ngal_mminhalo_halogalmatch()

    #calc spatial density from ntot and survey properties
    # svolume_mpc3 = survey_volume_mpc3(surveyarea_sqdeg,z_min,z_max)
    # n_mpc3 = N_tot_to_n_mpc3(N_tot,svolume_mpc3)

    #calc spatial density from angular density
    # zz = 0.75
    # n = convert_n_arcmin2_mpc3(n_arcmin2_LSST_goldsample(zz),zz)
    # print ("From n_arcmin2: density n Mpc-3", n)


    
if __name__ == "__main__":
    main()
