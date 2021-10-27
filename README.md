# SZ_cosmo

Code for calculation of CMB secondaries (kSZ,tSZ,lensing,moving lens) and CIB, cross-correlations
with galaxy surveys and reconstruction of radial velocity and transverse velocity
on the lightcone.

spectra.py generates Cls
sz_estim2.py  processes the Cls to calculate biases and noise to reconstructed 
velocity fields and provides pipeline for Gaussian simulations.

Not all files are included in the repo, current state is just for looking at the code.

Future update:
-user guide
-fully integrate remote dipole and remote quadrupole reconstructions
(basic functions in remote_spectra.py)