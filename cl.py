# Script to calculate the angular power spectrum of redshift drift fluctuations

import matplotlib.pyplot as plt
import numpy as np 
import time

from classy import Class 
from functools import lru_cache
from fftlogx import fftlog

import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import spherical_jn

c = 299792.458 #speed of light in km/s

OBSERVER_TERMS = True #should the H_0 term be inlcuded in the def. of the redshift drift
l_max = 1000

#parameters for CLASS 
params = {
    'A_s': 2.215e-9,
    'n_s': 0.9619,
    'k_pivot': 0.05,           # in units of inverse Mpc (not h/Mpc!)
    'h': 0.67556,
    'omega_b': 0.022032,
    'omega_cdm': 0.12038,
    'T_cmb': 2.7255,           # in units of K
    'N_ur': 3.046,
    'output': 'mPk',           # Specify to compute the matter power spectrum
    'P_k_max_1/Mpc': 1000,     # Maximum k (in 1/Mpc) for which P(k) is calculated
    'z_pk': 0.0                # Redshift at which to calculate the power spectrum
}

#intialize CLASS 
cosmo = Class()
cosmo.set(params)
cosmo.compute()

def a(z): 
    return 1/(1+z)

def H(z): 
    return cosmo.Hubble(z)*c

def Dplus(z): 
    return cosmo.scale_independent_growth_factor(z)

def f(z):
    return cosmo.scale_independent_growth_factor_f(z)

def df_da(z): 
    eps = 1e-5
    a_plus = np.minimum(a(z)+eps,1.0)
    a_minus = a(z) - eps 
    z_plus = 1/a_plus-1
    z_minus = 1/a_minus-1
    return (f(z_plus)-f(z_minus))/(a_plus-a_minus)

def transfer(z): 
    if (OBSERVER_TERMS):
        return Dplus(z)*H(z)*a(z)*f(z)*(1+f(z)+a(z)/f(z)*df_da(z))*H(z)/(H(0)*(1+z)-H(z))
    else: 
        return Dplus(z)*H(z)*a(z)*f(z)*(1+f(z)+a(z)/f(z)*df_da(z))

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def tophat(z, z_mean, dz):
    mask = (z <= z_mean + dz) & (z >= z_mean - dz)
    return np.where(mask, 1 / (2 * dz), 0)


@lru_cache(maxsize=None)
def window(z_mean, dz, type='tophat'):

    z_min = 0.01
    z_max = z_mean+1

    z = np.linspace(z_min,z_max,10000)

    #get co-moving distance and determinant from CLASS
    r, dzdr = cosmo.z_of_r(z)
    r_min = r.min()
    r_max = r.max()

    #interpolate, since we need to be able to call z(r) and dzdr(r)
    z_of_r = interp1d(r, z, kind='cubic')
    dzdr_of_r = interp1d(r, dzdr, kind='cubic')

    #r-values for the integration
    x = np.geomspace(r_min,r_max,10000)

    if (type=='tophat'): 
        f_of_x = dzdr_of_r(x)*transfer(z_of_r(x))*tophat(z_of_r(x),z_mean,dz)

    if (type=='gauss'): 
        f_of_x = dzdr_of_r(x)*transfer(z_of_r(x))*gaussian(z_of_r(x),z_mean,dz)

    #initialise FFTlog
    # -> x*f_of_x to cancel a factor 1/x in the definition of the integrand in FFTlog
    # -> nu=1.01 is the recommended value to avoid some numerical issues with nu=1
    # -> N_pad helps stabilize the integral 
    myfftlog = fftlog(x,x*f_of_x,nu=1.01,N_pad=5000) 


    # calculate the integral over r for each l 
    k = [None]
    result = [None]

    for l in range(1,l_max+1):
        k_i, result_i = myfftlog.fftlog_dj(l)
        k.append(k_i)
        result.append(result_i)
    
    return k, result


def Cl(l,z,dz=0.01,type='tophat'):

    if (type=='delta'): 
        k = np.geomspace(1e-5, 1e1, 10000)
        r = cosmo.comoving_distance(z)
        inner_int = transfer(z)*spherical_jn(l,k*r,True)
        Pk = [cosmo.pk(i,0) for i in k]

        #calculate final outer integral using simpsons rule
        integral = integrate.simpson(inner_int**2*Pk,x=k)

    else:

        k, inner_int = window(z,dz,type)

        Pk = [cosmo.pk(i, 0.0) for i in k[l]]

        #calculate final outer integral using simpsons rule
        integral = integrate.simpson(inner_int[l]**2*Pk,x=k[l])

    return 2/np.pi*integral/c**2    #factor c**2 to fix units


l = np.arange(1,l_max+1)

#compute C_l's for a tophat filter and different window sizes

CL1 = np.array([Cl(i,0.1,0.01)*i*(i+1)/2/np.pi for i in l])
CL2 = np.array([Cl(i,0.2,0.01)*i*(i+1)/2/np.pi for i in l])
CL3 = np.array([Cl(i,0.3,0.01)*i*(i+1)/2/np.pi for i in l])
np.savetxt('redshiftdrift_ps_theory_dz0.01.d',np.column_stack((l, CL1, CL2, CL3)),
         header='l\tl(l+1)Cl/2pi')

CL1 = np.array([Cl(i,0.1,0.001)*i*(i+1)/2/np.pi for i in l])
CL2 = np.array([Cl(i,0.2,0.001)*i*(i+1)/2/np.pi for i in l])
CL3 = np.array([Cl(i,0.3,0.001)*i*(i+1)/2/np.pi for i in l])
np.savetxt('redshiftdrift_ps_theory_dz0.001.d',np.column_stack((l, CL1, CL2, CL3)),
         header='l\tl(l+1)Cl/2pi')

CL1 = np.array([Cl(i,0.1,0.02)*i*(i+1)/2/np.pi for i in l])
CL2 = np.array([Cl(i,0.2,0.02)*i*(i+1)/2/np.pi for i in l])
CL3 = np.array([Cl(i,0.3,0.02)*i*(i+1)/2/np.pi for i in l])
np.savetxt('redshiftdrift_ps_theory_dz0.02.d',np.column_stack((l, CL1, CL2, CL3)),
         header='l\tl(l+1)Cl/2pi')


#compute C_l's for comparison with arXiv:2306.13911

OBSERVER_TERMS = False
CL1 = np.array([Cl(i,0.1,0.01,'gauss')*i*(i+1)/2/np.pi for i in l])
CL2 = np.array([Cl(i,0.5,0.01,'gauss')*i*(i+1)/2/np.pi for i in l])
CL3 = np.array([Cl(i,1,0.01,'gauss')*i*(i+1)/2/np.pi for i in l])
CL4 = np.array([Cl(i,5,0.01,'gauss')*i*(i+1)/2/np.pi for i in l])
np.savetxt('redshiftdrift_ps_theory_bessa2023.d',np.column_stack((l, CL1, CL2, CL3,CL4)),
         header='l\tl(l+1)Cl/2pi')


# Clean up
cosmo.struct_cleanup()
cosmo.empty()