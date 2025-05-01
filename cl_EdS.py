# Script to calculate the angular power spectrum of redshift drift fluctuations
# in an EdS universe

import matplotlib.pyplot as plt
import numpy as np 
import time

from classy import Class 
from functools import lru_cache
from fftlogx import fftlog

import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import spherical_jn

c = 299792.458

OBSERVER_TERMS = True
l_max = 100

#parameters for CLASS
params = {
    'h': 0.45,
    'output': 'mPk',          # Specify to compute the matter power spectrum
    'P_k_max_1/Mpc': 1000,    # Maximum k (in 1/Mpc) for which P(k) is calculated
    'z_max_pk': 1000.0        # Redshift at which to calculate the power spectrum
}

#Initialise CLASS
cosmo = Class()
cosmo.set(params)
cosmo.compute()

def a(z): 
    return 1/(1+z)

def H(z): 
    return a(z)**(-3/2)*params['h']*100

def dH_da(z): 
    return -3/2*a(z)**(-5/2)*params['h']*100

def f(z):
    return 1.0

def df_da(z): 
    return 0.0

def Dplus(z): 
    return a(z)

def r(z): 
    return 2*c/params['h']/100*(1-1/np.sqrt(1+z))

def z_of_r(r): 
    return (1-params['h']*100/2/c*r)**(-2)-1

def dzdr_of_r(r): 
    return params['h']*100/c*(1-params['h']*100/2/c*r)**(-3)

def transfer(z): 
    if (OBSERVER_TERMS):
        return Dplus(z)*H(z)*a(z)*f(z)*(1+f(z)+a(z)/f(z)*df_da(z))*H(z)/(H(0)*(1+z)-H(z))
    else: 
        return Dplus(z)*H(z)*a(z)*f(z)*(1+f(z)+a(z)/f(z)*df_da(z))

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def tophat(z,z_mean,dz):
    if ((z<=z_mean+dz) & (z>=z_mean-dz)):
        return 1/2/dz
    else:
        return 0

@lru_cache(maxsize=None)
def window(z_mean, dz, type='tophat'):

    z_min = 0.01
    z_max = z_mean+1

    r_min = r(z_min)
    r_max = r(z_max)

    #r-values for the integration
    x = np.geomspace(r_min,r_max,10000)

    if (type=='tophat'): 
        f_of_x = [dzdr_of_r(xi)*transfer(z_of_r(xi))*tophat(z_of_r(xi),z_mean,dz) for xi in x]

    if (type=='gauss'): 
        f_of_x = [dzdr_of_r(xi)*transfer(z_of_r(xi))*gaussian(z_of_r(xi),z_mean,dz) for xi in x]

    #initialise FFTlog
    # -> x*f_of_x to cancel a factor 1/x in the definition of the integrand in FFTlog
    # -> nu=1.01 is the recommended value to avoid some numerical issues with nu=1
    # -> N_pad helps stabilize the integral 
    myfftlog = fftlog(x,x*f_of_x,nu=1.01,N_pad=5000) 


    # calculate the integarl over the first derivative of j for each l 
    k = [None]
    result = [None]

    for l in range(1,l_max+1):
        k_i, result_i = myfftlog.fftlog_dj(l)
        k.append(k_i)
        result.append(result_i)
    
    return k, result


def Cl(l,z,dz=0.01,type='tophat'):

    if (type=='delta'): 
        k = np.geomspace(1e-5, 1e2, 10000)
        inner_int = transfer(z)*spherical_jn(l,k*r(z),True)
        Pk = [cosmo.pk(i,1000.0)/Dplus(1000.0)**2 for i in k]

        #calculate final outer integral using simpsons rule
        integral = integrate.simpson(inner_int**2*Pk,x=k)

    else: 
        k, inner_int = window(z,dz,type)
        Pk = [cosmo.pk(i,1000.0)/Dplus(1000.0)**2 for i in k[l]]

        #calculate final outer integral using simpsons rule
        integral = integrate.simpson(inner_int[l]**2*Pk,x=k[l])

    return 2/np.pi*integral/c**2    #factor c**2 to fix units

l = np.arange(1,l_max+1)

dz01 =  0.002
z01 = 0.108
dz05 =  0.0025
z05 = 0.508

CL1 = np.array([Cl(i,z01,type='delta')*i*(i+1)/2/np.pi for i in l])
CL2 = np.array([Cl(i,z05,type='delta')*i*(i+1)/2/np.pi for i in l])
CL3 = np.array([Cl(i,z01,dz01,'gauss')*i*(i+1)/2/np.pi for i in l])
CL4 = np.array([Cl(i,z05,dz05,'gauss')*i*(i+1)/2/np.pi for i in l])

np.savetxt('redshiftdrift_ps_theory_koksbang2024.d',np.column_stack((l, CL1, CL2, CL3, CL4)),
         header='l\tl(l+1)Cl/2pi')


# Clean up
cosmo.struct_cleanup()
cosmo.empty()