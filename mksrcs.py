#!/home/srandall/soft/anaconda3/bin/python

# Draw point sources from a logN-logS distribution and add them to an
# input image file

import sys
import argparse
import math
import scipy.integrate as integrate
import numpy as np
import scipy.optimize as optimize
import random
import astropy
from astropy.io import fits
from astropy.wcs import WCS

# dN/dS, takes S in 1e-14 erg/cm^2/s
# returns 10^14 deg^-2  (erg/cm^2/s)^-1
def dNdS(S, type, band):
    # From Lehmer et al. 2012
    # Change flux units to 1e-14 erg/cm^2/s
    if band == 'fb':
        K_agn = 562.2  # 1e14 deg^-2 (erg/cm^2/s)^-1
        beta1_agn = 1.34
        beta2_agn = 2.35
        f_break = 0.81
    else:
        sys.exit('Energy band not supported')
    
    # 10^-14 erg cm^-2 s^-1
    S_ref = 1

    if type == 'agn':
        if S <= f_break:
            dnds_agn = K_agn*(S/S_ref)**(-1*beta1_agn)
        else:
            dnds_agn = K_agn*(f_break/S_ref)**(beta2_agn - beta1_agn)*(S/S_ref)**(-1*beta2_agn)
        return dnds_agn
    else:
        sys.exit('Type not supported')

def dNdS_draw(S_draw, rand, norm, type, band):
#    return ((integrate.quad(dNdS, S_min, S_draw, args=(type,band))[0])/norm - rand)
    return ((integrate.quad(dNdS, S_draw, np.inf, args=(type,band))[0])/norm - rand)


def main():
    # exposure time, ksec
    t_exp = 4000
    # effective area, cm^2
    eff_area = 40000
    # mean photon energy, keV
    eph_mean = 1
    # edge of FOV, arcmin
    fov = 20
    sources = []
    inimage = 'foo.fits'
    outimage = 'goo.fits'
    draw_srcs = True
    image_srcs = True

    eph_mean_erg = eph_mean*1.6e-9

    S_min = eph_mean_erg/(t_exp*1000*eff_area)
    S_min = S_min/1e-14

    fov_area = fov**2

    # Read input image
    if image_srcs:
        print("Reading image...")
        try:
            hdus = fits.open(inimage)
        except:
            sys.exit('There was an error reading the input file', infile)

        xlen = hdus[0].header['naxis1']
        ylen = hdus[0].header['naxis2']
#        wcs = WCS(hdus[0].header)
#        pixscal = astropy.wcs.utils.proj_plane_pixel_scales(wcs)
        pixscal = abs(hdus[0].header['cdelt1'])
        if hdus[0].header['cunit1'] == 'degree':
            print("FOV is",xlen*60*pixscal,"x",ylen*60*pixscal,"arcmin.")
            fov_area = xlen*ylen*(60*pixscal)**2
        else:
            sys.exit('Image CUNIT1 type not recognized')

#    test = 1e14*dNdS(S_min, 'agn', 'fb')
#    print(test)


    # Calculate the number of sources with S>S_min in the FOV

    # dNdS returns 10^14 deg^-2 (erg/cm^2/s)^-1, but we get a factor of 
    # 10^-14 from dS in integral, so they cancel
    n_srcs = integrate.quad(dNdS, S_min, np.inf, args=('agn','fb'))[0]
    # scale to the FOV
    n_srcs_fov = n_srcs*fov_area/60**2
    print("Expect", n_srcs_fov, "sources per field.")

    # draw a random distribution of sources
    if draw_srcs:
        print("Drawing sources from distribution...")
        for i in range(0,int(round(n_srcs_fov,0))):
            rand = random.random()
            S = optimize.brentq(dNdS_draw, S_min, 1000, args=(rand, n_srcs, 'agn', 'fb'))
            sources.append(S)
            
    print("Placing sources in image...")
    if image_srcs:
        for sflux in sources:
            xrand = math.floor(xlen*random.random())
            yrand = math.floor(ylen*random.random())
            hdus[0].data[yrand, xrand] = sflux*1e-14*eff_area/eph_mean_erg
        hdus.writeto(outimage, clobber='y')

main()
