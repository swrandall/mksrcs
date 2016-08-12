#!/home/srandall/soft/anaconda3/bin/python

# Draw point sources from a logN-logS distribution and add them to an
# input image file.  Flux units are 10^-14 erg/cm^2/s.  Output image is
# in cts/s.
# NOTE: numerical integration and root finding could be done analytically
# to speed up the code at the expense of generality for the form of dNdS.
# Should eventually add an option to choose.

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
    if not (band == 'fb'):
        sys.exit('Energy band not supported')

    # From Lehmer et al. 2012
    # Change flux units to 1e-14 erg/cm^2/s
    if type == 'agn':
        K = 562.2  # 1e14 deg^-2 (erg/cm^2/s)^-1
        beta1 = 1.34
        beta2 = 2.35
        f_break = 0.81
    elif type == 'gal':
        K = 2.82
        beta1 = 2.4
    elif type == 'star':
        K = 4.07
        beta1 = 1.55
    else:
        sys.exit('Source type not supported')
    
    # 10^-14 erg cm^-2 s^-1
    S_ref = 1

    if type == 'agn' and S > f_break:
        dnds = K*(f_break/S_ref)**(beta2 - beta1)*(S/S_ref)**(-1*beta2)
    else:
        dnds = K*(S/S_ref)**(-1*beta1)

    return dnds

def dNdS_draw(S_draw, rand, norm, type, band):
    return ((integrate.quad(dNdS, S_draw, np.inf, args=(type,band))[0])/norm - rand)


def main():
    # exposure time, ksec
    t_exp = 50
    # effective area, cm^2
    eff_area = 40000
    # mean photon energy, keV
    eph_mean = 1
    # edge of FOV, arcmin
    fov = 20
    sources = []
    inimage = 'foo.fits'
    outimage = 'all_types_50ks.fits'
    draw_srcs = True
    image_srcs = True
    types = ['agn', 'gal', 'star']

    eph_mean_erg = eph_mean*1.6e-9

    # integrate down to a flux where we expect to have roughly one photon
    # during the exposure
    S_min = eph_mean_erg/(t_exp*1000*eff_area)
    S_min = S_min/1e-14

    print("Flux limit is",S_min*1e-14)

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

    # Calculate the number of sources with S>S_min in the FOV
    for type in types:
        # dNdS returns 10^14 deg^-2 (erg/cm^2/s)^-1, but we get a factor of 
        # 10^-14 from dS in integral, so they cancel
        n_srcs = integrate.quad(dNdS, S_min, np.inf, args=(type,'fb'))[0]
        # scale to the FOV
        n_srcs_fov = n_srcs*fov_area/60**2
        print("Expect sources", n_srcs_fov, "of type",type,"in field.")

        # draw a random distribution of sources
        n14 = 0
        if draw_srcs:
            print("Drawing sources from distribution...")
            for i in range(0,int(round(n_srcs_fov,0))):
                rand = random.random()
                S = optimize.brentq(dNdS_draw, S_min, 1000, args=(rand, n_srcs, type, 'fb'))
                sources.append(S)
                if S > 1:
                    n14 += 1
        print(integrate.quad(dNdS, 1, np.inf, args=(type,'fb'))[0]*fov_area/60**2, n14)


    print("Placing sources in image...")
    if image_srcs:
        for sflux in sources:
            xrand = math.floor(xlen*random.random())
            yrand = math.floor(ylen*random.random())
            hdus[0].data[yrand, xrand] = sflux*1e-14*eff_area/eph_mean_erg
        hdus.writeto(outimage, clobber='y')

main()
