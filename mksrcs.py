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
from xrs_tools import Spectrum, determine_apec_norm, determine_powerlaw_norm, write_simput_phlist, run_simx

class Bgsrc:
    def __init__(self, type, flux):
        self.type = type
        self.flux = flux
        self.model = ""
        self.params = []
        self.coords = []
        self.z = None

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
    t_exp = 50.0  # exposure time, ksec
    eff_area = 40000   # effective area, cm^2
    eph_mean = 1   # mean photon energy, keV
    fov = 20    # edge of FOV, arcmin
    sources = []
    inimage = 'foo.fits'
    outimage = 'moo.fits'
    draw_srcs = True
    mode = 'events' # 'events' or 'image', add idealized sources to input image or generate events file.  Set to something else to simply print out the number of expected BG sources of different types
    types = ['agn', 'gal', 'star']
    # parameters for making event file
    evt_prefix = "50ks_bgsrcs_" # event file prefix
    ra_cen = 96.6 # degress, RA of field center
    dec_cen = -53.73 #degrees, Dec of field center
    nH = 0.05 # Galactic absorption, 1e22 cm^-2
    fb_emin = 0.5  # keV, low energy bound for full band flux
    fb_emax = 8.0  # keV, high energy bound for full band flux
    spec_emin = 0.01 # keV, minimum energy of mock spectrum 
    spec_emax = 20.0 # keV, max energy of mock spectrum
    spec_nbins = 10000 # number of bins between spec_emin and spec_emax
    agn_ind = 1.2 # AGN photon index
    agn_z = 2.0 # AGN redshift
    gal_ind = 1.2 # galaxy photon index
    gal_z = 0.8 # galaxy redshift
    star_ind = 1.0 # star photon index

    eph_mean_erg = eph_mean*1.6e-9

    # integrate down to a flux where we expect to have roughly one photon
    # during the exposure
    S_min = eph_mean_erg/(t_exp*1000*eff_area)
    S_min = S_min/1e-14

    print("Flux limit is",S_min*1e-14)

    fov_area = fov**2

    if mode == 'image':
        # Read input image to get size of FOV
        print("Reading image...")
        try:
            hdus = fits.open(inimage)
        except:
            sys.exit('There was an error reading the input image')

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
        print("Expect", n_srcs_fov, "source of type",type,"in the field.")

        # draw a random distribution of sources
        if draw_srcs:
            print("Drawing sources from distribution...")
            for i in range(0,int(round(n_srcs_fov,0))):
                rand = random.random()
                S = optimize.brentq(dNdS_draw, S_min, 1000, args=(rand, n_srcs, type, 'fb'))
                thissrc = Bgsrc(type, S*1e-14)
                sources.append(thissrc)

    if mode == 'image':
        print("Placing sources in image...")
        for source in sources:
            xrand = math.floor(xlen*random.random())
            yrand = math.floor(ylen*random.random())
            hdus[0].data[yrand, xrand] = source.flux*eff_area/eph_mean_erg
        hdus.writeto(outimage, clobber='y')

    if mode == 'events':
        # assign spectral models
        print("Generating source spectra...")
        dec_scal = np.fabs(np.cos(dec_cen*np.pi/180))
        ra_min = ra_cen - fov/(2*60*dec_scal)
        dec_min = dec_cen - fov/(2*60)
        all_photons = np.array([])
        all_ra = np.array([])
        all_dec = np.array([])
        i = 0
        for source in sources:
            if source.type == 'agn':
                source.z = agn_z
                source.model = "zpowerlw"
                source.params = [agn_ind, source.z, 1.0]
            elif source.type == 'gal':
                source.z = gal_z
                source.model = "zpowerlw"
                source.params = [gal_ind, source.z, 1.0]
            elif source.type == 'star':
                source.z = 0
                source.model = "zpowerlw"
                source.params = [star_ind, source.z, 1.0]
 
            # generate spectrum and renormalize
            # assumes model has one mornalization, and it's the last parameter
            unabs_spec = Spectrum.from_xspec(source.model, source.params, emin=spec_emin, emax=spec_emax,
                                             nbins=spec_nbins)
            flux1 = unabs_spec.tot_flux
            unabs_spec.rescale_flux(source.flux, emin=fb_emin, emax=fb_emax, flux_type="energy")
            norm = unabs_spec.tot_flux/flux1
            source.model = '*'.join(["wabs", source.model])
            source.params = [nH] + source.params
            source.params.pop()
            source.params = source.params + [norm]
            abs_spec = Spectrum.from_xspec(source.model, source.params, emin=spec_emin, emax=spec_emax,
                                            nbins=spec_nbins)
            # draw photons and assign random position
            photons = abs_spec.generate_energies(t_exp*1000, eff_area)
            ph_ra = np.array([np.random.random()*fov/(60*dec_scal) + ra_min]*len(photons))
            ph_dec = np.array([np.random.random()*fov/60 + dec_min]*len(photons))

            all_photons = np.concatenate((all_photons, photons))
            all_ra = np.concatenate((all_ra, ph_ra))
            all_dec = np.concatenate((all_dec, ph_dec))

            i = i+1
            print(i)
        write_simput_phlist(evt_prefix, t_exp*1000, eff_area, all_ra, all_dec, all_photons, clobber=True)
        simput = evt_prefix + "_simput.fits"
        print("Running simx...")
        run_simx(simput, evt_prefix, t_exp*1000, [ra_cen, dec_cen], instrument='hdxi')
main()
