#!/home/srandall/soft/anaconda3/bin/python

# Draw point sources from a logN-logS distribution and add them to an
# input image file.  Flux units are 10^-14 erg/cm^2/s.  Output image is
# in cts/s.
# NOTE: numerical integration and root finding could be done analytically
# to speed up the code at the expense of generality for the form of dNdS.
# Should eventually add an option to choose.
# Author: Scott Randall (srandall@cfa.harvard.edu)

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
from soxs import Spectrum, write_photon_list,  instrument_simulator
from soxs.instrument import instrument_registry

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

    # lower limit on source flux so as not to overpredict unresolved CXRB.
    # Note that this number depends on the energy band
    S_cut = 9.8e-7

    if S < S_cut:
        return 0.0

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

# integral of dN/dS, takes S in 1e-14 erg/cm^2/s
# returns 10^14 deg^-2
def int_dNdS(S_lo, S_hi, type, band):
    if not (band == 'fb'):
        sys.exit('Energy band not supported')

    # lower limit on source flux so as not to overpredict unresolved CXRB.
    # Note that this number depends on the energy band
    S_cut = 9.8e-7

    if S_lo < S_cut:
        S_lo = S_cut
    if S_hi < S_cut:
        return 0.0

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

    if type == 'agn':
        if S_hi <= f_break:
            int_dnds = K*S_ref**beta1/(1-beta1)*(S_hi**(1-beta1) - S_lo**(1-beta1))
        else:
            int_dnds = K*S_ref**beta1/(1-beta1)*(f_break**(1-beta1) - S_lo**(1-beta1))
            int_dnds = int_dnds + K*(f_break/S_ref)**(beta2-beta1)*S_ref**beta2/(1-beta2)\
                *(S_hi**(1-beta2) - f_break**(1-beta2))
    else:
        int_dnds = K*S_ref**beta1/(1-beta1)*(S_hi**(1-beta1) - S_lo**(1-beta1))

    return int_dnds

def dNdS_draw(S_draw, rand, norm, type, band):
    return ((int_dNdS(S_draw, np.inf, type, band))/norm - rand)
#    return ((integrate.quad(dNdS, S_draw, np.inf, args=(type,band))[0])/norm - rand)

def main():
    t_exp = 10 # exposure time, ksec
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
    evt_prefix = "10ks_nobg_" # event file prefix
    ra_cen = 96.6 # degress, RA of field center
    dec_cen = -53.73 #degrees, Dec of field center
    nH = 0.05 # Galactic absorption, 1e22 cm^-2
    fb_emin = 0.5  # keV, low energy bound for full band flux
    fb_emax = 8.0  # keV, high energy bound for full band flux
    spec_emin = 0.01 # keV, minimum energy of mock spectrum 
    spec_emax = 20.0 # keV, max energy of mock spectrum
    spec_nbins = 10000 # number of bins between spec_emin and spec_emax
    apec_root = "./atomdb" # this and following for ApecGenerator (spectra without xspec)
    apec_vers = "3.0.3"
    agn_ind = 1.2 # AGN photon index
    agn_z = 2.0 # AGN redshift
    gal_ind = 1.2 # galaxy photon index
    gal_z = 0.8 # galaxy redshift
    star_ind = 1.0 # star photon index
    dither_size = 16.0 # dither circle radius or box width in arcsec
    dither_shape = 'square'
    test = False

    eph_mean_erg = eph_mean*1.6e-9

    # integrate down to a flux where we expect to have roughly 0.1 photons
    # during the exposure
    S_min = 0.1*eph_mean_erg/(t_exp*1000*eff_area)
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

    if test:
        n_srcs = integrate.quad(dNdS, S_min, np.inf, args=('agn','fb'))[0]
        print(n_srcs)
        n_srcs = int_dNdS(S_min, np.inf, 'agn', 'fb')
        print(n_srcs)
        n_srcs = integrate.quad(dNdS, S_min, np.inf, args=('gal','fb'))[0]
        print(n_srcs)
        n_srcs = int_dNdS(S_min, np.inf, 'gal', 'fb')
        print(n_srcs)
        n_srcs = integrate.quad(dNdS, S_min, np.inf, args=('star','fb'))[0]
        print(n_srcs)
        n_srcs = int_dNdS(S_min, np.inf, 'star', 'fb')
        print(n_srcs)
        dnds = dNdS(1.0, 'agn', 'fb')
        print(dnds)

    # Calculate the number of sources with S>S_min in the FOV
    for type in types:
        # dNdS returns 10^14 deg^-2 (erg/cm^2/s)^-1, but we get a factor of 
        # 10^-14 from dS in integral, so they cancel
#        n_srcs = integrate.quad(dNdS, S_min, np.inf, args=(type,'fb'))[0]
        n_srcs = int_dNdS(S_min, np.inf, type, 'fb')
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
    if test:
        ngt17 = 0
        ngt16 = 0
        ngt15 = 0
        ngt14 = 0
        for source in sources:
            if source.type == 'agn':
                if source.flux > 1.0e-17:
                    ngt17 = ngt17+1
                if source.flux > 1.0e-16:
                    ngt16 = ngt16+1
                if source.flux > 1.0e-15:
                    ngt15 = ngt15+1
                if source.flux > 1.0e-14:
                    ngt14 = ngt14+1
        print("N_AGN > 1e-17 erg/cm^2/s (deg^-2):",ngt17*60**2/fov_area)
        print("N_AGN > 1e-16 erg/cm^2/s (deg^-2):",ngt16*60**2/fov_area)
        print("N_AGN > 1e-15 erg/cm^2/s (deg^-2):",ngt15*60**2/fov_area)
        print("N_AGN > 1e-14 erg/cm^2/s (deg^-2):",ngt14*60**2/fov_area)
        ngt17 = 0
        ngt16 = 0
        ngt15 = 0
        ngt14 = 0
        for source in sources:
            if source.type == 'gal':
                if source.flux > 1.0e-17:
                    ngt17 = ngt17+1
                if source.flux > 1.0e-16:
                    ngt16 = ngt16+1
                if source.flux > 1.0e-15:
                    ngt15 = ngt15+1
                if source.flux > 1.0e-14:
                    ngt14 = ngt14+1
        print("N_GAL > 1e-17 erg/cm^2/s (deg^-2):",ngt17*60**2/fov_area)
        print("N_GAL > 1e-16 erg/cm^2/s (deg^-2):",ngt16*60**2/fov_area)
        print("N_GAL > 1e-15 erg/cm^2/s (deg^-2):",ngt15*60**2/fov_area)
        print("N_GAL > 1e-14 erg/cm^2/s (deg^-2):",ngt14*60**2/fov_area)
        ngt17 = 0
        ngt16 = 0
        ngt15 = 0
        ngt14 = 0
        for source in sources:
            if source.type == 'star':
                if source.flux > 1.0e-17:
                    ngt17 = ngt17+1
                if source.flux > 1.0e-16:
                    ngt16 = ngt16+1
                if source.flux > 1.0e-15:
                    ngt15 = ngt15+1
                if source.flux > 1.0e-14:
                    ngt14 = ngt14+1
        print("N_STAR > 1e-17 erg/cm^2/s (deg^-2):",ngt17*60**2/fov_area)
        print("N_STAR > 1e-16 erg/cm^2/s (deg^-2):",ngt16*60**2/fov_area)
        print("N_STAR > 1e-15 erg/cm^2/s (deg^-2):",ngt15*60**2/fov_area)
        print("N_STAR > 1e-14 erg/cm^2/s (deg^-2):",ngt14*60**2/fov_area)

        sys.exit('done testing')

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
            # assumes model has one nornalization, and it's the last parameter
#            unabs_spec = Spectrum.from_xspec(source.model, source.params, emin=spec_emin, emax=spec_emax,
#                                             nbins=spec_nbins)
            unabs_spec = Spectrum.from_powerlaw(source.params[0], source.z, 1.0, emin=spec_emin, 
                                                emax=spec_emax, nbins=spec_nbins)
            flux1 = unabs_spec.total_flux
            unabs_spec.rescale_flux(source.flux, emin=fb_emin, emax=fb_emax, flux_type="energy")
            unabs_spec.apply_foreground_absorption(nH)
            # draw photons and assign random position
            photons = unabs_spec.generate_energies(t_exp*1000, eff_area)
            if len(photons) > 0:
                ph_ra = np.array([np.random.random()*fov/(60*dec_scal) + ra_min]*len(photons))
                ph_dec = np.array([np.random.random()*fov/60 + dec_min]*len(photons))

                all_photons = np.concatenate((all_photons, photons))
                all_ra = np.concatenate((all_ra, ph_ra))
                all_dec = np.concatenate((all_dec, ph_dec))

        all_flux = np.sum(all_photons)*1.60218e-9/(t_exp*1000*eff_area)
        write_photon_list(evt_prefix, evt_prefix, all_flux, all_ra, all_dec, all_photons, clobber=True)
        simput = evt_prefix + "_simput.fits"
        evt_file = evt_prefix + "_evt.fits"
        print("Running instrument simulator...")
        instrument_simulator(simput, evt_file,  t_exp*1000, 'hdxi', [ra_cen, dec_cen], dither_size=dither_size, 
                             dither_shape=dither_shape, astro_bkgnd=None, instr_bkgnd_scale=0, clobber=True)
main()



