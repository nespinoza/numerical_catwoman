import numpy as np

import batman
import catwoman

def init_batman(t, ld_law, nresampling = None, etresampling = None):
     """
     This function initializes the batman code.
     """
     params = batman.TransitParams()
     params.t0 = 0.
     params.per = 3.
     params.rp = 0.1
     params.a = 10.
     params.inc = 90.
     params.ecc = 0.
     params.w = 90.
     if ld_law == 'linear':
         params.u = [0.5]
     else:
         params.u = [0.3,0.2]
     params.limb_dark = ld_law
     if nresampling is None or etresampling is None:
         m = batman.TransitModel(params, t)
     else:
         m = batman.TransitModel(params, t, supersample_factor=nresampling, exp_time=etresampling)
     return params,m

def init_catwoman(t, ld_law, nresampling = None, etresampling = None):
     """
     This function initializes the catwoman code.
     """
     params = batman.TransitParams()
     params.t0 = 0.
     params.per = 3.
     params.rp = 0.1
     params.rp2 = 0.1
     params.a = 10.
     params.inc = 90.
     params.ecc = 0.
     params.w = 90.
     params.phi = 45.
     if ld_law == 'linear':
         params.u = [0.5]
     else:
         params.u = [0.3,0.2]
     params.limb_dark = ld_law
     if nresampling is None or etresampling is None:
         m = catwoman.TransitModel(params, t)
     else:
         m = catwoman.TransitModel(params, t, supersample_factor=nresampling, exp_time=etresampling)
     return params,m

def numerical_batman(X,Y,rp,npixels = 1000):
    """
    This function receives X and Y coordinates (in stellar units, assuming zero is at the center of the star) of the 
    position of a planet, and calculates a numerical lightcurve at each of those positions assuming a star of radius 1 
    and a planet of radius rp. Idea is simple: we imagine a circumscribed circle in a npixels x npixels array. When a pixel 
    contains both the planet circle and the star circle, that pixel is painted (with a zero). Then we sum all the pixels. 

    npixels is always even (for easier modeling). If odd number is given, add 1.
    """
    # Set npixels to be even:
    if npixels % 2 != 0:
        npixels = npixels + 1

    star = np.ones([npixels, npixels])
    # Paint points outside the star with zeroes:
    rpixel = npixels/2
    for i in range(star.shape[0]):
        for j in range(star.shape[1]):
            if (i-rpixel)**2 + (j-rpixel)**2 > rpixel**2:
                star[i,j] = 0.

    # Set out-of-transit flux:
    oot_flux = np.sum(star)
    # Set array that will save the fluxes:
    fluxes = np.array([])
    # Iterate through X-coordinates
    for i in range(len(X)):
        # Rescale coordinates to pixel-space:
        cX, cY = (X+1)*rpixel, (Y+1)*rpixel
        d = np.sqrt(X[i]**2 + Y[i]**2)
        if d >= 1+rp:
            fluxes = np.append(fluxes, oot_flux)
        else:
            
