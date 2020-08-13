import numpy as np
import gc
import os

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

def numerical_batman(X,Y,rp,npixels = 1000,return_star = False, verbose = False):
    """
    This function receives X and Y coordinates (in stellar units, assuming zero is at the center of the star) of the 
    position of a planet, and calculates a numerical lightcurve at each of those positions assuming a star of radius 1 
    and a planet of radius rp. Idea is simple: we imagine a circumscribed circle in a npixels x npixels array. When a pixel 
    contains both the planet circle and the star circle, that pixel is painted (with a zero). Then we sum all the pixels. 

    npixels is always even (for easier modeling). If odd number is given, add 1.

    Parameters
    ----------

        X: Array of X distance from the host star in stellar radii

        X: Array of Y distance from the host star in stellar radii

        rp: planet-to-star radius ratio

        u: limb-darkening coefficients (quadratic law)

        npixels: number of pixels in X and Y direction to model the star on

    """
    u = [0.3,0.2]
    # Set npixels to be even:
    if npixels % 2 != 0:
        npixels = npixels + 1

    rpixel = npixels/2
    # Initialize array and star radii if not done already:
    if os.path.exists('star_'+str(npixels)+'.npy'):
        star = np.load('star_'+str(npixels)+'.npy')
        star_coords_X = np.load('star_coords_X_'+str(npixels)+'.npy')
        star_coords_Y = np.load('star_coords_Y_'+str(npixels)+'.npy')
    else:
        star = np.ones([npixels, npixels])
        star_coords_X = np.ones([npixels, npixels])
        star_coords_Y = np.ones([npixels, npixels])
        # Paint limb-darekning and points outside the star with zeroes:
        if verbose:
            print('Preparing array, LD...')
        for i in range(star.shape[0]):
            for j in range(star.shape[1]):
                # Get distance of current point:
                d = np.sqrt((i-rpixel)**2 + (j-rpixel)**2)
                # If distance outside rpixel, set to zero. If not, paint limb-darkening:
                if d > rpixel:
                    star[i,j] = 0.
                elif (d/rpixel)**2 <= 1.:
                    mu = np.sqrt(1. - (d/rpixel)**2)
                    star[i,j] = 1. - u[0]*(1-mu) - u[1]*((1-mu)**2)
                star_coords_X[i,j] = i
                star_coords_Y[i,j] = j
        if verbose:
            print('Done!')
        np.save('star_'+str(npixels)+'.npy',star)
        np.save('star_coords_X_'+str(npixels)+'.npy',star_coords_X)
        np.save('star_coords_Y_'+str(npixels)+'.npy',star_coords_Y)
    # Set out-of-transit flux:
    oot_flux = np.sum(star)
    # Set array that will save the fluxes:
    fluxes = np.array([])
    # Set planet radius in pixel space:
    rp_pixel = (npixels/2)*rp
    if return_star:
        stars = []
    # Iterate through X-coordinates
    if verbose:
        print('Iterating through coordinates')
    for i in range(len(X)):
        # Rescale coordinates to pixel-space:
        cX, cY = (X[i]+1)*rpixel, (Y[i]+1)*rpixel
        if cX != 0:
            cX -= 1
        if cY != 0:
            cY -= 1
        # Check where center of the planet is. If outside the star, do nothing, 
        # if they overlap, paint all points consistent with it as zeroes, sum relative flux 
        # and save:
        d = np.sqrt(X[i]**2 + Y[i]**2)
        if d >= 1+rp:
            fluxes = np.append(fluxes, oot_flux)
            if return_star:
                stars.append(star)
        else:
            # First, cut a square around the planetary position so we search for pixels to paint 
            # only around that area:
            left,right = np.max([int(cX - rp_pixel),0]), np.min([npixels-1, int(cX + rp_pixel)])
            down,up = np.max([0, int(cY - rp_pixel)]), np.min([npixels-1, int(cY + rp_pixel)])
            square = np.copy(star[left:right,down:up])
            square_X = star_coords_X[left:right,down:up]
            square_Y = star_coords_Y[left:right,down:up]
            square_dists = np.sqrt((square_X-cX)**2 + (square_Y-cY)**2)
            idx = np.where(square_dists <= rp_pixel)
            if not return_star:
                new_star = np.load('star_'+str(npixels)+'.npy')
            else:
                new_star = np.copy(star)
            square[idx] = 0.
            new_star[left:right,down:up] = square
            if return_star:
                stars.append(new_star)
            fluxes = np.append(fluxes, np.sum(new_star))
        if verbose:
            print('Coordinate number ',i)
    print('Done!')
    fluxes = fluxes/oot_flux
    if return_star:
        return fluxes, stars
    else:
        return fluxes

def get_slope(x,y,N=3):
    num = N*np.sum(x*y) - np.sum(x)*np.sum(y)
    den = N*np.sum(x**2) - (np.sum(x)**2)
    return num/den

def numerical_catwoman(X, Y, rp1, rp2, phi, npixels = 1000, return_star = False, verbose = False):
    """
    This function receives X and Y coordinates (in stellar units, assuming zero is at the center of the star) of the 
    position of a planet, and calculates a numerical lightcurve at each of those positions assuming a star of radius 1 
    and a two stacked semi-circles, one of radius rp1 and another of radius rp2, both of which are rotated by an angle phi. 
    Idea is simple: we imagine a circumscribed circle in a npixels x npixels array. When a pixel contains both the planet semi-circles 
    (modeled through a constraint that the pixel is within a circle *and* below a line) and the star circle, that pixel is painted (with 
    a zero). Then we sum all the pixels. 

    npixels is always even (for easier modeling). If odd number is given, add 1.

    Parameters
    ----------

        X: Array of X distance from the host star in stellar radii

        X: Array of Y distance from the host star in stellar radii

        rp1: planet-to-star radius ratio of the left-side of the semi-circle

        rp2: planet-to-star radius ratio of the right-side of the semi-circle

        phi: rotation angle of the semi-circles with respect to the orbital velocity vector

        u: limb-darkening coefficients (quadratic law)

        npixels: number of pixels in X and Y direction to model the star on

    """
    # First, compute theta --- the instantaneous rotation angle of the semi-circles as measured by the orthogonal coordinates X-Y. To this end, get the intantaneous slope 
    # of a line at each XY position:
    slopes = np.array([])
    nX = len(X)
    for i in range(nX):
        if i == 0:
            slopes = np.append((Y[1] - Y[0])/(X[1]-X[0]),slopes)
        elif i == nX-1:
            slopes = np.append(slopes, (Y[nX-1] - Y[nX-2])/(X[nX-1] - X[nX-2]))
        else:
            slopes = np.append(slopes, get_slope(X[i-1:i+2],Y[i-1:i+2]))

    # Angles of the orbital path at each XY:
    angles = np.arctan(slopes)*(180./np.pi)
    # Total angle of rotation with respect ot the XY coordinate system:
    thetas = angles + phi
    # XY plane slopes:
    XY_slopes = np.tan(thetas*np.pi/180.)

    u = [0.3,0.2]
    # Set npixels to be even:
    if npixels % 2 != 0:
        npixels = npixels + 1

    rpixel = npixels/2
    # Initialize array and star radii if not done already:
    if os.path.exists('star_'+str(npixels)+'.npy'):
        star = np.load('star_'+str(npixels)+'.npy')
        star_coords_X = np.load('star_coords_X_'+str(npixels)+'.npy')
        star_coords_Y = np.load('star_coords_Y_'+str(npixels)+'.npy')
    else:
        star = np.ones([npixels, npixels])
        star_coords_X = np.ones([npixels, npixels])
        star_coords_Y = np.ones([npixels, npixels])
        # Paint limb-darekning and points outside the star with zeroes:
        if verbose:
            print('Preparing array, LD...')
        for i in range(star.shape[0]):
            for j in range(star.shape[1]):
                # Get distance of current point:
                d = np.sqrt((i-rpixel)**2 + (j-rpixel)**2)
                # If distance outside rpixel, set to zero. If not, paint limb-darkening:
                if d > rpixel:
                    star[i,j] = 0.
                elif (d/rpixel)**2 <= 1.:
                    mu = np.sqrt(1. - (d/rpixel)**2)
                    star[i,j] = 1. - u[0]*(1-mu) - u[1]*((1-mu)**2)
                star_coords_X[i,j] = i
                star_coords_Y[i,j] = j
        if verbose:
            print('Done!')
        np.save('star_'+str(npixels)+'.npy',star)
        np.save('star_coords_X_'+str(npixels)+'.npy',star_coords_X)
        np.save('star_coords_Y_'+str(npixels)+'.npy',star_coords_Y)
    # Set out-of-transit flux:
    oot_flux = np.sum(star)
    # Set array that will save the fluxes:
    fluxes = np.array([])
    # Set maximum planet radius in pixel space (this defines the square to be cut around the planet's position):
    rp_pixel = (npixels/2)*np.max([rp1,rp2])
    # Get pixel lengths of rp1 and rp2 separately as well:
    rp1_pixel = (npixels/2)*rp1
    rp2_pixel = (npixels/2)*rp2
    if return_star:
        stars = []
    # Iterate through X-coordinates
    if verbose:
        print('Iterating through coordinates')
    for i in range(len(X)):
        # Rescale coordinates to pixel-space:
        cX, cY = (X[i]+1)*rpixel, (Y[i]+1)*rpixel
        if cX != 0:
            cX -= 1
        if cY != 0:
            cY -= 1
        # Check where center of the planet is. If outside the star, do nothing, 
        # if they overlap, paint all points consistent with it as zeroes, sum relative flux 
        # and save:
        d = np.sqrt(X[i]**2 + Y[i]**2)
        if d >= 1+rp:
            fluxes = np.append(fluxes, oot_flux)
            if return_star:
                stars.append(star)
        else:
            # First, cut a square around the planetary position so we search for pixels to paint 
            # only around that area:
            left,right = np.max([int(cX - rp_pixel),0]), np.min([npixels-1, int(cX + rp_pixel)])
            down,up = np.max([0, int(cY - rp_pixel)]), np.min([npixels-1, int(cY + rp_pixel)])
            square = np.copy(star[left:right,down:up])
            square_X = star_coords_X[left:right,down:up]
            square_Y = star_coords_Y[left:right,down:up]
            square_dists = np.sqrt((square_X-cX)**2 + (square_Y-cY)**2)
            # Compute angles between center of the planet and each coordinate:
            angles_dists = np.arctan((square_Y-cY)/(square_X-cX))*(180./np.pi)
            # Identify indexes to the left of the line separating the two semi-circles:
            idx_left = np.where((square_dists <= rp1_pixel)&(angles_dists-thetas[i]>0)&(angles_dists-thetas[i]<180.))
            # Same, to the right:
            idx_right = np.where((square_dists > rp2_pixel)&(angles_dists-thetas[i]<=0)&(angles_dists-thetas[i]>=-180.))
            if not return_star:
                new_star = np.load('star_'+str(npixels)+'.npy')
            else:
                new_star = np.copy(star)
            square[idx_left] = 0.
            square[idx_right] = 0.
            new_star[left:right,down:up] = square
            if return_star:
                stars.append(new_star)
            fluxes = np.append(fluxes, np.sum(new_star))
        if verbose:
            print('Coordinate number ',i)
    print('Done!')
    fluxes = fluxes/oot_flux
    if return_star:
        return fluxes, stars
    else:
        return fluxes
