numerical-catwoman
------------------

This algorithm returns the lightcurve of two stacked semi-circles. The axis that forms the base of those semi-cirlces is rotated by an angle (var)phi with respect to the direction of the 
orbital motion. Usage:

   numerical_flux, stars = ncw.numerical_catwoman(X,Y,rp1, rp2, phi,npixels=npixels, verbose = True, return_star = True)

Where X and Y are the coordinates of the planet in the plane of the sky (in stellar units), rp1 and rp2 are the semi-circle radii in stellar units, and phi is the angle of rotation. `npixels` 
defines the number of pixels used to numerically simulate the star being transited (the larger, the better the precision) --- `return_star` returns snapshots of the model at each X,Y coordinate.
