import numpy as np
import matplotlib.pyplot as plt

import ncw


# Define time-steps:
t = np.linspace(-0.5,0.5,100)

# Define parameters:
period,t0,u1,u2,ecc,omega =  3., 0., 0.3, 0.2, 0., 90.
#rp1, rp2, phi, a, inc, npixels = 0.1, 0.09, -45, 1.5, 55.0, 40000
rp1, rp2, phi, a, inc, npixels = 0.1, 0.09, -45, 1.5, 89.0, 40000
#rp1, rp2, phi, a, inc, npixels = 0.1, 0.09, -45, 1.5, 72.0, 40000
# Initialize catwoman and batman:
params, m = ncw.init_catwoman(t, 'quadratic', rp1 = rp1, rp2 = rp2, phi = phi, a = a, inc = inc, per = period, t0 = t0, u1 = u1, u2 = u2, ecc=ecc, w=omega)
paramsbm, mbm = ncw.init_batman(t, 'quadratic', rp = np.sqrt(0.5*(rp1**2)+0.5*(rp2**2)), a = a, inc = inc, per = period, t0 = t0, u1 = u1, u2 = u2, ecc=ecc, w=omega)

# Evaluate fluxes:
flux = m.light_curve(params)
fluxbm = mbm.light_curve(paramsbm)

# Extract X-Y coordinates:
X,Y = m.X, m.Y

counter = 0
njump = 4
for i in range(0,100,njump):
  if counter >= 15:
    # Get numerical batman:
    numerical_flux = ncw.numerical_catwoman(X[i:i+njump],Y[i:i+njump],rp1, rp2, phi,npixels=npixels, verbose = True, return_star = False)
    tt = t[i:i+njump]
    fflux = flux[i:i+njump]
    ffluxbm = fluxbm[i:i+njump]
    fout = open('results_'+str(npixels)+'_part'+str(counter)+'.dat','w')
    fout.write('# Transit paramenters: rp1: {0:.3f} rp2: {1:.3f}, phi: {2:.3f}, inc: {3:.3f}, a: {4:.3f}, period: {5:.3f}, t0: {6:.3f}, u1: {7:.3f}, u2: {8:.3f}, ecc: {9:.3f}, omega: {10:.3f}\n'.format(rp1,rp2,phi,inc,a,period,t0,u1,u2,ecc,omega))
    fout.write('# Time \t CatwomanFlux \t NumericalFlux \t BatmanFlux\n')
    for i in range(len(tt)):
        print(tt,fflux,numerical_flux,ffluxbm)
        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(tt[i],fflux[i],numerical_flux[i],ffluxbm[i]))
    fout.close()
  counter += 1
