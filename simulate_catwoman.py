import numpy as np
import matplotlib.pyplot as plt

import ncw


# Define time-steps:
t = np.linspace(-0.5,0.5,300)

# Define parameters:
period,t0,u1,u2,ecc,omega =  3., 0., 0.3, 0.2, 0., 90.
rp1, rp2, phi, a, inc, npixels = 0.05, 0.025, -45, 1.5, 55.0, 10000

# Initialize catwoman and batman:
params, m = ncw.init_catwoman(t, 'quadratic', rp1 = rp1, rp2 = rp2, phi = phi, a = a, inc = inc, per = period, t0 = t0, u1 = u1, u2 = u2, ecc=ecc, w=omega)
paramsbm, mbm = ncw.init_batman(t, 'quadratic', rp = np.sqrt(0.5*(rp1**2)+0.5*(rp2**2)), a = a, inc = inc, per = period, t0 = t0, u1 = u1, u2 = u2, ecc=ecc, w=omega)

# Evaluate fluxes:
flux = m.light_curve(params)
fluxbm = mbm.light_curve(paramsbm)

# Extract X-Y coordinates:
X,Y = m.X, m.Y

# Get numerical batman:
numerical_flux, stars = ncw.numerical_catwoman(X,Y,rp1, rp2, phi,npixels=npixels, verbose = True, return_star = True)

# Plot:
plt.subplot(211)
plt.plot(t,flux,label='catwoman',color='cornflowerblue')
plt.plot(t,numerical_flux,label='numerical',color='black')
plt.plot(t,fluxbm,label='batman',color='orangered')
plt.ylabel('Relative flux')
plt.legend()

plt.subplot(212)
plt.plot(t,(flux-numerical_flux)*1e6,color='cornflowerblue')
plt.plot(t,(fluxbm-numerical_flux)*1e6,color='orangered')
#plt.plot(t,(fluxbm-flux)*1e6,color='grey')
plt.xlabel('Time (days since mid-transit)')
plt.ylabel('Numerical - algorithm (ppm)')
plt.show()

fout = open('results_'+str(npixels)+'.dat','w')
fout.write('# Transit paramenters: rp1: {0:.3f} rp2: {1:.3f}, phi: {2:.3f}, inc: {3:.3f}, a: {4:.3f}, period: {5:.3f}, t0: {6:.3f}, u1: {7:.3f}, u2: {8:.3f}, ecc: {9:.3f}, omega: {10:.3f}\n'.format(rp1,rp2,phi,inc,a,period,t0,u1,u2,ecc,omega))
fout.write('# Time \t CatwomanFlux \t NumericalFlux \t BatmanFlux\n')
for i in range(len(t)):
    fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(t[i],flux[i],numerical_flux[i],fluxbm[i]))

im = plt.imshow(stars[150])
plt.show()
