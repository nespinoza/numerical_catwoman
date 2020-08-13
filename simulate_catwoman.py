import numpy as np
import matplotlib.pyplot as plt

import ncw


# Define time-steps:
t = np.linspace(-0.1,0.1,100)

# Initialize catwoman and batman:
params, m = ncw.init_catwoman(t, 'quadratic', rp1 = 0.2, rp2 = 0.1, phi = 90.)
paramsbm, mbm = ncw.init_batman(t, 'quadratic')

# Evaluate fluxes:
flux = m.light_curve(params)
fluxbm = mbm.light_curve(paramsbm)

# Extract X-Y coordinates:
X,Y = m.X, m.Y

# Get numerical batman:
numerical_flux, stars = ncw.numerical_catwoman(X,Y,0.2,0.1,90.,npixels=1000, verbose = True, return_star = True)

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
plt.plot(t,(fluxbm-flux)*1e6,color='grey')
plt.xlabel('Time (days since mid-transit)')
plt.ylabel('Numerical - algorithm (ppm)')
plt.show()

im = plt.imshow(stars[30])
plt.show()
