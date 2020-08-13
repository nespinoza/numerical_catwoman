import numpy as np
import matplotlib.pyplot as plt

import ncw


# Define time-steps:
t = np.linspace(-0.1,0.1,1000)

# Initialize catwoman:
params, m = ncw.init_catwoman(t, 'quadratic')

# Extract X-Y coordinates:
X,Y = m.X, m.Y

# Evaluate lightcurve:
flux = m.light_curve(params)

# Plot:
plt.plot(t,flux)
plt.show()
