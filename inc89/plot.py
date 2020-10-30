from pylab import *
import numpy as np
import glob

files = glob.glob('results*')
times = np.array([])
cw = np.array([])
num_cw = np.array([])
bm = np.array([])
for f in files:
    ct,ccw,cn,cbm = np.loadtxt(f,unpack=True)
    times = np.append(times,ct)
    cw = np.append(cw,ccw)
    num_cw = np.append(num_cw,cn)
    bm = np.append(bm,cbm)

idx = np.argsort(times)
plt.plot(times[idx],(cw[idx]-num_cw[idx])*1e6)
plt.show()
