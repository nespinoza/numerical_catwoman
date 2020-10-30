import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

import seaborn as sns
sns.set_style('ticks')
#my_cmap = sns.color_palette('crest_r', as_cmap=True)


import ncw

def read_data(inc):
    inc_int = int(inc)
    files = glob.glob('inc'+str(inc_int)+'/results*')
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
    return times[idx], cw[idx], num_cw[idx], bm[idx]

# Define time-steps; more finely sampled than the data ones for plotting purposes:
t = np.linspace(-0.5,0.5,300)
# Define also time sampled as the data (for plotting purposes)
#tt = np.linspace(-0.5,0.5,100)

# Define which of all the time-stamps images you will plot on the plot:
nplot = 150

# Define common parameters:
period,t0,u1,u2,ecc,omega =  3., 0., 0.3, 0.2, 0., 90.
rp1, rp2, phi, a, inc, npixels = 0.1, 0.09, -45, 1.5, 72.0, 20000

# Define inclinations, names, labels, etc.:
inclinations = [55., 72., 89.]
names = ['$i = 55^{o}$', '$i = 72^{o}$', '$i = 89^{o}$']
files = []

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(18,9))
gs = gridspec.GridSpec(ncols=3, nrows=6, figure = fig)
print(gs)
for i in range(len(inclinations)):
    inc = inclinations[i]
    # Compute catwoman model to get the X and Y's to plot both for the 2D image and for the X,Y of the simulations:
    params, m = ncw.init_catwoman(t, 'quadratic', rp1 = rp1, rp2 = rp2, phi = phi, a = a, inc = inc, per = period, t0 = t0, u1 = u1, u2 = u2, ecc=ecc, w=omega)
    X,Y = m.X, m.Y

    # Select dataset:
    tt, cw, num_cw, bm = read_data(inc)

    params_sim, m_sim = ncw.init_catwoman(tt, 'quadratic', rp1 = rp1, rp2 = rp2, phi = phi, a = a, inc = inc, per = period, t0 = t0, u1 = u1, u2 = u2, ecc=ecc, w=omega)
    X_sim,Y_sim = m_sim.X, m_sim.Y 

    # Get numerical catwoman, but only for the previously-defined nplot X/Y --- that's the one we want to plot:
    numerical_flux, stars = ncw.numerical_catwoman(X[nplot:nplot+2], Y[nplot:nplot+2],rp1, rp2, phi,npixels=npixels, verbose = True, return_star = True)


    # Plot top plot:
    ax1 = fig.add_subplot(gs[0:4,i])
    im = ax1.imshow(stars[0].T[::-1,:], extent=[-1.,1.,-1.,1.],zorder=1)
    ax1.plot(X[:140],Y[:140],color='grey',lw=2, alpha=0.5,zorder=2)
    ax1.plot(X[162:],Y[162:],color='grey',lw=2, alpha=0.5,zorder=2)
    ax1.set_xlim([-1., 1.])
    ax1.set_ylim([-1., 1.])
    if inc == 55.:
        ax1.set_ylabel('Stellar coordinate ($R_*$)')
    ax1.set_title(names[i])

    # Middle plot:
    ax2 = fig.add_subplot(gs[4,i])
    ax2.plot(X_sim,num_cw,lw=3,color='black',zorder=1,label='Numerical')
    ax2.plot(X_sim,cw,color='orangered',zorder=3,label='Catwoman')
    print(X_sim,num_cw)
    print(cw)
    if inc == 55.:
        ax2.set_ylabel('Relative flux')
    ax2.set_xlim([-1.,1.])
    if inc == 55.:
        ax2.legend(frameon=False)
        # Plot inset to show planet limbs:
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        axins = ax1.inset_axes([0.6, 0.3, 0.33, 0.33])
        x1, x2, y1, y2 = -0.111, 0.125, -0.96, -0.75
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        axins.imshow(stars[0].T[::-1,:], extent=[-1.,1.,-1.,1.],zorder=1)
        axins.plot(X[:140],Y[:140],color='grey',lw=2, alpha=0.5,zorder=2)
        axins.plot(X[162:],Y[162:],color='grey',lw=2, alpha=0.5,zorder=2)
        mark_inset(ax1, axins, loc1=1, loc2=2, linewidth=0.7, fc="None", ec='k', alpha=0.4, clip_on=True, zorder=3)
    # Residual plot:
    ax3 = fig.add_subplot(gs[5,i])
    ax3.plot(X_sim,(num_cw-cw)*1e6,lw=3,color='orangered',zorder=1)
    print((num_cw-cw)*1e6)
    ax3.set_xlabel('Stellar coordinate ($R_*$)')
    if inc == 55.:
        ax3.set_ylabel('N - C (ppm)')
    ax3.set_xlim([-1.,1.])
    ax3.set_ylim([-10,10])

plt.tight_layout()
plt.savefig('plot.pdf', dpi=400)
