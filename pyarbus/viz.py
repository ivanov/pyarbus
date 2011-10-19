import matplotlib.pyplot as plt
import numpy as np

def plot_saccade_scatter(sac, ax=None):
    """
    Make a scatter plot of saccades with all starting points moved to the origin.

    see pyarbus.data for details on the Saccade class
    """
    if ax is None:
        f,ax = plt.subplots(subplot_kw=dict(polar=True))
    dx = sac.xf - sac.xi
    dy = sac.yf - sac.yi
    radii = sac.amplitude
    thetas = np.arctan2(dy, dx)
    ax.scatter(thetas,radii,alpha=.5,s=sac.vpeak/10.0)

def plot_saccade_hist(sac, bins=36, ax=None):
    """
    Make a histogram of saccades with all starting points moved to the origin.

    see pyarbus.data for details on the Saccade class
    """
    if ax is None:
        f,ax = plt.subplots(subplot_kw=dict(polar=True))
    dx = sac.xf - sac.xi
    dy = sac.yf - sac.yi
    radii = sac.amplitude
    thetas = np.arctan2(dy, dx)
    ax.set_title("Directional histogram")
    theta_bins = np.linspace(-np.pi,np.pi,bins+1)
    theta_hist = np.histogram(thetas,bins=theta_bins)
    bars = plt.bar(theta_bins[:-1],theta_hist[0],width=(2*np.pi)/bins, alpha=.5)

def plot_saccade_stats(sac,bins=36, fig=None):
    """
    Draws individual saccades (polar coordinatess), directional histogram, 
    individual saccades (cartesian coordinates), and a histogram of saccade
    peak velocities
    """
    if fig is None:
        fig = plt.figure()
    dx = sac.xf - sac.xi
    dy = sac.yf - sac.yi
    radii = sac.amplitude
    thetas = np.arctan2(dy, dx)
    theta_bins = np.linspace(-np.pi,np.pi,bins+1)
    theta_hist = np.histogram(thetas,bins=theta_bins)

    plt.jet()
    plt.title("individual saccades")
    # XXX: there's an interesting artifact if we use sac.amplitude as the
    # sac_length in the scatter plot below
    sac_length = np.sqrt((dx**2+dy**2))
    plt.subplot(221,polar=True)
    plt.scatter(thetas,sac_length,alpha=.5,c=sac.amplitude,s=sac.vpeak/10.0)
    plt.colorbar()
    plt.subplot(222,polar=True)
    plt.title("Directional histogram")
    bars = plt.bar(theta_bins[:-1],theta_hist[0],width=(2*np.pi)/bins, alpha=.5)
    plt.subplot(223)
    plt.scatter(dx,dy,alpha=.5,c=sac.amplitude,s=sac.vpeak/10.0)

    #plt.hist(sac_length,bins=100)
    #plt.xlabel("saccade lengths (pixels)")
    plt.subplot(224)
    plt.hist(sac.vpeak,bins=100)
    plt.xlabel("saccade peak velocities")
