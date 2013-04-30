"""
Visualization routines for pyarbus objects.
"""

# If you are running nosetests right now, you might want to use 'agg' as a
# backend:
import matplotlib
import sys
if "nose" in sys.modules:
    matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import nitime
import pyarbus

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


def plot_xyp(eye, axes=None, subtract_t0=True):
    """
    Plots, on three separate subplots, the pupil area, x, and y position
    reported by the eyetracker as a function of time.
    """
    if isinstance(eye, pyarbus.Eyelink):
        raise ValueError(
          """This method can plot an Eye object, but you provided an Eyelink
          object. Eyelink objects have either a left or a right eye (or both),
          so pick one.

          Example: if you called this function with:

              viz.plot_xyp(o)

          you should instead call it with:

              plot_xyp(o.l)     # for left eye

          or

              plot_xyp(o.r)     # for right eye""")
    if axes is None:
        fig,axes = plt.subplots(3,1,sharex=True)
    t = eye.time
    if subtract_t0:
        t -= eye.time[0]
    axes[0].plot(eye.time, eye.pupA)
    axes[1].plot(eye.time, eye.x)
    axes[2].plot(eye.time, eye.y)
    eye.time.time_unit = 's'
    make_time_axis(axes[-1].xaxis, time_unit=eye.time.time_unit)
    axes[2].set_xlabel("Time (%s)" % eye.time.time_unit)

def plot_xy_p(eye, axes=None, subtract_t0=True):
    """
    Same as :meth:`plot_xyp`, but plots the position on one axis, and the
    pupil area on another.

    See Also
    --------
    plot_xyp : plots each quantity on it own subplot by default
    """
    if axes is None:
        fig, axes = plt.subplots(2,1, sharex=True)
    axes = axes[0], axes[1], axes[-1]
    plot_xyp(eye, axes, subtract_t0)

def make_time_axis(axis=None,time_unit='s'):
    """
    Change formatter to interpret axis as a nitime TimeArray object

    Parameters:
    -----------
    axis : None, matplotlib.axis.XAxis, or matplotlib.axis.YAxis
        Axis whose ticks will be reinterpreted as being nitime-based time. If
        None, ``plt.gca().xaxis`` will be used
    time_unit : str
        A valid ``nitime`` time format, for possibilities, see
        ``nitime.timeseries.time_unit_conversion.keys()``

    """
    if axis is None:
        axis = plt.gca().xaxis
    class TimeFormatter(matplotlib.ticker.FuncFormatter):
        def __init__(self,time_unit):
            self.time_unit = time_unit
            conv = nitime.time_unit_conversion
            self.func = lambda x,y: (1.0*x) / conv[self.time_unit]
        def format_data_short(self,value):
            'return a short string version'
            return "%-12g"%self.format_data(value)
    f = TimeFormatter(time_unit)
    axis.set_major_formatter(f)
