import numpy as np

def velocity(x,y,use_central=True, sampling_rate=None, xres=None,yres=None):
    """Returns the velocity extracted from ``x`` and ``y`` samples.

    Parameters
    ----------
    x : array
        Horizontal position of the eye.
    y : array
        Vertical position of the eye.
    use_central : bool
        Whether to use central difference-based computation, or the noisier
        adjacent sample difference. If ``True``, the first and last points in
        the output array will be masked out, otherwise, only the first point
        will be masked out.
    sampling_rate: None or float
        If provided, returns the velocity as units/second, rather than
        units/sample
    xres : array
        The viewing location-dependent horizontal resolution, used to convert
        ``x`` from pixels to degrees via: $ \frac{x}{xres} $.
    yres : array
        same as above for vertical resolution

    Output
    ------
    v : masked array
        This array is the same length as ``x`` and ``y``, regardless of the
        value of use_central.
    
    See Also
    --------
    acceleration
    """
    full_vel = np.empty_like(x)
    if use_central:
        # mask out the first and last points
        full_vel[0] = full_vel[-1] = np.nan
        valid = slice(1,-1)
        vel = full_vel[valid]
        vel[:] = y[2:]-y[:-2]
        velx = x[2:]-x[:-2]
        if xres is not None:
            velx /= (xres[2:] + xres[:-2])
            velx *= 2.
            vel /= (yres[2:] + yres[:-2])
            vel *= 2.
    else:
        # mask out the first point
        full_vel[0] = np.nan
        valid = slice(1,None)
        vel = full_vel[valid]
        vel[:] = np.diff(y)
        velx = np.diff(x)
        if xres is not None:
            velx /= (xres[1:] + xres[:-1])
            velx *= 2.
            vel /= (yres[1:] + yres[:-1])
            vel *= 2.


    velx*= velx
    vel *= vel
    vel +=velx
    del(velx)

    vel = np.sqrt(vel, vel)
    # keep velocity in terms of units per sample
    if use_central:
        vel /= 2.

    # convert velocity to units per second
    if sampling_rate:
        vel *= sampling_rate

    vel = np.ma.masked_invalid(full_vel,copy=False)
    return vel

def acceleration(v,use_central=True):
    """
    Calculate the acceleration from a velocity ``v``.

    Parameters
    ----------
    v : array
        Velocity of the eye.
    use_central : bool
        Whether to use central difference-based computation, or the noisier
        adjacent sample difference. If ``True``, the first and last points in
        the output array will be masked out, otherwise, only the first point
        will be masked out.

    Output
    ------
    v : masked array
        This array is the same length as ``v`` regardless of the value of
        ``use_central``.

    See Also
    --------
    velocity
    """
    full_accel = np.empty_like(v)
    if use_central:
        full_accel[0] = full_accel[-1] = np.nan
        full_accel[1:-1] = v[2:]-v[:-2]
        full_accel /= 2.
    else:
        full_accel[0] = np.nan
        full_accel[1:] = np.diff(v)
    return np.ma.masked_invalid(full_accel,copy=False)
