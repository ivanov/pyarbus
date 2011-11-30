import numpy as np

def velocity(x,y,use_central=True, sampling_rate=None, xres=None,yres=None):
    """Returns the velocity extracted from x and y samples

    Parameters
    ----------
    x : array
    y : array
    use_central : bool
        whether to use central difference-based computation, or the noisier
        adjacent sample difference
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
