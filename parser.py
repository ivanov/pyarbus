import numpy as np

# from http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Params:
    "Eyelink Parser Parameters"
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        #return self.__dict__.__repr__()
        "pi added this repr, it's not pretty, but works"
        #return self.__doc__ + self.__dict__.__repr__()
        return self.__doc__ + "".join(["\n\t%s = %r" % (attr, val) 
            for (attr, val) in self.__dict__.items()])


def get_default_params(**kw):
    p = Params(
            v_thresh = 30.,
            a_thresh = 8000.,
            saccade_onset_verify_time = 4,
            saccade_offset_verify_time = 20,
            )
    p.__dict__.update(kw)
    return p

testParams = get_default_params(
        # for testing purposed, these need to be
        saccade_onset_verify_time = 2,
        saccade_offset_verify_time = 4,
        )

def eyelink_event_parser(v, a, p=None):
    """Parser which should be equivalent to the Eyelink event parser with the
    standard (cognitive settings)

    Parameters
    ----------
    v : array
        Velocities (in degrees/second)

    a : array
        Acceleration (in degrees/second$^2$)

    p : Bunch
        Eyelink parameters, module defaults are used if this is None

    #d : Events
    #    the timestamped data which should have attributes .v for velocity and
    #    .a for acceleration

    Returns
    -------
    startidx,stopidx : array,array
        Indexes of the start and stop epochs for the saccade

    #saceps : Epochs
    #    The saccades as defined by Eyelink's heuristics / criteria

    Notes:
    ------
    Assumes 1ms sampling rate
    """
    if p is None:
        p = get_default_params()
    # following https://www.sr-support.com/forums/showthread.php?t=37
    saccade_signal = np.logical_or(v >= p.v_thresh, a >= p.a_thresh)

    start_sig = np.convolve(saccade_signal, np.ones(p.saccade_onset_verify_time))
    start_sig_on = start_sig==p.saccade_onset_verify_time
    prev_start_off = np.hstack([False,start_sig[:-1]!=p.saccade_onset_verify_time])
    start = np.logical_and(start_sig_on,prev_start_off)
    start_sig
    stop_sig = np.convolve(True-saccade_signal, np.ones(p.saccade_offset_verify_time))
    stop_sig_on = stop_sig==p.saccade_offset_verify_time
    prev_stop_off = np.hstack([False,stop_sig[:-1]!=p.saccade_offset_verify_time])
    stop = np.logical_and(stop_sig_on,prev_stop_off)

    startidx = np.where(start)[0] - p.saccade_onset_verify_time +1
    stopidx= np.where(stop)[0] - p.saccade_offset_verify_time+1
    starts,stops = [],[]

    while len(startidx):
        beg = startidx[0]
        if (stopidx>beg).any():
            end= stopidx[np.where(stopidx>beg)][0]
        else:
            end= len(v)-1
        # before we declare this a saccade, let's check if there are any
        # missing values in this slice, which would suggest a blink
        if hasattr(v, 'mask') and v.mask[beg:end+1].any() or np.isnan(v[beg:end+1]).any():
            # blink!
            print "found a blink"
            pass
        else:
            starts.append(beg)
            stops.append(end)
        #starts.append(beg)
        #stops.append(end)
        #idx = end # the next saccade should start AFTER this one ended
        startidx = startidx[np.where(startidx>end)]
    # exhausted the start indexes, so we ignore whatever stops were left
    #end = stopidx[np.where(stopidx>beg)][0]
    return starts,stops

    #try:
    #    while 1:
    #        beg = startidx[np.where(startidx>idx)][0]
    #        end = stopidx[np.where(stopidx>beg)][0]
    #        print beg,end
    #        idx = end
    #except IndexError:
    #    end = 1
    #    return starts,stops

