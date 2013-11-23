from __future__ import print_function
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
            online_vthresh = 2,
            sacslope =  1.70622,
            sacintercept = -12.089,
            framelag=5, # the amount of time to anticipate
            )
    p.__dict__.update(kw)
    return p

testParams = get_default_params(
        # for testing purposed, these need to be
        saccade_onset_verify_time = 2,
        saccade_offset_verify_time = 4,
        )

def timeleft_from_vpeak(vpeak,m,b):
    """ 
    Predict how much time is left in the saccade from its peak velocity
    `vpeak`.  `m` and `b` are the slope and intercept of the linear predictor,
    respectively.
    
    For now, we're just using a linear fit: y=mx+b, so x = (y-b)/m
    """

    return (vpeak-b)/m

def pitjb_online_parser_fast(v, p=None):
    """Online parser which extracts saccades based on peak velocity that is
    above some treshold.

    v : array
        velocity in *pixels* per sample

    XXX: not working at the moment
    """
    from Dimstim.multipylink import timeleft_from_vpeak
    if p is None:
        p = get_default_params()
    v_over_thresh, = np.where(v > p.online_vthresh)

    tleft = timeleft_from_vpeak(v[v_over_thresh],
            p.sacslope,p.sacintercept).astype(np.int)
    ends = v_over_thresh+tleft
    # is the sample larger that all other velocities before the predicted end
    # time 
    # XXX: this needs to be modified by our framelag / framerate, see
    # Dimstim.FixBarsAsync for details , in particular lines that include
    # either "ttl < 10 and ttl > 4" or "ttl < 5 and ttl >= 0"
    # this won't work - we'll get a ragged array depending on when stuff ends
    #mask = v[v_over_thresh] > v[v_over_thresh+np.arange(ends)].all()
    mask = [(v[b] > v[b+1:e]).all() for b,e in zip(v_over_thresh, ends)]
    mask = np.array(mask)
    # if the mask condition is false, that means we would have avoided this
    # sample, but if it's true, we still may have avoided this sample, if
    # there's a sample before it that caused us to make a prediction further
    # into the past.
    print(mask)
    print(v_over_thresh)
    if mask.sum() > 1:
        # now every peak thats at mask=True is the maximum until the end of its
        # predicted saccade. If there are any other peaks before this end, we
        # will ignore them (unless they occur after we've gone below
        # threshold), in which case we'll re-predict a new saccade. How do we
        # account for this in the code? what a headache!
        mask2 = ends[mask][:-1] < v_over_thresh[mask][1:]
        mask[1:][~mask2] = False
        print(mask2)

    return v_over_thresh[mask]

def pitjb_online_parser(v, p=None):
    if p is None:
        p = get_default_params()
    v_peak = p.online_vthresh
    peaks,ends = [],[]
    pe = 0,0 # peaktime, sacendtime tuple which we'll use 
    # XXX: it's slow to iterate over the velocity like this, but it's the
    # easiest way to have code that does the same thing as what our
    # online-parser currently does
    for i in range(len(v)):
        if i == pe[1] - p.framelag:
            ends.append(pe[1])
            peaks.append(pe[0])
        if v[i] >p.online_vthresh:
                #print "Got higher than v_thresh"
                if v[i] > v_peak:
                    v_peak = v[i]
                    peaktime = i 
                    sacendtime = peaktime +  timeleft_from_vpeak(v_peak,
                            p.sacslope,p.sacintercept)
                    sacendtime = int(sacendtime)
                    pe = peaktime,sacendtime
                    # communicate the predicted fixation position to visionegg display loop
                    # just put a random position for now
                    #fix_position[:] = np.random.rand() * 800, np.random.rand() * 600
                    #fix_position[:] = rbuf[i-1]
                    #print "updating v_peak", v[i], " \t curtime:", i, "end time: ",sacendtime
                    #peaks.append(i)

        else:
            # we're slower than threshold again, must be in a fixation epoch
            v_peak = p.online_vthresh
    return np.array(peaks),np.array(ends)

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

