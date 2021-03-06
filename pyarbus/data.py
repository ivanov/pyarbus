"""
Library of functions related to eye movement data input/output
"""
from __future__ import with_statement

import re
import os
import numpy as np
import nitime
from nitime import Events, Epochs
from io import BytesIO
import gzip
import inspect
from . import utils
from . import viz

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

__all__ = [ 'Eye','Eyelink', 'reprocess_eyelink_msgs', 'findall_loadtxt',
'read_eyelink', 'read_eyelink_cached', 'EyelinkReplayer']

# a dictionary to cache eyelink data in memory, used by read_eyelink_cached
# XXX: come up with a better repr for the cache - maybe subclass dict and just
# override the repr for saying how many objects are stored.  Something like
# "expname.asc: subj: duration, binocular" - this would then be useful for
# summarizing multiple experiments.
_cache = {}

# gaze_dtype dictionary which is meant to be indexed by the `binocular`
# boolean value
gaze_dtype = {
        False : np.dtype([ # dtype for monocular data
            ('time', 'uint64'),
            ('x','float64'),
            ('y','float64'),
            ('pupA','float64')]),
        True : np.dtype([ #dtype for binocular data
            ('time', 'uint64'),
            ('x','float64'),
            ('y','float64'),
            ('pupA','float64'),
            ('x2','float64'),
            ('y2','float64'),
            ('pupA2','float64')])
        }

vel_dtype = {
        # dtype for monocular velocity
        False : [ ('xv','float64'), ('yv','float64'), ],
        # dtype for binocular velocity
        True : [ ('xv','float64'), ('yv','float64'),
            ('x2v','float64'), ('y2v','float64'), ]
        }

res_dtype = [ ('xres','float64'), ('yres','float64'), ]

# gaze_col dictionary which is mean to be indexed by the `binocular`,
# `velocity`, and `res` boolean values, in that order
gaze_cols = {
        # Monocular:
        #        <time> <xp> <yp> <ps>
        (False,False,False) : (0,1,2,3), # gaze columns for monocular data
        # Monocular, with velocity
        #        <time> <xp> <yp> <ps> <xv> <yv>
        (False,True ,False) : (0,1,2,3,4,5),
        # Monocular, with resolution
        #     <time> <xp> <yp> <ps> <xr> <yr>
        (False,False,True ) : (0,1,2,3,4,5),
        # Monocular, with velocity and resolution
        #     <time> <xp> <yp> <ps> <xv> <yv> <xr> <yr>
        (False,True ,True ) : (0,1,2,3,4,5,6,7),
        # Binocular
        #     <time> <xpl> <ypl> <psl> <xpr> <ypr> <psr>
        (True ,False,False) : (0,1,2,3,4,5,6), # gaze columns for binocuclar data
        # Binocular, with velocity
        #     <time> <xpl> <ypl> <psl> <xpr> <ypr> <psr> <xvl> <yvl> <xvr> <yvr>
        (True ,True ,False) : (0,1,2,3,4,5,6,7,8,9,10),
        # Binocular, with and resolution
        #     <time> <xpl> <ypl> <psl> <xpr> <ypr> <psr> <xr> <yr>
        (True ,False,True ) : (0,1,2,3,4,5,6,7,8),
        # Binocular, with velocity and resolution
        #      <time> <xpl> <ypl> <psl> <xpr> <ypr> <psr> <xvl> <yvl> <xvr> <yvr> <xr> <yr>
        (True ,True,True ) : (0,1,2,3,4,5,6,7,8,9,10,11,12),
        }

def get_gaze_col_dtype(binocular, velocity, res):
    """Build complex dtype for reading samples, depending on the binary values
    of the 3 parameters
    """
    gcols = gaze_cols[binocular, velocity, res]
    gdtype = gaze_dtype[binocular]
    if velocity:
        gdtype = np.dtype(gdtype.descr+vel_dtype[binocular])
    if res:
        gdtype = np.dtype(gdtype.descr+res_dtype)
    return gcols, gdtype

class Saccades(nitime.Events):
    """contains several field specific to saccades"""
    @property
    def xi(self):
        return self.data['xi']
    @property
    def xf(self):
        return self.data['xf']
    @property
    def yi(self):
        return self.data['yi']
    @property
    def yf(self):
        return self.data['yf']
    @property
    def amplitude(self):
        return self.data['amplitude']
    @property
    def start(self):
        return self.data['start']
    @property
    def stop(self):
        return self.data['stop']
    @property
    def vpeak(self):
        return self.data['vpeak']

    def __getitem__(self,k):
        return Saccades(self.time[k], start=self.start[k],stop=self.stop[k],
                amplitude=self.amplitude[k], vpeak=self.vpeak[k],
                xi=self.xi[k], xf=self.xf[k], yi=self.yi[k], yf=self.yf[k])

class Eye(nitime.Events, nitime.descriptors.ResetMixin):
    """Class used for monocular eyetracking data

    Under the hood, it uses the nitime.Events class, but presents the event
    data keys as attributes, so that after you declare an eye instance,

    >>> time = np.linspace(0,10,11)
    >>> x,y = np.sin(time),np.cos(time)
    >>> eye = Eye(time, **dict(x=x,y=y))

    You can then access the data member directly using ``eye.x`` instead of the
    more awkward ``eye.data['x']``.

    Typically, an Eye will have the following attributes:
    x : horizontal component
    y : vertical component
    pupA : pupil area
    xres : horizontal resolution (used for converting to degrees)
    yres : vertical resolution (used for converting to degrees)

    Additionally, an instance of an Eye class has the following attributes

    >>> eye.blinks   # None or Epochs
    >>> eye.discard  # None or Epochs when data should be discarded
    >>> eye.saccades # None or Saccades
    >>> eye.sampling_rate # None or float, used in vel calculation
    >>> eye.sacepochs   # None or Epochs which correspond to the saccades

    XXX: That last bit above is hairy - saccades should subclass epochs, and
    then have extra attributes hanging off of them.

    There are also attributes which are computed from the components listed
    above. You can compare the smoothed velocity provided by Eyelink parser
    with that calculated from the samples themselves using something like this:

    >>>  plt.hist((eye.eyelink_vel-eye.vel),bins=np.linspace(-40,40,100))

    """
    def __init__(self, time, sampling_rate=None, *args, **kwargs):
        nitime.Events.__init__(self,time,**kwargs)
        self.sampling_rate = sampling_rate

    blinks = None
    discard = None
    saccades = None
    sacepochs = None
    sampling_rate = None
    vel_type = 'central'
    def __getattr__(self,k):
        return self.data[k]

    @property
    def vel(self):
        """
        Eye velocity calculated from the samples.

        ``eye.vel`` is calculated and cached the first time it is
        accessed, using the sample-centered difference by default. The
        sample-centered difference velocity at point ``t`` is ::

            $ v[t] = \frac{p[t+1] - p[t-1]}{2} $

        Where p is the x,y coordinates of the eye.

        To have ``eye.vel`` be recalculated as an adjacent sample difference,
        such that ::

            $ y[t] = p[t] - p[t-1] $

        you will have to change ``eye.vel_type`` to something other than
        'central'. Here's an example:

        >>> eye.reset() # reset the cached eye.vel property, if it's been set
        >>> eye.vel_type = 'noncentral'
        >>> eye.vel     # will now be v[1] = p[1] - p[0]

        If ``eye.xres`` and ``eye.yres`` are available, they are included in
        the velocity calculation, by dividing the change in x and y direction
        by the average of the xres and yres of those two point, as per the
        Eyelink manual. See ``pyarbus.utils.velocity`` code for details.

        If ``eye.sampling_rate`` is available then the velocity returned is in
        units/second, otherwise the quanitity is in units/sample.
        """
        return self._vel

    @nitime.descriptors.auto_attr
    def _vel(self):
        return utils.velocity(self.x,self.y,
                use_central= (self.vel_type=='central'),
                sampling_rate=self.sampling_rate, xres=self.xres,
                yres=self.yres)

    @property
    def accel(self):
        """
        Eye acceleration calculated from the data samples.

        See ``eye.vel`` for details on how this is calculated, cached, and
        what the options are for changing it. ``eye.vel_type`` determines
        which method is used in calculating both the acceleration and the
        velocity.

        Here's an illustrative plot on the affects of the central-difference
        method on the adjacent sample noise.

        >>> eye.reset()
        >>> eye.vel_type='central'
        >>> plt.plot(eye.accel,'b')
        >>> eye.reset()
        >>> eye.vel_type=''
        >>> plt.plot(eye.accel,'r')
        """
        return self._accel

    @nitime.descriptors.auto_attr
    def _accel(self):
        return utils.acceleration(self.vel,
                use_central=(self.vel_type=='central'))

    @property
    def eyelink_vel(self):
        """Velocity as provided by the Eyelink EDF2ASC parser

        See ``eye.vel`` for details on how this is cached.
        """
        return self._eyelink_vel

    @nitime.descriptors.auto_attr
    def _eyelink_vel(self):
        return np.ma.sqrt(self.xv**2 + self.yv**2)

    @property
    def eyelink_accel(self):
        """
        Acceleration as calcutated from the velocity provided by the
        Eyelink EDF2ASC parser.

        See ``eye.vel`` for details on how this is calculated, cached, and what the
        options are for changing it. ``eye.vel_type`` determines which method
        is used in calculating the acceleration
        """
        return self._eyelink_accel

    @nitime.descriptors.auto_attr
    def _eyelink_accel(self):
        return utils.acceleration(self.eyelink_vel)

class Eyelink(object):
    """
class for Eyelink data

Has x,y,pupA time series, as well as saccade and discard epochs,  msgs strings
extracted from an Eyelink .asc file. All members are also attributes (you can
use e.x and e.saccades as a shorthand for e['x'] and
e.['saccades']

:Members:
    XXX: these are actually a level further down, depending on the eye_used
    e["x"] : TimeSeries
        The x-axis trace
    e["y"] : TimeSeries
        The y-axis trace
    e["pupA"] : TimeSeries
        Pupil area trace

    e["saccades"] : Epochs or None
        Epochs which contain saccades
    e["blinks_l"] : Epochs or None
    e["blinks_r"] : Epochs or None
        Blinks from the left and right eyes
    e["discard"] : Epochs or None
        Epochs to be discarded because they contain a blink (see Notes below)


    e["msgs"] : tuple(str) or None
        Lines in Eyelink file which start with MSG (such as "Stimulus paused"),
    e["raw"] : str
        Contains the text which was not processed into one of the above

:SeeAlso:
    - read_eyelink : function which returns an Eyelink object

Notes
-----
According to Eyelink manual 1.3 4.5.3.5 Blinks (p. 98)

    "Blinks are always preceded and followed by partial occlusion of the and
    pupil, causing artificial changes in pupil position.  These are sensed by
    the EyeLink 1000 parser, and marked as saccades. The sequence of events
    produced is always:
    - start saccade (SSACC)
    - start blink   (SBLINK)
    - end blink     (EBLINK)
    - end saccade   (ESACC)
    ... All data between SSACC and ESSAC events should be discarded. "

In the Eyelink 1.4 manual, p 108:

    Blinks are always embedded in saccades, caused by artificial motion as the
    eyelids progressively occlude the pupil of the eye. Such artifacts are best
    eliminated by labeling and SSACC...ESACC pair with one or more SBLINK
    events between them as a blink, not a saccade.

    It is also useful to eliminate any short (less than 120 millisecond
    duration) fixations that precede or follow a blink. These may be artificial
    or be corrupted by the blink.

XXX: We do not currently eliminate short (<120ms) fixations that precede or
follow a blink

    """

    msgs = None
    raw = None

    # subclasses can extend this list, and the object will repr properly
    _extra_attrs = ['msgs']


    def __init__(self,fname='dummyfile', binocular=False, have_right=True,
            have_left=False,have_vel=False,have_res=False,samplingrate=None,
            from_eyelink=None, msgs=None):

        if from_eyelink:
            self.__dict__.update(from_eyelink.__dict__)
            return
        #Container.__init__(self)
        self._samplingrate = samplingrate
        self.binocular = have_left and have_right
        self.have_right = have_right
        self.have_left = have_left
        self.have_vel = have_vel
        self.have_res = have_res
        # XXX: this is crap, we should not define the heirarchy like this be
        # hand on init. Split off the "eyes" into their own class and
        # have that init do some of this work for us.
        self.r = None
        self.l = None
        self.msgs = msgs
        self.raw = None
        self._fnamelong = fname
        self._fname = fname.split('/')[-1].split('.')[0]

    def __repr__(self):
        rep  =  "Eyelink Data:"
        if self.binocular: rep += " binocular"
        elif self.have_right: rep += " monocular (right eye)"
        else: rep += " monocular (left eye)"
        for x in self._extra_attrs:
            if self.__getattribute__(x) is not None:
                rep +="\n %d %s ["%(len(self.__getattribute__(x)),x)
                rep += self.__getattribute__(x)[0].__repr__()
                rep += " ...]"
        raw = self.raw.split('\n')

        if self.have_left:
            rep += "\nLeft " +  repr(self.l)
        if self.have_right:
            rep += "\nRight "+ repr(self.r)
        rep += "\n"+raw[0]
        rep += "\n"+raw[1]
        rep += "\n"+raw[5]
        return rep

    def replayer(self):
        return EyelinkReplayer(el=self)

    def do_discard(self, pad=25):
        """ Mask out data in eye.x and eye.y during discard epochs
        """
        eye = self.eye_used

        if eye.discard is None:
            log.info("nothing to discard")
            return
        
        #XXX: should masks be shared (from the beginning) -that way we'd have
        # something like MaskedEvents, which all data attributes look onto
        # toward
        #eye.x.v.mask = eye.y.v.mask = mask
        #eye.xres.v.mask = eye.yres.v.mask = mask
        #eye.pupA.v.mask = mask

        for ep in eye.discard:
            # XXX: the current nitime implementation will cause the next line
            # to return events again, instead of another Eye
            discarded_data=eye[ep]
            # this relies on the above slicing operation to return a view on
            # the same data as 'eye' and to not make a copy
            for k in eye.data:
                discarded_data.__getattr__(k).mask[:] = True
            #mask[start-pad:stop+pad] = True

    @property
    def eye_used(self):
        """ Return self.l or self.r, depending on which eye has data, raises an
        error if binocular"""

        if self.binocular:
            raise AttributeError("Container is binocular, both eyes are available")
        if self.have_right:
            return self.r
        else:
            return self.l

    @property
    def samplingrate(self):
        """ the sampling rate taken from the RECCFG message"""
        return self._samplingrate or float([msg.split()[4] for msg in self.msgs if 'RECCFG' in msg][0])


    @property
    def fname(self):
        """short filename for this recording"""
        return self._fname

    @property
    def fnamelong(self):
        """Long (full) filename of this recording"""
        return self._fnamelong

    def plot_xyp(self, **kwargs):
        """Plot the position and pupil area for this eye, each in its own
        subplot.

        If this Eyelink object has binocular data, two figures will be created
        """
        self._viz_defer(**kwargs)
    
    def plot_xy_p(self, **kwargs):
        """Plot the position and pupil area for this eye, in two seperate
        subplots.

        If this Eyelink object has binocular data, two figures will be created
        """
        self._viz_defer(**kwargs)

    def _viz_defer(self, **kwargs):
        """
        A helper method to unify calls to viz.plot_* methods for each eye

        Notes
        -----
        This method should only be called by functions with names matching
        functions in viz, that take an Eye as the first argument, since
        _viz_defer uses the :mod:`inspect` module to resolve the calling
        function's name.
        """
        if self.binocular:
            eyes = [self.l, self.r]
        else:
            eyes = [self.eye_used]
        for e in eyes:
            viz.__dict__[inspect.stack()[1][3]](e, **kwargs)

def reprocess_eyelink_msgs(pattern, msgs, cols=(0,), dtype=None):
    """
    Takes the messages, creates a temporary buffer out of them, and reuses
    the machinery of findall_loadtxt to search for the pattern

    The pattern will be prepended with the `(?<=MSG.)\d+ ` regular expression,
    and appended with `.*` by default.

    If you don't supply a cols argument, you'll just get the message timestamp
    (column 0) as a float.

    Example:
    --------
        Grab just the timestamp of messages which start with pupArt:
        >>> reprocess_eyelink_msgs('pupArt', el.msgs)

    XXX: there's probably a better way of doing this than writing to a
    temporary buffer and reprocessing it, but this allows me to reuse machinery
    that's already there. and LTS. -pi
    """
    if not isinstance(pattern, bytes):
        pattern = pattern.encode('ascii') # if this breaks, please report bug
    pat = b"(?<=MSG.)\d+ "+pattern+b".*"
    return findall_loadtxt(pat, b"\n".join(msgs), cols, dtype)


def findall_loadtxt(pattern, raw, cols, dtype=None):
    matches = re.findall(pattern,raw, re.M)
    str = b"\n".join(matches)
    tmp = BytesIO(str)
    if len(matches) == 0:
        return np.array([])

    #ret = np.empty(len(matches),dtype=dtype)
    #for i,match in enumerate(matches):
    #    ret[i] = np.loadtxt(StringIO(match),dtype=dtype,usecols=cols)

    if cols == "all":
        #f = tempfile.TemporaryFile()
        #f.write(str)
        #f.seek(0)
        #ret = np.fromfile(f,dtype=dtype,count=len(matches), sep='\n')
        #f.close()
        #ret = np.fromiter(iter(matches),dtype=dtype,count=len(matches))
        # lelt's just read it all as strings, and then deal with the
        # consequences later

        # get rid of trailing 'flags' on newlines, flags look like
        # .C.C. and ICC.. at the end of lines
        regex = re.compile(b"\t[\.RIC]{2,5}")
        str_new = regex.sub(b'', str)
        retfloat = np.fromstring(str_new, dtype=float, sep=' ')
        retfloat.shape=len(matches),-1
        ret = np.empty(len(matches),dtype=dtype)

        for i,c in enumerate(retfloat.T):
            ret[dtype.names[i]] = dtype[i].type(c)
        ret
    else:
        ret = np.loadtxt(tmp,dtype=dtype,usecols=cols)
    #tmp.close()
    return ret

def read_eyelink(filename,Eyelink=Eyelink):
    """
Read in Eyelink .asc file,

Returns obect containing eye position time series, along with pupil area,
saccades and blink data as parsed by Eyelink, and any messages

:Parameters:
    filename : str
        name of .asc file to be read

    Eyelink : Eyelink (optional)
        Eyelink class as defined in pyarbus.data, or a subclass thereof

:Returns:
    eye : Eyelink (as passed)
        container with 'x','y','pupA' TimeSeries, 'saccade' and 'discard'
        Epochs.

:SeeAlso:
  - Eyelink : class of the return object of this function

Notes
-----

Though the functionality here *should* work with all formats produced by
``edf2asc`` -- it is most extensively tested with with .edf files proccessed
with the following edf2asc parameter set:  ``-y -z -vel -res -nflags``

Examples
--------

>>> eye = read_eyelink('data/pi.asc')
>>> eye['x'], eye['y'], eye['saccades'][:2]
(TimeSeries([ 398.8,  398.8,  398.8, ...,  350.2,  355.5,  361.1]),
 TimeSeries([ 301.1,  301. ,  300.9, ...,  547.5,  512.4,  478.9]),
 Epochs([(18984432.0, 74.0, 0.0, 18984506.0),
       (18984800.0, 6.0, 0.0, 18984806.0)],
      dtype=[('tstart', '<f8'), ('duration', '<f8'), ('t0_offset', '<f8'), ('tstop', '<f8')]))
>>>  print g['raw'][:200]
** CONVERTED FROM BALDI009.EDF using edfapi 3.0 Linux Jun 18 2008 on Wed Oct 22 16:47:03 2008
** DATE: Thu Mar  2 15:38:30 2006
** TYPE: EDF_FILE BINARY EVENT SAMPLE TAGGED
** VERSION: EYELINK II 1
**
"""
    if filename.endswith('gz'):
        f = gzip.open(filename)
        # looks like mmaping all of this makes no difference
        # mmap needs a fileno, so we'll read into a tempfile
        #import mmap
        #import tempfile
        #tempf = tempfile.TemporaryFile()
        #tempf.write(f.read())
        ## ensure that we mmap the whole file, not a partially written one
        #tempf.flush()
        #raw = mmap.mmap(tempf.fileno(), 0, access=mmap.ACCESS_READ)
        #raw = f.read()
    else:
        f = open(filename, 'rb')
        #import mmap
        #raw = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        #raw = f.read()
    # let's try mmap for efficiency
    # XXX: seems like mmaping was not necessary here - at least i can't see a
    # difference right now - need to test with reading lots of large files in
    # a row - so that they would flush one another out of cache to see if
    # mmaping makes a difference
    raw = f.read()

    # filter these redundant lines early, their E prefix counterparts contain all of the data
    # XXX: not true in HINAK075.ASC as parsed by before 2006 edf2asc - SSACS followed by blinks do not
    # have  a proper ESSAC after (but instead have a single sample long ESSAC)
    for x in [b"SFIX",b"SSACC",b"SBLINK"]:
        raw = re.sub(b"\n"+x+b".*",b"", raw)

    # from Eyelink manual 1.4 p 106:
    # Each type of event has its own line format. These use some of the data
    #items listed below. Each line begins with a keyword (always in uppercase) and
    #items are separated by one or more tabs or spaces.
    #DATA NOTATIONS
    #<eye>                 which eye caused event ("L" or "R")
    #<time>                timestamp in milliseconds
    #<stime>               timestamp of first sample in milliseconds
    #<etime>               timestamp of last sample in milliseconds
    #<dur>                 duration in milliseconds
    #<axp>, <ayp>          average X and Y position
    #<sxp>, <syp>          start X and Y position data
    #<exp>, <eyp>          end X and Y position data
    #<aps>                 average pupil size (area or diameter)
    #<av>, <pv>            average, peak velocity (degrees/sec)
    #<ampl>                saccadic amplitude (degrees)
    #<xr>, <yr>            X and Y resolution (position units/degree)



    gc_gaze_dtype = np.dtype([('time', 'uint64'),
                           ('x','float64'),
                           ('y','float64')])

    gcdisp = findall_loadtxt(b"(?<=MSG.)\d+\ GCDISP.\d+.*",raw,(0,2,3),
            dtype=gc_gaze_dtype)





    #grab binocularity from the EVENTS GAZE line
    start = re.search(b"^START.\d+\ [^\d].*", raw, re.M).group().upper()
    samples_line = re.search(b"^SAMPLES.*", raw, re.M).group()
    log.info(samples_line)

    binocular = False
    have_right = start.find(b'RIGHT') != -1
    have_left= start.find(b'LEFT') != -1
    if have_right and have_left:
        binocular = True

    velocity = b"VEL" in samples_line
    res = b"RES" in samples_line

    gcols, gdtype = get_gaze_col_dtype(binocular, velocity, res)
    verbose_string = " ".join(
            [str(x) for x in (binocular, velocity, res, gcols, gdtype)])
    log.debug( verbose_string )

    # EL1.4 p 103: 4.9.2 Sample Line Format (see DATA NOTATIONS for <codes>)
    # -----------------------
    #Sample lines contain time, position, and pupil size data. Optionally, velocity and
    #resolution data may be included. Several possible sample line formats are
    #possible. These are listed below.
    #Essentially, each sample line begins with a timestamp. Recordings done with a
    #2000 hz sampling rate will have two consecutive rows of the same time stamps.
    #The second row refers to the sample collected at 0.5 ms after the reported time
    #stamp. The time stamp field is followed by X and Y position pairs and pupil size
    #data for the tracked eye, and optionally by X and Y velocity pairs for the eye,
    #and resolution X and Y values. Missing data values are represented by a dot
    #("."), or the text specified by the "-miss" option to EDF2ASC.
    #         SAMPLE LINE FORMATS
    # Monocular:
    #        <time> <xp> <yp> <ps>
    # Monocular, with velocity
    #        <time> <xp> <yp> <ps> <xv> <yv>
    # Monocular, with resolution
    #     <time> <xp> <yp> <ps> <xr> <yr>
    # Monocular, with velocity and resolution
    #     <time> <xp> <yp> <ps> <xv> <yv> <xr> <yr>
    # Binocular
    #     <time> <xpl> <ypl> <psl> <xpr> <ypr> <psr>
    # Binocular, with velocity
    #     <time> <xpl> <ypl> <psl> <xpr> <ypr> <psr> <xvl> <yvl> <xvr> <yvr>
    # Binocular, with and resolution
    #     <time> <xpl> <ypl> <psl> <xpr> <ypr> <psr> <xr> <yr>
    # Binocular, with velocity and resolution
    #      <time> <xpl> <ypl> <psl> <xpr> <ypr> <psr> <xvl> <yvl> <xvr> <yvr> <xr> <yr>
    # -----------------------
    # XXX: for now, we'll only support monocular and binocular formats, will include others later

    #replace missing fields. with NaN XXX: should this NaN be user defined, instead of hardcoded?
    raw = re.sub(b'   .\t',b'  nan\t',raw)
    raw = re.sub(b'\.\.\.',b'',raw)


    # the bulk of the time for this function is spent here - XXX: any speedup
    # in this code / approach will be very welcome. For example - setting cols
    # to None here speeds up the calls to np.loadtxt by about 50%
    gaze = findall_loadtxt(b"^\d+.*",raw,'all',gdtype)
    raw = re.sub(b"\n\d+.*",b"",raw) # get rid of lines which we've already

    link_fix = findall_loadtxt(b"(?<=MSG.)\d+\ Fixation",raw,(0,),
            dtype=np.uint64).astype(float)

    # get MSGs
    msgsstr = re.findall(b"^MSG.\d+\ .*", raw, re.M)

    #NOTE: If the eyetracking data contains calibrations, then saccade
    #and blinks times will be off. Time series assumes all data sampled
    #continuously, but no data is sent during a calibration. we'll take
    #the approach of figuring out when there's missing data and adding it
    #into the timeseries with zeros (so it stays one continuous thing)
    prev = 0
    tmp = {}
    for fn in gaze.dtype.names:
        tmp[fn] = np.array([])
    # tmp[0] will be our time - change its dtype to be unsigned int
    t = tmp["time"]
    t.dtype='uint64'
    missing_tstamp= np.array([])

    #Use the eyelink-reported samplerate
    samplingrate = findall_loadtxt(b"RATE[\t ]*\d+.\d*",raw,(1,))
    dt = int(1000/samplingrate[0])

    #XXX: throw error if diff.gaze(['time']) is ever either 0 or negative (samples repeated or out of order)
    for D in np.where(np.diff(gaze['time']) != dt)[0]:
        log.warn("Discontinuity in eyetracker time series")
        log.warn("   at sample %d, time %s",D,str(gaze['time'][D:D+2]))
        ## XXX: implement a "fill-discontinuous" flag to have this
        ## functionality again - it was needed for neuropy's events and
        ## timeseries implementation, but no longer necessary with nitime
        missing_tstamp= np.concatenate((missing_tstamp,gaze['time'][D:D+2]))
        t = np.concatenate((t,gaze['time'][prev:D+1],
                            np.arange(gaze['time'][D]+dt,
                                      gaze['time'][D+1],
                                      dt,dtype='uint64')))
        # missing values stored as NaNs
        z = np.ones((gaze['time'][D+1]- gaze['time'][D]-dt)/dt) * np.nan
        # .names[1:] skips over the 'time' field
        for fn in gaze.dtype.names[1:]:
            tmp[fn] = np.concatenate((tmp[fn],gaze[fn][prev:D+1], z))
        prev = D+1

    # iterate over all fields
    tmp['time'] = t
    for fn in gaze.dtype.names:
        tmp[fn] = np.concatenate((tmp[fn],gaze[fn][prev:]))

    gaze = np.zeros(len(tmp['time']),dtype=gdtype)
    for fn in gaze.dtype.names:
        gaze[fn] = tmp[fn]

    raw= re.sub(b"\nMSG.*", b"", raw) # extracted

    # See Notes in the Eyelink class docstring
    # Basically, all we need to do is find those ESACC events which are
    # preceded by EBLINK events, and discard the entire saccade

    #TODO: a better way of doing this is to separate the streams (as we do
    #later anyway) and call a function to parse the two streams seperately...
    #...but life's too short for now

    # convert endblinks to fixed length (so we can use lookback)
    blinks_r = findall_loadtxt(b"(?<=EBLINK.)R\ \d+\t\d+",raw,(1,2))
    blinks_l = findall_loadtxt(b"(?<=EBLINK.)L\ \d+\t\d+",raw,(1,2))
    blinks_r.shape = -1,2
    blinks_l.shape = -1,2


    #XXX: previous approach does not worksince R and L can be staggered relative to one another
    # soln: split the left and right streams, and process them individually
    def grab_raw_monocular(raw,eye='R'):
        """ this function removes any lines that pertain to the oposite eye"""
        raw_ret = raw
        omit = b" L"
        if eye=='L':
            omit = b" R"
        for x in [b"EBLINK",b"ESACC",b"EFIX"]:
            raw_ret = re.sub(x+omit+b".*\n",b"",raw_ret)

        return raw_ret

    raw_l = grab_raw_monocular(raw,'L')
    raw_r = grab_raw_monocular(raw,'R')
    raw_l= re.sub(b"EBLINK.*",b"EBLINK",raw_l)
    raw_r= re.sub(b"EBLINK.*",b"EBLINK",raw_r)
    #1/0
    # lookback and return endsaccades which are preceded by an endblink
    discard_r = findall_loadtxt(b"(?<=EBLINK\nESACC...).*\d+\t\d+",raw_r,(0,1))
    discard_l = findall_loadtxt(b"(?<=EBLINK\nESACC...).*\d+\t\d+",raw_l,(0,1))
    # XXX: separate timestamp gaps with blinks (and maybe have a method that
    # reports the OR of all the crap
    if have_right:
        discard_r = np.append(missing_tstamp,discard_r)
    if have_left:
        discard_l = np.append(missing_tstamp,discard_l)
    discard_l.shape = -1,2
    discard_r.shape = -1,2

    #get rid of lines containing falsely reported ESACCS which were preceded by EBLINK
    # XXX: now I'm getting paranoid - are we excluding more ESACCs than we
    # should if there were multiple blinks ?
    raw_r = re.sub(b"(?<=EBLINK)\nESACC.*",b"",raw_r)
    raw_l = re.sub(b"(?<=EBLINK)\nESACC.*",b"",raw_l)
    # lookback and return endsaccades (see DATA NOTATIONS for <codes>)
    #ESACC   <eye> <stime> <etime> <dur> <sxp> <syp> <exp> <eyp> <ampl> <pv>

    saccades_r = findall_loadtxt(b"(?<=ESACC...).*",raw_r,(0,1,7,8,3,4,5,6)).astype(float)
    saccades_l = findall_loadtxt(b"(?<=ESACC...).*",raw_l,(0,1,7,8,3,4,5,6)).astype(float)
    saccades_r.shape = -1,8
    saccades_l.shape = -1,8

    # (see DATA NOTATIONS for <codes>)
    #EFIX <eye> <stime> <etime> <dur> <axp> <ayp> <aps>
    fix_r= findall_loadtxt(b"(?<=EFIX...).*",raw_r,(0,1,3,4,5)).astype(float)
    fix_l= findall_loadtxt(b"(?<=EFIX...).*",raw_l,(0,1,3,4,5)).astype(float)
    fix_r.shape = -1,5
    fix_l.shape = -1,5

    raw = re.sub(b"\n[ES]SACC.*",b"",raw)  # get rid of lines which we've already
    raw = re.sub(b"\n[ES]BLINK.*",b"",raw) # extracted

    el = Eyelink(fname=filename, binocular=binocular, have_right=have_right,
            have_left=have_left, have_vel=velocity, have_res=res,
            samplingrate=samplingrate[0], msgs=msgsstr)

    time = gaze['time'] 
    if binocular:
        mi = np.ma.masked_invalid
        # names[0] is 'time', everything else is x,y,pupA,
        names = list(gaze.dtype.names)
        left_eye = names[1:4]
        right_eye = names[4:7]
        if velocity:
            left_eye.extend(names[7:9])
            right_eye.extend(names[9:11])
        if res:
            # res fields are always the last two, if present
            left_eye.extend(names[-2:])
            right_eye.extend(names[-2:])

        dl = dict([(n,mi(gaze[n])) for n in left_eye])
        el.l = Eye(time, time_unit='ms', sampling_rate=samplingrate[0], **dl)

        # so far we've called right eye data x2,y2,pupA2 - rename these to x,y,pupA
        dr = dict([(n.replace('2',''),mi(gaze[n])) for n in right_eye])
        el.r = Eye(time, time_unit='ms', sampling_rate=samplingrate[0], **dr)

    else:
        mi = np.ma.masked_invalid
        d = dict([(n,mi(gaze[n])) for n in gaze.dtype.names[1:]])
        if have_right:
            el.r = Eye(time, time_unit='ms', sampling_rate=samplingrate[0], **d)
        else:
            el.l = Eye(time, time_unit='ms', sampling_rate=samplingrate[0], **d)

    #XXX: wrap this up into a loop for neatness / brevity
    if blinks_r.size:
        el.r.blinks = Epochs(blinks_r[:,0], blinks_r[:,1],time_unit='ms')

    if blinks_l.size:
        el.l.blinks = Epochs(blinks_l[:,0], blinks_l[:,1], time_unit='ms')

    if discard_l.size:
        el.l.discard = Epochs(discard_l[:,0], discard_l[:,1], time_unit='ms')
    if discard_r.size:
        el.r.discard = Epochs(discard_r[:,0], discard_r[:,1], time_unit='ms')

    if saccades_l.size:
        el.l.saccades = Saccades(saccades_l[:,0],
                start=saccades_l[:,0],stop=saccades_l[:,1], time_unit='ms',
                amplitude=saccades_l[:,2], vpeak=saccades_l[:,3],
                #epochs=Epochs(saccades_l[:,0],saccades_l[:,1]), time_unit='ms',
                xi=saccades_l[:,3], yi=saccades_l[:,4],
                xf=saccades_l[:,5], yf=saccades_l[:,6],
                )
        el.l.sacepochs = Epochs(saccades_l[:,0],saccades_l[:,1], time_unit='ms')
    if saccades_r.size:
        el.r.saccades = Saccades(saccades_r[:,0],
                start=saccades_r[:,0],stop=saccades_r[:,1], time_unit='ms',
                amplitude=saccades_r[:,2], vpeak=saccades_r[:,3],
                #epochs=Epochs(saccades_r[:,0],saccades_r[:,1]), time_unit='ms',
                xi=saccades_r[:,3], yi=saccades_r[:,4],
                xf=saccades_r[:,5], yf=saccades_r[:,6],
                )
        el.r.sacepochs = Epochs(saccades_r[:,0],saccades_r[:,1], time_unit='ms')
    if fix_l.size:
        el.l.fixations = Events(fix_l[:,0],
                start=fix_l[:,0],stop=fix_l[:,1], time_unit='ms',
                xavg=fix_l[:,2], yavg=fix_l[:,3], pavg=fix_l[:,4]
                #epochs=Epochs(fix_l[:,0],fix_l[:,1]), time_unit='ms')
                )
        el.l.fixepochs=Epochs(fix_l[:,0],fix_l[:,1], time_unit='ms')
    if fix_r.size:
        el.r.fixations = Events(fix_r[:,0],
                start=fix_r[:,0],stop=fix_r[:,1], time_unit='ms',
                xavg=fix_r[:,2], yavg=fix_r[:,3], pavg=fix_r[:,4],
                #epochs=Epochs(fix_r[:,0],fix_r[:,1], time_unit='ms')
                )
        el.r.fixepochs=Epochs(fix_r[:,0],fix_r[:,1], time_unit='ms')

    if link_fix.size:
        el.link_fixations = Events(link_fix, time_unit='ms')

    el.raw = raw # contains all the lines which have not been processed
    if gcdisp.size:
        gcdisp =  gcdisp.view(np.recarray)
        el.gcdisp = Events(gcdisp.time, x=gcdisp.x, y=gcdisp.y, time_unit='ms')

    # XXX: kind of wish I could do some interval math on epochs (union,
    # intersection, a la pyinterval)

    try:
        el.do_discard()
    except AttributeError:
        log.warn("Discarding failed")

    f.close()
    return el

def read_eyelink_cached(fname,d=None, **kwargs):
    """
    Read the asc file in `fname` into the dictionary `d` using
    read_eyelink(fname) and cache the results there. On subsequent calls, no
    reading is performed, and the cached results are returned.

    If passed only `fname`, a module-level dictionary is used for the
    in-memory caching. That dictionary can be accessed as pyarbus.data._cache
    """
    if d is None:
        d = _cache

    if (fname in d) == False:
        d[fname] = read_eyelink(fname,**kwargs)
    else:
        log.info("Using cached version of %s", fname)
    return d[fname]

class EyelinkReplayer(object):
    """ Class which implements the pylink API but plays back eyelink .asc files
    as though they are streaming in real time.

    This is useful for testing VisionEgg scripts which use the streaming data
    coming from the eyetracker, such as the gaze-contingent and the saccade
    prediction paradigms.

    Even though some of the methods should return "sample" type, we'll keep
    returning the reference to this object and implement all of the necessary
    methods as one flat object, for brevity.

    """
    def __init__(self, replayfile='', el=None, SAMPLE_TYPE=200):
        if replayfile=='' and el==None:
            return
        if el ==  None:
            el = read_eyelink(replayfile)
        self.el = el
        self.i=0
        self.bitflip=False


        self.x =  el.eye_used.x.v
        self.y =  el.eye_used.y.v
        self.t = el.eye_used.x #because x is an event array, indexing into it will just return single points in time

        self.SAMPLE_TYPE = SAMPLE_TYPE

    def startRecording(self, recSamples,recEvents,streamSamples,streamEvents):
        # I might have screwed up the order of parameters here
        return 0

    def getDataCount(self,samples=True):
        """Returns True every other time it is called (to give out one sample at
        a time
        """
        if self.bitflip:
            # get roughly 1000Hz replay
            time.sleep(.001)
        else:
            # increment the internal index
            self.i += 1
        self.bitflip = not self.bitflip
        return self.bitflip

    def getNewestSample(self):
        # increment the internal index
        self.i += 1
        return self


    def getNextData(self):
        return self.SAMPLE_TYPE

    def getFloatData(self):
        return self

    def isRightSample(self):
        return self.el.have_right


    def isLeftSample(self):
        return self.el.have_left

    def getRightEye(self):
        return self

    def getLeftEye(self):
        return self

    def getGaze(self):
        if self.i < len(self.x):
            return self.x[self.i], self.y[self.i]
        else:
            raise RuntimeError("Index %d exceeds  length of eyelink data (%d)"
                    % (self.i, len(self.x)))

    def getTime(self):
        # convert back to milliseconds, which is how pylink represents time
        return int(self.t[self.i]*1000)

    def sendMessage(self, txt):
        log.info("Replayer got Eyelink MSG: '%s'",txt)


def get_sample_data():
    "A simple way to get a handle on "
    sample_data_file = get_sample_data_filename()
    return read_eyelink(sample_data_file)

def get_sample_data_filename(short=False):
    "return a sample filename, which can be read using read_eyelink"
    import pyarbus
    if short:
        return os.path.join(pyarbus.path, 'data/pi_short.asc')
    else:
        return os.path.join(pyarbus.path, 'data/pi.asc.gz')

