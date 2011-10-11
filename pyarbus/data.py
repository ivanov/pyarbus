"""
Library of functions related to eye movement analysis
"""
from __future__ import with_statement

import re
import numpy as np
import nitime
from nitime import Events, Epochs
from StringIO import StringIO
import gzip

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

__all__ = [ 'Eyelink', 'reprocess_eyelink_msgs', 'findall_loadtxt',
'read_eyelink', 'EyelinkReplayer']

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
        (True ,False,True ) : (0,1,2,3,4,5,6,7,8,9,10,11,12),
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
    #@desc.setattr_on_read
    #def amplitude(self):
    #    dx = (self.data['xi'] - self.data['xf'])
    #    dx *= dx
    #    dx /= (self.data['xresi'] - self.data['xresf'])
    #    dx *= 2
    #    dy = (self.data['yi'] - self.data['yf'])**2
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
                xi=self.xi[k],
                xf=self.xf[k],
                yi=self.yi[k],
                yf=self.yf[k]
                )

class Eyelink(object):
    """
class for Eyelink data

Has x,y,pupA time series, as well as saccade and discard epochs, frames
events and msgs strings extracted from an Eyelink .asc file. All members are
also attributes (you can use e.x and e.saccades as a shorthand for e['x'] and
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


    e["frames"] : Events or None
        Onset of frames, e.g. e["frames"][29] returns when 29th frame was shown
    e["msgs"] : tuple(str) or None
        Lines in Eyelink file which start with MSG (such as "Stimulus paused"),
        other than frame onsets (which are parsed out)
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
    _fittype = None

    def get_surface(self):
        """Return the parameter surface specified by surftype"""
        if self._fittype not in self._surfaces:
            raise ValueError("fittype '" + str(self._fittype) +
                    "' has no surface associated with it")
        return self._surfaces[self._fittype]

    def set_surface(self,s):
        """Set the parameter surface specified by surftype"""
        self._surfaces[self._fittype] = s

    _surfdoc = "parameter surface currently in use (specified by surftype)"
    surface = property(get_surface,set_surface,_surfdoc)

    _fits=None

    def get_fits(self):
        """Return the fit parameters"""
        if self._fits is None:
            try:
                f = np.load(self.fname+'fits.npz')
                print "using cached fits found in "+self.fname+'fits.npz'
                self._fits = f['fits'][()]
            except IOError:
                print "did not find cached calculations"
                self._fits = {}
        return self._fits

    def set_fits(self,f):
        """set the fit parameters"""
        self._fits = f

    _fitdoc = """fit parameters keys by (xpos,ypos), which are loaded from
        disk when possible"""
    fits = property(get_fits,set_fits,_fitdoc)

    def save_fits(self):
        """save the fits to disk, so they get automatically loaded in the
        future"""
        np.savez(self.fname+'fits',fits=self._fits)

    @property
    def xsurf(self):
        """Return the X parameter surface specified by surftype"""
        return self._surfaces[self._fittype]['xsurf']

    @property
    def ysurf(self):
        """Return the Y parameter surface specified by surftype"""
        return self._surfaces[self._fittype]['ysurf']

    def get_fittype(self):
        """Get the current type of fit used for the parameter surface"""
        return self._fittype

    def set_fittype(self, ft):
        """Set the current type of fit to use for parameter surface"""
        self._fittype = ft

    _fittypedoc = """ What fit to use for the parameter surface (e.g. 'linear',
    'interpolated_quadratic', 'combined_cubic')
    """
    fittype = property(get_fittype, set_fittype, doc=_fittypedoc)

    msgs = None
    frames = None
    raw = None

    def list_fittypes(self):
        "Return the fit types which have been computed for this experiment"
        return self._surfaces.keys()

    def __init__(self,fname='dummyfile', binocular=False, have_right=True,
            have_left=False,have_vel=False,have_res=False,samplingrate=None,
            from_eyelink=None):

        if from_eyelink:
            self.__dict__.update(from_eyelink.__dict__)
            return
        #Container.__init__(self)
        self._surfaces = {}
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
        #self.r.saccades = None
        #self.l.saccades = None
        #self.r.fixepochs = None
        #self.l.fixepochs = None
        #self.r.sacepochs = None
        #self.l.sacepochs = None
        #self.r.discard = []
        #self.l.discard = []
        #self["discard"] = self["saccades"] = None
        self.msgs = None
        self.frames = None
        self.raw = None
        self._fnamelong = fname
        self._fname = fname.split('/')[-1].split('.')[0]

    def __repr__(self):
        rep  =  "Eyelink Data:"
        if self.binocular: rep += " binocular"
        elif self.have_right: rep += " monocular (right eye)"
        else: rep += " monocular (left eye)"
        for x in ['frames','msgs']:
            if self.__getattribute__(x) is not None:
                rep +="\n %d %s ["%(len(self.__getattribute__(x)),x)
                rep += self.__getattribute__(x)[0].__repr__()
                rep += " ...]"
        raw = self.raw.split('\n')

        if self.have_left:
            rep += "\nLeft " +  __eye_repr__(self.l)
        if self.have_right:
            rep += "\nRight "+ __eye_repr__(self.r)
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
        for ep in eye.discard:
            sl=eye.x.epoch2slice(ep)
            start,stop = sl.start,sl.stop
            mask = eye.x.v.mask
            mask[start-pad:stop+pad] = True
            eye.x.v.mask = eye.y.v.mask = mask
            eye.xres.v.mask = eye.yres.v.mask = mask
            eye.pupA.v.mask = mask

    @property
    def eye_used(self):
        """ Return self.l or self.r, depending on which eye has data, raises an
        error if binocular"""

        if self.binocular:
            raise Exception, "Container is binocular, both eyes are available"
        if self.have_right:
            return self.r
        else:
            return self.l

    @property
    def experiment(self):
        """ the name of the python file that ran at the beginning of this
        recording.

        Just grabs it from the .msg[0] """
        return self.msgs[0].split('/')[-1]

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
    pat = "(?<=MSG.)\d+ "+pattern+".*"
    return findall_loadtxt(pat, "\n".join(msgs), cols, dtype)


def findall_loadtxt(pattern, raw, cols, dtype=None):
    matches = re.findall(pattern,raw, re.M)
    str = "\n".join(matches)
    tmp = StringIO(str)
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
        retfloat = np.fromstring(str, dtype=float, sep=' ')
        retfloat.shape=-1,8
        ret = np.empty(len(matches),dtype=dtype)

        for i,c in enumerate(retfloat.T):
            ret[dtype.names[i]] = dtype[i].type(c)
        ret
    else:
        ret = np.loadtxt(tmp,dtype=dtype,usecols=cols)
    #tmp.close()
    return ret

def read_eyelink(filename):
    """
Read in Eyelink .asc file,

Returns obect containing eye position time series, along with pupil area,
saccades and blink data as parsed by Eyelink, and any messages (such as frame
information)


:Parameters:
    filename : str
        name of .asc file to be read

:Returns:
    eye : Eyelink
        container with 'x','y','pupA' TimeSeries, 'saccade' and 'discard'
        Epochs, and 'frames' Events.

:SeeAlso:
  - Eyelink : class of the return object of this function

Notes
-----

Assumes edf2asc ran with '-nflags -miss -9999.9'


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
        f = file(filename)
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
    for x in ["SFIX","SSACC","SBLINK"]:
        raw = re.sub("\n"+x+".*","", raw)

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

    gcdisp = findall_loadtxt("(?<=MSG.)\d+\ GCDISP.\d+.*",raw,(0,2,3),
            dtype=gc_gaze_dtype)





    #grab binocularity from the EVENTS GAZE line
    start = re.search("^START.\d+\ [^\d].*", raw, re.M).group().upper()
    samples_line = re.search("^SAMPLES.*", raw, re.M).group()
    print samples_line

    binocular = False
    have_right = start.find('RIGHT') != -1
    have_left= start.find('LEFT') != -1
    if have_right and have_left:
        binocular = True

    velocity = "VEL" in samples_line
    res = "RES" in samples_line

    gcols, gdtype = get_gaze_col_dtype(binocular, velocity, res)
    print (binocular, velocity, res), get_gaze_col_dtype(binocular, velocity, res)

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
    raw = re.sub('   .\t','  nan\t',raw)
    raw = re.sub('\.\.\.','',raw)


    # the bulk of the time for this function is spent here - XXX: any speedup
    # in this code / approach will be very welcome. For example - setting cols
    # to None here speeds up the calls to np.loadtxt by about 50%
    gaze = findall_loadtxt("^\d+.*",raw,'all',gdtype)
    raw = re.sub("\n\d+.*","",raw) # get rid of lines which we've already

    # frame data - MSG <time> <frame>
    # get first field for time, get second field as value dict
    frames = findall_loadtxt("(?<=MSG.)\d+\ \d+",raw,(0,1), dtype=np.uint64).astype(float)
    link_fix = findall_loadtxt("(?<=MSG.)\d+\ Fixation",raw,(0,),
            dtype=np.uint64).astype(float)

    # get MSGs which are not like frames or GCDISP
    msgsstr = re.findall("^MSG.\d+\ [^\dG].*", raw, re.M)

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
    samplingrate = findall_loadtxt("RATE[\t ]*\d+.\d*",raw,(1,))
    dt = 1000/samplingrate[0]

    #XXX: throw error if diff.gaze(['time']) is ever either 0 or negative (samples repeated or out of order)
    for D in np.arange(len(gaze['time']))[np.diff(gaze['time']) != dt]:
        log.warn("Discontinuity in eyetracker time series")
        log.warn("   at sample %d, time %s",D,str(gaze['time'][D:D+2]))
        ## XXX: implement a "fill-discontinuous" flag to have this
        ## functionality again - it was needed for neuropy's events and
        ## timeseries implementation, but no longer necessary with nitime
        #missing_tstamp= np.concatenate((missing_tstamp,gaze['time'][D:D+2]))
        #t = np.concatenate((t,gaze['time'][prev:D+1],
        #                    np.arange(gaze['time'][D],
        #                              gaze['time'][D+1],
        #                              dt,dtype='uint64')))
        ## missing values stored as NaNs
        #z = np.ones((gaze['time'][D+1]- gaze['time'][D])/dt) * np.nan
        ## .names[1:] skips over the 'time' field
        #for fn in gaze.dtype.names[1:]:
        #    tmp[fn] = np.concatenate((tmp[fn],gaze[fn][prev:D+1], z))
        prev = D+1

    # iterate over all fields
    tmp['time'] = t
    for fn in gaze.dtype.names:
        tmp[fn] = np.concatenate((tmp[fn],gaze[fn][prev:]))

    gaze = np.zeros(len(tmp['time']),dtype=gdtype)
    for fn in gaze.dtype.names:
        gaze[fn] = tmp[fn]

    raw= re.sub("\nMSG.*","", raw) # extracted

    # See Notes in the Eyelink class docstring
    # Basically, all we need to do is find those ESACC events which are
    # preceded by EBLINK events, and discard the entire saccade

    #TODO: a better way of doing this is to separate the streams (as we do
    #later anyway) and call a function to parse the two streams seperately...
    #...but life's too short for now

    # convert endblinks to fixed length (so we can use lookback)
    blinks_r = findall_loadtxt("(?<=EBLINK.)R\ \d+\t\d+",raw,(1,2))
    blinks_l = findall_loadtxt("(?<=EBLINK.)L\ \d+\t\d+",raw,(1,2))
    blinks_r.shape = -1,2
    blinks_l.shape = -1,2


    #XXX: previous approach does not worksince R and L can be staggered relative to one another
    # soln: split the left and right streams, and process them individually
    def grab_raw_monocular(raw,eye='R'):
        """ this function removes any lines that pertain to the oposite eye"""
        raw_ret = raw
        omit = " L"
        if eye=='L':
            omit = " R"
        for x in ["EBLINK","ESACC","EFIX"]:
            raw_ret = re.sub(x+omit+".*\n","",raw_ret)

        return raw_ret

    raw_l = grab_raw_monocular(raw,'L')
    raw_r = grab_raw_monocular(raw,'R')
    raw_l= re.sub("EBLINK.*","EBLINK",raw_l)
    raw_r= re.sub("EBLINK.*","EBLINK",raw_r)
    #1/0
    # lookback and return endsaccades which are preceded by an endblink
    discard_r = findall_loadtxt("(?<=EBLINK\nESACC...).*\d+\t\d+",raw_r,(0,1))
    discard_l = findall_loadtxt("(?<=EBLINK\nESACC...).*\d+\t\d+",raw_l,(0,1))
    # XXX: separate timestamp gaps with blinks (and maybe have a method that
    # reports the OR of all the crap
    discard_r = np.append(missing_tstamp,discard_r)
    discard_l = np.append(missing_tstamp,discard_l)
    discard_l.shape = -1,2
    discard_r.shape = -1,2

    #get rid of lines containing falsely reported ESACCS which were preceded by EBLINK
    # XXX: now I'm getting paranoid - are we excluding more ESACCs than we
    # should if there were multiple blinks ?
    raw_r = re.sub("(?<=EBLINK)\nESACC.*","",raw_r)
    raw_l = re.sub("(?<=EBLINK)\nESACC.*","",raw_l)
    # lookback and return endsaccades (see DATA NOTATIONS for <codes>)
    #ESACC   <eye> <stime> <etime> <dur> <sxp> <syp> <exp> <eyp> <ampl> <pv>

    saccades_r = findall_loadtxt("(?<=ESACC...).*",raw_r,(0,1,7,8,3,4,5,6)).astype(float)
    saccades_l = findall_loadtxt("(?<=ESACC...).*",raw_l,(0,1,7,8,3,4,5,6)).astype(float)
    saccades_r.shape = -1,8
    saccades_l.shape = -1,8

    # (see DATA NOTATIONS for <codes>)
    #EFIX <eye> <stime> <etime> <dur> <axp> <ayp> <aps>
    fix_r= findall_loadtxt("(?<=EFIX...).*",raw_r,(0,1,3,4,5)).astype(float)
    fix_l= findall_loadtxt("(?<=EFIX...).*",raw_l,(0,1,3,4,5)).astype(float)
    fix_r.shape = -1,5
    fix_l.shape = -1,5

    raw = re.sub("\n[ES]SACC.*","",raw)  # get rid of lines which we've already
    raw = re.sub("\n[ES]BLINK.*","",raw) # extracted

    el = Eyelink(filename, binocular, have_right, have_left, velocity, res,
            samplingrate=samplingrate[0])

    # XXX - remove this senseless division - nitime does time properly
    time = gaze['time'] / 1000.0
    if binocular:
        # names[0] is 'time', everything else is x,y,pupA,
        for name in gaze.dtype.names[1:4]:
            # XXX: grr! using a masked array doesn't work with neuropy's
            # implementation of Events
            el.l[name] = Events(time,
                    v=np.ma.masked_invalid(gaze[name], copy=False))
        # so far we've called right eye data x2,y2,pupA2 - rename these to x,y,pupA
        for name,field in zip(('x','y','pupA'),gaze.dtype.names[4:7]):
            el.r[name] = Events(time,
                    v=np.ma.masked_invalid(gaze[field]), copy=False) # typecast as Events,
        if velocity:
            for name in gaze.dtype.names[7:9]:
                el.l[name] = Events(time,
                        v=np.ma.masked_invalid(gaze[name], copy=False))
            # so far we've called right eye data x2v,y2v - rename these to xv,yv
            for name,field in zip(('xv','yv'),gaze.dtype.names[9:11]):
                el.r[name] = Events(time,
                        v=np.ma.masked_invalid(gaze[field], copy=False))
        if res:
            # res fields are always the last two, if present
            for name in gaze.dtype.names[-2],gaze.dtype.names[-1]:
                el.l[name] = Events(time,v=np.ma.masked_invalid(gaze[name],
                        copy=False)) # typecast as Events,
                el.r[name] = Events(time,v=np.ma.masked_invalid(gaze[name],
                    copy=False)) # typecast as Events,

    else:
        if have_right:
            mi = np.ma.masked_invalid
            d = dict([(n,mi(gaze[n])) for n in gaze.dtype.names[1:]])
            el.r = nitime.Events(time,**d)
        else:
            for name in gaze.dtype.names[1:]:
                el.l[name] = Events(time,v=np.ma.masked_invalid(gaze[name]),
                        copy=False) # typecast as Events,
                el.r[name] = None


    # eye["x"] = TimeSeries(gaze['x'],
    #                       t0=gaze['time'][0]/1000.0,
    #                       samplingrate=500)#gaze['time'][1]-gaze['time'][0])
    # eye["y"] = TimeSeries(gaze['y'],
    #                       t0=gaze['time'][0]/1000.0,
    #                       samplingrate=500)#gaze['time'][1]-gaze['time'][0])
    # eye["pupA"] = TimeSeries(gaze['pupA'],
    #                          t0=gaze['time'][0]/1000.0,
    #                          samplingrate=500)#gaze['time'][1]-gaze['time'][0])

    #XXX: wrap this up into a loop for neatness / brevity
    if blinks_r.size:
        el.r.blinks = Epochs(blinks_r[:,0]/1000.0,blinks_r[:,1]/1000.0)

    if blinks_l.size:
        el.l.blinks = Epochs(blinks_l[:,0]/1000.0,blinks_l[:,1]/1000.0)

    if discard_l.size:
        el.l.discard = Epochs(discard_l[:,0]/1000.0,discard_l[:,1]/1000.0)
    if discard_r.size:
        el.r.discard = Epochs(discard_r[:,0]/1000.0,discard_r[:,1]/1000.0)

    if saccades_l.size:
        el.l.saccades = Saccades(saccades_l[:,0]/1000.0, start=saccades_l[:,0]/1000.0,stop=saccades_l[:,1]/1000.0,
                amplitude=saccades_l[:,2], vpeak=saccades_l[:,3],
                #epochs=Epochs(saccades_l[:,0]/1000.0,saccades_l[:,1]/1000.0),
                xi=saccades_l[:,3], yi=saccades_l[:,4],
                xf=saccades_l[:,5], yf=saccades_l[:,6],
                )
        el.l.sacepochs = Epochs(saccades_l[:,0]/1000.0,saccades_l[:,1]/1000.0)
    if saccades_r.size:
        el.r.saccades = Saccades(saccades_r[:,0]/1000.0, start=saccades_r[:,0]/1000.0,stop=saccades_r[:,1]/1000.0,
                amplitude=saccades_r[:,2], vpeak=saccades_r[:,3],
                #epochs=Epochs(saccades_r[:,0]/1000.0,saccades_r[:,1]/1000.0),
                xi=saccades_r[:,3], yi=saccades_r[:,4],
                xf=saccades_r[:,5], yf=saccades_r[:,6],
                )
        el.r.sacepochs = Epochs(saccades_r[:,0]/1000.0,saccades_r[:,1]/1000.0)
    if fix_l.size:
        el.l.fixations = Events(fix_l[:,0]/1000.0, start=fix_l[:,0]/1000.0,stop=fix_l[:,1]/1000.0,
                xavg=fix_l[:,2], yavg=fix_l[:,3], pavg=fix_l[:,4]
                #epochs=Epochs(fix_l[:,0]/1000.0,fix_l[:,1]/1000.0))
                )
        el.l.fixepochs=Epochs(fix_l[:,0]/1000.0,fix_l[:,1]/1000.0)
    if fix_r.size:
        el.r.fixations = Events(fix_r[:,0]/1000.0, start=fix_r[:,0]/1000.0,stop=fix_r[:,1]/1000.0,
                xavg=fix_r[:,2], yavg=fix_r[:,3], pavg=fix_r[:,4],
                #epochs=Epochs(fix_r[:,0]/1000.0,fix_r[:,1]/1000.0)
                )
        el.r.fixepochs=Epochs(fix_r[:,0]/1000.0,fix_r[:,1]/1000.0)

    if frames.size:
        el.frames = Events(frames[:,0]/1000.0,v=frames[:,1])

    if link_fix.size:
        el.link_fixations = Events(link_fix/1000.0)

    el.msgs = msgsstr
    el.raw = raw # contains all the lines which have not been processed
    if gcdisp.size:
        gcdisp =  gcdisp.view(np.recarray)
        el.gcdisp = Events(gcdisp.time/1000.0, x=gcdisp.x, y=gcdisp.y)

    # XXX: kind of wish I could do some interval math on epochs (union,
    # intersection, a la pyinterval)

    try:
        el.do_discard()
    except AttributeError:
        log.warn("Discarding failed")

    f.close()
    return el

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
        print "Replayer got Eyelink MSG: '%s'"%txt
