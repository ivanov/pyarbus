import numpy as np
import numpy.testing as npt

import eyelib.parser as parser
p = parser.testParams

def test_startstop():
    "return no saccades for all zero data"
    #d = Events(range(10), v=np.zeros(10), a = np.zeros(10))
    v=np.zeros(21)
    starts,stops = parser.eyelink_event_parser(v[:-1],np.diff(v),p=p)
    npt.assert_equal(starts, [])
    npt.assert_equal(stops, [])

def test_nostart_stops():
    "return no saccades for multiple stop data"
    #d = Events(range(10), v=np.zeros(10), a = np.zeros(10))
    v=np.zeros(21)
    v[1]=v[13] = 30
    starts,stops = parser.eyelink_event_parser(v[:-1],np.diff(v),p=p)
    npt.assert_equal(starts, [])
    npt.assert_equal(stops, [])

def test_start_at_end():
    "saccade starts at the end of data (no stop)"
    #d = Events(range(10), v=np.zeros(10), a = np.zeros(10))
    v=np.zeros(21)
    v[17:19] = [30,40]
    starts,stops = parser.eyelink_event_parser(v[:-1],np.diff(v),p=p)
    print len(v)
    npt.assert_equal(starts, [17])
    npt.assert_equal(stops, [19])

def test_one_sac():
    "one saccade (velocity only) (simple)"
    v=np.zeros(21)
    v[1:4] =  [30,40,30]
    starts,stops = parser.eyelink_event_parser(v[:-1],np.diff(v),p=p)
    npt.assert_equal(starts, [1])
    npt.assert_equal(stops, [4])

def test_blink():
    "one blink (velocity only) (simple)"
    v=np.zeros(21)
    v[1:4] =  [30,40,30]
    v[3] = np.nan
    starts,stops = parser.eyelink_event_parser(v[:-1],np.diff(v),p=p)
    npt.assert_equal(starts, [])
    npt.assert_equal(stops, [])

def test_mult_start():
    "multiple starts for one stop"
    v=np.zeros(21)
    v[1:4] =  [30,40,30]
    v[4:7] =  [10,40,30]
    starts,stops = parser.eyelink_event_parser(v[:-1],np.diff(v),p=p)
    npt.assert_equal(starts, [1])
    npt.assert_equal(stops, [7])


def test_mult_end():
    "multiple stops for one start"
    v=np.zeros(21)
    v[1:4] =  [30,40,30]
    v[15] = 30 #monkey wrench to see if we throw away extra 'stop' signal
    starts,stops = parser.eyelink_event_parser(v[:-1],np.diff(v),p=p)
    npt.assert_equal(starts, [1])
    npt.assert_equal(stops, [4])


def test_mult_start_end():
    "multiple stops and starts, with overlaps"
    v=np.zeros(21)
    v[1:4] =  [30,40,30]
    v[5:7] =  [30,40]
    v[11] =  30 #monkey wrench to see if we throw away extra 'stop' signal
    v[15:17] = [30,40]

    starts,stops = parser.eyelink_event_parser(v[:-1],np.diff(v),p=p)
    npt.assert_equal(starts, [1,15])
    npt.assert_equal(stops, [7,19])

def test_pitjb():
    "online parser"
    v=np.zeros(21)
    v[1:4] =  [30,40,30]
    peaks = parser.pitjb_online_parser(v)
    npt.assert_equals([4],peaks)
