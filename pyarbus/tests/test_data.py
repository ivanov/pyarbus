from os import path
import pyarbus

pyarbus_path = path.dirname(pyarbus.__file__)
data_path = path.join(pyarbus_path,'data')
def test_read_eyelink():
    "Read a short eyelink file with discontinuous time points"
    test_file = path.join(data_path,'pi_short.asc')
    pyarbus.read_eyelink(test_file)

def test_read_eyelink_large():
    "Read a gunzipped .asc file (~10MB)"
    test_file = path.join(data_path,'pi.asc.gz')
    pyarbus.read_eyelink(test_file)
