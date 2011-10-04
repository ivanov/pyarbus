from os import path
import pyarbus

pyarbus_path = path.dirname(pyarbus.__file__)
data_path = path.join(pyarbus_path,'data')
def test_read_eyelink():
    test_file = path.join(data_path,'pi.asc.gz')
    pyarbus.read_eyelink(test_file)
