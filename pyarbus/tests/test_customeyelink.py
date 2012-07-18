import numpy.testing as npt
import pyarbus

from pyarbus.tests.test_data import short_test_file

def test_read_custom_eyelink():
    "Read a short eyelink file with a customized Eyelink class"
    class CustomizedEyelink(pyarbus.Eyelink):
        """
        A sample custom  Eyelink class.  Additional attributes defined by this
        class:

        e.experiment : string
        """
        @property
        def experiment(self):
            """
            The name of the python file that ran at the beginning of this
            recording.

            Just grabs it from the `e.msg[0]`, which is something like:

               'MSG\t13898560 Experiment: test_movieqt.py'

            ...and makes e.experiment be just `test_movieqt`
            """
            return self.msgs[0].split(' ')[-1].split('.')[0]

    customized = pyarbus.read_eyelink(short_test_file, Eyelink=CustomizedEyelink)
    npt.assert_(hasattr(customized, 'experiment'))
    npt.assert_equal(customized.experiment, 'test_movieqt')
