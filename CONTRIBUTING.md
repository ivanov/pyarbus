Thank you for helping to make pyarbus better.

If you are adding new functionality, please add tests for it in the
`pyarbus/tests` directory, with a filename called `test_module.py` if you are
adding a new module (otherwise add your tests to the `test_module.py` file that
already exists there). New test functions should look something like this:


    import numpy.testing as npt

    def test_add1():
        "adding 1 to a variable whos value starts at 10 makes it go to 11"
        x = 10
        npt.assert_equal(x+1,1, "something's wrong with the python interpreter")

If you think you have found a bug, please consider including a simple test which
demonstrates the faulty behavior (in the same format as described above). This
will make it easier to track down the issue at hand and verify when it has been
fixed.
