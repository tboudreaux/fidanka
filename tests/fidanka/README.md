# Testing Fidanka
Fidanka Tests are currently immature and the overall code coverage of them is still poor. However, we are working on improving that.

# Testing Dependencies
fidanka tests are written using pytest and coverage. In order to run them you will need to install both packages. These are *not* dependencies of fidanka as a whole as they are only needed for testing. Therefore, if they are not already installed you will need to manually install them.

```bash
pip install pytest
pip install coverage
```

# Running Tests
Once you install pytest and coverage, you can simply execute the runTest.sh script.

```bash
chmod u+x runTest.sh
./runTest.sh
```

# Expected Results
If all is working, then you should see some number of sucsesses and some number of xFailers (expected failure). If you get any regular failure (which will produce a lot of red output) then something is wrong. Its possible that this is an issue with the build on github itself (check the build status on the main page) or its possible this is an issue with your installation. Feel free to email me at Emily@boudreauxmail.com with any questions or submit an issue if you think there is a bug in fidanka which we have not identified.
