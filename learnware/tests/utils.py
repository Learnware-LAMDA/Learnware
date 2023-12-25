import unittest

def parametrize(test_class, **kwargs):
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(test_class)
    _suite = unittest.TestSuite()
    for name in test_names:
        _suite.addTest(test_class(name, **kwargs))
    return _suite