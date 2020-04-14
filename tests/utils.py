import unittest

class BasicObjectTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Adapted from https://gist.github.com/jonashaag/834a5f6051094dbed3bc
    @classmethod
    def setUpClass(cls):
        if cls is BasicObjectTest: raise unittest.SkipTest("base class")

    def setUp(self):
        self.obj = self.CLASS(**self.KWARGS)

    def test_constructs(self):
        # Technically this assert is barely necessary; the real test happens in the setup, but
        # this makes for nicer readability.
        self.assertIsInstance(self.obj, self.CLASS)
