import unittest

from agents.image_analyst import image_analyst

class TestImageAnalyst(unittest.TestCase):
    def test_image_analyst(self):
        self.assertTrue(callable(image_analyst))

if __name__ == '__main__':
    unittest.main()