import unittest

from agents.image_generator import image_generator

class TestImageGenerator(unittest.TestCase):
    def test_image_generator(self):
        self.assertTrue(callable(image_generator))

if __name__ == '__main__':
    unittest.main()