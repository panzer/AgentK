import unittest
import os
from tools import read_file

class TestReadFile(unittest.TestCase):
    def setUp(self):
        self.file_path = 'test_file.txt'
        self.content = 'Hello, World!'
        with open(self.file_path, 'w') as f:
            f.write(self.content)

    def test_read_file(self):
        result = read_file.read_file.invoke({"file": self.file_path})
        self.assertEqual(result, self.content)

    def tearDown(self):
        os.remove(self.file_path)

if __name__ == '__main__':
    unittest.main()