import unittest
import os
from tools import write_to_file

import tempfile

class TestWriteToFile(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            self.file_path = temp_file.name

    def tearDown(self):
        os.remove(self.file_path)

    def test_write_to_file(self):
        content = 'Hello, World!'
        result = write_to_file.write_to_file.invoke({"file": self.file_path, "file_contents": content})
        self.assertEqual(result, f"File {self.file_path} written successfully.")

if __name__ == '__main__':
    unittest.main()