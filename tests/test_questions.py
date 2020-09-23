import unittest

class TestQuestions(unittest.TestCase):

    def TestQuestions(self):
        with open('observations.txt') as fh:
            lines = fh.readlines()

        self.assertTrue(lines[0].startswith('In') or lines[0].startswith('in'))
        self.assertTrue(lines[1].startswith('0.1'))

if __name__ == '__main__':
    unittest.main()
