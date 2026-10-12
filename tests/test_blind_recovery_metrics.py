import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_blind_recovery import local_escape


class EscapeMetricTests(unittest.TestCase):
    def test_sustained_escape_and_short_excursion(self):
        origin=[0,201,0]
        self.assertFalse(local_escape([[3,201,0]]*20,origin)['success'])
        result=local_escape([[3,201,0]]+[[2,201,0]]*20+[[8,201,0]],origin)
        self.assertTrue(result['success']);self.assertEqual(result['verified_offset'],21)
        self.assertEqual(result['maximum_distance'],8.)

    def test_return_resets_and_two_block_motion_is_insufficient(self):
        origin=[0,201,0]
        self.assertFalse(local_escape([[2.5,201,0]]*30,origin)['success'])
        result=local_escape([[3,201,0]]*19+[[1.9,201,0]]+[[3,201,0]]*19,origin)
        self.assertFalse(result['success'])


if __name__=='__main__':unittest.main()
