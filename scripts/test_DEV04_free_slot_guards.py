"""Scientific-slot fencing rejects overlap, foreign work and repeated closed replay."""
import copy
import unittest
from verify_DEV04_archived_cases import free_scientific_slot
from dev04_execution_registry import FAILED_STARTS,RECOVERY_RUN

class Contracts(unittest.TestCase):
    def fixture(self):
        jobs=[dict(id=i+1,run_id=RECOVERY_RUN,name=c,status='in_progress' if c.endswith('403') else 'completed',conclusion=None if c.endswith('403') else 'success') for i,c in enumerate(sorted(FAILED_STARTS))]
        return jobs,dict(status='completed',conclusion='cancelled')

    def test_one_training_plus_one_serial_replay_admitted(self):
        self.assertTrue(free_scientific_slot(*self.fixture()))

    def test_two_training_or_live_observer_not_admitted(self):
        j,o=self.fixture();j[0].update(status='in_progress',conclusion=None);self.assertFalse(free_scientific_slot(j,o))
        j,o=self.fixture();o.update(status='in_progress',conclusion=None);self.assertFalse(free_scientific_slot(j,o))

    def test_already_closed_successful_verifier_not_repeated(self):
        j,o=self.fixture();o['conclusion']='success'
        with self.assertRaises(ValueError):free_scientific_slot(j,o)

    def test_foreign_duplicate_and_failed_recovery_rejected(self):
        for mutation in ('foreign','duplicate','failed'):
            j,o=self.fixture()
            if mutation=='foreign':j[0]['run_id']=0
            elif mutation=='duplicate':j[0]['id']=j[1]['id']
            else:j[0]['conclusion']='failure'
            with self.assertRaises(ValueError):free_scientific_slot(j,o)

    def test_all_science_terminal_allows_serial_replay(self):
        j,o=self.fixture();j[-1].update(status='completed',conclusion='success');self.assertTrue(free_scientific_slot(j,o))

if __name__=='__main__':unittest.main()
