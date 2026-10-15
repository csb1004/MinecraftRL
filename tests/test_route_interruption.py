import unittest
from minecraftrl.adaptive_route import AdaptiveRoute
from minecraftrl.navigation_choice import choice_packet,validate_packet
from test_exploration_route import pose


class RouteInterruptionTests(unittest.TestCase):
    def test_root_probe_failure_is_not_evidence_of_failed_navigation(self):
        route=AdaptiveRoute(pose(),[])
        route.replan(pose(),sequence=21,reason='root_probe_no_access')
        self.assertEqual(route.failures,[])
        self.assertEqual(len(route.interruptions),1)
        self.assertEqual(route.interruptions[0]['reason'],'root_probe_no_access')
        self.assertEqual(sum(route.failed.values()),0)
        packet=choice_packet(route,'episode:after-probe',21)
        self.assertEqual(packet['previous_failures'],[])
        self.assertEqual(packet['interrupted_candidates'],route.interruptions)
        self.assertEqual(packet['protocol'],'navigation_choice_v2')
        # Old paid responses remain verifiable, without mislabelling new packets as v1.
        legacy=dict(packet,protocol='navigation_choice_v1');del legacy['interrupted_candidates']
        self.assertEqual(validate_packet(legacy),legacy)
        with self.assertRaises(ValueError):validate_packet(dict(packet,protocol='navigation_choice_v1'))

    def test_observed_navigation_timeout_still_counts_as_failure(self):
        route=AdaptiveRoute(pose(),[])
        for sequence in range(1,41):route.observe(pose(),sequence=sequence)
        self.assertEqual(len(route.failures),1)
        self.assertEqual(route.failures[0]['reason'],'candidate_no_progress')
        self.assertEqual(route.interruptions,[])


if __name__=='__main__':unittest.main()
