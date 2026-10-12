import math
import unittest

import torch

from minecraftrl.correction_diagnostics import CommandMetrics, aggregate, annotate_correction


def example(mirror=False):
    world = dict(chunk=[0, 0], curriculum=dict(wall_end=5, mirror_z=mirror))
    poses = [(6.5, 3.5, -90, -10, 0), (5.5, 3.5, 90, 0, 0),
             (5.5, 3.5, 0, 0, 1), (5.5, 6.5, 0, 0, 0),
             (5.5, 6.5, -90, 0, 1), (9.5, 6.5, -90, 0, 0),
             (9.5, 6.5, 180, 0, 1), (9.5, 3.5, 180, 0, 0)]
    rows = []
    for i, (x, z, yaw, turn, forward) in enumerate(poses):
        if mirror:
            z, yaw, turn = 16-z, 180-yaw, -turn
        rows.append(dict(frame=f'frames/{i:06d}.png', owner='teacher', loss_eligible=True, phase='move',
                         server_before=dict(server_x=x, server_z=z, yaw=yaw, pitch=0),
                         action=dict(camera=[0, turn], forward=forward)))
    rows.append(dict(frame='frames/000008.png', owner='teacher', loss_eligible=True, phase='harvest'))
    return rows, world


POINTS = [(5.5,3.5),(5.5,6.5),(9.5,6.5),(9.5,3.5)]


class CorrectionDiagnosticTests(unittest.TestCase):
    def test_waypoint_boundaries_use_pre_action_state(self):
        rows, world = example()
        labels = annotate_correction(rows, world, POINTS)
        self.assertEqual([r['leg'] for r in labels], ['reposition']*2+['gap_approach']*2+['gap_crossing']*2+['reapproach']*2+['harvest'])
        self.assertEqual([r['motion'] for r in labels[:4]], ['turn', 'arrival', 'advance', 'arrival'])

    def test_mirror_changes_turn_sign_but_keeps_route_labels(self):
        normal, a = example(); mirrored, b = example(True)
        self.assertEqual(annotate_correction(normal, a, POINTS), annotate_correction(mirrored, b, POINTS))
        self.assertEqual(mirrored[0]['action']['camera'][1], 10)

    def test_source_action_mismatch_is_rejected(self):
        rows, world = example()
        rows[0]['action']['camera'][1] = 10
        with self.assertRaisesRegex(ValueError, 'navigation contract mismatch'):
            annotate_correction(rows, world, POINTS)

    def test_incomplete_route_and_injected_student_target_rejected(self):
        rows, world = example()
        with self.assertRaises(ValueError):
            annotate_correction(rows[:3], world, POINTS)
        rows[0]['owner'] = 'student'
        with self.assertRaisesRegex(ValueError, 'owner/target'):
            annotate_correction(rows, world, POINTS)

    def test_student_prefix_has_no_teacher_index(self):
        rows, world = example()
        prefix = [dict(owner='student', loss_eligible=False), dict(owner='handoff_neutral', loss_eligible=False)]
        all_rows = prefix + rows
        for i, r in enumerate(all_rows):r['frame'] = f'frames/{i:06d}.png'
        labels = annotate_correction(all_rows, world, POINTS)
        self.assertIsNone(labels[0]['teacher_index'])
        self.assertTrue(labels[0]['pre_handoff_last32'])
        self.assertFalse(labels[1]['pre_handoff_last32'])
        self.assertEqual(labels[2]['teacher_index'], 0)

    def test_reposition_uses_handoff_z_not_fixed_start_z(self):
        rows, world = example()
        rows[0]['server_before']['server_z'] = 3.15
        rows[1]['server_before']['server_z'] = 3.15
        with self.assertRaisesRegex(ValueError, 'route receipt'):
            annotate_correction(rows, world, POINTS)
        points = [(5.5,3.15), *POINTS[1:]]
        self.assertEqual(annotate_correction(rows, world, points)[0]['motion'], 'turn')

    def test_camera_direction_probability_includes_button_gate(self):
        metric = CommandMetrics([[0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,1], [0,0,0,0,0]],
                                [True,True,True,False], [[0,0],[0,1],[0,2]],
                                ['forward','back','left','right','attack'], 1, 'cpu')
        pd = dict(buttons=torch.tensor([.2,.1,.5,.2]).log(), camera=torch.tensor([.25,.25,.5]).log())
        result = metric.measure(pd, dict(buttons=torch.tensor(3), camera=torch.tensor(2)))
        self.assertAlmostEqual(result['p_target_yaw_sign'], .1, places=6)
        self.assertEqual(result['yaw_sign_top1_match'], 0)
        self.assertAlmostEqual(result['nll'], -math.log(.1), places=6)
        self.assertAlmostEqual(result['p_move'], .1, places=6)
        self.assertEqual(result['top1_attack'], 1)
        pd['camera'] = torch.tensor([.1,.1,.1]).log()
        with self.assertRaisesRegex(ValueError, 'unnormalized'):metric.measure(pd)

    def test_missing_metric_is_not_a_zero_or_a_target(self):
        values = aggregate([dict(p_move=0), dict(p_move=1, nll=2)])
        self.assertEqual(values['p_move'], dict(mean=.5, count=2))
        self.assertEqual(values['nll'], dict(mean=2., count=1))
        self.assertEqual(aggregate([]), {})


if __name__ == '__main__':unittest.main()
