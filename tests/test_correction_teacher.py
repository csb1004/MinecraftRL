import unittest
from types import SimpleNamespace

from minecraftrl.correction_teacher import CorrectionTeacher, validate_teacher_world


def world(mirror=False):
    return dict(kind='recovery',chunk=[0,0],seed=104743,experimental_split='train',
                curriculum=dict(wall_end=3,mirror_z=mirror))


class CorrectionTeacherTests(unittest.TestCase):
    def test_reserved_development_seed_cannot_be_training(self):
        w=dict(world(),seed=104761,experimental_split='development')
        validate_teacher_world(w,'eval')
        with self.assertRaises(ValueError):validate_teacher_world(w,'train')
        with self.assertRaises(ValueError):validate_teacher_world(dict(w,seed=2000003),'eval')

    def test_only_explicit_train_arenas_are_allowed(self):
        validate_teacher_world(world(),'train')
        for w,split in ((world(),'eval'),(dict(world(),experimental_split='eval'),'train'),
                        (dict(world(),seed=3000017),'train'),(dict(world(),kind='break'),'train')):
            with self.assertRaises(ValueError):validate_teacher_world(w,split)

    def test_mirrored_actions_and_safe_gap(self):
        raw=SimpleNamespace(action_space=SimpleNamespace(noop=lambda:{}))
        actions=[]
        for mirror in (False,True):
            p=CorrectionTeacher(raw,world(mirror));p.reset()
            p.set_reward_context([0.]*6)
            self.assertEqual(p.waypoints,[(5.5,3.5),(5.5,4.5),(9.5,4.5),(9.5,3.5)])
            g=dict(server_x=6.5,server_z=12.5 if mirror else 3.5,
                   yaw=-90,pitch=-13,eye_z=12.5 if mirror else 3.5,
                   hit_z=12 if mirror else 3)
            p.environment=SimpleNamespace(gui=g)
            actions.append(p.act(None))
            self.assertIs(p.environment.gui,g)
        self.assertEqual(actions[0]['camera'][1],-actions[1]['camera'][1])
        self.assertEqual(actions[0]['forward'],0)
