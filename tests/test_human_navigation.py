import math,unittest
from minecraftrl import human_navigation as module


class HumanReferenceTests(unittest.TestCase):
    def sample(self,**kwargs):return dict(dict(xpos=0.,ypos=64.,zpos=0.,yaw=0.,pitch=0.,tick=1,isGuiOpen=False),**kwargs)

    def test_geometry_motion_and_budget(self):
        old=self.sample(tick=0,zpos=-.2,xpos=-.1,yaw=-9.)
        g=module.reference_motion_goal(self.sample(),old,[0.,64.,4.],87)
        expected=[0.,1.,.25,0.,0.,1.,.1,.2,.05,.87]
        for a,b in zip(g,expected):self.assertAlmostEqual(a,b)
        g=module.reference_motion_goal(self.sample(yaw=90.),self.sample(tick=0,yaw=90.),[-4.,65.,0.],100)
        self.assertAlmostEqual(g[0],0.);self.assertAlmostEqual(g[1],1.);self.assertEqual(g[3],.25)

    def test_wrapped_yaw_and_bounds(self):
        g=module.reference_motion_goal(self.sample(yaw=-179.,pitch=90.,xpos=4.),self.sample(tick=0,yaw=179.),[0.,100.,100.],0)
        self.assertAlmostEqual(g[8],2/180);self.assertEqual(g[9],0.)
        self.assertEqual(g[2],1.);self.assertEqual(g[3],1.);self.assertEqual(g[4],1.)
        self.assertTrue(all(-1<=v<=1 for v in g))

    def test_unknown_or_nonconsecutive_inputs_rejected(self):
        for change in (dict(tick=2),dict(tick=1.),dict(tick=True),dict(isGuiOpen=True),dict(isGuiOpen=None),dict(pitch=91.),dict(yaw=float('nan')),dict(xpos=True)):
            with self.assertRaises(ValueError):module.reference_motion_goal(self.sample(**change),self.sample(tick=0),[0.,64.,4.],100)
        for remaining in (101,-1,True,1.5):
            with self.assertRaises(ValueError):module.reference_motion_goal(self.sample(),self.sample(tick=0),[0.,64.,4.],remaining)
        for target in ([0.,64.],[0.,64.,float('inf')]):
            with self.assertRaises(ValueError):module.reference_motion_goal(self.sample(),self.sample(tick=0),target,100)


if __name__=='__main__':unittest.main()
