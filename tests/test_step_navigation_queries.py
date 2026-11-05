import copy
import unittest
from tests.test_step_navigation import fixture,pose
from minecraftrl.navigation_refinement import motion_goal
from minecraftrl.step_navigation_queries import step_goal_requests


def sample(rotation=0,index=0,g=None):
    world,config=fixture(rotation);g=g or pose(rotation=rotation)
    target=config['points'][index];goal=motion_goal(g,target,remaining=47,pose_source='client')
    goal[6:9]=[.01,.02,.03]
    action=dict(attack=[0],use=[0]);row=dict(owner='student',index=index,motion_goal=goal,env_action=action,
        action={'untouched':'behavior'},features='preserved',logp=-12.)
    observed=[dict(owner='student',index=index,goal=goal,action=action,target=target,server_before=g)]
    data=dict(schema='navigation_trials_v1',split='train',eligible=True,world=world,config=config,rows=[row],
        receipts=[dict(outcome='completed') for _ in range(6)])
    return data,observed


class StepGoalQueriesTests(unittest.TestCase):
    def test_only_other_five_declared_goals_and_originals_unchanged(self):
        for rotation in (0,90,180,270):
            data,observed=sample(rotation);before=copy.deepcopy((data,observed))
            requests,receipt=step_goal_requests(data,observed)
            self.assertEqual((data,observed),before)
            self.assertEqual(len(requests),5);self.assertEqual(receipt['selected_rows'],[0])
            self.assertEqual([r['target_index'] for r in requests],[1,2,3,4,5])
            for r in requests:
                self.assertEqual(r['target'],data['config']['points'][r['target_index']])
                self.assertEqual(r['motion_goal'][6:],[.01,.02,.03,1.])

    def test_same_view_changes_jump_stop_and_turn_by_declared_goal(self):
        data,observed=sample(index=2)
        requests,_=step_goal_requests(data,observed)
        upward=next(r for r in requests if r['target_index']==0)
        self.assertEqual((upward['changes']['forward'],upward['changes']['jump']),(1,1))
        self.assertTrue(any(r['changes']['jump']==0 for r in requests))
        data,observed=sample(index=1,g=pose(x=8.5,y=202))
        requests,_=step_goal_requests(data,observed)
        self.assertEqual(next(r for r in requests if r['target_index']==0)['changes'],{})
        self.assertTrue(any(r['changes'].get('camera',[0,0])[1] for r in requests))
        self.assertTrue(all(not r['changes'].get('jump',0) for r in requests))

    def test_neutral_states_excluded_and_air_never_gets_ground_jump(self):
        data,observed=sample(g=pose(y=201.5,ground=0))
        requests,_=step_goal_requests(data,observed)
        self.assertTrue(all(not r['changes'].get('jump',0) for r in requests))
        neutral=copy.deepcopy(data['rows'][0]);neutral['owner']='neutral';data['rows'].append(neutral)
        raw=copy.deepcopy(observed[0]);raw['owner']='neutral';observed.append(raw)
        requests,selection=step_goal_requests(data,observed)
        self.assertEqual(len(requests),5);self.assertEqual(selection['excluded'],{'neutral_handoff':1})

    def test_nonstep_development_failed_or_mutated_terrain_rejected(self):
        for mutate in ('natural','flat','development','failed','attack','use','target','goal'):
            data,observed=sample()
            if mutate in ('natural','flat'):data['world']['kind']=mutate+'_navigation'
            elif mutate=='development':data['split']='development'
            elif mutate=='failed':data['receipts'][2]['outcome']='failed'
            elif mutate in ('attack','use'):
                observed[0]['action'][mutate]=[1]
            elif mutate=='target':observed[0]['target']=[0.,0.,0.]
            else:observed[0]['goal'][2]=.9
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):step_goal_requests(data,observed)


if __name__=='__main__':unittest.main()
