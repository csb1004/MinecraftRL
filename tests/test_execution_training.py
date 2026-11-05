import unittest


class ExecutionTrainingTests(unittest.TestCase):
    def test_longer_goal_does_not_dominate_balanced_objective(self):
        from minecraftrl.execution_training import objective_weights
        rows=[dict(execution_request=dict(target_ref=t,effect=e)) for t,e in
              [('wood','acquire')]*8+[('dirt','acquire')]*2+[('wood','possess')]]
        weights,receipt=objective_weights(rows)
        totals={}
        for row,weight in zip(rows,weights):
            key=(row['execution_request']['target_ref'],row['execution_request']['effect'])
            totals[key]=totals.get(key,0.)+weight
        self.assertAlmostEqual(sum(weights),len(rows))
        self.assertLess(max(totals.values())-min(totals.values()),1e-12)
        self.assertEqual(receipt['groups']['wood/acquire']['rows'],8)
        with self.assertRaises(ValueError):objective_weights([])

    def test_retention_is_train_only_and_never_a_ppo_source(self):
        from minecraftrl.execution_training import check_retention_source
        data=dict(scope='execution_contract_e2',split='train',eligible=True,base_model_sha256='official',
            context_fields=['a'],guidance=6.,status='target_reached')
        summary=dict(execution_valid=True,worker_exit_code=0,status='target_reached')
        check_retention_source(data,summary,data,source='anchor',ppo_sources={'ppo'})
        for key,value in [('split','development'),('eligible',False),('base_model_sha256','other'),('guidance',1.)]:
            bad=dict(data,**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):
                check_retention_source(bad,summary,data,source='anchor',ppo_sources={'ppo'})
        with self.assertRaises(ValueError):
            check_retention_source(data,summary,data,source='ppo',ppo_sources={'ppo'})
        with self.assertRaises(ValueError):
            check_retention_source(data,dict(summary,worker_exit_code=2),data,source='anchor',ppo_sources={'ppo'})

    def test_critic_calibration_changes_value_without_actor_updates(self):
        import torch
        from minecraftrl.execution_training import calibrate_value
        torch.set_num_threads(2);torch.manual_seed(113003)
        value=torch.nn.Linear(2,1);torch.nn.init.zeros_(value.weight);torch.nn.init.zeros_(value.bias)
        actor=torch.nn.Linear(2,2);before={k:v.clone() for k,v in actor.state_dict().items()}
        features=torch.tensor([[[1.,0.],[0.,1.],[1.,1.],[0.,0.]]])
        targets=torch.tensor([.5,.5,1.,0.])
        receipt=calibrate_value(value,features,targets,epochs=10,check_stop=lambda:None)
        self.assertGreater(receipt['updates'],0)
        self.assertLess(receipt['mse_after'],receipt['mse_before'])
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in actor.state_dict().items()))
        self.assertEqual(receipt['policy_updates'],0)

    def test_calibration_stop_and_bound(self):
        import torch
        from minecraftrl.execution_training import calibrate_value
        value=torch.nn.Linear(2,1);f=torch.ones(1,2,2);t=torch.ones(2)
        def stop():raise InterruptedError('STOP')
        with self.assertRaises(InterruptedError):calibrate_value(value,f,t,epochs=1,check_stop=stop)
        with self.assertRaises(ValueError):calibrate_value(value,f,t,epochs=11,check_stop=lambda:None)


if __name__=='__main__':unittest.main()
