import json
from pathlib import Path
import tempfile
import unittest

from minecraftrl.stall_replay import replay_run, wood_sample, action_signature


def observation(tick=0, logs=0, kind=1):
    return dict(server_available=1, render_ready=0, error=0, server_tick=tick,
                server_x=0.,server_y=64.,server_z=0.,yaw=0.,
                server_inventory={'oak_log':logs}, wood_v3_kind=kind,wood_v3_distance=8.,
                wood_v3_reachable=0,wood_v3_target_x=8.,wood_v3_target_y=64.,wood_v3_target_z=0.,
                wood_server_mining=1)


class StallReplayTests(unittest.TestCase):
    def test_pending_chunks_do_not_invalidate_server_evidence(self):
        s=wood_sample(observation(),{'attack':[1]},'test')
        self.assertTrue(s.valid)
        self.assertIsNone(s.mining_progress)

    def test_missing_server_is_distinct_from_stall(self):
        obs=observation();obs['server_available']=0
        self.assertFalse(wood_sample(obs,{},'test').valid)

    def test_target_disappearance_is_not_claimed_for_switch(self):
        before=observation();after=observation(1,kind=2)
        self.assertEqual(wood_sample(after,{},'test',previous=before).block_removed,(8.,64.,0.))
        after['wood_v3_target_x']=9.
        self.assertIsNone(wood_sample(after,{},'test',previous=before).block_removed)

    def test_action_signature_preserves_direction_and_ignores_tiny_camera_noise(self):
        self.assertEqual(action_signature({'attack':[1],'forward':[0],'camera':[[.01,-10.]]}),('attack','yaw-'))

    def test_teacher_and_policy_action_shapes_have_same_signature(self):
        scalar=action_signature({'attack':1,'forward':0,'camera':[.01,-10.]})
        batched=action_signature({'attack':[1],'forward':[0],'camera':[[.01,-10.]]})
        self.assertEqual(scalar,batched)

    def test_recorded_episode_full_adapter_detector_chain(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            (run/'initial-state.json').write_text(json.dumps({'server':observation()}))
            (run/'summary.json').write_text(json.dumps(dict(steps=202,status='target_reached',server_target_reached=True)))
            rows=[dict(event='transition',step=i,monotonic_s=i*.1,done=False,
                       action={'attack':[1]},info={'workshop':observation(i,logs=int(i==202))})
                  for i in range(1,203)]
            rows.append(dict(event='stop_acknowledged'))
            source=''.join(json.dumps(row)+'\n' for row in rows)
            (run/'events.jsonl').write_text(source)
            result=replay_run(run,run/'decisions.jsonl')
            self.assertEqual(result['alarms'][0]['step'],200)
            self.assertEqual(result['actual_inventory_delta'],1)
            self.assertEqual(result['new_game_actions'],0)
            self.assertEqual(result['observation_gaps'],[])
            self.assertEqual((run/'events.jsonl').read_text(),source)


if __name__=='__main__':unittest.main()
