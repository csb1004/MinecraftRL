import unittest
import hashlib
import tempfile
import contextlib
import io
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from minecraftrl.policies import Steve1Policy


class PromptTests(unittest.TestCase):
    def policy(self,profile):
        policy=Steve1Policy.__new__(Steve1Policy)
        policy.prompt_profile=profile; policy.agent=Mock(); policy.mineclip=object(); policy.prior=object(); policy.device='cuda'
        policy.embed=Mock(return_value=Mock(tobytes=lambda:b'fixed-embedding'))
        return policy

    def test_explicit_guidance_controls_text_reset_and_metadata(self):
        for scale in (0.,2.):
            policy=self.policy('official');policy.guidance=scale
            policy.reset('collect wood')
            policy.agent.reset.assert_called_once_with(cond_scale=scale)
            self.assertEqual(policy.condition_metadata['cond_scale'],scale)

    def test_invalid_guidance_rejected_before_model_loading(self):
        for value in (True,-1,8,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                Steve1Policy(None,None,guidance=value)

    def test_cli_guidance_is_finite_bounded_and_opt_in(self):
        from minecraftrl.cli import parser,episode
        self.assertIsNone(parser().parse_args(['session','--policy','steve1']).steve_cond_scale)
        self.assertEqual(parser().parse_args(['session','--policy','steve1','--steve-cond-scale','2']).steve_cond_scale,2.)
        for value in ('nan','inf','-1','8'):
            with self.subTest(value=value),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parser().parse_args(['session','--policy','steve1','--steve-cond-scale',value])
        for argv in (['episode','--policy','vpt'],['episode','--policy','steve1','--rl-collect']):
            args=parser().parse_args(argv+['--steve-cond-scale','2'])
            with self.assertRaisesRegex(ValueError,'guidance'):
                episode(args,{})

    def test_official_profile_changes_only_condition_and_records_actual_text(self):
        policy=self.policy('official')
        policy.reset('collect dirt')
        expected='get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt'
        self.assertEqual(policy.embed.call_args.args[0],expected)
        policy.agent.reset.assert_called_once_with(cond_scale=6.0)
        self.assertEqual(policy.condition_metadata['requested_goal'],'collect dirt')
        self.assertEqual(policy.condition_metadata['effective_prompt'],expected)
        self.assertEqual(len(policy.condition_metadata['embedding_sha256']),64)

    def test_short_default_and_unsupported_official_goal(self):
        policy=self.policy('short'); policy.reset('collect dirt')
        self.assertEqual(policy.embed.call_args.args[0],'collect dirt')
        official=self.policy('official')
        with self.assertRaises(ValueError): official.reset('teleport home')
        official.agent.reset.assert_not_called()

    def test_cli_profile_defaults_and_validation(self):
        from minecraftrl.cli import parser
        args=parser().parse_args(['session','--policy','steve1'])
        self.assertEqual(args.steve_prompt_profile,'short')
        args=parser().parse_args(['session','--policy','steve1','--steve-prompt-profile','official'])
        self.assertEqual(args.steve_prompt_profile,'official')

    def test_isolated_world_root_parser(self):
        from minecraftrl.cli import parser
        args=parser().parse_args(['session','--policy','steve1','--world-store-root','results/probe/worlds'])
        self.assertEqual(args.world_store_root,Path('results/probe/worlds'))

    def test_official_visual_uses_verified_array_without_calling_text_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy=self.policy('official-visual');policy.root=Path(tmp)
            path=policy.root/'weights/steve1/visual-prompts/wood.npy';path.parent.mkdir(parents=True)
            value=np.ones((1,512),dtype=np.float32);np.save(path,value)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(Steve1Policy,'OFFICIAL_VISUAL',{'collect wood':('wood.npy',sha)},create=True):
                policy.reset('collect wood')
            policy.embed.assert_not_called()
            policy.agent.reset.assert_called_once_with(cond_scale=7.0)
            np.testing.assert_array_equal(policy.goal_embed,value)
            self.assertEqual(policy.condition_metadata['source_file_sha256'],sha)
            self.assertEqual(policy.condition_metadata['source'],'official_video_embedding')
            self.assertIsNone(policy.condition_metadata['effective_prompt'])
            policy.agent.reset.reset_mock();policy.guidance=2.
            with patch.object(Steve1Policy,'OFFICIAL_VISUAL',{'collect wood':('wood.npy',sha)},create=True):
                policy.reset('collect wood')
            policy.agent.reset.assert_called_once_with(cond_scale=2.)
            self.assertEqual(policy.condition_metadata['cond_scale'],2.)
            np.testing.assert_array_equal(policy.goal_embed,value)
            policy.embed.assert_not_called()

    def test_visual_rejects_tampering_invalid_array_and_unsupported_goal_before_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy=self.policy('official-visual');policy.root=Path(tmp)
            path=policy.root/'weights/steve1/visual-prompts/wood.npy';path.parent.mkdir(parents=True)
            for value,wrong_hash in [(np.zeros((1,512),np.float32),True),
                                     (np.zeros((512,),np.float32),False),
                                     (np.full((1,512),np.nan,np.float32),False),
                                     (np.zeros((1,512),np.float64),False)]:
                np.save(path,value);sha='0'*64 if wrong_hash else hashlib.sha256(path.read_bytes()).hexdigest()
                with patch.object(Steve1Policy,'OFFICIAL_VISUAL',{'collect wood':('wood.npy',sha)},create=True):
                    with self.assertRaises(ValueError):policy.reset('collect wood')
            with self.assertRaises(ValueError):policy.reset('teleport home')
            policy.agent.reset.assert_not_called();policy.embed.assert_not_called()
