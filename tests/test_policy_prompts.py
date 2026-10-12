import unittest
from pathlib import Path
from unittest.mock import Mock
from minecraftrl.policies import Steve1Policy


class PromptTests(unittest.TestCase):
    def policy(self,profile):
        policy=Steve1Policy.__new__(Steve1Policy)
        policy.prompt_profile=profile; policy.agent=Mock(); policy.mineclip=object(); policy.prior=object(); policy.device='cuda'
        policy.embed=Mock(return_value=Mock(tobytes=lambda:b'fixed-embedding'))
        return policy

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
