import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from minecraftrl.hazard_condition import load_condition
from minecraftrl.world_store import digest


class HazardConditionTests(unittest.TestCase):
    def test_tool_and_hazard_conditions_are_not_silently_interchangeable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,m=self.fixture(root)
            with self.assertRaises(ValueError):load_condition(path,root,kind='experimental_tool_condition')
            m['kind']='experimental_tool_condition';path.write_text(json.dumps(m))
            self.assertEqual(load_condition(path,root,kind='experimental_tool_condition')['provenance']['kind'],m['kind'])
            with self.assertRaises(ValueError):load_condition(path,root)

    def test_cli_rejects_hazard_condition_without_guard_before_creating_output(self):
        from minecraftrl.cli import parser,episode
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            args=parser().parse_args(['session','--policy','steve1','--task-hazard-condition','candidate.json','--output',str(output)])
            with self.assertRaisesRegex(ValueError,'requires threat guard'):episode(args,{})
            self.assertFalse(output.exists())

    def fixture(self,root):
        folder=root/'results/condition';folder.mkdir(parents=True)
        weights=root/'weights/steve1';weights.mkdir(parents=True)
        for name in ('steve1.weights','steve1_prior.pt','attn.pth'):(weights/name).write_bytes(name.encode())
        value=np.ones((1,512),dtype=np.float32);np.save(folder/'value.npy',value)
        m=dict(schema=1,kind='experimental_hazard_condition',prompt='run away from monsters',seed=9911,
            path='results/condition/value.npy',file_sha256=digest(folder/'value.npy'),
            embedding_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
            base_model_sha256=digest(weights/'steve1.weights'),prior_sha256=digest(weights/'steve1_prior.pt'),
            mineclip_sha256=digest(weights/'attn.pth'),source_commit='903b244796322f4d0073a8f62c05f51eac3aed52',validated_skill=False)
        path=folder/'condition.json';path.write_text(json.dumps(m));return path,m

    def test_bound_tensor_is_loaded_and_modified_source_or_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,m=self.fixture(root)
            loaded=load_condition(path,root)
            self.assertEqual(loaded['sha256'],m['embedding_sha256'])
            self.assertFalse(loaded['provenance']['validated_skill'])
            (root/'weights/steve1/steve1.weights').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'model mismatch'):load_condition(path,root)
            (root/m['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_condition(path,root)

    def test_invalid_numeric_array_and_outside_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,m=self.fixture(root)
            for value in (np.full((1,512),np.nan,dtype=np.float32),np.ones((512,),dtype=np.float32),np.ones((1,512),dtype=np.float64)):
                np.save(root/m['path'],value);m['file_sha256']=digest(root/m['path'])
                path.write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError,'invalid hazard embedding'):load_condition(path,root)
            m['path']='../outside.npy';path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):load_condition(path,root)

    def test_guidance_is_explicit_bounded_and_zero_is_not_defaulted_to_six(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,m=self.fixture(root)
            self.assertEqual(load_condition(path,root)['cond_scale'],6.)
            m['cond_scale']=0.;path.write_text(json.dumps(m))
            self.assertEqual(load_condition(path,root)['cond_scale'],0.)
            for invalid in (True,-1,8,float('nan'),'0'):
                m['cond_scale']=invalid;path.write_text(json.dumps(m))
                with self.assertRaises(ValueError):load_condition(path,root)
