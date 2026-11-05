"""Source diversity must not alter local supervision or leak development labels."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from minecraftrl.human_source_bc import load_train_source_groups, reweight_sources
from minecraftrl.human_onset_bc import reweight_onsets
from minecraftrl.world_store import digest


class HumanSourceBCTests(unittest.TestCase):
    def test_equal_and_single_source_identity_preserve_exact_weights(self):
        weights=[.123456789,2.7,.003]
        flags=[False,True,True]
        for groups,exponent in ((['a','b'],1),(['a','a'],.5)):
            result,receipt=reweight_sources(weights,flags,groups,exponent)
            self.assertEqual(result,weights)
            self.assertTrue(receipt['identity'])

    def test_mass_ratios_tempered_and_local_weights_bitwise_unchanged(self):
        weights=[1.3,36.,.7,64.,1.,3.]
        flags=[False,True,False,True,True,True]
        result,receipt=reweight_sources(weights,flags,['large','large','small','small'])
        self.assertEqual([result[0],result[2]],[weights[0],weights[2]])
        self.assertAlmostEqual(math.fsum(result[i] for i in (1,3,4,5)),104.)
        self.assertAlmostEqual((result[1]+result[3])/(result[4]+result[5]),5.)
        self.assertAlmostEqual(result[1]/result[3],36./64.)
        self.assertAlmostEqual(result[4]/result[5],1./3.)
        self.assertTrue(receipt['within_source_ratios_preserved'])

    def test_composes_after_onset_weighting_without_erasing_onset_preference(self):
        values=[3.,1.,1.,8.,8.];flags=[False,True,True,True,True]
        onset,_=reweight_onsets(values,flags,['onset','nonjump','onset','nonjump'],4)
        result,_=reweight_sources(onset,flags,['a','a','b','b'])
        self.assertAlmostEqual(result[1]/result[2],4.)
        self.assertAlmostEqual(result[3]/result[4],4.)
        self.assertAlmostEqual(math.fsum(result[1:]),18.)
        self.assertEqual(result[0],3.)

    def test_invalid_weights_groups_and_exponents_rejected(self):
        for exponent in (-1,2,True,float('nan'),float('inf')):
            with self.subTest(exponent=exponent),self.assertRaises(ValueError):
                reweight_sources([1.,1.],[False,True],['a'],exponent)
        for weights,flags,groups in (([],[],[]),([1,0],[False,True],['a']),
                ([1,float('inf')],[False,True],['a']),([1,1],[False,1],['a']),
                ([1,1],[True,True],['a','b']),([1,1],[False,True],[]),
                ([1,1],[False,True],[' '])):
            with self.subTest(weights=weights,groups=groups),self.assertRaises(ValueError):
                reweight_sources(weights,flags,groups)

    def test_hash_bound_train_aliases_only_and_no_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'pairs.json'
            name='source-a-abcdef012345-20220101-010000'
            pair=dict(session=name,split='train')
            row=dict(owner='human',bc_source='hindsight_reference',human_session=name,human_frame=10)
            def contract():
                path.write_text(json.dumps(dict(pairs=[pair])),encoding='utf-8')
                return dict(pair_manifest=dict(path='pairs.json',sha256=digest(path)))
            groups,receipt=load_train_source_groups(root,contract(),[row])
            self.assertEqual(groups,['source-a'])
            self.assertFalse(receipt['development_used'])
            with self.assertRaisesRegex(ValueError,'duplicate human'):
                load_train_source_groups(root,contract(),[row,row])
            pair['split']='development'
            with self.assertRaisesRegex(ValueError,'cannot use development'):
                load_train_source_groups(root,contract(),[row])
            pair['split']='train';pair['alias']='other'
            with self.assertRaisesRegex(ValueError,'disagrees'):
                load_train_source_groups(root,contract(),[row])
            bound=contract();path.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_train_source_groups(root,bound,[row])


if __name__=='__main__':unittest.main()
