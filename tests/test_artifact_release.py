from pathlib import Path
import tempfile
import unittest
import zipfile

from minecraftrl.artifact_release import (confined, digest, inspect_public, plan, prepare,
    relative_path, restore, validate_manifest, verify_bundle)


class ArtifactReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'results/source').mkdir(parents=True)
        (self.root/'results/source/checkpoint.pt').write_bytes(b'opaque checkpoint, never unpickle')
        (self.root/'results/source/training.json').write_text('{"completed":true}')
        self.spec = dict(repository='owner/repo', tag='run-1',
            groups=[dict(id='policy', sources=[dict(path='results/source')])],
            experiments={'case':dict(groups=['policy'])})
        self.commit = 'a'*40

    def build(self, name='out', previous=None):
        return prepare(self.root, self.spec, self.root/name, self.commit, previous)

    def test_real_zip_round_trip_and_repeat_skip(self):
        manifest = self.build()
        target = self.root/'recipient'; target.mkdir()
        self.assertEqual(len(plan(target, manifest, 'case')['groups']), 1)
        result = restore(target, manifest, 'case', self.root/'out')
        self.assertEqual(result['files_created'], 2)
        self.assertEqual(restore(target, manifest, 'case', self.root/'out')['files_created'], 0)
        self.assertEqual((target/'results/source/checkpoint.pt').read_bytes(), (self.root/'results/source/checkpoint.pt').read_bytes())

    def test_content_is_deterministic_and_unchanged_bundle_references_old_release(self):
        first = self.build()
        second = self.build('out2')
        self.assertEqual(first['groups'], second['groups'])
        self.spec['tag'] = 'run-2'
        reused = self.build('out3', first)
        self.assertEqual(reused['groups'][0]['release_tag'], 'run-1')
        self.assertEqual(list((self.root/'out3').glob('*.zip')), [])

    def test_changed_checkpoint_gets_new_bundle(self):
        first = self.build()
        (self.root/'results/source/checkpoint.pt').write_bytes(b'new training result')
        self.spec['tag'] = 'run-2'
        second = self.build('out2', first)
        self.assertNotEqual(first['groups'][0]['asset'], second['groups'][0]['asset'])
        self.assertEqual(second['groups'][0]['release_tag'], 'run-2')

    def test_conflict_aborts_before_any_other_file_is_installed(self):
        manifest = self.build(); target = self.root/'recipient'
        (target/'results/source').mkdir(parents=True)
        protected = target/'results/source/training.json'; protected.write_text('existing unrelated content')
        with self.assertRaisesRegex(ValueError, 'conflict'): restore(target, manifest, 'case', self.root/'out')
        self.assertEqual(protected.read_text(), 'existing unrelated content')
        self.assertFalse((target/'results/source/checkpoint.pt').exists())

    def test_corrupt_archive_installs_nothing(self):
        manifest = self.build(); group = manifest['groups'][0]
        (self.root/'out'/group['asset']).write_bytes(b'broken')
        target = self.root/'recipient'; target.mkdir()
        with self.assertRaisesRegex(ValueError, 'bundle_hash'): restore(target, manifest, 'case', self.root/'out')
        self.assertEqual(list(target.iterdir()), [])

    def test_windows_path_escape_ads_reserved_and_case_collisions(self):
        for value in ('../outside', 'C:/outside', '/outside', 'a\\b', 'a/file:ads', 'a/CON.txt', 'a/file.', '.git/config', 'a/.env', 'a//b'):
            with self.subTest(path=value), self.assertRaises(ValueError): relative_path(value)
        manifest = self.build()
        row = dict(manifest['groups'][0]['files'][0]); row['path'] = row['path'].upper()
        manifest['groups'][0]['files'].append(row)
        with self.assertRaisesRegex(ValueError, 'duplicate_destination'): validate_manifest(manifest)

    def test_unexpected_zip_entry_rejected_even_if_archive_digest_is_updated(self):
        manifest = self.build(); group = manifest['groups'][0]; archive = self.root/'out'/group['asset']
        with zipfile.ZipFile(archive, 'a') as z: z.writestr('../escape', 'bad')
        group['sha256'] = digest(archive); group['bytes'] = archive.stat().st_size
        with self.assertRaisesRegex(ValueError, 'unexpected_bundle'): verify_bundle(archive, group)

    def test_member_hash_checked_independently_of_archive_hash(self):
        manifest = self.build(); group = manifest['groups'][0]
        group['files'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'member_hash'): verify_bundle(self.root/'out'/group['asset'], group)

    def test_public_binaries_and_embedded_game_rejected(self):
        for name in ('client.jar', 'minerl.whl'):
            p = self.root/name; p.write_bytes(b'not public')
            with self.assertRaises(ValueError): inspect_public(p, 'results/'+name)
        archive = self.root/'world.zip'
        with zipfile.ZipFile(archive, 'w') as z: z.writestr('client.jar', 'bad')
        with self.assertRaisesRegex(ValueError, 'world_member'): inspect_public(archive, 'results/world.zip')

    def test_original_world_dot_prefix_is_preserved_but_parent_escape_rejected(self):
        archive = self.root/'world.zip'
        with zipfile.ZipFile(archive, 'w') as z: z.writestr('./saves/world/DIM-1/data/raids.dat', b'NBT')
        before = archive.read_bytes()
        inspect_public(archive, 'results/world.zip')
        self.assertEqual(archive.read_bytes(), before)
        with zipfile.ZipFile(archive, 'a') as z: z.writestr('./../outside.dat', b'bad')
        with self.assertRaises(ValueError): inspect_public(archive, 'results/world.zip')

    def test_prepare_resumes_verified_local_bundle_after_missing_source_failure(self):
        self.spec['groups'].append(dict(id='second', sources=[dict(path='results/missing.txt')]))
        with self.assertRaises(FileNotFoundError): self.build()
        archive = next((self.root/'out').glob('*.zip')); before = archive.stat().st_mtime_ns
        (self.root/'results/missing.txt').write_text('now available')
        manifest = self.build()
        self.assertEqual(archive.stat().st_mtime_ns, before)
        self.assertEqual(len(manifest['groups']), 2)

    def test_secret_scan_across_block_boundary(self):
        p = self.root/'secret.json'
        p.write_bytes(b' '*(1024**2-4)+b'sk-'+b'A'*40)
        with self.assertRaisesRegex(ValueError, 'potential_credential'): inspect_public(p, 'results/secret.json')

    def test_selected_run_diagnostic_logs_are_allowed_but_global_logs_are_not(self):
        p = self.root/'watcher.log'; p.write_text('child process exited')
        inspect_public(p, 'results/run/watcher-runtime/logs/watcher.log')
        with self.assertRaisesRegex(ValueError, 'excluded_public_path'): inspect_public(p, 'logs/watcher.log')

    def test_link_destination_rejected(self):
        outside = self.root/'outside'; outside.mkdir()
        link = self.root/'linked'
        try: link.symlink_to(outside, target_is_directory=True)
        except OSError: self.skipTest('OS does not permit creating a test symlink')
        with self.assertRaisesRegex(ValueError, 'link_in_path'): confined(self.root, 'linked/file')

    def test_experiment_cannot_reference_a_different_checkpoint(self):
        manifest = self.build()
        manifest['experiments']['case']['frozen_assets'] = {'results/source/checkpoint.pt':'0'*64}
        with self.assertRaisesRegex(ValueError, 'experiment_asset_mismatch'): validate_manifest(manifest)

    def test_component_name_cannot_escape_output_directory(self):
        self.spec['groups'][0]['id'] = '../escape'
        with self.assertRaisesRegex(ValueError, 'invalid_group_id'): self.build()
        self.assertFalse((self.root/'out').exists())

    def test_file_directory_collision_is_rejected_before_restoration(self):
        manifest = self.build()
        row = dict(manifest['groups'][0]['files'][0]); row['path'] += '/child'
        manifest['groups'][0]['files'].append(row)
        with self.assertRaisesRegex(ValueError, 'file_directory_collision'): validate_manifest(manifest)


if __name__ == '__main__': unittest.main()
