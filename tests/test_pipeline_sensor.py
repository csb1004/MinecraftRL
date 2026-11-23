"""SensorFrame and the sensor_v1 log (sensor-tier design 2026-11-19 §2, §3.1, §6; plan 2026-11-22 task 2)."""
import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.sensor import (ACTOR_TIERS, CROSSHAIR, SensorLog, check_actor_tiers, load_sensor_log,
                                         sensor_frame)

RAYS, VOXELS = 136, 27


def raw_sensor(**overrides):
    value = dict(sensor_available=1, sensor_client_tick=500, sensor_cols=15, sensor_rows=9, sensor_range=8., sensor_hfov=101.4,
                 sensor_vfov=70., sensor_eye_x=0.5, sensor_eye_y=65.62, sensor_eye_z=0.5, sensor_yaw=0., sensor_pitch=0.,
                 sensor_foot_x=0, sensor_foot_y=64, sensor_foot_z=0, sensor_tool_item='stone_pickaxe', sensor_tool_damage=3,
                 sensor_tool_max=131,
                 sensor_ray_kind=['stone'] * RAYS, sensor_ray_dist=[2.5] * RAYS, sensor_ray_x=[0] * RAYS,
                 sensor_ray_y=[65] * RAYS, sensor_ray_z=[3] * RAYS, sensor_ray_dx=[0.] * RAYS, sensor_ray_dy=[0.] * RAYS,
                 sensor_ray_dz=[1.] * RAYS, sensor_ray_fluid=['none'] * RAYS, sensor_ray_fluid_dist=[-1.] * RAYS,
                 sensor_voxel_kind=['air'] * 9 + ['air'] * 9 + ['air'] * 9, sensor_voxel_fluid=['none'] * VOXELS)
    value['sensor_voxel_kind'] = ['stone'] * 9 + ['air'] * 18     # floor below, open air at feet and head
    value.update(overrides)
    return value


def info(**sensor_overrides):
    workshop = dict(server_available=1, client_x=0.5, client_y=64., client_z=0.5, client_on_ground=1, server_fall_distance=0.,
                    server_health=20., server_food=20., screen=0, server_selected_item='stone_pickaxe',
                    server_inventory={'stone_pickaxe': 1, 'dirt': 0, 'cobblestone': 4},
                    # oracle / judge values a sensor frame must never read
                    server_x=9., server_y=9., server_z=9., pursuit_count=3, pursuit_distance=4., wood_target_x=7)
    demo = dict(demo_available=1, demo_mining_block='iron_ore', demo_mining_x=1, demo_mining_y=2, demo_mining_z=3,
                demo_last_destroy_harvestable=1, demo_hit_block='stone', demo_fixture_sha52=123)
    return dict(workshop=workshop, demo=demo, sensor=raw_sensor(**sensor_overrides))


class SensorFrameTests(unittest.TestCase):
    def test_frame_reads_client_pose_and_sensor_values(self):
        f = sensor_frame(info(), 1.5)
        self.assertEqual((f.t, f.tick, f.position, f.eye, f.yaw), (1.5, 500, (0.5, 64., 0.5), (0.5, 65.62, 0.5), 0.))
        self.assertEqual((f.foot, f.selected_item, f.tool_item, f.tool_damage, f.tool_max), ((0, 64, 0), 'stone_pickaxe',
                                                                                          'stone_pickaxe', 3, 131))
        self.assertEqual(f.inventory, {'stone_pickaxe': 1, 'cobblestone': 4})
        self.assertEqual((len(f.ray_kinds), len(f.ray_cells), len(f.voxel_kinds)), (RAYS, RAYS, VOXELS))
        self.assertEqual(f.crosshair, ('stone', (0, 65, 3)))
        self.assertEqual(f.layout, dict(cols=15, rows=9, range=8., hfov=101.4, vfov=70.))
        self.assertTrue(f.on_ground and not f.gui_open)

    def test_crosshair_face_comes_with_a_crosshair_block_only(self):
        value = info(); value['workshop']['hit_face'] = 2
        self.assertEqual(sensor_frame(value, 0.).hit_face, 2)
        self.assertIsNone(sensor_frame(info(), 0.).hit_face)
        xs = [-999] * RAYS
        missed = info(sensor_ray_x=xs, sensor_ray_y=xs, sensor_ray_z=xs); missed['workshop']['hit_face'] = 2
        self.assertIsNone(sensor_frame(missed, 0.).hit_face)

    def test_missed_ray_has_no_cell(self):
        kinds = ['air'] * RAYS; dist = [-1.] * RAYS; xs = [-999] * RAYS
        f = sensor_frame(info(sensor_ray_kind=kinds, sensor_ray_dist=dist, sensor_ray_x=xs, sensor_ray_y=xs, sensor_ray_z=xs), 0.)
        self.assertEqual(f.crosshair, ('air', None))
        self.assertIsNone(f.ray_cells[0])

    def test_unavailable_or_missing_sensor_gives_no_frame(self):
        self.assertIsNone(sensor_frame(info(sensor_available=0), 0.))
        self.assertIsNone(sensor_frame({'workshop': info()['workshop']}, 0.))

    def test_oracle_and_judge_values_do_not_change_the_frame(self):
        # Leak check 3 (design §6): change only what the server knows and the sensors cannot measure.
        base = info(); changed = info()
        changed['workshop'].update(server_x=-50., server_y=3., server_z=40., pursuit_count=0, wood_target_x=-1)
        changed['demo'].update(demo_mining_block='diamond_ore', demo_mining_x=-5, demo_last_destroy_harvestable=0,
                               demo_fixture_sha52=999, demo_hit_block='dirt')
        changed['fixture'] = {'targets': [[1, 2, 3]]}
        self.assertEqual(sensor_frame(base, 0.), sensor_frame(changed, 0.))
        # Non-vacuous: a sensor value does change it.
        self.assertNotEqual(sensor_frame(base, 0.), sensor_frame(info(sensor_tool_damage=4), 0.))

    def test_actor_tiers_are_sensor_or_instruction_only(self):
        check_actor_tiers(ACTOR_TIERS)
        self.assertTrue(set(ACTOR_TIERS.values()) <= {'sensor', 'instruction'})
        with self.assertRaises(ValueError): check_actor_tiers(dict(ACTOR_TIERS, target=['oracle']))
        with self.assertRaises(ValueError): check_actor_tiers(dict(ACTOR_TIERS, server_x='judge'))
        self.assertEqual(CROSSHAIR, RAYS - 1)


class SensorLogTests(unittest.TestCase):
    def test_log_round_trip_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SensorLog(Path(tmp) / 'sensor', chunk=2)
            rows = [raw_sensor(), raw_sensor(sensor_client_tick=501, sensor_tool_damage=4),
                    raw_sensor(sensor_ray_kind=['iron_ore'] * RAYS, sensor_ray_fluid=['lava'] * RAYS), raw_sensor(sensor_available=0)]
            indices = [log.add(step, raw) for step, raw in zip((10, 11, 12, 13), rows)]
            manifest = log.close()
            self.assertEqual(indices, [0, 1, 2, 3])
            self.assertEqual((manifest['schema'], manifest['rows'], manifest['complete'], len(manifest['chunks'])),
                             ('sensor_v1', 4, True, 2))
            self.assertEqual(manifest['layout'], dict(cols=15, rows=9, range=8., hfov=101.4, vfov=70.))
            self.assertEqual(json.loads((Path(tmp) / 'sensor/manifest.json').read_text())['rows'], 4)
            loaded = list(load_sensor_log(Path(tmp) / 'sensor'))
            self.assertEqual([r['step'] for r in loaded], [10, 11, 12, 13])
            self.assertEqual(loaded[1]['sensor_tool_damage'], 4)
            self.assertEqual(loaded[2]['sensor_ray_kind'][0], 'iron_ore')
            self.assertEqual(loaded[2]['sensor_ray_fluid'][5], 'lava')
            self.assertEqual(loaded[0]['sensor_ray_dist'][0], 2.5)
            self.assertEqual(loaded[3]['sensor_available'], 0)
            # A frame rebuilt from the log equals the live frame (S2 shadow records use the log).
            live = sensor_frame(info(), 0.); again = sensor_frame(dict(info(), sensor=loaded[0]), 0.)
            self.assertEqual(live, again)

    def test_entity_rays_are_logged_and_old_raws_without_them_read_as_none(self):
        # Entity rays (design §3.1, user request 2026-11-23): first entity before the block on each ray, no ids.
        ents = ['none'] * RAYS; ents[7] = 'item:beacon'; ents[CROSSHAIR] = 'cow'
        dist = [-1.] * RAYS; dist[7] = 1.75; dist[CROSSHAIR] = 3.25
        xs = [-999.] * RAYS; xs[7] = 551.6123456789; xs[CROSSHAIR] = 2.5
        ys = [-999.] * RAYS; ys[7] = 64.0; ys[CROSSHAIR] = 65.
        zs = [-999.] * RAYS; zs[7] = 831.4; zs[CROSSHAIR] = 4.
        raw = raw_sensor(sensor_ray_entity=ents, sensor_ray_entity_dist=dist, sensor_ray_entity_x=xs,
                         sensor_ray_entity_y=ys, sensor_ray_entity_z=zs)
        live = sensor_frame(info(), 0.)
        live = sensor_frame(dict(info(), sensor=raw), 0.)
        self.assertEqual((live.ray_entities[7], live.ray_entity_points[7], live.ray_entity_dist[7]),
                         ('item:beacon', (551.6123456789, 64.0, 831.4), 1.75))
        self.assertIsNone(live.ray_entity_points[0])
        self.assertEqual(live.entity_sightings(), [('item:beacon', (551.6123456789, 64.0, 831.4), 1.75),
                                                   ('cow', (2.5, 65., 4.), 3.25)])
        old = sensor_frame(info(), 0.)                        # logs before 2026-11-23 have no entity rays
        self.assertEqual((set(old.ray_entities), old.entity_sightings()), ({'none'}, []))
        with tempfile.TemporaryDirectory() as tmp:
            log = SensorLog(Path(tmp) / 'sensor'); log.add(1, raw); log.add(2, raw_sensor()); log.close()
            loaded = list(load_sensor_log(Path(tmp) / 'sensor'))
            self.assertEqual(sensor_frame(dict(info(), sensor=loaded[0]), 0.), live)     # exact: points are float64
            self.assertEqual(sensor_frame(dict(info(), sensor=loaded[1]), 0.), old)
        self.assertEqual({ACTOR_TIERS[k] for k in ('ray_entities', 'ray_entity_dist', 'ray_entity_points')}, {'sensor'})

    def test_unavailable_raw_with_empty_arrays_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = SensorLog(Path(tmp) / 'sensor')
            empty = dict(raw_sensor(sensor_available=0), sensor_ray_kind=[], sensor_ray_dist=[], sensor_voxel_kind=[])
            self.assertEqual(log.add(1, empty), 0)
            self.assertEqual(log.close()['rows'], 1)


if __name__ == '__main__':
    unittest.main()
