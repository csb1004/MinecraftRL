"""Opt-in client sensor telemetry in the patched jar (sensor-tier design 2026-11-19 §3.1, plan 2026-11-22 task 1)."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from minecraftrl.world_save_patch import build_save_jar, patched_sources

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / 'src/minecraftrl/SensorObservation.java'
DEMO = dict(craft_monitor=True, workshop_monitor=True, demo_monitor=True)


def minecraft_dir():
    try:
        from minerl.env.malmo import InstanceManager
    except ImportError:
        return None
    path = Path(InstanceManager.MINECRAFT_DIR)
    return path if (path / 'build/libs/mcprec-6.13.jar').exists() else None


class SensorPatchTests(unittest.TestCase):
    def test_sensor_monitor_requires_the_demo_monitor(self):
        for call in (build_save_jar, patched_sources):
            with self.subTest(call=call.__name__), self.assertRaises(ValueError):
                call(ROOT, ROOT / 'missing-minecraft', craft_monitor=True, workshop_monitor=True, sensor_monitor=True)

    def test_sensor_java_reads_only_the_client_world(self):
        text = JAVA.read_text(encoding='utf-8')
        for server in ('getIntegratedServer', 'ServerPlayerEntity', 'getServerWorld', 'DemoFixture', 'interactionManager'):
            self.assertNotIn(server, text)
        self.assertIn('mc.world.rayTraceBlocks', text)
        self.assertIn('RayTraceContext.FluidMode.NONE', text)   # solid pass: like the crosshair
        self.assertIn('RayTraceContext.FluidMode.ANY', text)    # fluid pass: the first fluid nearer than the solid hit

    def test_sensor_sources_are_opt_in_additions(self):
        mc = minecraft_dir()
        if mc is None: self.skipTest('MCP-Reborn source is unavailable')
        plain = patched_sources(ROOT, mc, **DEMO)
        self.assertEqual(plain[2], patched_sources(ROOT, mc, **DEMO, sensor_monitor=False)[2])  # earlier jars unchanged
        for place in (False, True):
            with self.subTest(place=place):
                _, extra, digest = patched_sources(ROOT, mc, **DEMO, place_monitor=place, sensor_monitor=True)
                self.assertNotEqual(digest, patched_sources(ROOT, mc, **DEMO, place_monitor=place)[2])
                self.assertEqual(extra['SensorObservation.java'], JAVA.read_text(encoding='utf-8'))
                self.assertEqual(extra['EnvServer.java'].count('SensorObservation.snapshot()'), 1)
                self.assertEqual(extra['WorkshopObservation.java'].count('SensorObservation.capture();'), 1)
                self.assertEqual(extra['WorkshopObservation.java'].count('DemoObservation.capture();'), 1)
                self.assertEqual(extra['WorkshopObservation.java'].count('PlaceObservation.capture();'), int(place))

    def test_sensor_class_compiles_against_the_upstream_jar(self):
        mc = minecraft_dir()
        if mc is None or shutil.which('javac') is None: self.skipTest('MCP-Reborn jar or javac is unavailable')
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run(['javac', '-encoding', 'UTF-8', '-source', '8', '-target', '8', '-cp',
                                     str(mc / 'build/libs/mcprec-6.13.jar'), '-d', out, str(JAVA)],
                                    capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(sys.platform == 'linux', 'MineRL specification is installed in WSL')
    def test_workshop_survival_sensor_monitor_is_opt_in(self):
        from minecraftrl.workshop_monitor import SENSOR_RAYS, SENSOR_VOXELS, workshop_survival
        self.assertNotIn('sensor', {m.to_string() for m in workshop_survival(demo=True).create_monitors()})
        monitors = {m.to_string(): m for m in workshop_survival(demo=True, sensor=True).create_monitors()}
        spaces = monitors['sensor'].space.spaces
        for key in ('sensor_available', 'sensor_eye_x', 'sensor_yaw', 'sensor_foot_y', 'sensor_hfov', 'sensor_tool_damage',
                    'sensor_tool_max', 'sensor_tool_item'):
            self.assertIn(key, spaces)
        self.assertEqual(spaces['sensor_ray_dist'].shape, (SENSOR_RAYS,))
        self.assertEqual(len(spaces['sensor_ray_kind'].spaces), SENSOR_RAYS)
        self.assertEqual(len(spaces['sensor_voxel_kind'].spaces), SENSOR_VOXELS)
        with self.assertRaises(ValueError): workshop_survival(sensor=True)


if __name__ == '__main__':
    unittest.main()
