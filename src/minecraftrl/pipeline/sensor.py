"""Sensor-tier actor input (sensor-tier design 2026-11-19 §2–§3, §6).

`SensorFrame` is the only observation type the sensor path's guards and features read. It is built from the client
sensor telemetry (`info['sensor']`, opt-in jar) and the body-sensor fields of the workshop telemetry: client pose,
health, food, feet, fall distance, screen, inventory and held item. Server pose, mining targets, destroy records,
fixtures and 360-degree mob scans are never read here (leak checks 1 and 3).

Inventory, held item, health, food and fall distance are written by the server player (about 4 ticks late); the design
classes them as body sensors (load, proprioception), so the execution side treats them as sensor values.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = 'sensor_v1'
RAYS, VOXELS = 136, 27
CROSSHAIR = RAYS - 1          # the last ray is the client's own crosshair pick (reach-limited), like demo_hit_block
FLUIDS = ('none', 'water', 'lava')
MISS = -999
# Voxel index = (dy + 1) * 9 + (dz + 1) * 3 + (dx + 1), around the block at the feet.
VOXEL_OFFSETS = tuple((dx, dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1) for dx in (-1, 0, 1))
# Workshop telemetry keys a frame reads; every other key (server_x/y/z, pursuit_*, wood_*, threat_*) is ignored.
WORKSHOP_SENSOR_KEYS = ('client_x', 'client_y', 'client_z', 'client_on_ground', 'server_fall_distance', 'server_health',
                        'server_food', 'screen', 'server_selected_item', 'server_inventory', 'hit_face')

# Tier of every actor-side input of the sensor path (leak check 4). Instruction values come from the plan, not the world.
ACTOR_TIERS = {
    'position': 'sensor', 'eye': 'sensor', 'yaw': 'sensor', 'pitch': 'sensor', 'tick': 'sensor', 'on_ground': 'sensor',
    'fall_distance': 'sensor', 'health': 'sensor', 'food': 'sensor', 'gui_open': 'sensor', 'selected_item': 'sensor',
    'inventory': 'sensor', 'tool_item': 'sensor', 'tool_damage': 'sensor', 'tool_max': 'sensor', 'foot': 'sensor',
    'voxel_kinds': 'sensor', 'voxel_fluids': 'sensor', 'ray_kinds': 'sensor', 'ray_dist': 'sensor', 'ray_cells': 'sensor',
    'ray_dirs': 'sensor', 'ray_fluids': 'sensor', 'ray_fluid_dist': 'sensor', 'layout': 'sensor', 'seen_map': 'sensor',
    'hit_face': 'sensor', 'blueprint': 'instruction',
    'guarded': 'instruction', 'needs': 'instruction', 'harvest_requires': 'instruction', 'planned_uses': 'instruction',
    'skill_kind': 'instruction',
}


def check_actor_tiers(tiers):
    """Refuse an execution path whose actor inputs include an oracle or judge value."""
    bad = sorted(k for k, v in tiers.items() if v not in ('sensor', 'instruction', 'screen'))
    if bad: raise ValueError('actor inputs must be screen, instruction or sensor tier: %s' % ', '.join(bad))


@dataclass(frozen=True)
class SensorFrame:
    t: float
    tick: int
    position: tuple
    eye: tuple
    yaw: float
    pitch: float
    on_ground: bool
    fall_distance: float
    health: float
    food: float
    gui_open: bool
    selected_item: str
    inventory: dict
    tool_item: str
    tool_damage: int
    tool_max: int
    foot: tuple
    voxel_kinds: tuple
    voxel_fluids: tuple
    ray_kinds: tuple
    ray_dist: tuple
    ray_cells: tuple
    ray_dirs: tuple
    ray_fluids: tuple
    ray_fluid_dist: tuple
    layout: dict
    hit_face: int | None = None      # face ordinal of the crosshair block (client pick), None without one

    @property
    def crosshair(self): return self.ray_kinds[CROSSHAIR], self.ray_cells[CROSSHAIR]

    def voxel(self, dx, dy, dz):
        """(kind, fluid) of the block at feet + offset, each offset in -1..1."""
        i = (dy + 1) * 9 + (dz + 1) * 3 + (dx + 1)
        return self.voxel_kinds[i], self.voxel_fluids[i]


def _cell(x, y, z):
    x, y, z = int(x), int(y), int(z)
    return None if (x, y, z) == (MISS, MISS, MISS) else (x, y, z)


def sensor_frame(info, t):
    """The sensor-tier view of one observation, or None when the client sensors are not available."""
    s = info.get('sensor') or {}; w = info.get('workshop') or {}
    if int(s.get('sensor_available', 0)) != 1 or len(s.get('sensor_ray_kind') or ()) != RAYS: return None
    w = {k: w.get(k) for k in WORKSHOP_SENSOR_KEYS}
    rays = range(RAYS)
    picked = _cell(s['sensor_ray_x'][CROSSHAIR], s['sensor_ray_y'][CROSSHAIR], s['sensor_ray_z'][CROSSHAIR])
    face = w.get('hit_face')
    hit_face = int(face) if picked is not None and face is not None and 0 <= int(face) < 6 else None
    return SensorFrame(
        t=float(t), tick=int(s['sensor_client_tick']),
        position=tuple(float(w.get('client_' + a) or 0.) for a in 'xyz'),
        eye=tuple(float(s['sensor_eye_' + a]) for a in 'xyz'),
        yaw=float(s['sensor_yaw']), pitch=float(s['sensor_pitch']),
        on_ground=bool(w.get('client_on_ground')), fall_distance=float(w.get('server_fall_distance') or 0.),
        health=float(w.get('server_health', 20.) if w.get('server_health') is not None else 20.),
        food=float(w.get('server_food') or 0.), gui_open=(w.get('screen') or 0) != 0,
        selected_item=w.get('server_selected_item') or 'air',
        inventory={k: int(v) for k, v in (w.get('server_inventory') or {}).items() if v},
        tool_item=s.get('sensor_tool_item') or 'air', tool_damage=int(s['sensor_tool_damage']), tool_max=int(s['sensor_tool_max']),
        foot=tuple(int(s['sensor_foot_' + a]) for a in 'xyz'),
        voxel_kinds=tuple(s['sensor_voxel_kind']), voxel_fluids=tuple(s['sensor_voxel_fluid']),
        ray_kinds=tuple(s['sensor_ray_kind']), ray_dist=tuple(float(d) for d in s['sensor_ray_dist']),
        ray_cells=tuple(_cell(s['sensor_ray_x'][i], s['sensor_ray_y'][i], s['sensor_ray_z'][i]) for i in rays),
        ray_dirs=tuple((float(s['sensor_ray_dx'][i]), float(s['sensor_ray_dy'][i]), float(s['sensor_ray_dz'][i])) for i in rays),
        ray_fluids=tuple(s['sensor_ray_fluid']), ray_fluid_dist=tuple(float(d) for d in s['sensor_ray_fluid_dist']),
        layout=layout_of(s), hit_face=hit_face)


def layout_of(raw):
    return dict(cols=int(raw['sensor_cols']), rows=int(raw['sensor_rows']), range=float(raw['sensor_range']),
                hfov=float(raw['sensor_hfov']), vfov=float(raw['sensor_vfov']))


class SensorLog:
    """Raw client sensor values per dispatched step: chunked compressed npz files and a manifest kept current per chunk.
    Rows carry only the returned row index (design §3.1: not in the row JSON)."""

    SCALARS = (('sensor_available', np.int8), ('sensor_client_tick', np.int64), ('sensor_eye_x', np.float64),
               ('sensor_eye_y', np.float64), ('sensor_eye_z', np.float64), ('sensor_yaw', np.float64),
               ('sensor_pitch', np.float64), ('sensor_foot_x', np.int32), ('sensor_foot_y', np.int32),
               ('sensor_foot_z', np.int32), ('sensor_tool_damage', np.int32), ('sensor_tool_max', np.int32),
               ('sensor_cols', np.int16), ('sensor_rows', np.int16), ('sensor_range', np.float64),
               ('sensor_hfov', np.float64), ('sensor_vfov', np.float64))
    RAY_NUMBERS = (('sensor_ray_dist', np.float32, -1.), ('sensor_ray_x', np.int32, MISS), ('sensor_ray_y', np.int32, MISS),
                   ('sensor_ray_z', np.int32, MISS), ('sensor_ray_dx', np.float32, 0.), ('sensor_ray_dy', np.float32, 0.),
                   ('sensor_ray_dz', np.float32, 0.), ('sensor_ray_fluid_dist', np.float32, -1.))
    KINDS = (('sensor_ray_kind', RAYS), ('sensor_voxel_kind', VOXELS))
    FLUID_KEYS = (('sensor_ray_fluid', RAYS), ('sensor_voxel_fluid', VOXELS))

    def __init__(self, folder, *, chunk=512):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True); self.chunk = int(chunk)
        self.rows = []; self.chunks = []; self.total = 0; self.vocab = ['air']; self.codes = {'air': 0}; self.layout = None

    def _code(self, kind):
        if kind not in self.codes: self.codes[kind] = len(self.vocab); self.vocab.append(kind)
        return self.codes[kind]

    def add(self, step, raw):
        """Store one raw `info['sensor']` dict; returns its row index in this log."""
        row = dict(step=int(step))
        for key, dtype in self.SCALARS: row[key] = dtype(raw.get(key, 0) or 0)
        for key, dtype, missing in self.RAY_NUMBERS:
            values = list(raw.get(key) or ()); values += [missing] * (RAYS - len(values))
            row[key] = np.asarray(values, dtype)
        for key, size in self.KINDS:
            values = list(raw.get(key) or ()); values += ['air'] * (size - len(values))
            row[key] = np.asarray([self._code(v) for v in values], np.int16)
        for key, size in self.FLUID_KEYS:
            values = list(raw.get(key) or ()); values += ['none'] * (size - len(values))
            row[key] = np.asarray([FLUIDS.index(v) for v in values], np.int8)
        row['sensor_tool_item'] = np.int16(self._code(raw.get('sensor_tool_item') or 'air'))
        if self.layout is None and int(raw.get('sensor_available', 0)) == 1: self.layout = layout_of(raw)
        self.rows.append(row); index = self.total + len(self.rows) - 1
        if len(self.rows) >= self.chunk: self._flush()
        return index

    def _flush(self):
        if not self.rows: return
        name = 'sensor-%03d.npz' % len(self.chunks); path = self.folder / name
        np.savez_compressed(path, **{k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]})
        self.chunks.append(dict(file=name, rows=len(self.rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.total += len(self.rows); self.rows = []
        self._manifest(complete=False)

    def _manifest(self, *, complete):
        manifest = dict(schema=SCHEMA, rays=RAYS, voxels=VOXELS, crosshair=CROSSHAIR, layout=self.layout,
                        voxel_offsets=[list(o) for o in VOXEL_OFFSETS], kinds=list(self.vocab), fluids=list(FLUIDS),
                        tiers={'sensor_v1': 'sensor'}, rows=self.total, chunks=list(self.chunks), complete=complete,
                        source='client world and client player only (SensorObservation.java)')
        (self.folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        return manifest

    def close(self):
        self._flush()
        return self._manifest(complete=True)


def load_sensor_log(folder):
    """Yield the logged rows as raw-sensor dicts (plus `step`), in order; checks each chunk's hash."""
    folder = Path(folder); manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA: raise ValueError('not a %s log' % SCHEMA)
    kinds = manifest['kinds']; fluids = manifest['fluids']
    for chunk in manifest['chunks']:
        path = folder / chunk['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk['sha256']: raise ValueError('modified sensor chunk ' + chunk['file'])
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        for i in range(chunk['rows']):
            row = {}
            for key, value in arrays.items():
                v = value[i]
                if key in ('sensor_ray_kind', 'sensor_voxel_kind'): row[key] = [kinds[c] for c in v.tolist()]
                elif key in ('sensor_ray_fluid', 'sensor_voxel_fluid'): row[key] = [fluids[c] for c in v.tolist()]
                elif key == 'sensor_tool_item': row[key] = kinds[int(v)]
                else: row[key] = v.tolist()
            yield row
