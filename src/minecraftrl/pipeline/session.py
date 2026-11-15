"""One demo game session: prepare the fixture, run plan steps, stop and release.

Success is read from server telemetry only. Rule (GUI) and scripted steps are
recorded with their own controller and never count as policy success.
"""
import hashlib
import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..runtime import jsonable
from .rm import sample_from
from .steps import block_matches, item_count

CONTROLLERS = {'policy': 'policy', 'rule': 'rule', 'scripted': 'scripted_probe'}
STOP_STATUSES = ('stop_file', 'time_limit', 'environment_done')
WORKSHOP_SAMPLE = ('world_time', 'server_tick', 'server_x', 'server_y', 'server_z', 'yaw', 'pitch', 'screen',
                   'server_selected_item', 'server_health', 'server_available', 'hit_x', 'hit_y', 'hit_z',
                   'held_key_count', 'server_on_ground', 'server_fall_distance',
                   # jump course (game check 2026-11-13): the client pose leads the server pose by about 4 ticks
                   'client_x', 'client_y', 'client_z', 'client_on_ground', 'client_tick',
                   # block placing (design 2026-11-14): the crosshair face says where a placed block lands
                   'hit_face')


def fixture_sha52(path):
    """Same 52-bit prefix DemoFixture reports (first 13 hex digits of SHA-256)."""
    return int(hashlib.sha256(Path(path).read_bytes()).hexdigest()[:13], 16)


def set_condition(policy, embedding, cond_scale):
    """Switch STEVE-1's goal like tool_collection.RememberedToolPolicy: reset recurrent state, set the embedding."""
    policy.agent.reset(cond_scale=cond_scale); policy.goal_embed = embedding
    meta = dict(cond_scale=float(cond_scale),
                embedding_sha256=hashlib.sha256(np.ascontiguousarray(embedding).tobytes()).hexdigest())
    policy.condition_metadata = meta
    return meta


def nonzero(values): return {k: int(v) for k, v in values.items() if v}


def compact_action(action):
    result = {}
    for key, value in action.items():
        value = np.asarray(value)
        if key == 'camera': result[key] = [float(v) for v in value.reshape(-1)]
        elif value.any(): result[key] = jsonable(value.reshape(-1)[0]) if value.size == 1 else jsonable(value)
    return result


def attacking(action): return bool(np.asarray(action.get('attack', 0)).any())


# Keys that can take the equipped tool out of hand: slot switch, Q-drop, and the inventory GUI (crafting is a rule step).
TOOL_KEYS = ('drop', 'inventory', *('hotbar.%d' % i for i in range(1, 10)))


# Blocks whose `use` opens a GUI; under the tool lock the policy cannot close one (the inventory key is locked).
GUI_BLOCKS = ('crafting_table', 'furnace', 'blast_furnace', 'smoker', 'chest', 'trapped_chest', 'barrel')


def lock_tool(action, hit_block=None, allowed=()):
    """Remove the policy's tool-losing keys so the given/equipped tool stays held; returns (action, masked keys).
    `use` is removed only while aiming at a GUI block (real check P3, 2026-11-05); placing blocks stays the policy's.
    `allowed` hotbar keys pass (building round 5 switches between the block and the pickaxe, design 2026-11-15)."""
    masked = [k for k in TOOL_KEYS if k not in allowed and np.asarray(action.get(k, 0)).any()]
    if hit_block in GUI_BLOCKS and np.asarray(action.get('use', 0)).any(): masked.append('use')
    if not masked: return action, masked
    action = dict(action)
    for key in masked: action[key] = np.zeros_like(np.asarray(action[key]))
    return action, masked


class DemoSession:
    def __init__(self, env, policy, catalog, recorder, gate, stop_file, *, limit_s, skills, clock=time.monotonic,
                 rule_controller=None, fixture_sha52=None, settle_steps=20, prepare_limit=400, tool_lock=False, oversight=None,
                 reacquire=None, course=None, place=None, bridge=None, build=None, tool_allowed=()):
        if not 0 < limit_s <= 900: raise ValueError('limit_s must be in (0, 900]')
        if any(not k.startswith('hotbar.') or k not in TOOL_KEYS for k in tool_allowed):
            raise ValueError('only hotbar keys can pass the tool lock: %s' % list(tool_allowed))
        if rule_controller is None:
            from ..workshop import WorkshopController as rule_controller
        self.env = env; self.policy = policy; self.catalog = catalog; self.recorder = recorder; self.gate = gate
        self.stop_file = stop_file; self.limit_s = limit_s; self.skills = skills; self.clock = clock
        self.rule_controller = rule_controller; self.fixture_sha52 = fixture_sha52
        self.settle_steps = settle_steps; self.prepare_limit = prepare_limit
        # Explicit action filter on policy steps only; every masked key is logged and never counts as policy skill.
        self.tool_lock = tool_lock; self.tool_allowed = tuple(tool_allowed)
        # Goal-derived guard and monitors (None = plain probe session). A hold ends policy dispatch for this run.
        self.oversight = oversight; self.last_sample = None; self.next_steps = None; self.surface_y = None
        self.last_kind = None  # kind of the step that ran last (a policy step after a rule step gets a level view)
        if oversight: oversight.emit = recorder.event
        # Reacquire hook (stage_live.ReacquireHook): live stage input for the residual head and the latent record.
        # It only acts on policy steps whose skill has a `reacquire` block.
        self.reacquire = reacquire
        # Jump course (design 2026-11-13): the fixture's course; its stage feed ends `until='course'` steps on the goal
        # platform (success) or a fall (failure) and drives the preparation-only teacher check.
        if course is not None:
            from ..jump_course import JumpFeed
            course = JumpFeed(course)
        self.course = course
        # Block placing (design 2026-11-14): the fixture's task; its feed ends `until='place'` steps on a server placement
        # at the target and drives the preparation-only teacher check.
        if place is not None:
            from ..place_task import PlaceFeed
            place = PlaceFeed(place)
        self.place = place
        # Bridging (design 2026-11-14): its feed ends `until='bridge'` steps on the server goal platform or a fall and
        # drives the preparation-only teacher check.
        if bridge is not None:
            from ..bridge_task import BridgeFeed
            bridge = BridgeFeed(bridge)
        self.bridge = bridge
        # Building (design 2026-11-14): its feed ends `until='build'` steps when the server placement record covers the
        # blueprint and drives the preparation-only teacher check.
        if build is not None:
            from ..build_recover import feed_for    # the recovering teacher when the fixture names it (design 2026-11-15)
            build = feed_for(build)
        self.build = build
        self.steps = 0; self.obs = None; self.info = {}; self.started = None

    # Workshop controller adapter: it reads .gui, .env.raw.action_space and .steps.
    @property
    def gui(self): return self.info.get('workshop')

    def demo(self): return self.info.get('demo', {})

    def inventory(self): return dict(self.gui.get('server_inventory', {})) if self.gui else {}

    def server_valid(self): return bool(self.gui) and self.gui.get('server_available') == 1 and not self.gui.get('error')

    def noop(self): return self.env.raw.action_space.noop()

    def stop_reason(self):
        if self.stop_file is not None and Path(self.stop_file).exists(): self.gate.stop('stop_file')
        if self.started is not None and self.clock() - self.started >= self.limit_s: self.gate.stop('time_limit')
        return self.gate.reason if self.gate.requested.is_set() else None

    def sample(self, controller, action):
        w = self.gui or {}
        value = {k: w.get(k) for k in WORKSHOP_SAMPLE}
        value.update(controller=controller, action=compact_action(action),
                     inventory=nonzero(w.get('server_inventory', {})), demo=dict(self.demo()))
        if 'place' in self.info: value['place'] = dict(self.info['place'])   # opt-in place monitor (plan 2026-11-13 §1)
        return value

    def dispatch(self, action, controller, masked=()):
        """Send one action unless STOP or the deadline is latched; record frame and telemetry."""
        if self.stop_reason(): return False
        dispatched, transition = self.gate.dispatch(lambda: self.env.step(action))
        if not dispatched: return False
        self.steps += 1; self.obs, _, done, self.info = transition
        self.recorder.frame(self.steps, self.obs['pov'])
        sample = self.sample(controller, action)
        if masked: sample['masked'] = list(masked)
        row = self.recorder.telemetry(self.steps, sample)
        if self.reacquire is not None: self.reacquire.observe(row)
        if self.course is not None: self.course.observe(row)
        if self.place is not None: self.place.observe(row)
        if self.bridge is not None: self.bridge.observe(row)
        if self.build is not None: self.build.observe(row)
        if done: self.gate.stop('environment_done')
        return True

    def prepare(self):
        """Neutral steps until server telemetry (and the requested fixture) is live, then settle for rendering."""
        for _ in range(self.prepare_limit):
            if not self.dispatch(self.noop(), 'prepare'): return self.gate.reason
            if not self.server_valid(): continue
            if self.fixture_sha52 is not None:
                state = self.demo().get('demo_fixture_state')
                if state == -1: return 'fixture_failed'
                if state != 2: continue
                if int(self.demo().get('demo_fixture_sha52', -1)) != self.fixture_sha52: return 'fixture_failed'
            # Reference height for pit depth (privileged, disclosed): the fixture's start block when the game reports it
            # (a check fixture may dig the shaft under it), else where the session starts standing.
            origin = self.demo().get('demo_fixture_origin_y') if self.fixture_sha52 is not None else None
            self.surface_y = float(origin) if origin is not None else float(self.gui['server_y'])
            self.recorder.event('fixture_ready' if self.fixture_sha52 is not None else 'telemetry_ready',
                                blocks=self.demo().get('demo_fixture_blocks'), surface_y=self.surface_y)
            for _ in range(self.settle_steps):
                if not self.dispatch(self.noop(), 'prepare'): return self.gate.reason
            return None
        return 'fixture_timeout'

    def run(self, steps):
        report = dict(status='initializing', execution_valid=False, steps=0, prepare_steps=0, stage_results=[],
                      all_steps_completed=False, final_inventory=None, release='not_attempted', tool_lock=self.tool_lock,
                      tool_allowed=list(self.tool_allowed),
                      oversight=self.oversight.summary() if self.oversight else None,
                      stop_contract='no new dispatch after stop acknowledgment; an in-flight step may finish')
        try:
            self.obs = self.env.reset(); self.started = self.clock()
            self.recorder.event('session_started', plan=[asdict(s) for s in steps], limit_s=self.limit_s)
            self.kb_snapshot('session_start')
            failure = self.prepare(); report['prepare_steps'] = self.steps
            if failure: report['status'] = failure
            else:
                queue = list(steps)
                while queue:
                    step = queue.pop(0)
                    result = self.run_step(step); report['stage_results'].append(result)
                    if result['status'] == 'completed' and self.oversight:
                        self.oversight.verify(step, result['gained'], self.last_sample if step.kind == 'policy' else None)
                        self.kb_snapshot('step_verified')
                    if result['status'] == 'anomaly_hold' and not self.gate.requested.is_set():
                        resolution = result['slow_path'].get('resolution')
                        # Slow path replanned (new plan) or retries the step: release the hold and continue.
                        if self.next_steps:  # 'insert' keeps the rest of the queue behind the sub-goal
                            queue = list(self.next_steps) + (queue if resolution == 'insert' else []); self.next_steps = None
                        elif resolution == 'retry':
                            if result['anomaly'].get('kind') == 'guard_violation_persistent' and not self.turn_away(step):
                                break  # stopped while turning
                            queue.insert(0, step)
                        if self.next_steps is None and resolution in ('replan', 'insert', 'retry'):
                            self.recorder.event('resume', resolution=resolution, plan_id=result['slow_path'].get('plan_id'),
                                                detect_to_resume_s=self.clock() - result['slow_path']['held_at'])
                            continue
                    if result['status'] != 'completed': break
                done = not queue and report['stage_results'][-1]['status'] == 'completed'
                report['all_steps_completed'] = done
                reason = self.gate.reason if self.gate.requested.is_set() else None
                held = report['stage_results'] and report['stage_results'][-1]['status'] == 'anomaly_hold'
                report['status'] = ('completed' if done else reason if reason in STOP_STATUSES
                                    else 'anomaly_hold' if held else 'step_failed')
        except BaseException as exc:
            report['status'] = 'error'; report['error'] = type(exc).__name__ + ': ' + str(exc)
            (self.recorder.output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        finally:
            ended = self.clock()
            self.gate.stop(report['status'])
            self.recorder.event('stop_acknowledged', reason=self.gate.reason, steps=self.steps)
            try: report['release'] = self.env.release() or 'neutral_command_sent'
            except BaseException as exc: report['release'] = 'failed: ' + repr(exc)
            self.recorder.event('input_release', controller='cleanup', outcome=report['release'])
            if self.oversight: report['oversight'] = self.oversight.summary()  # API calls happen during the run
            if self.reacquire is not None:
                # Flush the last latent chunk on every ending (time limit, STOP, error) so the record stays checkable.
                try: report['reacquire'] = self.reacquire.close()
                except BaseException as exc: report['reacquire'] = dict(error=type(exc).__name__ + ': ' + str(exc))
            report.update(steps=self.steps, final_inventory=nonzero(self.inventory()) if self.gui else None,
                          stop_reason=self.gate.reason,
                          elapsed_control_s=None if self.started is None else ended - self.started)
            # A terminal episode (death) has no neutral step to send; like runtime.run_episode it stays a valid outcome.
            released = report['release'] == 'neutral_command_sent' or (
                report['status'] == 'environment_done' and report['release'] == 'environment_done_no_neutral_step')
            report['execution_valid'] = (self.steps > 0 and released
                and report['status'] not in ('error', 'fixture_failed', 'fixture_timeout'))
        return report

    def run_step(self, step):
        result = dict(id=step.id, skill=step.skill, kind=step.kind, target=step.target, count=step.count,
                      block=step.block, controller=CONTROLLERS[step.kind], status='active', start_step=self.steps)
        start = item_count(self.inventory(), step.target); before = nonzero(self.inventory())
        fields = lambda: {('step_kind' if k == 'kind' else k): v for k, v in result.items()}
        self.recorder.event('step_started', **fields(), inventory=nonzero(self.inventory()))
        try: status, extra = getattr(self, 'run_' + step.kind)(step, start)
        except Exception as exc:
            if step.kind != 'rule': raise
            status, extra = 'rule_skill_failed', dict(reason=type(exc).__name__ + ': ' + str(exc))
        if status == 'rule_skill_failed' and self.oversight and not self.gate.requested.is_set():
            status, extra = 'anomaly_hold', dict(extra, **self.hold(self.rule_failure(step, extra['reason']), step))
        after = nonzero(self.inventory())
        self.last_kind = step.kind
        result.update(extra, status=status, steps=self.steps - result['start_step'],
                      gain=item_count(self.inventory(), step.target) - start,
                      gained={k: n - before.get(k, 0) for k, n in after.items() if n > before.get(k, 0)})
        self.recorder.event('step_finished', **fields())
        return result

    def run_policy(self, step, start):
        skill = self.skills[step.skill]
        meta = set_condition(self.policy, skill['embedding'], skill['cond_scale'])
        self.recorder.event('condition_set', id=step.id, skill=step.skill, **meta)
        until = step.params.get('until'); masked_steps = 0; gui_closed = 0; prev_y = None
        if self.last_kind == 'rule' and self.gui and abs(float(self.gui.get('pitch', 0.))) > 5.:
            # Evaluation round 2, E01: after a craft the view still pointed at the table/floor and the iron policy kept
            # digging down. Level the view first (disclosed rule action, recorded).
            pitch = float(self.gui.get('pitch', 0.)); self.recorder.event('level_view', step_id=step.id, pitch_before=pitch)
            level = self.noop(); level['camera'] = [max(-90., min(90., -pitch)), 0.]
            if not self.dispatch(level, 'rule'): return self.gate.reason, {}
        # Probe only: `until='rise'` succeeds when the server feet height gains `count` blocks (pit escape).
        start_y = float(self.gui['server_y']) if until == 'rise' else None; peak_y = start_y
        # An escape climb ends on the surface (dev check D1: a fixed +3 after reaching it built a tower above the ground).
        surface = self.surface_y if until == 'rise' and step.params.get('to_surface') else None
        if surface is not None and start_y >= surface and abs(start_y - round(start_y)) < 1e-6:
            return 'completed', dict(completion='already_at_surface', rise=0.)
        # Probe only: `until='clear_above'` counts server block breaks in the start column above head height.
        column = None; cleared = 0; seen = int(self.demo().get('demo_destroys', 0))
        if until == 'clear_above':
            column = tuple(math.floor(float(self.gui['server_' + a])) for a in 'xyz')
        if until == 'course' and self.course is None: raise ValueError("until='course' needs a course fixture")
        if until == 'place' and self.place is None: raise ValueError("until='place' needs a place fixture")
        if until == 'bridge' and self.bridge is None: raise ValueError("until='bridge' needs a bridge fixture")
        if until == 'build' and self.build is None: raise ValueError("until='build' needs a build fixture")

        oversight = self.oversight; attack_masks = 0
        if oversight:
            oversight.start_step(step); self.last_sample = sample_from(self.info, self.noop(), self.clock())
        if self.reacquire is not None: self.reacquire.activate(bool(skill.get('reacquire')))
        hook = self.reacquire if self.reacquire is not None and skill.get('reacquire') else None
        if hook is not None: hook.start_step(self.recorder.clock())

        def finish(status, **extra):
            if start_y is not None: extra['rise'] = peak_y - start_y
            if column is not None: extra['cleared'] = cleared
            if oversight: extra['attack_masks'] = attack_masks
            if gui_closed: extra['gui_closed'] = gui_closed
            if until == 'place': extra['wrong'] = self.place.wrong()
            if until == 'bridge':
                extra.update(filled=self.bridge.filled(), wrong=self.bridge.wrong(), unknown_placements=self.bridge.unknown)
            if until == 'build':
                extra.update(filled=self.build.filled(), wrong=self.build.wrong(), unknown_placements=self.build.unknown)
                if hasattr(self.build, 'repair_extra'): extra.update(self.build.repair_extra())   # round 5 (2026-11-15)
            return status, dict(extra, tool_masked=masked_steps) if self.tool_lock else extra

        while True:
            if self.stop_reason(): return finish(self.gate.reason)
            if self.tool_lock and self.gui and self.gui.get('screen', 0) != 0:
                # A GUI the locked policy cannot close: the session presses and releases the inventory key (recorded).
                press = self.noop(); press['inventory'] = 1; gui_closed += 1
                if gui_closed == 1: self.recorder.event('gui_closed', step_id=step.id, screen=self.gui.get('screen'))
                if not self.dispatch(press, 'gui_close') or not self.dispatch(self.noop(), 'gui_close'): return finish(self.gate.reason)
                continue
            if hook is not None: hook.before_act()
            action = self.policy.act(self.obs['pov'])
            masked = ()
            if self.tool_lock: action, masked = lock_tool(action, self.demo().get('demo_hit_block'), self.tool_allowed)
            if oversight:
                # Last reversible moment: the aimed block from the previous observation plus this action's attack.
                event = oversight.pre_dispatch(self.last_sample, action)
                if event and event['mask_attack']:
                    action = dict(action); action['attack'] = np.zeros_like(np.asarray(action['attack'])); attack_masks += 1
                    if attack_masks == 1: self.recorder.event('monitor_event', step_id=step.id, event=event)  # first mask per attempt
                if event and event['hold']: return finish('anomaly_hold', **self.hold(event, step))
            # Probe only: the pre-dispatch aim+attack moment is where the guard will decide.
            if until == 'mining_start' and attacking(action) and block_matches(self.demo().get('demo_hit_block', ''), step.block):
                return finish('completed', completion='aim_attack')
            if not self.dispatch(action, 'policy', masked): return finish(self.gate.reason)
            if hook is not None: hook.after_dispatch(self.steps)
            masked_steps += bool(masked)
            if oversight:
                self.last_sample = sample_from(self.info, action, self.clock())
                for event in oversight.observe(self.last_sample):
                    if event['hold']: return finish('anomaly_hold', **self.hold(event, step))
                    self.recorder.event('monitor_event', step_id=step.id, event=event)
            if not self.server_valid(): continue
            if start_y is not None:
                # Count standing heights only: two equal samples at a whole-block height. A jump apex is not a level
                # (real check P4), and the server can report the apex twice (P5: 68.24918707874468).
                y = float(self.gui['server_y']); prev, prev_y = prev_y, y
                standing = prev is not None and abs(y - prev) < 1e-3 and abs(y - round(y)) < 1e-6
                if standing:
                    peak_y = max(peak_y, y)
                    if surface is not None and y >= surface: return finish('completed', completion='server_surface')
                    if y - start_y >= step.count: return finish('completed', completion='server_rise')
            if until == 'course':
                view = self.course.server_view()       # success and falls on the server state only
                if view is not None and view.stage == 'goal':
                    return finish('completed', completion='server_course_goal', best_platform=self.course.best())
                if view is not None and view.stage == 'fallen':
                    return finish('course_fell', best_platform=self.course.best(), fell_from=view.k)
            if until == 'place' and self.place.done():       # success on the server placement record only
                return finish('completed', completion='server_place_target')
            if until == 'bridge':
                view = self.bridge.server_view()       # success and falls on the server state only
                if view is not None and view.stage == 'goal': return finish('completed', completion='server_bridge_goal')
                if view is not None and view.stage == 'fallen': return finish('bridge_fell')
            if until == 'build' and self.build.done():       # success on the server placement record only
                return finish('completed', completion='server_build_done')
            if column is not None and int(self.demo().get('demo_destroys', 0)) > seen:
                seen = int(self.demo()['demo_destroys'])
                x, y, z = (int(self.demo()['demo_last_destroy_' + a]) for a in 'xyz')
                cleared += (x, z) == (column[0], column[2]) and y > column[1] + 1
                if cleared >= step.count: return finish('completed', completion='server_clear_above')
            if item_count(self.inventory(), step.target) - start >= step.count:
                return finish('completed', completion='server_inventory')
            if until == 'mining_start' and block_matches(self.demo().get('demo_mining_block', ''), step.block):
                return finish('completed', completion='server_mining')

    def turn_away(self, step):
        """Disclosed rule action before retrying a step whose policy kept hitting a protected block (evaluation round 1,
        E01: the wood policy kept facing the stone wall): turn 180 degrees, level the view, then a few neutral steps."""
        pitch = float(self.gui.get('pitch', 0.)) if self.gui else 0.
        self.recorder.event('turn_away', step_id=step.id, pitch_before=pitch)
        action = self.noop(); action['camera'] = [max(-90., min(90., -pitch)), 180.]
        if not self.dispatch(action, 'rule'): return False
        for _ in range(3):
            if not self.dispatch(self.noop(), 'rule'): return False
        return True

    def kb_snapshot(self, reason):
        if self.oversight: self.recorder.event('kb_snapshot', reason=reason, claims=self.oversight.kb.snapshot())

    def where(self):
        """Server height and depth below the start surface (privileged observation, disclosed in the report)."""
        y = float(self.gui['server_y']) if self.gui else None
        return dict(y=y, surface_y=self.surface_y, depth=None if y is None or self.surface_y is None else self.surface_y - y)

    def rule_failure(self, step, reason):
        """A failed rule skill blocks the next transition (spec 5.3)."""
        return dict(kind='rule_skill_failed', hold=True, tick=int(self.gui.get('world_time', 0)) if self.gui else None, step=step.id,
                    evidence=dict(reason=reason))

    def hold(self, event, step):
        """Stop policy dispatch, record the anomaly with its goal context, then the slow path up to the knowledge query."""
        detected = self.clock()
        event = dict(event, evidence=dict(self.where(), **event.get('evidence', {})))  # where the player stands
        self.recorder.event('anomaly', step_id=step.id, event=event, rm=self.oversight.rm_state(step))
        self.recorder.event('hold', step_id=step.id, controller='hold', reason=event['kind'],
                            mining_ticks_at_hold=event.get('mining_ticks'), hold_latency_ms=(self.clock() - detected) * 1000)
        self.recorder.event('kb_query', step_id=step.id, subject=self.oversight.query_for(event)[0],
                            relation=self.oversight.query_for(event)[1])
        slow = self.oversight.slow_path(event, nonzero(self.inventory()), step)
        self.next_steps = slow.pop('plan_steps')  # Step objects stay out of the JSON report
        self.recorder.event('slow_path', step_id=step.id, **{k: v for k, v in slow.items() if k != 'plan'})
        self.kb_snapshot('slow_path')
        return dict(anomaly=event, slow_path=dict(slow, held_at=detected))

    def settle_inventory(self, quiet=5, limit=40):
        """Close an open GUI and wait until pickups stop changing the server inventory. Real run A4 (2026-11-05): a dirt
        drop landed one step after the workshop snapshot and the recipe commit never matched."""
        last, same = None, 0
        for _ in range(limit):
            if self.gui and self.gui.get('screen', 0) != 0:
                action = self.noop(); action['inventory'] = 1
                if not self.dispatch(action, 'rule'): return self.gate.reason
                last, same = None, 0; continue
            now = nonzero(self.inventory()); same = same + 1 if now == last else 0; last = now
            if same >= quiet: return None
            if not self.dispatch(self.noop(), 'rule'): return self.gate.reason
        return None

    def run_rule(self, step, start):
        stopped = self.settle_inventory()
        if stopped: return stopped, {}
        if step.skill == 'clear_above': return self.clear_above(step)
        goals = list(step.params['goals']); equip = step.params.get('equip'); equip_only = bool(step.params.get('equip_only'))
        # Equip-only (escape pillar stacks): move the owned stack to hotbar slot 9 and select it; no crafting, no table.
        extra_kw = dict(equip_only=True, equip_slot=9) if equip_only else {}
        controller = self.rule_controller(self.catalog, nonzero(self.inventory()), goals, self.recorder.output, equip=equip,
                                          **extra_kw)
        controller.environment = self; controller.reset()
        while True:
            if self.stop_reason(): return self.gate.reason, {}
            if not self.dispatch(next(controller.sequence), 'rule'): return self.gate.reason, {}
            if controller.stage == 'finish' and self.gui['screen'] == 0 and controller.committed():
                inventory = self.inventory()
                problems = [g for g in goals if inventory.get(g, 0) < 1]
                observed = tuple(int(self.gui['table_' + a]) for a in 'xyz')
                if not equip_only and (controller.table is None or observed != tuple(controller.table)):
                    problems.append('workstation')
                if equip and self.gui['server_selected_item'] != equip: problems.append('equip:' + equip)
                extra = dict(milestones=controller.milestones, table=controller.table)
                if problems: return 'rule_skill_failed', dict(extra, reason='final requirements: ' + ','.join(problems))
                return 'completed', dict(extra, completion='server_inventory')

    def clear_above(self, step):
        """Disclosed rule skill (STEVE-1 prompts cleared 0/9 ceilings): look straight up and break what the view up hits
        until it hits nothing within reach. Success is the server's destroy count; the climb after it is the policy's."""
        for _ in range(12):
            pitch = float(self.gui.get('pitch', 0.))
            if pitch <= -85.: break
            action = self.noop(); action['camera'] = [max(-180., -90. - pitch), 0.]
            if not self.dispatch(action, 'rule'): return self.gate.reason, {}
        else: return 'rule_skill_failed', dict(reason='could not look straight up')
        cleared, seen, limit = 0, int(self.demo().get('demo_destroys', 0)), int(step.params.get('limit_blocks', 4))
        def count():
            nonlocal cleared, seen
            destroys = int(self.demo().get('demo_destroys', 0))
            if destroys > seen: cleared += destroys - seen; seen = destroys
        for _ in range(int(step.params.get('limit_steps', 600))):
            if self.demo().get('demo_hit_block', 'air') in ('air', None):
                # The crosshair shows air before the server confirms the destroy and before the next block shows
                # (real checks C1/C2): settle, count confirmed destroys, then judge the view again.
                for _ in range(8):
                    if not self.dispatch(self.noop(), 'rule'): return self.gate.reason, dict(cleared=cleared)
                    count()
                if self.demo().get('demo_hit_block', 'air') in ('air', None):
                    return 'completed', dict(cleared=cleared, completion='view_up_open')  # an already open view counts
                continue
            if cleared >= limit: return 'rule_skill_failed', dict(reason='more than %d blocks above the head' % limit, cleared=cleared)
            action = self.noop(); action['attack'] = 1
            if not self.dispatch(action, 'rule'): return self.gate.reason, dict(cleared=cleared)
            count()
        return 'rule_skill_failed', dict(reason='clear above timed out', cleared=cleared)

    def run_scripted(self, step, start):
        """Preparation-only telemetry check: hold attack until the server records a block destroy."""
        if step.params.get('driver') == 'course_teacher': return self.run_course_teacher(step)
        if step.params.get('driver') == 'place_teacher': return self.run_place_teacher(step)
        if step.params.get('driver') == 'bridge_teacher': return self.run_bridge_teacher(step)
        if step.params.get('driver') == 'build_teacher': return self.run_build_teacher(step)
        attack = self.noop(); attack['attack'] = 1
        destroys = int(self.demo().get('demo_destroys', 0)); mining_start = None
        for _ in range(int(step.params.get('limit_steps', 600))):
            if not self.dispatch(dict(attack), 'scripted_probe'): return self.gate.reason, {}
            demo = self.demo()
            if mining_start is None and block_matches(demo.get('demo_mining_block', ''), step.block):
                mining_start = int(self.gui['world_time'])
            if int(demo.get('demo_destroys', 0)) > destroys:
                tick = int(demo['demo_last_destroy_tick'])
                destroy = dict(block=demo['demo_last_destroyed_block'], harvestable=int(demo['demo_last_destroy_harvestable']),
                               tool=demo['demo_last_destroy_tool'], tick=tick, mining_start_tick=mining_start,
                               mining_ticks=None if mining_start is None else tick - mining_start)
                status = 'completed' if block_matches(destroy['block'], step.block) else 'wrong_block_destroyed'
                return status, dict(destroy=destroy)
        return 'scripted_limit', {}

    def run_course_teacher(self, step):
        """Preparation-only check of the jump course teacher (design 2026-11-13 §4): the privileged teacher drives until
        the goal, a fall or the step limit. Recorded as scripted_probe; never a policy result."""
        if self.course is None: raise ValueError('the course teacher needs a course fixture')
        for _ in range(int(step.params.get('limit_steps', 1200))):
            server = self.course.server_view()     # success and falls on the server state only
            if server is not None and server.stage == 'goal':
                return 'completed', dict(completion='server_course_goal', best_platform=self.course.best())
            if server is not None and server.stage == 'fallen':
                return 'course_fell', dict(best_platform=self.course.best(), fell_from=server.k)
            view = self.course.decision_view()     # labels on the client pose (the player's own position)
            action = self.noop()
            label = None if view is None else self.course.label(self.course.last_row, view)
            for k, v in (label or {}).items():
                if k in action: action[k] = np.asarray(v, dtype=np.float32) if k == 'camera' else v
            if not self.dispatch(action, 'scripted_probe'): return self.gate.reason, {}
        return 'scripted_limit', dict(best_platform=self.course.best())

    def run_place_teacher(self, step):
        """Preparation-only check of the block placing teacher (design 2026-11-14 §5): the privileged teacher drives until
        a server placement at the target or the step limit. Recorded as scripted_probe; never a policy result."""
        if self.place is None: raise ValueError('the place teacher needs a place fixture')
        for _ in range(int(step.params.get('limit_steps', 900))):
            if self.place.done(): return 'completed', dict(completion='server_place_target', wrong=self.place.wrong())
            view = self.place.decision_view(); action = self.noop()
            label = None if view is None else self.place.label(self.place.last_row, view)
            for k, v in (label or {}).items():
                if k in action: action[k] = np.asarray(v, dtype=np.float32) if k == 'camera' else v
            if not self.dispatch(action, 'scripted_probe'): return self.gate.reason, dict(wrong=self.place.wrong())
        return 'scripted_limit', dict(wrong=self.place.wrong())

    def run_bridge_teacher(self, step):
        """Preparation-only check of the bridging teacher (design 2026-11-14 §5): the privileged teacher drives until the
        server goal platform, a fall or the step limit. Recorded as scripted_probe; never a policy result."""
        if self.bridge is None: raise ValueError('the bridge teacher needs a bridge fixture')

        def extra(): return dict(filled=self.bridge.filled(), wrong=self.bridge.wrong(), unknown_placements=self.bridge.unknown)
        for _ in range(int(step.params.get('limit_steps', 1200))):
            server = self.bridge.server_view()     # success and falls on the server state only
            if server is not None and server.stage == 'goal': return 'completed', dict(completion='server_bridge_goal', **extra())
            if server is not None and server.stage == 'fallen': return 'bridge_fell', extra()
            view = self.bridge.decision_view(); action = self.noop()
            label = None if view is None else self.bridge.label(self.bridge.last_row, view)
            for k, v in (label or {}).items():
                if k in action: action[k] = np.asarray(v, dtype=np.float32) if k == 'camera' else v
            if not self.dispatch(action, 'scripted_probe'): return self.gate.reason, extra()
        return 'scripted_limit', extra()

    def run_build_teacher(self, step):
        """Preparation-only check of the building teacher (design 2026-11-14 §4): the privileged teacher drives until the
        server placement record covers the blueprint or the step limit. Recorded as scripted_probe; never a policy result."""
        if self.build is None: raise ValueError('the build teacher needs a build fixture')

        def extra():
            out = dict(filled=self.build.filled(), wrong=self.build.wrong(), unknown_placements=self.build.unknown)
            if hasattr(self.build, 'repair_extra'): out.update(self.build.repair_extra())    # round 5 (2026-11-15)
            return out
        for _ in range(int(step.params.get('limit_steps', 1200))):
            if self.build.done(): return 'completed', dict(completion='server_build_done', **extra())
            view = self.build.decision_view(); action = self.noop()
            label = None if view is None else self.build.label(self.build.last_row, view)
            for k, v in (label or {}).items():
                if k in action: action[k] = np.asarray(v, dtype=np.float32) if k == 'camera' else v
            if not self.dispatch(action, 'scripted_probe'): return self.gate.reason, extra()
        return 'scripted_limit', extra()
