"""SQLite knowledge store: conditional, non-monotonic claims with evidence and status history.

Nothing is deleted. Status changes go through `claim_events`; evidence rows stay attached.
"""
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 'blocks_movement' added 2026-11-04 for the pit scenario (a solid block in the path blocks movement).
# 'escape_by' added 2026-11-05: how a player leaves a 1-wide vertical pit (subject 'vertical_pit').
RELATIONS = {'harvest_requires', 'drops', 'recipe', 'stage_duration_s', 'blocks_movement', 'escape_by'}
STATUSES = ('proposed', 'supported', 'disputed', 'retracted')
USABLE = ('proposed', 'supported')
EVIDENCE_KINDS = ('web', 'local_source', 'game_observation', 'llm_assertion')
CONDITION_KEYS = {'game_version': True, 'tool': True, 'enchantment': True, 'difficulty': True}  # value: observable

SCHEMA = """
create table if not exists claims(id integer primary key, subject text not null, relation text not null,
  object_json text not null, conditions_json text not null, status text not null, revision integer not null,
  created_run text, unique(subject, relation, object_json, conditions_json));
create table if not exists evidence(id integer primary key, claim_id integer not null references claims(id), kind text not null,
  stance text not null, ref text not null, excerpt text, retrieved_at text, context_json text, run_id text, t real);
create table if not exists claim_events(id integer primary key, claim_id integer not null, from_status text, to_status text not null,
  reason text, run_id text, t real);
create table if not exists plans(plan_id text primary key, steps_json text not null, t real);
create table if not exists plan_claims(plan_id text, step_id text, claim_id integer);
"""


@dataclass(frozen=True)
class Evidence:
    kind: str
    stance: str  # 'support' | 'refute'
    ref: str
    excerpt: str = ''
    retrieved_at: str | None = None


@dataclass(frozen=True)
class Claim:
    id: int
    subject: str
    relation: str
    object: Any
    conditions: dict
    status: str
    revision: int
    support: int
    refute: int


def dumps(value): return json.dumps(value, sort_keys=True, separators=(',', ':'))


def version(text): return tuple(int(part) for part in str(text).split('.'))


def condition_matches(conditions, context):
    """Every condition key must be present in the context; strings match exactly, {'gte'|'lte': v} compare versions."""
    for key, want in conditions.items():
        if key not in context: return False
        have = context[key]
        if isinstance(want, dict):
            if 'gte' in want and version(have) < version(want['gte']): return False
            if 'lte' in want and version(have) > version(want['lte']): return False
        elif have != want: return False
    return True


class KnowledgeBase:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path)
        if readonly:
            self.db = sqlite3.connect('file:%s?mode=ro' % self.path.as_posix(), uri=True)
        else:
            self.db = sqlite3.connect(self.path)
            self.db.execute('pragma journal_mode=wal'); self.db.executescript(SCHEMA); self.db.commit()

    def __enter__(self): return self

    def __exit__(self, *exc): self.close()

    def close(self): self.db.close()

    def _event(self, claim_id, before, after, reason, run_id):
        self.db.execute('insert into claim_events(claim_id, from_status, to_status, reason, run_id, t) values(?,?,?,?,?,?)',
                        (claim_id, before, after, reason, run_id, time.time()))

    def _status(self, claim_id): return self.db.execute('select status from claims where id=?', (claim_id,)).fetchone()[0]

    def _counts(self, claim_id):
        rows = self.db.execute('select kind, stance from evidence where claim_id=?', (claim_id,)).fetchall()
        return (sum(s == 'support' for _, s in rows), sum(s == 'refute' for _, s in rows),
                sum(k == 'game_observation' and s == 'support' for k, s in rows))

    def _store(self, claim_id, ev, run_id, context=None):
        if ev.kind not in EVIDENCE_KINDS or ev.stance not in ('support', 'refute'): raise ValueError('invalid evidence')
        self.db.execute('insert into evidence(claim_id, kind, stance, ref, excerpt, retrieved_at, context_json, run_id, t) '
                        'values(?,?,?,?,?,?,?,?,?)', (claim_id, ev.kind, ev.stance, ev.ref, ev.excerpt, ev.retrieved_at,
                                                    dumps(context) if context is not None else None, run_id, time.time()))

    def _promote(self, claim_id, run_id):
        support, refute, observed = self._counts(claim_id)
        before = self._status(claim_id)
        if before == 'proposed' and observed >= 1 and refute == 0:
            self.db.execute('update claims set status=? where id=?', ('supported', claim_id))
            self._event(claim_id, before, 'supported', 'game_observation', run_id)
        return self._status(claim_id)

    def add_claim(self, subject, relation, obj, conditions, evidence, *, run_id):
        """Same subject/relation/object/conditions only adds evidence to the existing claim."""
        if relation not in RELATIONS: raise ValueError('unknown relation: ' + relation)
        key = (subject, relation, dumps(obj), dumps(conditions))
        row = self.db.execute('select id from claims where subject=? and relation=? and object_json=? and conditions_json=?', key).fetchone()
        if row is None:
            claim_id = self.db.execute('insert into claims(subject, relation, object_json, conditions_json, status, revision, created_run) '
                                       'values(?,?,?,?,?,?,?)', key + ('proposed', 1, run_id)).lastrowid
            self._event(claim_id, None, 'proposed', 'created', run_id)
        else: claim_id = row[0]
        for ev in evidence: self._store(claim_id, ev, run_id)
        self._promote(claim_id, run_id); self.db.commit()
        return claim_id

    def add_evidence(self, claim_id, ev, *, run_id):
        self._store(claim_id, ev, run_id); status = self._promote(claim_id, run_id); self.db.commit()
        return status

    def _claim(self, row):
        claim_id, subject, relation, obj, conditions, status, revision = row
        support, refute, _ = self._counts(claim_id)
        return Claim(claim_id, subject, relation, json.loads(obj), json.loads(conditions), status, revision, support, refute)

    def query(self, subject, relation, *, context, usable_only=True):
        rows = self.db.execute('select id, subject, relation, object_json, conditions_json, status, revision from claims '
                               'where subject=? and relation=? order by id', (subject, relation)).fetchall()
        claims = [self._claim(r) for r in rows]
        return [c for c in claims if condition_matches(c.conditions, context) and (not usable_only or c.status in USABLE)]

    def set_status(self, claim_id, status, reason, *, run_id):
        if status not in STATUSES: raise ValueError('unknown status')
        before = self._status(claim_id)
        self.db.execute('update claims set status=?, revision=revision+1 where id=?', (status, claim_id))
        self._event(claim_id, before, status, reason, run_id); self.db.commit()

    def split_conditional(self, claim_id, condition, *, run_id):
        """Narrow a claim to `condition`: a new conditioned claim keeps its support; the broad one is retracted."""
        subject, relation, obj, conditions = self.db.execute(
            'select subject, relation, object_json, conditions_json from claims where id=?', (claim_id,)).fetchone()
        narrowed = dict(json.loads(conditions), **condition)
        support = [Evidence(k, s, r, e or '', ra) for k, s, r, e, ra in self.db.execute(
            'select kind, stance, ref, excerpt, retrieved_at from evidence where claim_id=? and stance=?', (claim_id, 'support'))]
        new_id = self.add_claim(subject, relation, json.loads(obj), narrowed, support, run_id=run_id)
        self.set_status(claim_id, 'retracted', 'conditionalized as claim %d' % new_id, run_id=run_id)
        return new_id

    def resolve_conflict(self, claim_id, observation, context, *, candidate_condition, observation_valid, run_id):
        """User-confirmed order: observation error -> observable condition split -> repeated refutation -> disputed."""
        if not observation_valid:
            self._event(claim_id, self._status(claim_id), self._status(claim_id), 'observation_error', run_id); self.db.commit()
            return 'observation_error'
        self._store(claim_id, observation, run_id, context)
        if (candidate_condition and all(CONDITION_KEYS.get(k) for k in candidate_condition)
                and not condition_matches(candidate_condition, context)):
            self.split_conditional(claim_id, candidate_condition, run_id=run_id)
            return 'conditionalized'
        same = self.db.execute('select count(*) from evidence where claim_id=? and stance=? and context_json=?',
                               (claim_id, 'refute', dumps(context))).fetchone()[0]
        outcome = 'retracted' if same >= 2 else 'disputed'
        self.set_status(claim_id, outcome, 'game observation refutes in context ' + dumps(context), run_id=run_id)
        return outcome

    def record_plan(self, plan_id, steps):
        self.db.execute('insert or replace into plans(plan_id, steps_json, t) values(?,?,?)', (plan_id, dumps(steps), time.time()))
        for step in steps:
            for claim_id in step.get('claim_ids', ()):
                self.db.execute('insert into plan_claims(plan_id, step_id, claim_id) values(?,?,?)', (plan_id, step['id'], claim_id))
        self.db.commit()

    def dependents(self, claim_id):
        return [tuple(r) for r in self.db.execute('select plan_id, step_id from plan_claims where claim_id=? order by rowid', (claim_id,))]

    def snapshot(self):
        """Every claim with its web sources and game-observation counts (the replay's knowledge table)."""
        rows = self.db.execute('select id, subject, relation, object_json, conditions_json, status, revision from claims order by id')
        out = []
        for row in rows.fetchall():
            claim = dict(self._claim(row).__dict__)
            ev = self.db.execute('select kind, stance, ref from evidence where claim_id=? order by id', (claim['id'],)).fetchall()
            claim['sources'] = list(dict.fromkeys(ref for kind, _, ref in ev if kind == 'web'))
            claim['observed'] = {stance: sum(k == 'game_observation' and s == stance for k, s, _ in ev) for stance in ('support', 'refute')}
            out.append(claim)
        return out
