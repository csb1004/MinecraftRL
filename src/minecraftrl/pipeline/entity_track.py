"""Entity tracks from entity-ray sightings (sensor-tier design 2026-11-19 §2.6-3 and §3.1, built 2026-11-23).

A sensor frame names, per view ray, the first entity it met (kind and hit point; no game id). Sightings of one kind
within MERGE of each other in one frame are one observation (several rays on one entity). An observation joins the
track of its kind whose last point is within MATCH, else it starts a new track; the ids are this tracker's own counter,
like a real tracker's. Nothing is known about an entity no ray met.
"""
import math

MERGE = .5      # blocks: rays of one frame this close on one kind meet the same entity
MATCH = 1.      # blocks: an observation this close to a track's last point is the same entity


class EntityTracks:
    def __init__(self):
        self.tracks = []; self.next_id = 1

    def update(self, frame, sightings):
        """Fold one frame's sightings [(kind, point, distance)] into the tracks."""
        groups = []
        for kind, point, _ in sightings:
            group = next((g for g in groups if g['kind'] == kind and math.dist(g['points'][0], point) <= MERGE), None)
            if group is None: groups.append(dict(kind=kind, points=[tuple(point)]))
            else: group['points'].append(tuple(point))
        for g in groups:
            point = tuple(sum(p[a] for p in g['points']) / len(g['points']) for a in range(3))
            near = [t for t in self.tracks if t['kind'] == g['kind'] and t['frame'] < frame
                    and math.dist(t['point'], point) <= MATCH]
            if near:
                track = min(near, key=lambda t: math.dist(t['point'], point))
                track.update(point=point, frame=frame, seen=track['seen'] + 1)
            else:
                self.tracks.append(dict(id=self.next_id, kind=g['kind'], point=point, frame=frame, first=frame, seen=1))
                self.next_id += 1

    def latest(self, kind, since=None):
        """The track of `kind` seen most recently (at or after frame `since`), or None."""
        found = [t for t in self.tracks if t['kind'] == kind and (since is None or t['frame'] >= since)]
        return max(found, key=lambda t: (t['frame'], t['id'])) if found else None
