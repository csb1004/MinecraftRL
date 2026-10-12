"""Read-only labels and executable-command probabilities for correction analysis."""
import math


LEGS = ('reposition', 'gap_approach', 'gap_crossing', 'reapproach')


def annotate_correction(rows, world, waypoints):
    """Recover the teacher's waypoint contract; never assign student action targets."""
    ox, oz = (v * 16 for v in world['chunk'])
    mirror = world['curriculum']['mirror_z']
    gap = world['curriculum']['wall_end'] + 1.5
    first = next(r for r in rows if r['owner'] == 'teacher')['server_before']
    handoff_z = first['server_z'] - oz
    if mirror:
        handoff_z = 16 - handoff_z
    points = [tuple(p) for p in waypoints]
    expected = [(5.5, handoff_z), (5.5, gap), (9.5, gap), (9.5, 3.5)]
    if points != expected or first['server_x'] - ox > 6.71:
        raise ValueError('handoff route receipt does not match supported student pose')
    cursor = teacher_index = 0
    student_count = sum(r['owner'] == 'student' for r in rows)
    labels = []
    for i, row in enumerate(rows):
        if row['frame'] != f'frames/{i:06d}.png':
            raise ValueError('non-contiguous source frame')
        owner = row['owner']
        if row['loss_eligible'] != (owner == 'teacher'):
            raise ValueError('owner/target mismatch')
        label = dict(frame=i, owner=owner, teacher_index=None, leg=owner, motion='context',
                     pre_handoff_last32=owner == 'student' and i >= student_count - 32)
        if owner == 'teacher':
            label['teacher_index'] = teacher_index
            teacher_index += 1
            if cursor < 4:
                if row['phase'] != 'move':
                    raise ValueError('teacher phase changed before route completion')
                server = row['server_before']
                x, z = server['server_x'] - ox, server['server_z'] - oz
                yaw = server['yaw']
                if mirror:
                    z, yaw = 16 - z, 180 - yaw
                tx, tz = points[cursor]
                arrived = math.hypot(tx - x, tz - z) < .22
                turn = (math.degrees(math.atan2(-(tx - x), tz - z)) - yaw + 180) % 360 - 180
                expected = [0., 0.] if arrived else [max(-10, min(10, -server['pitch'])), max(-10, min(10, turn))]
                if mirror:
                    expected[1] *= -1
                forward = int(not arrived and abs(turn) < 12)
                actual = row['action']
                if actual['forward'] != forward or any(abs(a - b) > 1e-5 for a, b in zip(actual['camera'], expected)):
                    raise ValueError(f'teacher navigation contract mismatch at {i}')
                label.update(leg=LEGS[cursor], motion='arrival' if arrived else 'advance' if forward else 'turn',
                             waypoint_index=cursor, heading_error_degrees=turn)
                cursor += int(arrived)
            else:
                if row['phase'] not in ('harvest', 'clear_stump', 'collect'):
                    raise ValueError('unknown post-route phase')
                label.update(leg=row['phase'], motion='harvest_or_collect')
        labels.append(label)
    if cursor != 4 or not teacher_index:
        raise ValueError('incomplete teacher route')
    return labels


class CommandMetrics:
    """Exact marginals of the official factored decoder, with camera gating."""
    def __init__(self, button_table, camera_off, camera_table, names, camera_null_bin, device):
        import torch
        self.torch = torch
        self.buttons = torch.as_tensor(button_table, device=device).bool()
        self.camera_on = ~torch.as_tensor(camera_off, device=device).bool()
        self.yaw_sign = torch.sign(torch.as_tensor(camera_table, device=device)[:, 1] - camera_null_bin)
        self.names = names
        self.masks = {key: self.buttons[:, names.index(key)] for key in ('forward', 'attack')}
        self.masks['move'] = self.buttons[:, [names.index(k) for k in ('forward', 'back', 'left', 'right')]].any(-1)

    def measure(self, pd, target=None):
        torch = self.torch
        logs = {k: v.flatten() for k, v in pd.items()}
        p, camera = logs['buttons'].exp(), logs['camera'].exp()
        if not torch.isfinite(p).all() or not torch.isfinite(camera).all():
            raise ValueError('nonfinite probabilities')
        if abs(float(p.sum()) - 1) > 1e-5 or abs(float(camera.sum()) - 1) > 1e-5:
            raise ValueError('unnormalized probabilities')
        button_mode, camera_mode = int(p.argmax()), int(camera.argmax())
        result = {f'p_{key}': float(p[mask].sum()) for key, mask in self.masks.items()}
        result['p_camera_on'] = float(p[self.camera_on].sum())
        result['top1_move'] = int(self.masks['move'][button_mode])
        result['top1_attack'] = int(self.masks['attack'][button_mode])
        if target is not None:
            button = int(target['buttons'].item())
            cam = int(target['camera'].item())
            b, c = float(-logs['buttons'][button]), float(-logs['camera'][cam])
            result.update(nll=b+c, button_nll=b, camera_nll=c, exact_label_probability=math.exp(-b-c),
                          button_top1_match=int(button_mode == button), camera_top1_match=int(camera_mode == cam),
                          target_forward=int(self.masks['forward'][button]), target_move=int(self.masks['move'][button]),
                          target_attack=int(self.masks['attack'][button]))
            sign = int(self.yaw_sign[cam]) if self.camera_on[button] else 0
            if sign:
                result['p_target_yaw_sign'] = result['p_camera_on'] * float(camera[self.yaw_sign == sign].sum())
                mode_sign = int(self.yaw_sign[camera_mode]) if self.camera_on[button_mode] else 0
                result['yaw_sign_top1_match'] = int(mode_sign == sign)
        return result


def aggregate(records):
    """A missing metric has no denominator; zero remains an observed value."""
    sums, counts = {}, {}
    for record in records:
        for key, value in record.items():
            if not math.isfinite(value):
                raise ValueError('nonfinite diagnostic value')
            sums[key] = sums.get(key, 0.) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: dict(mean=value / counts[key], count=counts[key]) for key, value in sums.items()}
