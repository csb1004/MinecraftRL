def aggregate(reports, *, planned, has_target):
    valid = [x for x in reports if x.get('execution_valid')]
    # Early planner pauses and cancellation are separate from complete trials.
    technical = [x for x in reports if x.get('status') in ('blocked', 'error', 'hard_timeout', 'worker_failed')
                 or (x.get('status') not in ('cancelled','planning_paused') and not x.get('execution_valid'))]
    return {'planned': planned, 'attempted': len(reports), 'unrun': max(0, planned - len(reports)),
            'valid_gameplay_episodes': len(valid), 'technical_failures': len(technical),
            'cancelled': sum(x.get('status') == 'cancelled' for x in reports),
            'planning_paused': sum(x.get('status') == 'planning_paused' for x in reports),
            'technical_failure_rate': len(technical) / len(reports) if reports else None,
            'task_failure_rate': sum(not x.get('target_observed') for x in valid) / len(valid) if has_target and valid else None}
