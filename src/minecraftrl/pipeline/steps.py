"""Plan steps shared by the planner, the session, and the replay."""
from dataclasses import dataclass, field

LOGS = ('oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log')
# 'scripted' is a preparation-only probe controller (never planned, never policy success).
KINDS = ('policy', 'rule', 'scripted')


@dataclass(frozen=True)
class Step:
    id: str
    skill: str
    kind: str
    target: str
    count: int
    block: str | None = None
    claim_ids: tuple[int, ...] = ()
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS: raise ValueError('unknown step kind: ' + str(self.kind))
        if type(self.count) is not int or self.count < 1: raise ValueError('step count must be a positive integer')
        if self.kind == 'rule' and not self.params.get('goals'): raise ValueError('rule step requires goals')


def item_count(inventory, target):
    """'log' means any of the six log kinds; other targets are exact item ids."""
    if target == 'log': return sum(int(inventory.get(item, 0)) for item in LOGS)
    return int(inventory.get(target, 0))


def block_matches(block, wanted):
    return bool(wanted) and (block.endswith('_log') if wanted == 'log' else block == wanted)
