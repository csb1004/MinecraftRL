"""Training-only supervision of physical movement bits in the joint action head."""
import torch


MOVEMENT_BUTTONS = ('forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint')


class HumanMovementMarginalBC(torch.nn.Module):
    def __init__(self, physical_buttons, training_labels):
        super().__init__()
        bits = torch.as_tensor(physical_buttons)
        labels = torch.as_tensor(training_labels)
        if (bits.ndim != 2 or bits.shape[1] != len(MOVEMENT_BUTTONS) or
                bits.shape[0] < 2 or not ((bits == 0) | (bits == 1)).all()):
            raise ValueError('binary joint-action movement mapping required')
        if (labels.ndim != 1 or not labels.numel() or labels.dtype != torch.long or
                labels.min() < 0 or labels.max() >= bits.shape[0]):
            raise ValueError('nonempty valid human training action indices required')
        bits = bits.bool()
        positives = bits[labels].sum(0)
        active = (positives > 0) & (positives < labels.numel())
        if not active.any():
            raise ValueError('human training labels have no varying movement buttons')
        selected = positives[active].float()
        self.register_buffer('bits', bits[:, active])
        self.register_buffer('positive_weight', (labels.numel() / (2 * selected)).clamp(max=10))
        self.register_buffer('negative_weight', (labels.numel() / (2 * (labels.numel() - selected))).clamp(max=10))
        self.receipt = dict(training_labels=labels.numel(), class_weight_cap=10.,
            counts={name:dict(positive=int(pos), negative=labels.numel()-int(pos), active=bool(use))
                    for name,pos,use in zip(MOVEMENT_BUTTONS, positives, active)},
            active_buttons=[name for name,use in zip(MOVEMENT_BUTTONS,active) if use],
            positive_weights=self.positive_weight.tolist(), negative_weights=self.negative_weight.tolist(),
            scope='Human train labels only; balanced physical-button marginals; constant buttons excluded; whole-batch mean')

    @classmethod
    def from_mapper(cls, mapper, button_names, training_labels):
        mapping = torch.as_tensor(mapper.BUTTON_IDX_TO_FACTORED)
        return cls(mapping[:, [button_names.index(name) for name in MOVEMENT_BUTTONS]], training_labels)

    def forward(self, button_logits, action_indices, is_human, effective_weights):
        if (button_logits.ndim < 2 or button_logits.shape[-1] != self.bits.shape[0] or
                action_indices.shape != button_logits.shape[:-1] or action_indices.dtype != torch.long):
            raise ValueError('joint logits and action index shapes must match')
        logits = button_logits.reshape(-1, self.bits.shape[0])
        actions = action_indices.flatten()
        flags = torch.as_tensor(is_human, device=logits.device)
        if (not actions.numel() or flags.dtype != torch.bool or flags.shape != actions.shape or
                effective_weights.shape != actions.shape or
                not torch.isfinite(effective_weights).all() or not (effective_weights > 0).all() or
                actions.min() < 0 or actions.max() >= self.bits.shape[0]):
            raise ValueError('valid per-row human flags, weights and actions required')
        # Indexing first makes local rows contribute exactly zero auxiliary gradient.
        selected = logits[flags]
        if not selected.numel():
            return selected.sum()
        if not torch.isfinite(selected).all():
            raise ValueError('finite human action logits required')
        logp = selected - torch.logsumexp(selected, -1, keepdim=True)
        targets = self.bits[actions[flags]]
        terms = []
        for index in range(self.bits.shape[1]):
            mask = self.bits[:, index]
            positive = torch.logsumexp(logp[:, mask], -1)
            negative = torch.logsumexp(logp[:, ~mask], -1)
            terms.append(-torch.where(targets[:, index],
                self.positive_weight[index] * positive, self.negative_weight[index] * negative))
        per_row = torch.stack(terms, -1).mean(-1)
        return (per_row * effective_weights[flags]).sum() / actions.numel()
