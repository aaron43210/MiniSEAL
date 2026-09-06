"""
Loss functions for SFT and DPO.

SFT:
    Standard causal LM loss on response tokens.

DPO:
    Direct Preference Optimization:
        -log sigmoid(beta * [(log pi chosen - log ref chosen)
                            - (log pi rejected - log ref rejected)])
"""

import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculates causal LM loss.

    Input:
        logits: [batch, seq, vocab]
        labels: [batch, seq]
            Labels use -100 for ignored positions.

    Output:
        scalar loss.

    Important:
        logits[:, t] predicts labels[:, t+1]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    return loss


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Computes sequence log probability of only supervised answer tokens.

    Input:
        model: policy or reference model.
        input_ids: [batch, seq]
        labels: [batch, seq], -100 on ignored tokens.

    Output:
        log probability per sequence: [batch]
    """
    logits, _ = model(input_ids, targets=None)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)

    mask = shift_labels != -100

    safe_labels = shift_labels.masked_fill(~mask, 0)

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs * mask

    return token_log_probs.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """
    Direct Preference Optimization loss.

    Input:
        policy_chosen_logps: log pi_theta(chosen)
        policy_rejected_logps: log pi_theta(rejected)
        ref_chosen_logps: log pi_ref(chosen)
        ref_rejected_logps: log pi_ref(rejected)
        beta: KL trade-off parameter.

    Output:
        loss
        metrics dictionary
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (policy_logratios - ref_logratios)

    losses = -F.logsigmoid(logits)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

    metrics = {
        "reward_accuracy": reward_accuracy.detach().item(),
        "chosen_reward": chosen_rewards.mean().detach().item(),
        "rejected_reward": rejected_rewards.mean().detach().item(),
        "margin": (chosen_rewards - rejected_rewards).mean().detach().item(),
    }

    return losses.mean(), metrics
