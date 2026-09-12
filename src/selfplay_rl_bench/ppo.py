from __future__ import annotations

import math

import numpy as np
import torch

from .connect4 import Batch, PolicyNet


def ppo_update(
    net: PolicyNet,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    *,
    epochs: int = 1,
    minibatch: int = 256,
    clip: float = 0.2,
    ent_coef: float = 0.002,
    vf_coef: float = 1.0,
    max_grad: float = 0.5,
    target_kl: float = 0.04,
) -> dict[str, float]:
    adv = (batch.adv - batch.adv.mean()) / (batch.adv.std() + 1e-8)

    obs = torch.from_numpy(batch.obs).float()
    masks = torch.from_numpy(batch.mask).bool()
    actions = torch.from_numpy(batch.action).long()
    old_logp = torch.from_numpy(batch.old_logp).float()
    advantages = torch.from_numpy(adv).float()
    returns = torch.from_numpy(batch.ret).float()

    indices = np.arange(len(obs))
    kls, clipfracs, ents, pgl, vfl, grads = [], [], [], [], [], []
    stopped = False

    for _ in range(epochs):
        np.random.shuffle(indices)
        epoch_kls = []
        for start in range(0, len(indices), minibatch):
            ix = torch.from_numpy(indices[start : start + minibatch])
            dist, value = net.dist_value(obs[ix], masks[ix])
            new_logp = dist.log_prob(actions[ix])
            ratio = (new_logp - old_logp[ix]).exp()
            pg1 = -advantages[ix] * ratio
            pg2 = -advantages[ix] * torch.clamp(ratio, 1 - clip, 1 + clip)
            pg = torch.maximum(pg1, pg2).mean()
            vf = ((value - returns[ix]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + vf_coef * vf - ent_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            optimizer.step()

            with torch.no_grad():
                kl = (old_logp[ix] - new_logp).mean().item()
                clipfrac = ((ratio - 1.0).abs() > clip).float().mean().item()
            kls.append(kl)
            epoch_kls.append(kl)
            clipfracs.append(clipfrac)
            ents.append(float(ent.detach()))
            pgl.append(float(pg.detach()))
            vfl.append(float(vf.detach()))
            grads.append(float(grad.detach()))

        if np.mean(epoch_kls) > 1.5 * target_kl:
            stopped = True
            break

    return {
        "kl": float(np.mean(kls)),
        "clipfrac": float(np.mean(clipfracs)),
        "entropy": float(np.mean(ents)),
        "pg": float(np.mean(pgl)),
        "vf": float(np.mean(vfl)),
        "grad": float(np.mean(grads)),
        "early_stop": float(stopped),
    }
