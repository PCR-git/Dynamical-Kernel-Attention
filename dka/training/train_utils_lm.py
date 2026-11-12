import math

import torch
import torch.nn as nn
from tqdm import tqdm

##########################################################################################
##########################################################################################

# Per-head RFA/DKA parameters logged to `history` (output-dict key -> history key); keys a layer does not
# return are skipped (tau is RFA only; prior_weight c_h, zeta and the mean step are DKA only)
_RFA_LOG_KEYS = {"mu_v": "mu", "sigma_sq_v": "sigma", "sigma_tilde_sq_v": "sigma_tilde",
                 "eta_sq_v": "eta", "gamma_sq_v": "gamma", "tau": "tau", "nu_over_d": "nu_over_d",
                 "prior_weight": "prior_weight", "zeta": "zeta", "saturation": "saturation",
                 "step_mean": "step", "v_inf_all": "v_inf_all", "v_inf_test_all": "v_inf_test_all", "b_all": "b_all"}
_PER_BLOCK_KEYS = ("prior_weight", "zeta", "saturation", "step_mean", "v_inf_all", "v_inf_test_all", "b_all")   # logged for every block, [n_blocks, H]; the rest last block, [H]


def single_epoch_lm(model, train_loader, history, optimizer, args, scheduler=None, log_every=100):
    """
    One epoch of next-token language-model training (RFA or baseline backbone).
    Appends the loss of every step to history['loss'], and for RFA/DKA models, every `log_every`
    steps, the per-head parameters of the last block (history['mu'], ['sigma'], ..., [H] arrays as the
    plotting code expects); for DKA also c_h, zeta and the mean step of every block (history['prior_weight'],
    ['zeta'], ['step'], [n_blocks, H] arrays).
    """
    model.train()
    output_dict = None

    pbar = tqdm(train_loader, desc="Training", leave=False, mininterval=5.0)
    for it, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)

        optimizer.zero_grad()
        loss, output_dict = model.loss(input_ids, labels, causal=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        history['loss'].append(loss.item())
        if it % 100 == 0:
            pbar.set_postfix({"loss": f"{history['loss'][-1]:.4f}"})

        # Backbone returns one info dict per block; the notebooks' RFATransformerNetwork only the last one
        infos = output_dict if isinstance(output_dict, list) else [output_dict]
        if it % log_every == 0 and isinstance(infos[-1], dict) and "mu_v" in infos[-1]:
            for key, hist_key in _RFA_LOG_KEYS.items():
                if key not in infos[-1]:
                    continue
                if key in _PER_BLOCK_KEYS:
                    value = torch.stack([info[key].detach() for info in infos])
                else:
                    value = infos[-1][key].detach()
                history.setdefault(hist_key, []).append(value.cpu().numpy())

    return output_dict, history

##########################################################################################
##########################################################################################

@torch.no_grad()
def evaluate_lm(model, loader, args):
    """Token-weighted mean next-token loss and perplexity over `loader`: (avg_nll, ppl)."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        loss, _ = model.loss(input_ids, labels, causal=True)
        n = (labels[:, 1:] != -100).sum().item()
        total_nll += loss.item() * n
        total_tokens += n
    avg_nll = total_nll / total_tokens
    return avg_nll, math.exp(avg_nll)
