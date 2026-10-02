"""Original batch-mean AMTPP training loss."""
import torch

def train_one_epoch(model, loader, optimizer, device: str):
    model.train()
    totals = {"loss": 0.0, "nll_tau": 0.0, "nll_o": 0.0, "nll_d": 0.0, "n": 0}
    for batch in loader:
        cond = batch["cond"].to(device)
        tau = batch["tau"].to(device)
        hour = batch["hour"].to(device)
        dow = batch["dow"].to(device)
        origin = batch["origin"].to(device)
        dest = batch["dest"].to(device)
        mask = batch["mask"].to(device)
        out = model(cond=cond, tau=tau, hour=hour, dow=dow, origin=origin, dest=dest, mask=mask)
        eos = torch.zeros_like(tau)
        losses = model.nll(out, tau, origin, dest, eos, mask)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

        totals["loss"] += float(losses["total"].item())
        totals["nll_tau"] += float(losses["nll_tau"].item())
        totals["nll_o"] += float(losses["nll_o"].item())
        totals["nll_d"] += float(losses["nll_d"].item())
        totals["n"] += 1

    for k in ["loss", "nll_tau", "nll_o", "nll_d"]:
        totals[k] /= max(totals["n"], 1)
    return totals


@torch.no_grad()
def eval_one_epoch(model, loader, device: str):
    model.eval()
    totals = {"loss": 0.0, "nll_tau": 0.0, "nll_o": 0.0, "nll_d": 0.0, "n": 0}
    for batch in loader:
        cond = batch["cond"].to(device)
        tau = batch["tau"].to(device)
        hour = batch["hour"].to(device)
        dow = batch["dow"].to(device)
        origin = batch["origin"].to(device)
        dest = batch["dest"].to(device)
        mask = batch["mask"].to(device)
        out = model(cond=cond, tau=tau, hour=hour, dow=dow, origin=origin, dest=dest, mask=mask)
        eos = torch.zeros_like(tau)
        losses = model.nll(out, tau, origin, dest, eos, mask)

        totals["loss"] += float(losses["total"].item())
        totals["nll_tau"] += float(losses["nll_tau"].item())
        totals["nll_o"] += float(losses["nll_o"].item())
        totals["nll_d"] += float(losses["nll_d"].item())
        totals["n"] += 1

    for k in ["loss", "nll_tau", "nll_o", "nll_d"]:
        totals[k] /= max(totals["n"], 1)
    return totals

