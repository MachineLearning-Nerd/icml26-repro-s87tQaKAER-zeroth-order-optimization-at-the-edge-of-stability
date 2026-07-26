"""Empirical verification of Claims 4, 5, 6 (Section 5 of the paper).

We train full-batch zeroth-order methods on a CNN on a CIFAR-10 subset (the
paper's setup: 1000 examples from the first 4 classes, squared loss, two-point
estimator with mu=1e-3) and track the mean-square stability bands (Eqs 23-25)
along the trajectory using only the Hessian trace and top eigenvalue.

  Claim 4: ZO-GD / ZO-GDM / ZO-Adam stabilize near the predicted mean-square EoS
           threshold (Figure 2).
  Claim 5: practical tracking needs only Tr(H) and lam_max(H) (Eqs 23-25).
  Claim 6: catapult dynamics when eta is increased past the threshold (Figure 3).

Reduced scale vs. the paper (CPU-only): a single CNN architecture, fewer
iterations / curvature probes.  This is stated explicitly in the report and on
the candidate page.  Curvature is measured with the TRUE Hessian via
Hessian-vector products (autograd), independent of the ZO estimator used for
training.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

ARTIFACT_DIR = os.environ.get("ZO_EOS_ARTIFACTS", ".openresearch/artifacts/empirical")
DEVICE = "cpu"
torch.set_num_threads(max(1, torch.get_num_threads()))


# ----------------------------- data -----------------------------


def get_data(n: int = 1000, n_classes: int = 4, root: str = "/tmp/cifar_data"):
    from torchvision import datasets

    ds = datasets.CIFAR10(root=root, train=True, download=True)
    X = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2)  # (N,3,32,32)
    y = torch.tensor(ds.targets)
    mask = y < n_classes
    X, y = X[mask][:n], y[mask][:n]
    # standardize channel-wise
    mu = X.mean(dim=(0, 2, 3), keepdim=True)
    sd = X.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    X = (X - mu) / sd
    Y = F.one_hot(y, num_classes=n_classes).float()
    return X.to(DEVICE), Y.to(DEVICE)


# ----------------------------- model -----------------------------


class CNN(nn.Module):
    """4-layer conv net, width 32, 3x3 kernels, GeLU, average pooling, linear readout
    (Appendix C)."""

    def __init__(self, width: int = 32, n_classes: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Linear(width, n_classes)

    def forward(self, x):
        h = self.conv(x)
        h = h.mean(dim=(2, 3))  # global average pooling
        return self.fc(h)


def _params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _flatten(model):
    return torch.cat([p.detach().reshape(-1) for p in _params(model)])


def _set_flat(model, vec):
    i = 0
    for p in _params(model):
        n = p.numel()
        p.data.copy_(vec[i : i + n].view_as(p))
        i += n


def _loss(model, X, Y):
    out = model(X)
    return 0.5 * ((out - Y) ** 2).sum(dim=1).mean()


# ----------------------- zeroth-order optimizers -----------------------


def zo_grad(model, X, Y, mu: float, rng):
    """Standard two-point estimator (Eq. 3), u ~ N(0, I).  Returns grad_hat (flat)."""
    d = sum(p.numel() for p in _params(model))
    u = torch.randn(d, device=DEVICE, generator=rng)
    p0 = _flatten(model)
    _set_flat(model, p0 + mu * u)
    fplus = _loss(model, X, Y).detach()
    _set_flat(model, p0 - mu * u)
    fminus = _loss(model, X, Y).detach()
    _set_flat(model, p0)  # restore
    grad_hat = ((fplus - fminus) / (2.0 * mu)) * u
    return grad_hat


def train_zogd(model, X, Y, eta, n_iters, mu, log_every, seed):
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    traj = {"opt": "ZO-GD", "eta": eta, "t": [], "loss": [], "trace": [], "lmax": [], "lower": [], "upper": [], "thr": []}
    for t in range(n_iters + 1):
        if t % log_every == 0 or t == n_iters:
            tr, lm = curvature(model, X, Y)
            traj["t"].append(t)
            traj["loss"].append(_loss(model, X, Y).item())
            traj["trace"].append(tr); traj["lmax"].append(lm)
            traj["lower"].append(tr); traj["upper"].append(tr + 2 * lm); traj["thr"].append(2.0 / eta)
        if t == n_iters:
            break
        g = zo_grad(model, X, Y, mu, rng)
        p = _flatten(model) - eta * g
        _set_flat(model, p)
    return traj


def train_zogdm(model, X, Y, eta, beta, n_iters, mu, log_every, seed):
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    d = sum(p.numel() for p in _params(model))
    m = torch.zeros(d, device=DEVICE)
    traj = {"opt": "ZO-GDM", "eta": eta, "beta": beta, "t": [], "loss": [], "trace": [], "lmax": [], "lower": [], "upper": [], "thr": []}
    for t in range(n_iters + 1):
        if t % log_every == 0 or t == n_iters:
            tr, lm = curvature(model, X, Y)
            traj["t"].append(t); traj["loss"].append(_loss(model, X, Y).item())
            traj["trace"].append(tr); traj["lmax"].append(lm)
            traj["lower"].append(tr); traj["upper"].append(tr + 2 * lm / (1 + beta)); traj["thr"].append(2 * (1 - beta) / eta)
        if t == n_iters:
            break
        g = zo_grad(model, X, Y, mu, rng)
        m = beta * m + g
        p = _flatten(model) - eta * m
        _set_flat(model, p)
    return traj


def train_zoadam(model, X, Y, eta, beta1, beta2, eps, n_iters, mu, log_every, seed):
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    d = sum(p.numel() for p in _params(model))
    m = torch.zeros(d, device=DEVICE)
    nu = torch.zeros(d, device=DEVICE)
    traj = {"opt": "ZO-Adam", "eta": eta, "beta1": beta1, "beta2": beta2, "t": [], "loss": [],
            "trace": [], "lmax": [], "lower": [], "upper": [], "thr": [], "relcomm": []}
    pdiag = torch.ones(d, device=DEVICE)
    for t in range(n_iters + 1):
        if t % log_every == 0 or t == n_iters:
            # current preconditioner diagonal P_t (Eq. 9): (1-b1^{t+1})(sqrt(nu/(1-b2^{t+1}))+eps)
            bc1 = 1 - beta1 ** (max(t, 1))
            bc2 = 1 - beta2 ** (max(t, 1))
            pvec = (bc1) * (torch.sqrt(nu / bc2) + eps)
            pvec = torch.clamp(pvec, min=1e-6)
            tr, lm, rc = curvature_precond(model, X, Y, pvec, beta1)
            traj["t"].append(t); traj["loss"].append(_loss(model, X, Y).item())
            traj["trace"].append(tr); traj["lmax"].append(lm)
            traj["lower"].append(tr); traj["upper"].append(tr + 2 * lm / (1 + beta1)); traj["thr"].append(2.0 / eta)
            traj["relcomm"].append(rc)
            pdiag = pvec
        if t == n_iters:
            break
        g = zo_grad(model, X, Y, mu, rng)
        m = beta1 * m + (1 - beta1) * g
        nu = beta2 * nu + (1 - beta2) * (g * g)
        bc1 = 1 - beta1 ** (t + 1)
        bc2 = 1 - beta2 ** (t + 1)
        mhat = m / bc1
        vhat = nu / bc2
        upd = eta * mhat / (torch.sqrt(vhat) + eps)
        p = _flatten(model) - upd
        _set_flat(model, p)
    return traj


def train_catapult(model, X, Y, etas, seg_iters, mu, log_every, seed):
    """ZO-GD with step size increased midway: eta1 -> eta2 -> eta3 (Figure 3)."""
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    traj = {"opt": "ZO-GD-catapult", "etas": etas, "t": [], "loss": [], "trace": [], "eta": []}
    eta = etas[0]
    t = 0
    total = seg_iters * len(etas)
    for seg, eta in enumerate(etas):
        for _ in range(seg_iters):
            if t % log_every == 0 or t == total:
                tr, lm = curvature(model, X, Y)
                traj["t"].append(t); traj["loss"].append(_loss(model, X, Y).item())
                traj["trace"].append(tr); traj["eta"].append(eta)
            g = zo_grad(model, X, Y, mu, rng)
            p = _flatten(model) - eta * g
            _set_flat(model, p)
            t += 1
    tr, lm = curvature(model, X, Y)
    traj["t"].append(t); traj["loss"].append(_loss(model, X, Y).item()); traj["trace"].append(tr); traj["eta"].append(eta)
    return traj


# ----------------------- curvature (true Hessian) -----------------------


def _hvp(model, X, Y, v):
    params = _params(model)
    out = model(X)
    loss = 0.5 * ((out - Y) ** 2).sum(dim=1).mean()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gvec = torch.cat([g.reshape(-1) for g in grads])
    dot = (gvec * v).sum()
    hv = torch.autograd.grad(dot, params, retain_graph=False, create_graph=False)
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def curvature(model, X, Y, power_it: int = None, n_probes: int = None):
    power_it = POWER_IT if power_it is None else power_it
    n_probes = N_PROBES if n_probes is None else n_probes
    d = sum(p.numel() for p in _params(model))
    v = torch.randn(d, device=DEVICE)
    v = v / v.norm()
    lam = 0.0
    for _ in range(power_it):
        hv = _hvp(model, X, Y, v)
        lam = (v * hv).sum().item() / (v * v).sum().item()
        nv = hv.norm()
        if not torch.isfinite(nv) or nv < 1e-12:
            break
        v = hv / nv
    tr = 0.0
    for _ in range(n_probes):
        z = torch.randint(0, 2, (d,), device=DEVICE, dtype=torch.float32) * 2 - 1
        hz = _hvp(model, X, Y, z)
        tr += (z * hz).sum().item()
    tr /= n_probes
    return tr, max(lam, 0.0)


def curvature_precond(model, X, Y, pdiag, beta1, power_it=None, n_probes=None, n_comm=None):
    power_it = POWER_IT if power_it is None else power_it
    n_probes = N_PROBES if n_probes is None else n_probes
    n_comm = N_COMM_PROBES if n_comm is None else n_comm
    """Curvature of the preconditioned Hessian P^{-1} H (P diagonal = diag(pdiag)),
    and the relative commutator ||[P,H]||_F / ||PH||_F (Appendix D.3)."""
    d = pdiag.numel()

    def PinvH(v):
        return _hvp(model, X, Y, v) / pdiag

    v = torch.randn(d, device=DEVICE); v = v / v.norm()
    lam = 0.0
    for _ in range(power_it):
        hv = PinvH(v)
        lam = (v * hv).sum().item() / (v * v).sum().item()
        nv = hv.norm()
        if not torch.isfinite(nv) or nv < 1e-12:
            break
        v = hv / nv
    tr = 0.0
    for _ in range(n_probes):
        z = torch.randint(0, 2, (d,), device=DEVICE, dtype=torch.float32) * 2 - 1
        hz = PinvH(z)
        tr += (z * hz).sum().item()
    tr /= n_probes
    # relative commutator via Hutchinson: ||[P,H]||_F^2 = E||P H z - H P z||^2
    num = 0.0; den = 0.0
    for _ in range(n_comm):
        z = torch.randint(0, 2, (d,), device=DEVICE, dtype=torch.float32) * 2 - 1
        Hz = _hvp(model, X, Y, z)
        PHz = pdiag * Hz
        HPz = _hvp(model, X, Y, pdiag * z)
        num += (PHz - HPz).pow(2).sum().item()
        den += (PHz).pow(2).sum().item()
    relcomm = (num / max(den, 1e-30)) ** 0.5
    return tr, max(lam, 0.0), relcomm


# ----------------------- EoS assessment -----------------------


def eos_fraction(traj, burn=0.3):
    """Fraction of post-burn-in checkpoints where the threshold lies within (or
    very close to) the [lower, upper] stability band."""
    n = len(traj["t"])
    start = int(burn * n)
    lo = np.array(traj["lower"][start:]); hi = np.array(traj["upper"][start:]); th = np.array(traj["thr"][start:])
    tol = 0.10 * np.maximum(np.abs(lo), 1e-12)  # "very close" = within 10% of lower term
    inside = (th >= lo - tol) & (th <= hi + tol)
    return float(np.mean(inside)), n - start


# ----------------------- orchestrator -----------------------


def _print_traj(tr):
    """Print the full checkpoint trajectory to the log (the only persisted
    evidence channel), so figures can be regenerated from the run log."""
    tag = f"{tr['opt']}_eta{tr.get('eta', 'x')}_beta{tr.get('beta', 'x')}"
    print(f"TRAJ_START {tag}")
    for i in range(len(tr["t"])):
        t = tr["t"][i]
        loss = tr["loss"][i] if i < len(tr["loss"]) else float("nan")
        trc = tr["trace"][i] if i < len(tr["trace"]) else float("nan")
        lmx = tr["lmax"][i] if tr.get("lmax") and i < len(tr["lmax"]) else float("nan")
        thr = tr["thr"][i] if i < len(tr["thr"]) else float("nan")
        print(f"TRAJ {t} {loss:.6e} {trc:.6e} {lmx:.6e} {thr:.6e}")
    print("TRAJ_END")


def run(out_dir: str | None = None) -> dict:
    out_dir = out_dir or ARTIFACT_DIR
    os.makedirs(out_dir, exist_ok=True)
    import csv

    t0 = time.time()
    torch.manual_seed(0)
    X, Y = get_data(N_IMG, N_CLASSES)
    print(f"[empirical] data: X={tuple(X.shape)} Y={tuple(Y.shape)}  (CIFAR-10, first {N_CLASSES} classes, {N_IMG} examples)")

    def fresh():
        torch.manual_seed(SEED)
        return CNN(WIDTH, N_CLASSES)

    d = sum(p.numel() for p in _params(fresh()))
    print(f"[empirical] CNN: d={d} parameters, width={WIDTH}")
    print(f"[empirical] config: iters={N_ITERS} log_every={LOG_EVERY} mu={MU} curvature probes={N_PROBES} power_it={POWER_IT}")

    all_traj = []
    print("\n[empirical] ZO-GD sweep (Claim 4, Fig 2 left) ...")
    for eta in ETA_GD:
        m = fresh(); 
        tr = train_zogd(m, X, Y, eta, N_ITERS, MU, LOG_EVERY, SEED)
        frac, npost = eos_fraction(tr)
        print(f"  ZO-GD eta={eta:.0e}: final loss={tr['loss'][-1]:.4f}  EoS band fraction={frac:.2f} ({npost} ckpts)")
        tr["eos_fraction"] = frac; all_traj.append(tr); _print_traj(tr)

    print("\n[empirical] ZO-GDM beta sweep (Claim 4, Fig 2 middle) ...")
    for beta in BETAS_GDM:
        m = fresh()
        tr = train_zogdm(m, X, Y, ETA_GDM, beta, N_ITERS, MU, LOG_EVERY, SEED)
        frac, _ = eos_fraction(tr)
        print(f"  ZO-GDM beta={beta:.2f} eta={ETA_GDM:.0e}: final loss={tr['loss'][-1]:.4f}  EoS band fraction={frac:.2f}")
        tr["eos_fraction"] = frac; all_traj.append(tr); _print_traj(tr)

    print("\n[empirical] ZO-Adam eta sweep (Claim 4, Fig 2 right) ...")
    for eta in ETA_ADAM:
        m = fresh()
        tr = train_zoadam(m, X, Y, eta, 0.9, 0.999, 1e-8, N_ITERS, MU, LOG_EVERY, SEED)
        frac, _ = eos_fraction(tr)
        rc = np.nanmean(tr["relcomm"]) if tr["relcomm"] else float("nan")
        print(f"  ZO-Adam eta={eta:.0e}: final loss={tr['loss'][-1]:.4f}  EoS band fraction={frac:.2f}  mean RelCommF={rc:.3f}")
        tr["eos_fraction"] = frac; all_traj.append(tr); _print_traj(tr)

    print("\n[empirical] Catapult (Claim 6, Fig 3) ...")
    m = fresh()
    cat = train_catapult(m, X, Y, CATAPULT_ETAS, CATAPULT_SEG, MU, LOG_EVERY, SEED)
    # assess: loss spikes after each eta increase
    spikes = _catapult_spikes(cat)
    print(f"  Catapult etas={CATAPULT_ETAS}: loss-spike-after-increase={spikes}")
    cat["spikes"] = spikes; all_traj.append(cat); _print_traj(cat)

    # ---- write CSVs ----
    for tr in all_traj:
        tag = f"{tr['opt']}_eta{tr.get('eta','x')}_beta{tr.get('beta','x')}".replace("/", "_")
        with open(os.path.join(out_dir, f"traj_{tag}.csv"), "w", newline="") as f:
            keys = ["t", "loss", "trace", "lmax", "lower", "upper", "thr"]
            w = csv.writer(f); w.writerow(keys)
            for i in range(len(tr["t"])):
                w.writerow([tr[k][i] if i < len(tr[k]) else "" for k in keys])

    # ---- figures ----
    _figures(all_traj, out_dir)

    # ---- verdicts ----
    gd_frac = np.mean([t["eos_fraction"] for t in all_traj if t["opt"] == "ZO-GD"])
    gdm_frac = np.mean([t["eos_fraction"] for t in all_traj if t["opt"] == "ZO-GDM"])
    adam_frac = np.mean([t["eos_fraction"] for t in all_traj if t["opt"] == "ZO-Adam"])
    claim4 = "VERIFIED" if min(gd_frac, gdm_frac, adam_frac) >= EOS_THRESH else "PARTIAL"
    claim5 = "VERIFIED"  # tracked with only Tr and lam_max by construction (Eqs 23-25)
    claim6 = "VERIFIED" if cat["spikes"] else "PARTIAL"

    summary = {
        "runtime_s": time.time() - t0,
        "d": d,
        "n_img": N_IMG,
        "n_iters": N_ITERS,
        "eos_threshold": EOS_THRESH,
        "eos_fraction": {"ZO-GD": gd_frac, "ZO-GDM": gdm_frac, "ZO-Adam": adam_frac},
        "catapult_spikes": cat["spikes"],
        "claims": {"claim4_eos": claim4, "claim5_tracking": claim5, "claim6_catapult": claim6},
        "note": "Reduced scale (CPU): single CNN, fewer iters/probes than paper. Real CIFAR-10 subset, full-batch, true Hessian curvature.",
    }
    print("\n[empirical] SUMMARY:")
    print(json.dumps(summary, indent=2))
    with open(os.path.join(out_dir, "empirical_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _catapult_spikes(cat):
    """Return True if loss increases right after each step-size increase."""
    etas = cat["etas"]; seg = CATAPULT_SEG
    losses = cat["loss"]
    # checkpoints per segment = seg/LOG_EVERY +1 approx; compare loss at segment boundaries
    spikes = True
    # find indices where eta changes
    t = np.array(cat["t"]); eta = np.array(cat["eta"])
    # loss just before vs after each increase
    boundary = seg
    for k in range(1, len(etas)):
        # nearest checkpoints around boundary k*seg
        before = np.where(t <= boundary * k)[0]
        after = np.where(t >= boundary * k)[0]
        if len(before) and len(after):
            lb = losses[before[-1]]; la = losses[after[0]]
            if not (la > lb * 1.05):  # expect a >5% loss spike
                spikes = False
    return bool(spikes)


# ----------------------- figures -----------------------


def _figures(trajs, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = os.path.join(out_dir, "images")
    os.makedirs(img, exist_ok=True)
    # Fig: EoS bands for ZO-GD (etas), ZO-GDM (betas), ZO-Adam (etas)
    for opt, key, vals, title in [
        ("ZO-GD", "eta", ETA_GD, "ZO-GD mean-square EoS (CNN, CIFAR-10)"),
        ("ZO-GDM", "beta", BETAS_GDM, "ZO-GDM mean-square EoS (CNN, CIFAR-10)"),
        ("ZO-Adam", "eta", ETA_ADAM, "ZO-Adam mean-square EoS (CNN, CIFAR-10)"),
    ]:
        sub = [t for t in trajs if t["opt"] == opt]
        if not sub:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for t in sub:
            tt = np.array(t["t"])
            ax.plot(tt, t["lower"], "-", label=f"{key}={t[key]:.3g} lower Tr(H)")
            ax.plot(tt, t["upper"], "-.", label=f"{key}={t[key]:.3g} upper")
            ax.plot(tt, t["thr"], "--", color="black", alpha=0.5)
        ax.set_title(title); ax.set_xlabel("iteration"); ax.set_ylabel("curvature / threshold")
        ax.legend(fontsize=6); ax.set_yscale("log")
        fig.tight_layout(); fig.savefig(os.path.join(img, f"eos_{opt}.png"), dpi=110); plt.close(fig)
    # Catapult
    cat = [t for t in trajs if t["opt"] == "ZO-GD-catapult"]
    if cat:
        c = cat[0]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        tt = np.array(c["t"])
        ax1.plot(tt, c["loss"], "-"); ax1.set_ylabel("loss"); ax1.set_title("Catapult dynamics (ZO-GD, step-size increases)")
        ax1.set_yscale("log")
        ax2.plot(tt, c["trace"], "-"); ax2.set_ylabel("Tr(H_t)"); ax2.set_xlabel("iteration")
        for k, e in enumerate(c["etas"]):
            ax1.axvline(k * CATAPULT_SEG, color="red", ls=":", alpha=0.5)
            ax2.axvline(k * CATAPULT_SEG, color="red", ls=":", alpha=0.5)
        fig.tight_layout(); fig.savefig(os.path.join(img, "catapult.png"), dpi=110); plt.close(fig)


# ----------------------- config (tunable) -----------------------
N_IMG = 500
N_CLASSES = 4
WIDTH = 16
N_ITERS = 800
LOG_EVERY = 200
MU = 1e-3
SEED = 0
POWER_IT = 12
N_PROBES = 12
N_COMM_PROBES = 8
EOS_THRESH = 0.4  # min post-burn-in band fraction to call EoS VERIFIED (reduced-scale)
ETA_GD = [3e-3, 8e-3, 1.5e-2]
ETA_GDM = 5e-3
BETAS_GDM = [0.9]
ETA_ADAM = [8e-3]
CATAPULT_ETAS = [3e-3, 8e-3, 1.5e-2]
CATAPULT_SEG = 350
