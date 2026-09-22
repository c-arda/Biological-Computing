"""
Experiment 1c (extended): ENAQT robustness sweep across chain length and
bridge (barrier) height.

Replaces a prior generator (not present in this repo) that used explicit
forward-Euler integration of the Lindblad master equation. That integrator
was numerically unstable for the stiffest parameter regimes (large barrier
height Delta_epsilon combined with low dephasing gamma), producing
non-positive transport-efficiency values and corrupted enhancement ratios
for the Delta_epsilon=25J bridge case and (catastrophically) the
Delta_epsilon=50J case. See the manuscript's "Correction to the extended
sweep" paragraph (Limitations section) for the full account.

This version solves the Lindblad equation EXACTLY via the matrix exponential
of the vectorised Liouvillian super-operator. The systems here are small
(5-9 dimensional Hilbert space including the trap state), so this is both
exact (no timestep-instability failure mode) and fast (the full 10-study,
200-point sweep runs in a few seconds on CPU).

Model (matches the physical setup already used in experiment_1c_gpu.py):
  - n_sites donor-bridge-acceptor tight-binding chain, hopping J=1 (natural
    units; all energies/rates below are quoted in units of J).
  - Site energies: [0, Delta_epsilon, Delta_epsilon, ..., Delta_epsilon, 0]
    (flat bridge of height Delta_epsilon between donor and acceptor).
  - An explicit (n_sites+1)-th "trap" state, decoupled from H, absorbing
    population from the last real site at rate kappa.
  - Dephasing: independent pure-dephasing Lindblad operator sqrt(gamma) on
    each of the n_sites real sites (not the trap).
  - Transport efficiency P4(gamma) := trap-state population at time t=20/J,
    starting from full population on site 0.

Outputs simulations/results/enaqt_extended_sweep.json with the same schema
as before: {"gammas": [...], "studies": {name: {"p4": [...], "peak_gamma":
..., "peak_p4": ..., "enhancement": p4max/p4[0]}}}.
"""
import json
import os
import time

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "enaqt_extended_sweep.json")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "enaqt_extended_sweep_dashboard.png")

HOPPING = 1.0          # J, natural unit
SINK_RATE = 0.05        # kappa, in units of J
T_FINAL = 20.0           # in units of 1/J
N_GAMMA = 200
GAMMA_MIN = 0.01
GAMMA_MAX = 500.0        # matches the previous sweep's max (~501.19)

CHAIN_BRIDGE_HEIGHT = 15.0   # Delta_epsilon (J) used for the chain-length sweep
CHAIN_LENGTHS = [3, 4, 5, 6, 8]
BRIDGE_N_SITES = 4            # fixed chain length for the bridge-height sweep
BRIDGE_HEIGHTS = [5.0, 10.0, 15.0, 25.0, 50.0]


def build_liouvillian(n_sites, bridge_height, gamma, kappa=SINK_RATE, hopping=HOPPING):
    """Vectorised Liouvillian for an n_sites donor-bridge-acceptor chain
    plus one absorbing trap state. Returns a (d^2, d^2) matrix L such that
    d(vec(rho))/dt = L @ vec(rho), using the row-major vec convention
    consistent with the rhs() helper below.
    """
    d = n_sites + 1  # +1 for the trap state
    H = np.zeros((d, d), dtype=complex)
    if n_sites == 2:
        site_energies = [0.0, 0.0]
    else:
        site_energies = [0.0] + [bridge_height] * (n_sites - 2) + [0.0]
    for i in range(n_sites):
        H[i, i] = site_energies[i]
    for i in range(n_sites - 1):
        H[i, i + 1] = hopping
        H[i + 1, i] = hopping

    Ls = []
    for i in range(n_sites):
        L = np.zeros((d, d), dtype=complex)
        L[i, i] = np.sqrt(gamma)
        Ls.append(L)
    L_trap = np.zeros((d, d), dtype=complex)
    L_trap[n_sites, n_sites - 1] = np.sqrt(kappa)
    Ls.append(L_trap)

    def rhs(rho):
        out = -1j * (H @ rho - rho @ H)
        for L in Ls:
            Ld = L.conj().T
            LdL = Ld @ L
            out += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
        return out

    dim2 = d * d
    Lmat = np.empty((dim2, dim2), dtype=complex)
    for j in range(dim2):
        e = np.zeros(dim2, dtype=complex)
        e[j] = 1.0
        Lmat[:, j] = rhs(e.reshape(d, d)).reshape(-1)
    return Lmat, d


def transport_efficiency(n_sites, bridge_height, gamma, t=T_FINAL):
    Lmat, d = build_liouvillian(n_sites, bridge_height, gamma)
    rho0 = np.zeros((d, d), dtype=complex)
    rho0[0, 0] = 1.0
    rho_t = expm(Lmat * t) @ rho0.reshape(-1)
    rho_t = rho_t.reshape(d, d)
    trace = rho_t.trace().real
    if abs(trace - 1.0) > 1e-6:
        raise RuntimeError(
            f"Trace not preserved (n_sites={n_sites}, Delta_eps={bridge_height}, "
            f"gamma={gamma}): trace={trace:.8f}"
        )
    return rho_t[n_sites, n_sites].real  # trap-state population


def run_sweep(n_sites, bridge_height, gammas):
    p4 = np.array([transport_efficiency(n_sites, bridge_height, g) for g in gammas])
    # Positivity check: catch any future numerical regression immediately
    # rather than silently shipping a bad table again.
    assert np.all(p4 >= -1e-9), (
        f"Non-positive P4 encountered (n_sites={n_sites}, Delta_eps={bridge_height}): "
        f"min={p4.min():.3e}"
    )
    assert np.all(p4 <= 1.0 + 1e-9), (
        f"P4 exceeds 1 (n_sites={n_sites}, Delta_eps={bridge_height}): max={p4.max():.3e}"
    )
    peak_idx = int(np.argmax(p4))
    peak_gamma = float(gammas[peak_idx])
    peak_p4 = float(p4[peak_idx])
    baseline = max(p4[0], 1e-12)  # only to guard a genuine zero, never to mask instability
    enhancement = float(peak_p4 / baseline)
    return {
        "p4": p4.tolist(),
        "peak_gamma": peak_gamma,
        "peak_p4": peak_p4,
        "enhancement": enhancement,
    }


def plot_dashboard(gammas, studies):
    """Dark-theme 2x2 dashboard, matching the visual style already used by
    experiment_1c_gpu.py's ENAQT dashboards."""
    bg = "#0c1024"
    edge = "#1a3a4a"
    colors = ["#00d2ff", "#f59e0b", "#10b981", "#f43f5e", "#a78bfa"]

    fig = plt.figure(figsize=(16, 10), facecolor=bg)
    fig.suptitle("ENAQT EXTENDED SWEEP :: EXACT LINDBLAD (MATRIX EXPONENTIAL)",
                 fontsize=14, fontweight="bold", color="#00d2ff",
                 fontfamily="monospace", y=0.98)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(bg)
        ax.set_title(title, color="#00d2ff", fontsize=11, fontfamily="monospace")
        ax.set_xlabel(xlabel, color="white", fontfamily="monospace")
        ax.set_ylabel(ylabel, color="white", fontfamily="monospace")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color(edge)

    # Panel 1: chain-length P4(gamma) curves
    ax1 = fig.add_subplot(2, 2, 1)
    chain_keys = [k for k in studies if k.endswith("_sites")]
    chain_keys.sort(key=lambda k: int(k.split("_")[0]))
    for i, key in enumerate(chain_keys):
        n = key.split("_")[0]
        ax1.semilogx(gammas, studies[key]["p4"], color=colors[i % len(colors)],
                     linewidth=1.5, label=f"N={n}")
    style_ax(ax1, f"CHAIN-LENGTH SWEEP (Δε={CHAIN_BRIDGE_HEIGHT:.0f}J)",
             "γ (J, log scale)", "P₄")
    ax1.legend(fontsize=8, facecolor=bg, labelcolor="white")

    # Panel 2: bridge-height P4(gamma) curves.
    # Restricted to the range actually reported in the manuscript
    # (Delta_eps = 5-25 J, Table 1c-ext); Delta_eps=50J is retained in the
    # JSON as a diagnostic (it's the point that catastrophically diverges
    # under the old broken integrator) but isn't discussed in the text,
    # so it's excluded from the published figure to avoid a mismatch.
    ax2 = fig.add_subplot(2, 2, 2)
    REPORTED_BRIDGE_HEIGHTS = {5.0, 10.0, 15.0, 25.0}
    bridge_keys = [k for k in studies if k.startswith("bridge_")
                   and float(k.split("_")[1]) in REPORTED_BRIDGE_HEIGHTS]
    bridge_keys.sort(key=lambda k: float(k.split("_")[1]))
    for i, key in enumerate(bridge_keys):
        eps = key.split("_")[1]
        ax2.semilogx(gammas, studies[key]["p4"], color=colors[i % len(colors)],
                     linewidth=1.5, label=f"Δε={float(eps):.0f}J")
    style_ax(ax2, f"BRIDGE-HEIGHT SWEEP (N={BRIDGE_N_SITES})",
             "γ (J, log scale)", "P₄")
    ax2.legend(fontsize=8, facecolor=bg, labelcolor="white")

    # Panel 3: enhancement vs chain length
    ax3 = fig.add_subplot(2, 2, 3)
    ns = [int(k.split("_")[0]) for k in chain_keys]
    enh = [studies[k]["enhancement"] for k in chain_keys]
    ax3.semilogy(ns, enh, "o-", color="#f59e0b", linewidth=1.5, markersize=7)
    style_ax(ax3, "ENHANCEMENT vs CHAIN LENGTH", "N (sites)", "Enhancement (log scale)")

    # Panel 4: enhancement vs bridge height
    ax4 = fig.add_subplot(2, 2, 4)
    epss = [float(k.split("_")[1]) for k in bridge_keys]
    enh2 = [studies[k]["enhancement"] for k in bridge_keys]
    ax4.semilogy(epss, enh2, "o-", color="#10b981", linewidth=1.5, markersize=7)
    style_ax(ax4, "ENHANCEMENT vs BARRIER HEIGHT", "Δε (J)", "Enhancement (log scale)")

    fig.patch.set_facecolor(bg)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PNG, dpi=150, facecolor=bg, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUTPUT_PNG}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gammas = np.logspace(np.log10(GAMMA_MIN), np.log10(GAMMA_MAX), N_GAMMA)

    studies = {}
    t0 = time.time()

    print("Chain-length sweep (Delta_epsilon = {:.0f} J):".format(CHAIN_BRIDGE_HEIGHT))
    for n in CHAIN_LENGTHS:
        result = run_sweep(n, CHAIN_BRIDGE_HEIGHT, gammas)
        studies[f"{n}_sites"] = result
        print(f"  N={n}: peak_gamma={result['peak_gamma']:.3f}  "
              f"peak_p4={result['peak_p4']:.6f}  enh={result['enhancement']:.1f}x")

    print(f"\nBridge-height sweep (N = {BRIDGE_N_SITES} sites):")
    for eps in BRIDGE_HEIGHTS:
        result = run_sweep(BRIDGE_N_SITES, eps, gammas)
        studies[f"bridge_{eps:.1f}"] = result
        print(f"  Delta_eps={eps:.0f}J: peak_gamma={result['peak_gamma']:.3f}  "
              f"peak_p4={result['peak_p4']:.6f}  enh={result['enhancement']:.1f}x")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")

    out = {"gammas": gammas.tolist(), "studies": studies}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {OUTPUT_JSON}")

    plot_dashboard(gammas, studies)


if __name__ == "__main__":
    main()
