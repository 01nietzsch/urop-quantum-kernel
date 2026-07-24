"""
Reproduce Fig. 5 of Blank et al. (2020): theory curve, our noisy simulation,
and the paper's hardware data from ibmq_ourense (2019-09-29).

Saves Latex/figures/sweep.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from noise_model import run_sweep, fit_amplitude, theory

# ---------------------------------------------------------------------------
# Hardware data from the authors' result file
# experiment_results/exp_sim_regular_20190929T114806Z.py
# 'classification' key: 63 values, theta = 0, 0.1, ..., 6.2
# ---------------------------------------------------------------------------
HARDWARE = np.array([
    -0.0107421875, 0.022705078125, 0.059814453125, 0.0927734375,
     0.126220703125, 0.16796875, 0.195068359375, 0.213623046875,
     0.255615234375, 0.26171875, 0.28173828125, 0.29150390625,
     0.3115234375, 0.347900390625, 0.348876953125, 0.337890625,
     0.333984375, 0.35498046875, 0.347412109375, 0.336669921875,
     0.32275390625, 0.312255859375, 0.2861328125, 0.281494140625,
     0.263671875, 0.23974609375, 0.21630859375, 0.210693359375,
     0.1611328125, 0.142578125, 0.093994140625, 0.09033203125,
     0.053955078125, 0.038330078125, -0.007568359375, -0.04248046875,
    -0.065673828125, -0.08642578125, -0.124267578125, -0.150390625,
    -0.193603515625, -0.19970703125, -0.22412109375, -0.2421875,
    -0.26806640625, -0.264892578125, -0.311279296875, -0.31201171875,
    -0.32421875, -0.30517578125, -0.296630859375, -0.298095703125,
    -0.28466796875, -0.26953125, -0.258544921875, -0.2490234375,
    -0.216796875, -0.181396484375, -0.160400390625, -0.14599609375,
    -0.107177734375, -0.074951171875, -0.03369140625,
])

def fit_model(th, a, vartheta, w2):
    return a * (np.sin((th + vartheta) / 2 + np.pi / 4) ** 2 - w2)


def main():
    thetas = np.arange(0.0, 2 * np.pi, 0.1)   # 63 points
    assert len(thetas) == len(HARDWARE), "theta/hardware length mismatch"

    print("Running noisy sweep (seed=1234, 8192 shots) ...")
    th, sim_vals = run_sweep(noisy=True, seed=1234)

    # Fits
    a_sim, v_sim, w2_sim = fit_amplitude(th, sim_vals)
    popt_hw, _ = curve_fit(fit_model, thetas, HARDWARE, p0=[0.65, 0.0, 0.5])
    a_hw, v_hw, w2_hw = popt_hw

    # Smooth curves
    th_fine = np.linspace(0, 2 * np.pi, 400)
    theory_curve = theory(th_fine)
    sim_fit_curve = fit_model(th_fine, a_sim, v_sim, w2_sim)
    hw_fit_curve = fit_model(th_fine, a_hw, v_hw, w2_hw)

    print(f"Sim fit:  a={a_sim:.4f}, vartheta={v_sim:.4f}, w2={w2_sim:.4f}")
    print(f"HW  fit:  a={a_hw:.4f},  vartheta={v_hw:.4f},  w2={w2_hw:.4f}")
    print(f"Paper HW target: a≈0.6515, vartheta≈{2/51*np.pi:.4f}, w2≈0.5414")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(th_fine, theory_curve, color="black", lw=1.5, label="Theory ($a=1$)")
    ax.plot(th_fine, sim_fit_curve, color="C0", lw=1.2, ls="--",
            label=f"Sim fit ($a={a_sim:.3f}$)")
    ax.plot(th_fine, hw_fit_curve, color="C1", lw=1.2, ls="--",
            label=f"HW fit ($a={a_hw:.3f}$)")

    ax.scatter(th, sim_vals, color="C0", s=14, zorder=3,
               label="Our noisy sim (seed 1234)")
    ax.scatter(thetas, HARDWARE, color="C1", s=14, marker="s", zorder=3,
               label="Hardware (ibmq\\_ourense, 2019-09-29)")

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\langle \sigma_z^{(a)} \sigma_z^{(l)} \rangle$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                  ["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Noisy sweep: theory, simulation, and hardware", fontsize=10)

    fig.tight_layout()
    out = "Latex/figures/sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
