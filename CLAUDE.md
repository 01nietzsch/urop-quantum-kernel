# UROP: Quantum Kernel Methods — project context

## Who / what

Vlad, undergraduate research opportunity programme. Supervisor: Jack (Antoine Jacquier).
Roughly one month remaining as of late July 2026.

Background: I have taken a quantum mechanics course but I am **not** experienced in
quantum computing. When explaining, don't assume familiarity with Qiskit idioms,
NISQ hardware conventions, or QML terminology — spell them out. Physics/linear
algebra I'm fine with.

## Objective

Reproduce the results of:

> C. Blank, D. K. Park, J.-K. K. Rhee, F. Petruccione,
> *Quantum classifier with tailored quantum kernel*,
> npj Quantum Information **6**, 41 (2020).

**Important:** the published npj version does NOT contain the Supplementary
Information. Use the arXiv version, `arXiv:1909.02611` — Supplementary Notes I–V,
Table I, and Figs. 1–7 are appended to that PDF. Everything about the noise model,
the device calibration data, and the transpiled circuit is in there and nowhere else.

The paper builds a binary classifier whose kernel is the quantum state fidelity
|<x~|x_m>|^(2n), evaluated by a swap test, read out as a two-qubit observable
<sigma_z^(a) sigma_z^(l)> so that no post-selection is needed.

## Where things stand

| Step | Status |
|---|---|
| 1. Closed-form kernel (`closed_form_kernel.py`) | done |
| 2. Swap-test circuit, verified against closed form (`circuit_kernel.py`) | done |
| 2b. Generalised quantum forking for arbitrary M (`circuit_kernel.py`) | done, verified at M=4 |
| 3. Noise model reproducing the paper's noisy simulation (`noise_model.py`) | done — see below |
| 4. Run on real IBM hardware | not started |
| 5. Close the simulation-vs-hardware gap | not started |

### Step 3 result (current)

Fitting `a * (sin^2((theta + vartheta)/2 + pi/4) - w2)` to the noisy sweep.
Single seed (seed=1234) and 10-seed statistics (seeds 0–9, 8192 shots each):

|  | single seed (1234) | 10-seed mean ± std | paper |
|---|---|---|---|
| amplitude `a` | 0.8300 | 0.8213 ± 0.0087 | 0.8213 |
| phase `vartheta` | −0.0037 rad | ~0 | ~0 |
| offset `w2` | 0.4989 | 0.4968 ± 0.0017 | 0.5023 |
| transpiled CNOT count | 13 | 13 | 13 |

The 10-seed mean `a = 0.8213` matches the paper exactly. The single-seed result
of 0.8300 was 1 std dev above the mean — the offset is shot noise, not systematic.

Noiseless sanity check (`noisy=False`, seed=42): `a = 1.0025`, consistent with
ideal `a = 1.000` (deviation is shot noise at 8192 shots per point; noiseless
transpile gives 10 CNOTs because no coupling_map constraint is applied).

## The toy problem (paper Eq. 11)

    |x1>     = (i|0> + |1>)/sqrt(2),   y1 = 0
    |x2>     = (i|0> - |1>)/sqrt(2),   y2 = 1
    |x~(th)> = cos(th/2)|0> - i sin(th/2)|1>

with w1 = w2 = 1/2, n = 1. Theory: `<sz sz> = sin^2(th/2 + pi/4) - 1/2`.

This dataset is deliberately chosen so the *Hadamard* classifier of Schuld et al.
gives identically zero — only a fidelity-based kernel separates these classes.
That is the whole point of the example; don't "simplify" it.

## Two circuit variants — do not confuse them

**(a) The paper's 5-qubit toy circuit** (Fig. 6 / Supp. Fig. 2), in `noise_model.py`.
No quantum forking. Because x2 is x1 with a sign flip on |1>, the entire entangled
training register comes from a single `cz(index, data)`. Physical layout hand-picked
by the authors: `a->q0, d->q1, in->q2, m->q3, l->q4`.
**This is the one that reproduces Fig. 5.**

**(b) The general M-point forking circuit**, in `circuit_kernel.py`. Needs
`2 + ceil(log2 M) + M + 1` qubits — six for M=2. It therefore does **not** fit on
the 5-qubit device and cannot be used for the Fig. 5 comparison. It is our own
extension, to be run on an idealised all-to-all backend.

## Reproduction targets (arXiv Supp. Note III D)

Simulation: `a ≈ 0.8213`, `vartheta ≈ 0`, `w2 ≈ 0.50232985`
Hardware (ibmq_ourense, 2019-09-29): `a ≈ 0.6515`, `vartheta ≈ 2/51 * pi`, `w2 ≈ 0.5414`
Gate count: 27 total = 14 single-qubit + 13 CNOT
Sweep: theta from 0 to 2*pi in steps of 0.1 (63 points), 8192 shots each

(The Methods section says "8129 shots" once — that is a typo; the expectation-value
formula in the same paragraph divides by 8192.)

## Gotchas — these have already bitten us

- **Never use `Statevector.from_instruction` for anything noisy.** It is exact by
  construction and will silently give you the noiseless answer.
- **Never use `qc.initialize` in a circuit destined for a noise model.** It inserts a
  `reset` instruction, which does not correspond to any hardware operation and
  inflates the transpiled gate count. Use explicit `ry`/`rx` rotations.
- `u1`/`u2`/`u3` no longer exist in current Qiskit. `noise_model.py` remaps the 2019
  calibration onto `rz`/`sx`/`x`: `rz` is virtual (zero duration, error-free), and
  `sx`/`x` are each one pulse, inheriting the tabulated u2 duration and u2 error.
  Faithful in spirit; a candidate source of the residual 1%.
- Don't pass `backend` *and* `coupling_map`/`basis_gates` to `transpile` together —
  Qiskit warns and the backend's durations get invalidated. Specify the constraints
  explicitly and omit the backend.

## Open items

1. **`cx` duration for the q3–q1 pair is a placeholder (300 ns)** in `noise_model.py`.
   Not legible in the PDF table. We have searched every file in the authors' repo
   (`github.com/carstenblank/Quantum-classifier-with-tailored-quantum-kernels---Supplemental`)
   including `lib_experimental_utils.py` and all experiment result files — ibmq_ourense
   gate times were never stored statically; they were queried live from the IBM Q API
   at runtime in 2019, which is no longer accessible. 300 ns (interpolated between the
   neighbouring pairs 235 ns and 210 ns) remains our best estimate. This edge matters:
   it is the noisiest link (eps = 0.0116, roughly double the others) and the label
   qubit q4 sits behind q3.
2. ~~**Quantify shot noise on the fitted amplitude.**~~ **DONE.** 10-seed run gives
   `a = 0.8213 ± 0.0087`. The offset is shot noise — see Step 3 result table above.
3. ~~**Sanity check: `run_sweep(noisy=False)` should give `a ≈ 1.000`.**~~ **DONE.**
   `a = 1.003` at seed=42 with 8192 shots. Passes.
3. **Sanity check:** `run_sweep(noisy=False)` should give `a ≈ 1.000`.

## Direction for the remaining month — to be agreed with Jack

Two candidates, and they are genuinely different projects:

- **(A) Run on real IBM hardware.** Caveat that must not be glossed over:
  *ibmq_ourense has been retired.* Any hardware run happens on a different device with
  different calibration data, so it is not a like-for-like comparison with the paper's
  0.65. To make it meaningful we would have to rebuild the noise model from the
  *current* backend's calibration and compare our own hardware/simulation pair.
- **(B) Close the 0.82 vs 0.65 gap.** The supplementary states plainly that the basic
  noise model undershoots reality and attributes the difference to crosstalk,
  time-dependent drift, and non-Markovian noise — none of which the basic model
  contains. Extending the model toward any of these is a more interesting scientific
  question than re-running the toy example.

## Standing rule: code and write-up move together

**Every change to the code must be reflected in the LaTeX write-up in the same
session.** The `.tex` source lives in this folder. Concretely:

- New or changed numerical result → update the corresponding number, table, or figure
  in the `.tex`, and update the surrounding text if the change alters the conclusion.
- New method, circuit, or noise component → add or revise the methods subsection
  describing it, in enough detail that a reader could reimplement it.
- Any parameter that is a placeholder or an assumption (e.g. the 300 ns above, the
  u2 → sx remapping) → must appear explicitly in the write-up as a stated caveat,
  not left only as a code comment.
- If a code change makes something in the `.tex` wrong or stale, fix it rather than
  leaving it — flag it to me if the right fix isn't obvious.

Do not batch this up "for later". A code change without the matching LaTeX update is
an incomplete change.

## Repo layout

    closed_form_kernel.py   analytic fidelity + expectation value, toy dataset
    circuit_kernel.py       swap-test circuit; generalised forking for M > 2
    noise_model.py          ourense noise model + paper's 5-qubit circuit + sweep + fit
    02_circuit_check.ipynb  circuit vs closed-form verification (M=2 and M=4)
    *.tex                   write-up — keep in sync, see standing rule above
