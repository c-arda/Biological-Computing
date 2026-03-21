# PROJECT REVIEW — Biological Computing

> **Reviewed**: 2026-03-21 | **Status**: ◉ MATURE — Publication-ready
> **Path**: `/home/ardac/Projects/Biology/Biological Computing/`

## Health Score: 9/10

## Structure
- 11 experiments (1a–3b) with consistent naming
- Each experiment has: `experiment_XX.py` + `experiment_XX_metrics.json` + dashboard PNG
- Phase 2 (CL SDK) and Phase 3 (isotope/magnetic) complete
- Zenodo DOI assigned, preprint available

## Issues

### ◆ HIGH — Uncommitted files
```
?? simulations/experiment_1c_gpu.py    ← GPU ENAQT sweep (new from overnight)
?? simulations/results/                ← Dashboard PNGs not tracked
```
**Fix**: `git add simulations/experiment_1c_gpu.py simulations/results/ && git commit -m "feat: GPU-accelerated ENAQT Lindblad sweep"`

### ◆ MEDIUM — experiment_1c_gpu.py needs physics fix
The GPU Lindblad solver works (716K eval/s) but lacks the external trap state needed for the proper ENAQT peak. The original CPU version (experiment_1c) uses QuTiP's `mesolve` with a proper sink.
**Fix**: Add a 6th "trap" state to the Hilbert space with irreversible Lindblad operator `|trap⟩⟨4|`.

### ◆ LOW — No .gitignore for results/
Large PNGs and DCD files in results/ should have a gitignore entry.
**Fix**: Add `*.dcd` and optionally `*.png` to `.gitignore`

### ◆ LOW — experiment_3b null result not highlighted in README
README table is missing 3b (magnetic field) which gave a null result.
**Fix**: Add row for 3b in README table.

## Action Plan
1. Commit new GPU files
2. Fix ENAQT trap state physics
3. Add experiment 3b to README
4. Consider Phase 4 scope (real organoid hardware?)
