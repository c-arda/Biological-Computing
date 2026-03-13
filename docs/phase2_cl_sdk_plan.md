# Phase 2 — CL SDK Experiments: ENAQT on Biological Neural Networks

## Overview

Phase 1 established a computational bridge: ENAQT-optimal quantum transport in an ion channel model directly maximises memory capacity in an abstract reservoir (ESN). Phase 2 replaces the abstract ESN with **simulated biological neural networks (BNNs)** using the [Cortical Labs CL SDK](https://github.com/Cortical-Labs/cl-sdk), testing whether the same bridge holds when the reservoir is biological.

The CL SDK simulator is a **drop-in replacement** for the physical CL1 hardware — code developed locally runs on real organoid neurons without modification. This means Phase 2 experiments are immediately deployable on CL1 hardware when access becomes available.

---

## CL SDK API Capabilities

| Feature | API | Details |
|---------|-----|---------|
| Connect to BNN | `cl.open()` | Context manager, works identically on simulator and CL1 |
| Closed-loop processing | `neurons.loop()` | Up to 25 kHz tick rate with µs-latency spike detection |
| Spike detection | `tick.analysis.spikes` | Per-tick spike list with timestamps, channels, and waveforms |
| Stimulation | `neurons.stim()` | Current-based stim via `ChannelSet` + `StimDesign` + `BurstDesign` |
| Recording | `neurons.record()` | HDF5 output: raw samples, spikes, stims, custom data streams |
| Offline analysis | `RecordingView` | Load and inspect HDF5 recordings |
| Accelerated mode | `CL_SDK_ACCELERATED_TIME=1` | Fast simulation decoupled from wall clock |

---

## Proposed Experiments

### Experiment 2a — BNN Reservoir Computing Baseline

**Goal**: Establish memory capacity (MC) of the CL SDK biological reservoir as a baseline.

**Method**:
1. Open BNN connection via `cl.open()`
2. Drive the network with a random input signal via `neurons.stim()` — encode the input as biphasic current stim patterns
3. Record the neural response via `neurons.record()` — spike trains across channels
4. Offline: train ridge-regression readout on spike counts per channel (as reservoir states)
5. Compute MC using the same delay-correlation method as Experiment 1d

**CL SDK specifics**:
```python
import cl
from cl import ChannelSet, StimDesign

input_channels = ChannelSet(0, 1, 2, 3, 4)  # 5 input channels
stim = StimDesign(160, -1.0, 160, 1.0)       # biphasic, 160µs, 1µA

with cl.open() as neurons:
    recording = neurons.record()
    for tick in neurons.loop(ticks_per_second=1000, stop_after_seconds=30):
        # Encode random input as stimulation on/off gating
        if random_input[tick_idx] > threshold:
            neurons.stim(input_channels, stim)
        # Spike response is captured automatically by recording
    recording.stop()
```

**Output**: Baseline MC value for the simulated BNN reservoir.

---

### Experiment 2b — ENAQT-Gated BNN Reservoir

**Goal**: Apply ENAQT-derived synaptic gating to the biological reservoir (replicate Experiment 1e with BNN instead of ESN).

**Method**:
1. Use the P₄(γ) curve from Experiment 1c
2. At each dephasing rate γ, set `p_rel = P₄(γ)` as the stochastic stimulation probability
3. For each incoming stim: transmit with probability `p_rel`, skip otherwise
4. Measure MC at each γ value
5. Test whether the MC curve co-peaks with the ENAQT curve

**Key difference from Phase 1**: The reservoir is now biological neural dynamics (spiking, adaptation, plasticity) instead of a tanh-activated ESN.

```python
import cl
import numpy as np
from cl import ChannelSet, StimDesign

# ENAQT-derived release probability for this dephasing condition
p_rel = 0.418  # e.g., ENAQT peak

stim = StimDesign(160, -1.0, 160, 1.0)
input_channels = ChannelSet(range(8))

with cl.open() as neurons:
    recording = neurons.record()
    for tick in neurons.loop(ticks_per_second=1000, stop_after_seconds=30):
        if random_input[tick_idx] > 0:
            # ENAQT stochastic gating: transmit with probability p_rel
            if np.random.random() < p_rel:
                neurons.stim(input_channels, stim)
    recording.stop()
```

---

### Experiment 2c — Temperature Sweep Prediction Test

**Goal**: Test the falsifiable prediction that varying the effective "temperature" (dephasing strength) produces a non-monotonic MC curve.

**Method**:
1. Sweep p_rel across the full ENAQT curve (0.01 → 1.0)
2. At each p_rel, run the stim-gated reservoir, record, compute MC
3. Plot MC vs γ — if ENAQT is operative, MC should peak and then decline
4. Compare against a "classical" control where stim amplitude is scaled (not gated)

**Prediction**: MC peaks at the ENAQT-optimal p_rel ≈ 0.418, then decreases. The classical control shows monotonic MC increase with stronger driving.

---

### Experiment 2d — Closed-Loop Spike-Triggered ENAQT

**Goal**: Use real-time spike feedback to create a true closed-loop ENAQT reservoir.

**Method**:
1. Run at 25 kHz tick rate
2. When a spike is detected on an output channel, stim input channels with ENAQT-gated probability
3. This creates a biologically realistic feedback loop where neural activity drives its own inputs through quantum-modulated gating

```python
with cl.open() as neurons:
    recording = neurons.record()
    for tick in neurons.loop(ticks_per_second=25000, stop_after_seconds=10):
        for spike in tick.analysis.spikes:
            # Spike-triggered ENAQT-gated feedback
            if np.random.random() < p_rel:
                neurons.stim(spike.channel, stim)
    recording.stop()
```

---

## Transition to Real Hardware

Because the CL SDK simulator API is identical to the CL1 API:

| Phase | Platform | Change Required |
|-------|----------|-----------------|
| 2 (now) | CL SDK simulator | None — develop and test locally |
| 3 (future) | CL1 hardware | Remove `pip install cl-sdk`, run on CL1 device |

The only difference is that the simulator replays recordings or generates Poisson spikes, while the CL1 uses actual living neurons. All code is forward-compatible.

---

## Dependencies

```
cl-sdk          # CL API simulator
numpy
scipy
matplotlib
h5py            # HDF5 recording analysis
scikit-learn    # Ridge regression for readout
```

---

## Expected Outcomes

| Experiment | Hypothesis | If confirmed |
|---|---|---|
| **2a** | BNN reservoir has measurable MC | CL SDK is viable as a reservoir computing platform |
| **2b** | MC co-peaks with ENAQT curve | Phase 1 bridge holds for biological dynamics, not just ESN |
| **2c** | Non-monotonic MC vs temperature | Falsifiable prediction validated computationally |
| **2d** | Closed-loop ENAQT feedback works | Biologically realistic quantum-classical reservoir |

---

## Timeline Estimate

| Task | Effort |
|------|--------|
| Install CL SDK + familiarisation | 1 day |
| Experiment 2a (baseline) | 1–2 days |
| Experiment 2b (ENAQT-gated) | 2–3 days |
| Experiment 2c (temperature sweep) | 1–2 days |
| Experiment 2d (closed-loop) | 2–3 days |
| Analysis + dashboard figures | 1–2 days |
| **Total** | **~2 weeks** |
