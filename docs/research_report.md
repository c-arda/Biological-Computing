# Biological Computing — State of the Art Report

*Last updated: 2026-03-10*

---

## 1  Executive Summary

Biological computing — growing living neurons on silicon chips and using them to process information — has moved from laboratory curiosity to early commercial reality. In 2025 the first code-deployable biological computer shipped. By early 2026 brain organoids solved real-time control benchmarks and played 3D video games. The approach promises **orders-of-magnitude gains in energy efficiency** over silicon AI, but remains at a pre-production maturity level with fundamental open questions about scalability, longevity, and reproducibility.

This report covers:
- What has been demonstrated (Section 2)
- How computation is performed (Section 3)
- Available programming interfaces (Section 4)
- Simulation approaches (Section 5)
- Energy efficiency (Section 6)
- Quantum biology considerations (Section 7)
- What we can do today (Section 8)

---

## 2  Current Landscape

### 2.1  Cortical Labs (Melbourne, Australia)

| Attribute | Detail |
|---|---|
| **Product** | CL1 — first commercial biological computer |
| **Neurons** | ~800,000 human neurons (iPSC-derived) on custom silicon chip |
| **Interface** | Multi-Electrode Array (MEA) with real-time bidirectional I/O |
| **Lifespan** | Up to 6 months per culture with internal life-support |
| **OS** | biOS (Biological Intelligence Operating System) |
| **Cloud** | Cortical Cloud — Wetware-as-a-Service (WaaS), ~USD 300/week |
| **SDK** | Python API; GitHub repos `cl-api-doc`, `cl-sdk` |
| **Data centres** | Melbourne + Singapore (NUS partnership) |

**Key milestones:**
- **2022** — DishBrain learns to play Pong with higher sample efficiency than deep RL
- **2025 Q2** — CL1 commercial shipments begin
- **2026 Mar** — CL1 neurons learn to play Doom (3D, complex input space)

### 2.2  FinalSpark (Vevey, Switzerland)

| Attribute | Detail |
|---|---|
| **Product** | Neuroplatform — online research platform |
| **Organoids** | 16 human brain organoids on MEAs |
| **Lifespan** | ~100 days per experiment cycle |
| **Power** | Claims **1,000,000× less** than digital processors |
| **Access** | Monthly subscription for researchers/industry |
| **Architecture** | MEA + microfluidic life-support |

### 2.3  Academic Initiatives

| Group | Achievement (2024–2026) |
|---|---|
| **Johns Hopkins OI** | Demonstrated learning & memory building blocks in human brain organoids |
| **UC Santa Cruz** | Organoids solved cart-pole control benchmark in real-time (2026) |
| **Various** | Synaptic plasticity in thalamocortical assembloids; developmental neurotoxicity testing; preconfigured neuronal firing sequences |

### 2.4  Classification: OI vs BI

- **Organoid Intelligence (OI)** — *top-down*: grow miniature brains that mimic natural brain structures
- **Bioengineered Intelligence (BI)** — *bottom-up*: construct custom neural circuits from scratch for specific tasks (more controllable, task-optimised)

---

## 3  How Computation Works

### 3.1  The Hardware Stack

```
┌─────────────────────────────────────────────┐
│  Software Layer (biOS / experiment control)  │
├─────────────────────────────────────────────┤
│  Interface Layer (MEA - electrodes)          │
│  • Stimulation: electrical patterns → neurons│
│  • Recording: neuronal spikes → digital data │
├─────────────────────────────────────────────┤
│  Biological Layer (neurons on chip)          │
│  • iPSC-derived human neurons                │
│  • 2D cultures or 3D organoids               │
│  • Synaptic connections form spontaneously   │
├─────────────────────────────────────────────┤
│  Life Support (microfluidics)                │
│  • Temperature control (37°C)                │
│  • Nutrient media exchange                   │
│  • CO₂ atmosphere                            │
└─────────────────────────────────────────────┘
```

### 3.2  The Computation Loop

1. **Encode** — Map external data (game pixels, sensor readings) to electrical stimulation patterns across MEA electrodes
2. **Process** — Neurons integrate inputs, form/strengthen synaptic connections, and generate output spike patterns
3. **Decode** — Read spike activity from recording electrodes and map to actions (game controls, motor commands)
4. **Feedback** — Provide reward/error signals back to the culture (e.g., predictable vs unpredictable stimulation) to drive learning

In the DishBrain/Pong experiment:
- Ball position → spatially encoded voltage pulses
- Neuronal output activity → paddle movement
- **Reward**: predictable stimulation when the ball is hit
- **Penalty**: random noise when the ball is missed (violates the culture's "preference" for predictability — the Free Energy Principle)

### 3.3  What Makes Neurons Compute?

| Property | Silicon | Biological Neurons |
|---|---|---|
| **Signal type** | Binary voltage levels | Graded potentials + discrete spikes |
| **Connectivity** | Fixed wiring (HDL-defined) | Self-organising, plastic synapses |
| **Learning** | Gradient descent on parameters | Hebbian plasticity, STDP, homeostasis |
| **Parallelism** | Explicit pipeline/SIMD | Massive implicit parallelism (~10⁴ synapses/neuron) |
| **Power** | ~300 W per GPU | ~20 W for an entire human brain |
| **Adaptation** | Requires retraining | Continuous online adaptation |

---

## 4  Programming Interfaces

### 4.1  Cortical Labs Ecosystem

```python
# Conceptual — Cortical Cloud Python SDK usage
from cortical_cloud import CorticalClient

client = CorticalClient(api_key="your-key")
session = client.create_session(chip_id="CL1-unit-42")

# Stimulate neurons
session.stimulate(
    electrodes=[1, 5, 12, 23],
    waveform="biphasic",
    amplitude_uV=200,
    frequency_Hz=10
)

# Record neural activity
spikes = session.record(duration_ms=1000)
print(f"Recorded {len(spikes)} spikes across {spikes.channels} channels")

# Close-loop feedback
for step in range(1000):
    state = environment.get_state()
    session.stimulate(encode(state))
    spikes = session.record(duration_ms=50)
    action = decode(spikes)
    reward = environment.step(action)
    session.provide_feedback(reward)
```

**Available now:**
- GitHub: `cortical-labs/cl-api-doc` (API documentation)
- GitHub: `cortical-labs/cl-sdk` (CL API Simulator — run experiments without physical hardware)
- Cortical Cloud: browser-based Jupyter + Python SDK

### 4.2  FinalSpark Neuroplatform

- REST API for remote organoid access
- Real-time stimulation and recording
- Monthly subscription model

### 4.3  Open-Source Simulators

| Tool | Best For | Language | Scale |
|---|---|---|---|
| **BRIAN2** | Flexible spiking networks, custom equations | Python | Small–Medium |
| **NEST** | Large-scale networks, HPC clusters | Python/SLI | Large |
| **NEURON** | Detailed biophysical models, morphology | Python/HOC | Single cell–Medium |
| **NetPyNE** | Network building on NEURON | Python | Medium |

---

## 5  Simulation: What We Can Do Locally

### 5.1  BRIAN2 — Recommended Entry Point

BRIAN2 is ideal for our purposes because:
- Pure Python, easy to install (`pip install brian2`)
- Mathematically transparent — you write the differential equations directly
- GPU acceleration available via `brian2cuda`
- The same Leaky Integrate-and-Fire (LIF) and Hodgkin-Huxley models used to describe real neurons in organoids

**See** `simulations/brian2_spiking_demo.py` for a runnable example.

### 5.2  What Can Be Simulated

1. **Spiking Neural Networks** — LIF, AdEx, Hodgkin-Huxley neuron models
2. **Synaptic Plasticity** — STDP (Spike-Timing Dependent Plasticity), Hebbian learning
3. **Reservoir Computing** — random recurrent networks where only the readout layer is trained (most similar to organoid computing)
4. **DishBrain-like Experiments** — simulate the closed-loop feedback paradigm with a virtual environment
5. **Network Formation** — simulate how cultures self-organise connectivity

### 5.3  From Simulation to Real Hardware

```
Simulation (BRIAN2)          →  Software validation
Cortical Labs cl-sdk         →  API-compatible mock hardware
Cortical Cloud               →  Real neurons, remote access
CL1 hardware                 →  Full on-premise biological computer
```

---

## 6  Energy Efficiency

| System | Power | Notes |
|---|---|---|
| Human brain | ~20 W | 86 billion neurons, 100 trillion synapses |
| GPT-4 training | ~50 MW (est.) | ~25,000 GPUs for months |
| NVIDIA H100 GPU | ~700 W | Single chip |
| CL1 (800K neurons) | ~mW range | Life-support dominates |
| FinalSpark organoid | Claims 1M× less | Per compute operation, not total system |

The energy advantage is real but nuanced:
- **Per-operation**: neurons are extraordinarily efficient (~10 fJ per synaptic event)
- **Total system**: life-support (temperature control, media exchange, CO₂) adds overhead
- **Scale**: current systems have 10⁵–10⁶ neurons vs brain's 10¹¹
- **Speed**: biological timescales (ms) are slower than silicon (ns), but the massive parallelism compensates

---

## 7  Quantum Biology Considerations

### 7.1  The Core Question

> Can quantum information (superposition, entanglement) be extracted from or exploited in a neuronal petri dish?

**Short answer: Almost certainly not with current technology, and likely not in principle for computation.**

### 7.2  The Decoherence Problem

The neuronal environment is:
- **Warm**: 37°C (k_B T ≈ 26 meV — thermal noise overwhelms typical quantum energy scales)
- **Wet**: aqueous ionic solution (constant molecular collisions)
- **Dense**: ~10⁴ molecules per nm³ in cytoplasm

Estimated decoherence times in neural tissue:

| System | Decoherence Time | Required for Computation |
|---|---|---|
| Superposition of ion positions | ~10⁻¹³ s | ≫10⁻³ s for neural processing |
| Entanglement across synapses | ~10⁻²⁰ s | ≫10⁻³ s |
| Microtubule quantum states (Orch-OR) | Debated: 10⁻¹³ to 25 ms | Controversial |

For context, superconducting quantum computers operate at **15 mK** (millikelvin) — roughly **20,000× colder** than a petri dish.

### 7.3  Possible Exceptions

**Posner Molecules (Ca₉(PO₄)₆)**
- Phosphorus nuclear spins in Posner clusters may be shielded from decoherence
- Proposed by Matthew Fisher (UCSB) as quantum memory in biology
- Coherence times potentially hours to days (if the theory is correct)
- Brain organoids are being used to test this hypothesis experimentally

**Proton Tunnelling**
- Hydrogen bond dynamics in enzyme active sites show quantum tunnelling
- Relevant for metabolism, not computation per se

**Noise-Assisted Transport**
- Counter-intuitively, thermal noise may *help* maintain certain quantum coherences
- Demonstrated in photosynthesis (FMO complex)
- No evidence this applies to neuronal computation

### 7.4  Practical Assessment for Our Work

| Approach | Feasibility | Notes |
|---|---|---|
| Quantum computing with organoids | ❌ Not feasible | Decoherence far too rapid |
| Quantum sensing of neural activity | ⚠️ Emerging | NV-centre magnetometry can detect ion currents |
| Quantum simulation of neural models | ✅ Possible | Use quantum computer to simulate neuron physics |
| Classical organoid computing | ✅ Available now | CL1, FinalSpark |

**Recommendation**: Focus on classical biological computing. Quantum effects are a fascinating theoretical question but not a practical avenue for organoid intelligence work.

---

## 8  What We Can Do Right Now

### 8.1  Immediate (No Hardware Required)

1. **Run spiking network simulations** with BRIAN2 (see `simulations/`)
2. **Experiment with the CL SDK simulator** — `cortical-labs/cl-sdk` on GitHub
3. **Study reservoir computing** — the computational paradigm closest to organoid computing
4. **Read the DishBrain paper**: Kagan et al. (2022) "In vitro neurons learn and exhibit sentience when embodied in a simulated game-world" — *Neuron*

### 8.2  With Modest Investment

1. **Cortical Cloud access** (~USD 300/week) — remote access to real neurons via Python SDK
2. **FinalSpark Neuroplatform** — monthly subscription for organoid research
3. **Build a MEA interface** — open-source hardware projects exist for recording from neural cultures

### 8.3  Research Directions

1. **Reservoir computing benchmarks** — compare organoid-inspired architectures vs echo state networks
2. **Closed-loop learning** — implement the Free Energy Principle feedback paradigm in simulation
3. **Scaling laws** — how does computational capability scale with neuron count?
4. **Hybrid systems** — combine silicon AI with biological co-processors
5. **Longevity** — can cultures be maintained beyond 6 months?

---

## References

- Kagan, B.J., et al. (2022). "In vitro neurons learn and exhibit sentience when embodied in a simulated game-world." *Neuron*, 110(23), 3952–3969.
- Smirnova, L., et al. (2023). "Organoid intelligence (OI): the new frontier in biocomputing and intelligence-in-a-dish." *Frontiers in Science*, 1:1017235.
- Cortical Labs — [corticallabs.com](https://corticallabs.com)
- FinalSpark — [finalspark.com](https://finalspark.com)
- Organoid Intelligence — [organoidintelligence.org](https://organoidintelligence.org)
- BRIAN2 — [brian2.readthedocs.io](https://brian2.readthedocs.io)
- NEST — [nest-simulator.org](https://nest-simulator.org)
- Fisher, M.P.A. (2015). "Quantum cognition: The possibility of processing with nuclear spins in the brain." *Annals of Physics*, 362, 593–602.
- Tegmark, M. (2000). "Importance of quantum decoherence in brain processes." *Physical Review E*, 61(4), 4194.
