# Quantum Biology in Biological Computing — Deep Analysis

*Last updated: 2026-03-10*

---

## 1  The Central Thesis

Biological systems don't build quantum computers. They do something potentially more elegant: **they fold matter into geometries that sustain quantum effects just long enough to be useful**, then discard the coherence. Biology treats quantumness as a *transient computational tool*, not a persistent state.

This document explores:
- Confirmed quantum effects in biology (Section 2)
- The folding/pocket protection mechanism (Section 3)
- Experimental plan to probe quantum effects in organoids (Section 4)
- The quantum branching question (Section 5)
- Open questions and research directions (Section 6)

---

## 2  Confirmed Quantum Biology — The Evidence Is Real

These are no longer speculative. They are **experimentally verified**:

### 2.1  Photosynthesis — Quantum Coherent Energy Transfer

| Property | Detail |
|---|---|
| **System** | FMO complex in green sulfur bacteria (*Chlorobaculum tepidum*) |
| **Quantum effect** | Superposition of exciton states across multiple chromophores |
| **Duration** | ~600 femtoseconds at 277 K (near room temperature) |
| **Efficiency** | ~99% energy transfer from antenna to reaction centre |
| **Mechanism** | Protein scaffold positions chromophores at precise distances and angles; coherence allows energy to explore all pathways simultaneously |
| **Key paper** | Engel et al. (2007) *Nature* 446, 782–786 |

The FMO complex is the **gold standard** for quantum biology. The protein acts as a **quantum-coherent waveguide** — not by isolating from the environment, but by **using environmental noise constructively** (noise-assisted transport / ENAQT).

### 2.2  Avian Magnetoreception — Quantum Entanglement for Navigation

| Property | Detail |
|---|---|
| **System** | Cryptochrome proteins (CRY4) in European robin retina |
| **Quantum effect** | Radical pair mechanism — entangled electron spins |
| **Duration** | ~1–100 microseconds |
| **Function** | Detects Earth's magnetic field direction for seasonal migration |
| **Mechanism** | Blue photon creates singlet radical pair → spin precession in magnetic field → triplet/singlet ratio encodes field direction → protein fold maintains spin coherence |
| **Key paper** | Xu et al. (2021) *Nature* 594, 535–540 |

This is remarkable: the protein fold creates a **cavity** where entangled electron spins survive for microseconds at body temperature (41°C in birds). The geometry of the tryptophan chain acts as a quantum wire, and the surrounding protein shield reduces decoherence by ~1,000× compared to the same radical pair in aqueous solution.

### 2.3  Enzyme Catalysis — Proton Tunnelling

| Property | Detail |
|---|---|
| **System** | Alcohol dehydrogenase, aromatic amine dehydrogenase, many others |
| **Quantum effect** | Hydrogen/proton quantum tunnelling through energy barriers |
| **Evidence** | Anomalous kinetic isotope effects (H/D ratios >7, classical limit ~7) |
| **Function** | Accelerates reaction rates beyond classical transition state theory |
| **Mechanism** | Active site geometry compresses donor-acceptor distance to ~2.7 Å, enabling wavefunction overlap |
| **Key paper** | Scrutton et al. (2012) *Nature Chemistry* 4, 161–168 |

### 2.4  Olfaction — Quantum Vibration Sensing (Debated)

| Property | Detail |
|---|---|
| **System** | Odorant receptors in nasal epithelium |
| **Quantum effect** | Inelastic electron tunnelling spectroscopy (IETS) |
| **Claim** | Receptors distinguish molecular vibration frequencies, not just shapes |
| **Status** | **Controversial** — some experiments support, others contradict |
| **Key paper** | Turin (1996); Franco et al. (2011) *PNAS* 108, 3797 |

---

## 3  The Folding / Pocket Protection Mechanism

### 3.1  The Universal Pattern

Across all confirmed quantum biology systems, the same architectural pattern emerges:

```
┌─────────────────────────────────────────────────────┐
│  Warm, wet, noisy environment (300+ K)              │
│                                                      │
│    ┌──────────────────────────────────┐              │
│    │  PROTEIN FOLD / MOLECULAR CAGE   │              │
│    │                                  │              │
│    │  ● Precise atomic geometry       │              │
│    │  ● Ordered water exclusion       │              │
│    │  ● Vibrational mode coupling     │              │
│    │  ● Structured noise (ENAQT)      │              │
│    │                                  │              │
│    │    ┌─────────────────────┐       │              │
│    │    │  QUANTUM COHERENT   │       │              │
│    │    │  DOMAIN             │       │              │
│    │    │                     │       │              │
│    │    │  fs → µs lifetime   │       │              │
│    │    └─────────────────────┘       │              │
│    │                                  │              │
│    └──────────────────────────────────┘              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 3.2  Protection Strategies Biology Uses

| Strategy | Where | How |
|---|---|---|
| **Geometric confinement** | FMO, enzymes | Precise inter-chromophore distances maintained by rigid protein backbone |
| **Water exclusion** | Cryptochrome, microtubules | Hydrophobic pockets expel bulk water, removing the primary decoherence source |
| **Noise tuning (ENAQT)** | FMO | Environmental vibrations tuned to *assist* rather than destroy coherence |
| **Spin isolation** | Cryptochrome, Posner | Nuclear spins weakly coupled to environment; protein geometry minimises magnetic noise |
| **Topological protection** | Microtubules? | Hollow tube geometry may create protected modes (theoretical) |

### 3.3  Relevance to Neurons

In neurons, candidate structures for quantum pockets:

| Structure | Size | Quantum Candidate | Protection Mechanism |
|---|---|---|---|
| **Microtubules** | 25 nm diameter, µm length | Tubulin superposition (Orch-OR) | Hollow interior, ordered water shell, Debye screening |
| **Synaptic cleft** | ~20 nm | Neurotransmitter tunnelling | Confined geometry between pre/post-synaptic membranes |
| **Ion channels** | ~0.5 nm selectivity filter | Ion quantum superposition | Angstrom-scale confinement, single-file ion transit |
| **Posner molecules** | ~1 nm cages | ³¹P nuclear spin entanglement | Phosphate cage shields nuclear spins from electric fields |
| **Mitochondrial complexes** | ~10 nm | Electron tunnelling in respiratory chain | Protein scaffold, iron-sulfur cluster geometry |

---

## 4  Experimental Plan — Fully Virtual / Remote

> [!IMPORTANT]
> This plan is designed for a **private researcher without lab access**.
> All experiments use remote wetware platforms, computational simulation,
> or publicly available datasets. No BSL-2 lab, no culture facility needed.

### 4.1  Where We Are (TRL Assessment)

| Aspect | Technology Readiness | Our Access |
|---|---|---|
| Growing brain organoids | **TRL 7–8** — commercial | ✅ Via Cortical Cloud / FinalSpark |
| MEA recording from organoids | **TRL 7** — standard | ✅ Via remote platforms (data download) |
| Quantum sensing of neural activity | **TRL 3–4** — lab demos | ❌ Requires physical lab |
| Detecting quantum coherence in neural tissue | **TRL 1–2** — theoretical | ✅ Computational modelling |
| Manipulating quantum states in neurons | **TRL 1** — theoretical | ✅ Simulation only |

### 4.2  Proposed Experimental Programme

#### Phase 1 — Computational Modelling (Now)

**Objective**: Model whether quantum coherence in microtubules/Posner molecules could influence spiking dynamics.

**Status**: *We can start immediately — all tools available.*

| Experiment | Tool | What We Learn |
|---|---|---|
| **1a** LIF network with quantum-modified synaptic delays | BRIAN2 | Does a tunnelling-delay distribution change network computation vs classical fixed delays? |
| **1b** Posner molecule spin dynamics | QuTiP | Can ³¹P nuclear spin coherence survive ms-scale in a thermal bath model? |
| **1c** ENAQT in a model ion channel | QuTiP / Qiskit | Could noise-assisted quantum transport apply to K⁺/Na⁺ permeation? |
| **1d** Reservoir computing: quantum vs classical noise | BRIAN2 | Does quantum-scale noise injection improve computational capacity of a reservoir? |
| **1e** Cryptochrome radical pair in neuron context | QuTiP | Model radical pair spin dynamics at 37°C with realistic dephasing |

**Requirements**: Python, BRIAN2, QuTiP, NumPy, matplotlib. ✅ *Installed in project venv.*

Additional installs needed:
```bash
pip install qutip qiskit
```

#### Phase 2 — Remote Wetware Access (1–3 months)

**Objective**: Run experiments on real neurons without owning a lab.

| Platform | Access | Cost | What You Get |
|---|---|---|---|
| **Cortical Cloud** (Cortical Labs) | Python SDK, Jupyter in browser | ~USD 300/week | Stimulate & record from ~800K human neurons on CL1 hardware |
| **FinalSpark Neuroplatform** | REST API, web interface | Monthly subscription | Access to 16 brain organoids on MEAs |
| **CL SDK Simulator** | GitHub (free) | Free | Offline API-compatible mock — test code before paying for cloud time |

**Recommended workflow:**
1. Develop experiment scripts locally using the **CL SDK Simulator** (free)
2. Validate the pipeline end-to-end with simulated data
3. Book **Cortical Cloud** time (1–2 weeks initially) to run on real neurons
4. Download spike data for offline analysis

**Virtual experiments we can run remotely:**

| Experiment | Platform | What We Measure |
|---|---|---|
| **2a** Baseline spiking statistics | Cortical Cloud | Firing rates, burst patterns, inter-spike intervals — establish what "normal" looks like |
| **2b** Stimulation-response mapping | Cortical Cloud | How do neurons respond to different electrode patterns? Build input/output transfer functions |
| **2c** Reservoir computing benchmark | Cortical Cloud | Can the biological culture solve NARMA-10 or wave-form classification? Compare to our BRIAN2 simulation |
| **2d** Temporal learning test | Cortical Cloud | Free Energy Principle feedback — does the culture learn to minimise prediction error (like DishBrain/Pong)? |
| **2e** Multi-session plasticity | FinalSpark | Do organoids retain learned patterns across days? (FinalSpark cultures last ~100 days) |

#### Phase 3 — Computational Quantum Probes (3–12 months)

**Objective**: Since we can't physically measure quantum effects in organoids, we probe computationally — test whether quantum-modified models better explain *observed* organoid data than purely classical models.

| Experiment | Method | What We Learn |
|---|---|---|
| **3a** Classical vs quantum-delay model fitting | Fit BRIAN2 models (classical synaptic delays) and quantum-modified models (tunnelling delay distributions) to the real Cortical Cloud data from Phase 2. Compare goodness-of-fit. | If quantum-delay models fit better → indirect evidence for quantum contributions |
| **3b** Analyse published D₂O neuronal data | Search literature for existing heavy water experiments on neurons (they exist from anaesthesia research). Re-analyse for anomalous KIE signatures. | Leverage existing data — no lab needed |
| **3c** Posner molecule population model | Simulate Ca²⁺/phosphate dynamics in a neural context. Predict: if Posner entanglement is real, what statistical signature would appear in calcium waves? | Generate falsifiable predictions for future lab tests |
| **3d** Information-theoretic analysis of organismal data | Apply transfer entropy & mutual information measures to Cortical Cloud spike trains. Compare to predictions from quantum vs classical stochastic models. | Do real neurons carry more information than classical stochastic models predict? |
| **3e** Radical pair magnetometry simulation | Model whether Earth-strength magnetic fields (~50 µT) could modulate neural cryptochrome activity. Compare model predictions with any publicly available data. | Sets up the theoretical framework even without field apparatus |
| **3f** Quantum walk on microtubule lattice | Simulate quantum vs classical random walk on a 2D microtubule lattice (13 protofilaments × length). Compare transport efficiency. | Does the microtubule geometry favour quantum transport like FMO chromophore geometry? |

**Key insight**: We're not proving quantum effects exist in neurons (that requires a lab). We're building **predictive models** that distinguish quantum from classical contributions, so that when lab data becomes available, we already know what to look for.

#### Phase 4 — Publication & Collaboration (12+ months)

**Objective**: Contribute findings to the community; partner with labs that have physical access.

| Activity | How |
|---|---|
| **Preprint on arXiv/bioRxiv** | Publish computational results from Phases 1–3 |
| **Open-source simulation toolkit** | Release our BRIAN2/QuTiP models as a Python package for others to use |
| **Collaborate with wet labs** | Contact groups running organoid experiments (JHU OI, Cortical Labs, UCSC) — offer computational support in exchange for data access |
| **Join Organoid Intelligence community** | [organoidintelligence.org](https://organoidintelligence.org) — connect with researchers |
| **Propose experiments** | Write up Phase 3 predictions as testable hypotheses that wet labs can run (D₂O, ⁶Li/⁷Li, magnetic field experiments) |

### 4.3  Budget — What We Actually Need

**Total estimated cost for a private researcher:**

| Category | Item | Cost |
|---|---|---|
| **Software** | BRIAN2, QuTiP, Qiskit, Python | **Free** (open source) |
| **Hardware** | Existing compute cluster | **Already owned** |
| **Remote wetware** | CL SDK Simulator | **Free** (GitHub) |
| **Remote wetware** | Cortical Cloud (2 weeks initial) | ~€550 |
| **Remote wetware** | FinalSpark (1 month) | ~€500–1,500 (estimate) |
| **Literature** | Sci-Hub / arXiv / PubMed (open access) | **Free** |
| **Publication** | arXiv / bioRxiv preprint | **Free** |
| **Total** | | **€550 – €2,100** |

This is remarkably accessible. Five years ago this research would have required a €200K+ lab. Today the remote platforms make it possible from a home office.

---

## 5  The Quantum Branching Question

### 5.1  Are We Biological Machines in a Quantum Outcome?

You're asking one of the deepest questions in physics, and you're **not** mixing apples with bananas. You're touching the intersection of:

1. **Many-Worlds Interpretation (MWI)** of quantum mechanics
2. **Quantum Darwinism** (Zurek)
3. **Decoherent histories** (Griffiths, Gell-Mann, Hartle)
4. **Biological decision-making** as branch selection

Let me unpack this carefully.

### 5.2  The Standard View: Decoherence Selects a Branch

In the Many-Worlds picture, every quantum measurement causes the universal wavefunction to branch. But we never *experience* the branching — decoherence makes the branches mutually inaccessible almost instantly. The environment **selects** which branch we observe by interacting with the quantum system.

```
Quantum state: |ψ⟩ = α|A⟩ + β|B⟩

           decoherence
     |ψ⟩ ──────────────→  Branch A  (we experience this)
                           Branch B  (inaccessible to us)
```

For macroscopic biology, decoherence is so fast (~10⁻²⁰ s for a neuron) that we're always already "on a branch." Normal neural computation is fully classical.

### 5.3  Your Insight: Biology on the Quantum Ridge

What you're proposing is more subtle and interesting:

> What if biological systems don't compute ON a branch, but compute BY riding the ridge BETWEEN branches — using the quantum regime just before decoherence locks them in?

This is actually close to real physics. Here's the mapping:

| Your Concept | Physics Framework | Status |
|---|---|---|
| "Quantum ridge" | **Quantum criticality** — operating at the edge between quantum and classical | Active research area |
| "Pre-value of paths" | **Feynman path integral** — all paths contribute before measurement | Foundation of QM |
| "Once locked onto it, it runs" | **Decoherence / wavefunction branching** | Standard physics |
| "Computing the quantum line" | **Quantum coherent processing** before classical collapse | Exactly what FMO does |

The FMO complex literally does what you describe: the exciton explores **all energy transfer pathways simultaneously** (the "pre-value of paths"), and the protein fold keeps the quantum superposition alive just long enough for the optimal path to emerge, then decoherence "locks in" that path and the energy flows classically.

### 5.4  Could Neurons Do This?

The speculative but coherent picture:

```
Signal arrives at neuron
        │
        ▼
┌─────────────────────────────────────┐
│ QUANTUM EXPLORATION WINDOW          │
│ (fs to µs — inside protein folds)   │
│                                     │
│ Multiple states explored            │
│ simultaneously:                     │
│   • Ion channel conformations       │
│   • Neurotransmitter binding modes  │
│   • Calcium pathway options         │
│   • Microtubule processing?         │
│                                     │
│ Quantum coherence explores pathways │
│ → optimal route "selected"          │
└─────────────────────────────────────┘
        │
        ▼ decoherence collapses to one outcome
        │
Classical spike / no-spike decision
        │
        ▼
Network-level computation (classical)
```

In this model:
- **Quantum effects** provide the **micro-level optimisation** (which ion goes through which channel, how fast neurotransmitter binds, which calcium wave propagates)
- **Classical computation** handles the **macro-level network dynamics** (spike patterns, learning, behaviour)
- Biology operates at the **boundary** — the "quantum ridge" — where quantum effects inform classical decisions

This is not Many-Worlds in the cosmological sense. It's more like **quantum decision theory at the molecular scale**: the neuron runs a "quantum annealing" step in its protein machinery before committing to a classical output.

### 5.5  Why This Matters for Biological Computing

If this picture is even partially correct, it has profound implications for organoid computing:

1. **Silicon can't replicate it** — classical simulations of spiking networks miss this entire sub-spike optimisation layer
2. **Energy efficiency explains itself** — the 10⁶× energy advantage of biological computing may come from quantum-coherent shortcuts at the molecular level, not just architectural differences
3. **Organoids might already be doing it** — we just don't have instruments sensitive enough to detect it yet (→ Phase 3 experiments)
4. **True "biological quantum computing"** wouldn't require superconducting qubits — it would require engineering better protein folds

### 5.6  What's Apples and What's Bananas

Where your intuition is **correct**:
- ✅ Quantum effects are leveraged by biology at the micro-level
- ✅ Protein folding/pockets create protective environments
- ✅ There's something like "path exploration before commitment"
- ✅ The bird navigation example is real quantum biology
- ✅ This could matter for understanding biological computing efficiency

Where you need to be **careful**:
- ⚠️ MWI branching at the cosmic/neuronal level is different from molecular quantum effects — don't conflate the scales
- ⚠️ "Computing the quantum line" works at the protein/molecule scale but there's no evidence for neuron-scale superposition
- ⚠️ The FMO example is energy transfer (simple optimisation), not general computation — it's not a quantum computer, it's a quantum funnel

---

## 6  Open Questions — Research Programme

### Fundamental

1. Does quantum coherence in any neural structure persist long enough to influence spike timing?
2. Can the isotope experiments (D₂O, ⁶Li/⁷Li) distinguish quantum from classical contributions?
3. Is the "quantum ridge" a general principle of biological information processing?

### Engineering

4. Can we engineer organoids with enhanced quantum pockets (modified ion channels, increased Posner formation)?
5. Could quantum sensors (NV-centres) detect coherence signatures in real-time during organoid computation?
6. What would a hybrid quantum-biological computer architecture look like?

### Philosophical

7. If neurons use transient quantum coherence for micro-optimisation, does this change our understanding of consciousness?
8. Is the energy efficiency of biological computing fundamentally linked to quantum effects?
9. Are biological systems "quantum-enhanced classical computers" rather than either purely classical or quantum?

---

## Key References

### Confirmed Quantum Biology
1. Engel, G.S. et al. (2007). "Evidence for wavelike energy transfer through quantum coherence in photosynthetic systems." *Nature*, 446, 782–786.
2. Xu, J. et al. (2021). "Magnetic sensitivity of cryptochrome 4 from a migratory songbird." *Nature*, 594, 535–540.
3. Scrutton, N.S. et al. (2012). "Good vibrations in enzyme-catalysed reactions." *Nature Chemistry*, 4, 161–168.

### Neural Quantum Biology
4. Fisher, M.P.A. (2015). "Quantum cognition: The possibility of processing with nuclear spins in the brain." *Annals of Physics*, 362, 593–602.
5. Hameroff, S. & Penrose, R. (2014). "Consciousness in the universe: A review of the 'Orch OR' theory." *Physics of Life Reviews*, 11(1), 39–78.
6. Tegmark, M. (2000). "Importance of quantum decoherence in brain processes." *Physical Review E*, 61(4), 4194.

### Quantum Biology Reviews
7. Lambert, N. et al. (2013). "Quantum biology." *Nature Physics*, 9, 10–18.
8. Cao, J. et al. (2020). "Quantum biology revisited." *Science Advances*, 6(14).
9. Kim, Y. et al. (2021). "Quantum biology: An update and perspective." *Quantum Science and Technology*, 6(2).

### Quantum Branching / Decoherence
10. Zurek, W.H. (2009). "Quantum Darwinism." *Nature Physics*, 5, 181–188.
11. Schlosshauer, M. (2005). "Decoherence, the measurement problem, and interpretations of quantum mechanics." *Reviews of Modern Physics*, 76(4), 1267.
