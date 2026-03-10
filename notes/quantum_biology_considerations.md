# Quantum Biology Considerations for Organoid Computing

*Working notes — 2026-03-10*

---

## The Central Question

Can we extract **quantum information** (superposition, entanglement, coherence) from neurons in a petri dish, or use quantum effects to enhance organoid computing?

## Why It's Hard

### The Environment

| Parameter | Petri Dish | Superconducting QC | Required Ratio |
|---|---|---|---|
| **Temperature** | 310 K (37°C) | 0.015 K (15 mK) | ~20,000× colder |
| **Medium** | Aqueous ionic solution | Ultra-high vacuum | Fundamentally different |
| **Density** | ~10⁴ molecules/nm³ | ~10⁻¹⁰ molecules/nm³ | ~10¹⁴× less dense |

### Decoherence Timescales

The warm, wet, dense environment causes quantum states to decohere (lose their quantumness) extremely fast:

- **Ion superposition**: ~10⁻¹³ seconds (0.1 picoseconds)
- **Cross-synaptic entanglement**: ~10⁻²⁰ seconds (0.01 attoseconds)
- **Neural processing time**: ~10⁻³ seconds (1 millisecond)

**Gap**: 10 orders of magnitude between decoherence and processing.
This is why Max Tegmark (2000) concluded that "the brain is already too warm for quantum computing."

## Possible Exceptions

### 1. Posner Molecules — Ca₉(PO₄)₆

- **Proposed by**: Matthew Fisher (UCSB, 2015)
- **Mechanism**: ³¹P nuclear spins in Posner clusters may be magnetically isolated from the thermal environment
- **Predicted coherence time**: Hours to days (!)
- **Status**: Experimentally unconfirmed; some labs using brain organoids to test
- **Relevance**: If real, these would be naturally occurring quantum memory elements in neural tissue

### 2. Proton Tunnelling in Enzymes

- Hydrogen bonds in enzyme active sites show evidence of quantum tunnelling
- Affects metabolic rates, not information processing
- Experimentally established but not relevant to computation

### 3. Orch-OR (Penrose–Hameroff)

- **Claim**: Quantum computation occurs in microtubules inside neurons
- **Mechanism**: Tubulin dimers exist in quantum superposition; gravitational self-energy triggers "objective reduction" (consciousness)
- **Scientific consensus**: Largely rejected — decoherence times calculated by Tegmark are far too short
- **However**: Some anomalous anaesthetic sensitivity data remains unexplained

### 4. Noise-Assisted Quantum Transport

- In photosynthetic complexes (FMO), thermal noise *helps* maintain coherence
- This is real and experimentally verified
- **Not demonstrated** in neural tissue

## What IS Possible: Quantum Sensing of Neural Activity

Instead of using quantum effects *inside* the neurons, we can use quantum sensors *outside* to measure neural activity with unprecedented sensitivity:

| Sensor | Measurement | Sensitivity |
|---|---|---|
| **NV-centre magnetometry** | Magnetic fields from ion currents | Single-neuron action potentials |
| **Quantum SQUID arrays** | Magnetic fields from neural ensembles | Sub-fT resolution |
| **Entangled photon microscopy** | Fluorescence imaging | Sub-diffraction limit |

This is the more practical intersection of quantum technology and organoid computing.

## Summary Table

| Approach | Feasible? | Timescale |
|---|---|---|
| Quantum computation in organoids | ❌ No | Not foreseeable |
| Quantum memory via Posner molecules | ⚠️ Speculative | Years to validate |
| Orch-OR consciousness in organoids | ❌ Rejected by most physicists | — |
| Quantum sensing of organoid activity | ✅ Yes | Available now (lab scale) |
| Quantum simulation of neuron models | ✅ Yes | Available now (NISQ devices) |
| Classical organoid computing | ✅ Yes | Commercial now (CL1) |

## Key References

1. Tegmark, M. (2000). "Importance of quantum decoherence in brain processes." *Phys. Rev. E*, 61(4), 4194.
2. Fisher, M.P.A. (2015). "Quantum cognition: The possibility of processing with nuclear spins in the brain." *Annals of Physics*, 362, 593–602.
3. Hameroff, S. & Penrose, R. (2014). "Consciousness in the universe: A review of the 'Orch OR' theory." *Physics of Life Reviews*, 11(1), 39–78.
4. Cao, J. et al. (2020). "Quantum biology revisited." *Science Advances*, 6(14).
5. Lambert, N. et al. (2013). "Quantum biology." *Nature Physics*, 9, 10–18.
