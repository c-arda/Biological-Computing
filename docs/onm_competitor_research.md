# Open Neuromorphic & Competitor Research Notes

**Date:** 2026-03-22

---

## 1. Open Neuromorphic (ONM) Community Peer Review

### Submission Process

| Step | Detail |
|------|--------|
| **Platform** | OpenReview ([openreview.net](https://openreview.net)) |
| **Requirement** | All authors must have an OpenReview profile |
| **Format** | GitHub repo, Jupyter notebooks, whitepapers, or IEEE-formatted papers |
| **Fields** | Title (≤250 chars), Abstract (≤5000 chars), author info |
| **Schedule** | Rolling submissions; ~1 month turnaround |
| **Review** | 3–5 community volunteers |
| **Outcome** | "ONM Community Approved" badge + public registry listing |

### Review Criteria

- **Relevance** to neuromorphic computing
- **Clarity** — organization and readability
- **Reproducibility** — code, data, methods documented
- **Technical Rigor** — sound approach
- **Contribution** — non-trivial or novel
- **Openness** — permissive licensing, open tools
- **Community Value** — reuse and extension potential

### Our Fit

Our Biological-Computing repo meets all criteria:
- ✅ MIT license (permissive)
- ✅ Reproducible (all scripts self-contained, `requirements.txt`)
- ✅ Novel (ENAQT-gated reservoir computing — no one else has done this)
- ✅ Community value (bridges quantum biology and neuromorphic computing)
- ⚠️ Relevance framing needed — our work is more "quantum biology → reservoir computing" than "neuromorphic hardware", but the BNN/CL SDK experiments bring it into neuromorphic territory

### Action Items

1. Create OpenReview profiles
2. Prepare abstract (≤5000 chars) tailored to neuromorphic community
3. Submit via ONR portal

---

## 2. NICE 2026 Conference

### Details

| Field | Value |
|-------|-------|
| **Name** | Neuro Inspired Computational Elements (NICE) |
| **Date** | March 24–27, 2026 (tutorials March 27) |
| **Location** | Georgia Institute of Technology, Atlanta, GA |
| **Proceedings** | IEEE Conference Proceedings (indexed) |
| **Papers** | 4–8 pages (regular), 1–2 pages (WIP/abstract) |
| **Deadline** | Passed — camera-ready was Feb 27, 2026 |

### Relevant Topic Tracks

- Computational and Systems Neuroscience (neural circuits, plasticity)
- Neuromorphic Computing Applications (biosignal processing, BCI)
- Algorithms and Software Frameworks (benchmarks, neuromorphic datasets)
- Bio-Inspired Sensing (event-driven sensing)

### Quantum Biology Gap

NICE 2026 does **not** have an explicit quantum computing or quantum biology track. This means:
- Our work would be novel and distinctive if submitted to NICE 2027
- The "quantum → reservoir computing" bridge is underrepresented in this community
- Potential for a future targeted submission or invited talk

### Action Items

1. Monitor IEEE proceedings when published (April–May 2026)
2. Check for relevant reservoir computing or biological computing papers
3. Consider NICE 2027 submission (CFP expected ~September 2026)
4. Also consider: ICONS (International Conference on Neuromorphic Systems)

---

## 3. SNN Reservoir Computing Benchmarks

### Key Finding: No Direct ESN ↔ SNN-RC Memory Capacity Benchmark Exists

Our work fills a gap — there is no published direct comparison of memory capacity (MC) between:
- Echo-State Networks (ESN) — our Phase 1 (Experiments 1d, 1e)
- Spiking Neural Network reservoirs — our Phase 2 (Experiments 2a–2d)

### Available SNN Reservoir Frameworks

| Framework | Org | Notes |
|-----------|-----|-------|
| **Rockpool** | SynSense | SNN framework with recurrent network support, LIF modules, SynNet architecture. Supports Xylo hardware deployment. Has tutorials for spiking neurons but no dedicated MC benchmark |
| **snnTorch** | Open source | PyTorch SNN training. Leaky, Synaptic neuron models. GPU accelerated. No built-in MC metric |
| **Norse** | Open source | Bio-inspired PyTorch primitives. Good for LIF/AdEx neuron comparisons |
| **Intel Lava** | Intel | For Loihi hardware. Includes process-based SNN models |
| **SpikingJelly** | Open source | PyTorch SNN framework. ANN-to-SNN conversion. Focused on classification, not MC |
| **NEST** | Open source | Large-scale SNN simulator. Can complement Brian2 |

### Relevant Recent Papers

1. **Short-term MC of input-driven ESNs** (2025) — Edge-of-chaos analysis, ESN only, no SNN comparison
2. **Neuromorphic on-chip reservoir computing** (Jul 2024) — Integrate-and-fire neurons, topology impact, no MC vs ESN
3. **Analog spiking neuron for physical RC** (Sep 2024) — Short-term memory tasks on hardware
4. **Reservoir Memory Networks** (2024, ESANN) — Linear memory cell + nonlinear reservoir, compared to ESN
5. **ESN vs LSM for fault detection** (2023, MDPI) — Classification-focused, not MC-specific

### Our Unique Contribution

Our work provides:
1. **First quantum-gated MC benchmark** — using ENAQT P₄ as synaptic release probability
2. **ESN → BNN bridge** — directly comparing MC across abstract (ESN) and biological (CL SDK) reservoirs
3. **Falsifiable predictions** — isotope effect, temperature sweep, magnetic field
4. **Forward-compatible code** — CL SDK simulator → CL1 hardware without modification

### Action Items

1. Add Rockpool/snnTorch citations to manuscript Discussion for context
2. Highlight the MC benchmark gap as a novel contribution
3. Consider implementing a Rockpool-based spiking reservoir to compare MC directly (future work)

---

## 4. Competitor Landscape Summary

| Platform | Type | Status | Relevance |
|----------|------|--------|-----------|
| **Cortical Labs CL1** | Bio (organoid) | Commercial H2 2025 | Primary hardware target |
| **FinalSpark BioBit** | Bio (16 organoids, remote) | Available | Alternative platform (deferred to April 2026) |
| **Brainoware** (Cai et al.) | Bio (organoid RC) | Academic | Direct predecessor |
| **Intel Loihi 2** | Silicon neuromorphic | Available | Spiking chip |
| **BrainChip Akida** | Silicon neuromorphic | Commercial | Edge AI |
| **SynSense Xylo** | Silicon neuromorphic | Commercial | Rockpool target hardware |
| **Koniku** | Hybrid bio-silicon | R&D | DNA-programmed neurons |
| **SpiNNaker / EBRAINS** | Silicon neuromorphic HPC | Available | Large-scale SNN |
| **Open Neuromorphic** | Community + OSS tools | Active | Software ecosystem + peer review |

---

## 5. Recommended Next Actions (Priority Order)

1. **ONM Peer Review** — Create OpenReview profile + submit repo (free, immediate)
2. **Manuscript update** — Add keywords, integrate Phase 3, cite Rockpool/snnTorch context
3. **Monitor NICE 2026 proceedings** — Check IEEE Xplore in April–May 2026
4. **NICE 2027 submission** — Prepare when CFP opens (~September 2026)
5. **FinalSpark access** — Revisit in April 2026 when funding allows
6. **Rockpool MC experiment** — Future work: implement spiking reservoir MC benchmark
