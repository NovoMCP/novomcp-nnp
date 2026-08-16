# NovoMCP Neural Network Potentials Service

Fast molecular energy and force computation using pre-trained neural network potentials. Orders of magnitude faster than DFT, ~100x faster than xTB for single-point energies.

**Port:** 8032 | **Ingress:** Internal only | **Contacted by:** the NovoMCP engine

---

## Tool

### compute_energy
Compute molecular energy and atomic forces using neural network potentials. Accepts SMILES input, generates 3D geometry via RDKit, runs inference. Returns energy (eV, kcal/mol) and force magnitudes (eV/A).

- **Models available:**
  - `ani2x` — ANI-2x (University of Florida). Covers H, C, N, O, F, S, Cl. Fastest for organic drug-like molecules.
  - `mace` — MACE-MPA-0 (Materials Project + Alexandria universal potential, MIT-licensed). Covers all elements. Broader coverage, slightly slower.
  - `auto` — Selects ANI-2x for organic molecules (H/C/N/O/F/S/Cl only), MACE for anything with other elements.
- **Typical runtime:** 5-50ms per molecule (vs 1-10s for xTB, minutes for DFT)
- **Use cases:**
  - Fast conformer ranking (energy sort without waiting for xTB)
  - Rapid strain energy estimation for docked poses
  - Batch screening energetics (100 molecules in seconds)
  - Comparing potential energy surfaces across model chemistries
- **Credit cost:** 5

---

## Models

### ANI-2x
- **Source:** TorchANI (github.com/aiqm/torchani)
- **Paper:** Smith et al., "The ANI-2x Potential" (2020)
- **Elements:** H, C, N, O, F, S, Cl (covers >95% of drug-like molecules)
- **Architecture:** Ensemble of 8 neural networks with Behler-Parrinello symmetry functions + feed-forward layers
- **Training data:** ~10M DFT calculations (wB97X/6-31G*)
- **Accuracy:** ~1 kcal/mol MAE vs CCSD(T)/CBS for organic molecules
- **Speed:** ~5-20ms per molecule on CPU
- **Limitations:** Only 7 elements supported; no metals, no boron, no iodine/bromine

### MACE-MPA-0
- **Description:** MACE-MPA-0 (Materials Project + Alexandria training set), MIT-licensed medium-size foundation potential from ACEsuit.
- **Source:** MACE foundation models (github.com/ACEsuit/mace-foundations)
- **License:** MIT
- **Paper:** Batatia et al., "A foundation model for atomistic simulations" (2024)
- **Elements:** All elements (universal potential, trained on Materials Project + Alexandria DFT data)
- **Architecture:** Equivariant message passing with higher-order many-body interactions (E(3)-equivariant)
- **Training data:** Materials Project + Alexandria structures (PBE functional)
- **Accuracy:** Good for structures and relative energies; less accurate than ANI-2x for absolute energies of organic molecules
- **Speed:** ~20-100ms per molecule on CPU (medium-size; more accurate but heavier than the earlier small model)
- **Strengths:** Handles any element, metallic systems, inorganic molecules

---

## Architecture

```
the NovoMCP engine (tools.py)
  → _call_service("novomcp-nnp", "/api/compute-energy", {...})
  → novomcp-nnp:8032
```

### Request flow
1. the NovoMCP engine receives `compute_energy` MCP tool call
2. Executor sends HTTP POST to novomcp-nnp `/api/compute-energy`
3. Service converts SMILES → 3D coordinates (RDKit ETKDG + MMFF)
4. Determines element composition; selects model if `auto`
5. Runs PyTorch inference (no subprocess, pure Python)
6. Returns energy (eV, kcal/mol) and force statistics
7. the NovoMCP engine returns result to client

### Model selection logic (`auto` mode)
```
ANI-2x elements = {H, C, N, O, F, S, Cl}
if all atoms in molecule are ANI-2x elements → use ANI-2x (faster, more accurate for organics)
else → use MACE-MPA-0 (universal, handles any element)
```

### Startup sequence
1. Load ANI-2x via TorchANI (ensemble of 8 NNs, ~200MB in memory)
2. Load MACE-MPA-0 via mace_mp("medium-mpa-0") (medium-size checkpoint)
3. Both models run on CPU (no GPU needed for inference)
4. Health check reports which models loaded

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/compute-energy` | POST | Single molecule energy + forces |
| `/api/optimize-geometry` | POST | ASE BFGS geometry optimization (~100x faster than xTB) |
| `/api/batch-energy` | POST | Batch energy (up to 100 molecules) |
| `/api/compare-methods` | POST | Run all models on same molecule, compare results + speed |
| `/health` | GET | Health check with per-model availability |

---

## Speed comparison

| Method | Typical time per molecule | Accuracy (organic) | Elements |
|---|---|---|---|
| ANI-2x (novomcp-nnp) | 5-20ms | ~1 kcal/mol MAE | H/C/N/O/F/S/Cl |
| MACE-MPA-0 (novomcp-nnp) | 20-100ms | ~3-5 kcal/mol MAE | All |
| GFN2-xTB (novomcp-qm) | 1-10s | ~3-5 kcal/mol MAE | All (up to Rn) |
| DFT (B3LYP/6-31G*) | 1-60 min | Reference | All |
| CCSD(T)/CBS | Hours | Gold standard | Light elements |

---

## Relationship to novomcp-qm

novomcp-nnp (this service) and novomcp-qm (xTB/CREST) are complementary:

| Task | Use novomcp-nnp | Use novomcp-qm |
|---|---|---|
| Single-point energy | Fast screening | Accurate reference |
| Conformer ranking | Batch (>20 conformers) | Final ranking (<20) |
| Strain energy | Quick check | Production quality |
| Geometry optimization | ASE BFGS (~100x faster, neutral only) | xTB optimization (charged/open-shell) |
| Solvation free energy | Not supported | xTB ALPB model |
| HOMO-LUMO gap | Not supported | xTB orbital analysis |
| Batch energetics (>50 molecules) | Ideal (~seconds) | Too slow (~minutes) |

---

## Container specs
- **Image:** python:3.11-slim-bullseye + PyTorch 2.5.1 (CPU) + TorchANI + MACE + ASE
- **CPU/Memory:** 4 vCPU / 8Gi
- **Min/Max replicas:** 1 / 10
- **Startup time:** ~30-60s (model loading, first-time weight download cached)

---

## Files

```
novomcp-nnp/
├── main.py                      # FastAPI app, endpoints, SMILES→geometry conversion
├── app/
│   └── models/
│       └── registry.py          # Model loading, inference dispatch, ANI-2x + MACE-MPA-0
├── Dockerfile
├── requirements.txt
└── .github/workflows/deploy-azure.yml
```

---

## Funnel integration

`compute_energy` fits into the discovery funnel as a fast screening step:

1. **Target discovery** → `target_discovery`
2. **Literature search** → `search_literature`
3. **Lead optimization** → `lead_optimization` (scaffold hopping)
4. **Fast energy screening** → `compute_energy` (rank candidates by NNP energy, discard high-energy outliers)
5. **Conformer generation** → `run_conformer_search` (CREST on top candidates)
6. **Docking** → `dock_molecules` (AutoDock-GPU)
7. **Strain correction** → `dock_with_strain` (xTB, filter false positives)
8. **QM refinement** → `run_qm_calculation` (xTB solvation, orbital analysis)
9. **MD simulation** → `run_molecular_dynamics` (GROMACS)
10. **Patient stratification** → `stratify_patients`
