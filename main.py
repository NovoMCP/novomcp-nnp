"""
NovoMCP Neural Network Potentials Service

Fast energy and force computation using pre-trained neural network potentials:
- ANI-2x: Organic molecules (H, C, N, O, F, S, Cl)
- MACE-MP-0: Universal potential (all elements)

Orders of magnitude faster than DFT, ~10x faster than xTB for single-point energies.
"""

import os
import logging
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.models.registry import initialize, compute_energy, optimize_geometry, relax_batch, compute_energy_batch, get_info

logging.basicConfig(
    format="[NovoMCP] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("novomcp-nnp")

PORT = int(os.getenv("PORT", "8032"))
API_KEY = os.getenv("NNP_API_KEY", "")

app = FastAPI(
    title="NovoMCP Neural Potentials",
    description="Neural network potential inference: ANI-2x, MACE-MP-0 for fast energy/force computation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_key(key: Optional[str]):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# --- Request/Response Models ---

class EnergyRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    method: str = Field("auto", description="Model: auto, ani2x, or mace")
    engine: str = Field("ase", description="Execution engine: 'ase' (default) or 'alchemi' (ALCHEMI Toolkit GPU forward pass; falls back to ASE when unavailable)")
    charge: int = Field(0, description="Molecular charge (must be 0 — NNPs don't support charged species)")
    uhf: int = Field(0, description="Unpaired electrons (must be 0 — NNPs don't support open-shell)")

class EnergyResponse(BaseModel):
    smiles: str
    energy_ev: float
    energy_kcal_mol: float
    forces_max_ev_ang: Optional[float]
    forces_rms_ev_ang: Optional[float]
    method: str
    n_atoms: int
    wall_time_ms: Optional[float]
    engine_used: Optional[str] = None
    note: Optional[str] = None

class BatchEnergyRequest(BaseModel):
    smiles_list: list[str] = Field(..., max_length=100)
    method: str = Field("auto")
    engine: str = Field("alchemi", description="'alchemi' (GPU-batched forward, default) or 'ase' (per-molecule)")

class BatchEnergyResponse(BaseModel):
    results: list[Optional[EnergyResponse]]
    engine_used: str
    note: Optional[str] = None
    count: int
    elapsed_ms: int

class OptimizeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    method: str = Field("auto", description="Model (potential): auto, ani2x, or mace")
    engine: str = Field("ase", description="Execution engine: 'ase' (BFGS, default) or 'alchemi' (ALCHEMI Toolkit GPU-batched; falls back to ASE when unavailable)")
    charge: int = Field(0, description="Molecular charge (must be 0 — NNPs don't support charged species)")
    uhf: int = Field(0, description="Unpaired electrons (must be 0 — NNPs don't support open-shell)")
    fmax: float = Field(0.05, description="Force convergence threshold in eV/Å")
    max_steps: int = Field(200, description="Maximum optimization steps", ge=1, le=1000)

class OptimizeResponse(BaseModel):
    smiles: str
    energy_ev: float
    energy_kcal_mol: float
    optimized_xyz: Optional[str] = None
    forces_max_ev_ang: Optional[float] = None
    converged: bool
    n_steps: int
    method: str
    n_atoms: int
    wall_time_ms: Optional[float]
    engine_used: Optional[str] = None
    note: Optional[str] = None

class RelaxBatchRequest(BaseModel):
    smiles_list: list[str] = Field(..., max_length=100, description="SMILES to relax in one batched pass")
    method: str = Field("auto", description="Model (potential) applied to every system")
    engine: str = Field("alchemi", description="'alchemi' (GPU-batched, default) or 'ase' (sequential fallback)")
    fmax: float = Field(0.05, description="Force convergence threshold in eV/Å, per system")
    max_steps: int = Field(200, ge=1, le=1000)

class RelaxBatchResponse(BaseModel):
    results: list[Optional[OptimizeResponse]]
    engine_used: str
    note: Optional[str] = None
    count: int
    elapsed_ms: int

class CompareRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")

class CompareResponse(BaseModel):
    smiles: str
    results: dict
    fastest: str
    elapsed_ms: int


# --- Startup ---

@app.on_event("startup")
async def startup_event():
    logger.info("Starting NovoMCP Neural Potentials Service...")
    t0 = time.time()
    ok = initialize()
    elapsed = round((time.time() - t0) * 1000)
    info = get_info()
    logger.info(f"Initialization complete in {elapsed}ms — {info}")


# --- Health ---

@app.get("/health")
async def health():
    info = get_info()
    ready_models = [k for k, v in info["models"].items() if v["available"]]
    status = "healthy" if info["ready"] else "unhealthy"

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=200 if info["ready"] else 503,
        content={
            "status": status,
            "service": "novomcp-nnp",
            "version": "1.0.0",
            "port": PORT,
            "models": info["models"],
            "ready_models": ready_models,
        },
    )


@app.get("/")
async def root():
    return {"service": "novomcp-nnp", "version": "1.0.0"}


# --- Helpers ---

def _reject_charged(smiles: str, charge: int = 0, uhf: int = 0):
    """Reject charged or open-shell molecules with helpful error."""
    smiles_charge = 0
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            smiles_charge = Chem.GetFormalCharge(mol)
    except Exception:
        pass

    effective_charge = charge or smiles_charge
    if effective_charge != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Neural network potentials (ANI-2x, MACE) do not support charged species "
                f"(detected charge={effective_charge} from {'SMILES' if smiles_charge else 'parameter'}). "
                f"Use novomcp-qm /api/qm-calculate with charge and uhf parameters for "
                f"charged/open-shell systems."
            ),
        )
    if uhf != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Neural network potentials do not support open-shell systems (uhf={uhf}). "
                f"Use novomcp-qm /api/qm-calculate with uhf parameter for radical species."
            ),
        )


def _smiles_to_geometry(smiles: str):
    """Convert SMILES to atomic positions and species."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol)
        if status != 0:
            raise HTTPException(status_code=422, detail=f"Could not generate 3D coords for: {smiles}")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    positions = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n_atoms)])
    species = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)]

    return positions, species


# --- Endpoints ---

@app.post("/api/compute-energy", response_model=EnergyResponse)
async def api_compute_energy(req: EnergyRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    _reject_charged(req.smiles, req.charge, req.uhf)

    positions, species = _smiles_to_geometry(req.smiles)
    # Route through compute_energy_batch (batch of one) so engine selection lives
    # in one place. Stage: 'alchemi' falls back to ASE when the GPU/toolkit is absent.
    results, engine_used, note = compute_energy_batch(
        [(positions, species)], method=req.method, engine=req.engine)
    result = results[0]

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Computation failed")

    return EnergyResponse(
        smiles=req.smiles,
        energy_ev=result.energy_ev,
        energy_kcal_mol=result.energy_kcal_mol,
        forces_max_ev_ang=result.forces_max_ev_ang,
        forces_rms_ev_ang=result.forces_rms_ev_ang,
        method=result.method,
        n_atoms=result.n_atoms,
        wall_time_ms=result.wall_time_ms,
        engine_used=engine_used,
        note=note,
    )


@app.post("/api/optimize-geometry", response_model=OptimizeResponse)
async def api_optimize_geometry(req: OptimizeRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    _reject_charged(req.smiles, req.charge, req.uhf)

    positions, species = _smiles_to_geometry(req.smiles)
    # Route through relax_batch (batch of one) so the engine-selection logic lives
    # in one place. Stage 1: 'alchemi' transparently falls back to ASE.
    results, engine_used, note = relax_batch(
        [(positions, species)], method=req.method,
        engine=req.engine, fmax=req.fmax, max_steps=req.max_steps,
    )
    result = results[0]

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Optimization failed")

    return OptimizeResponse(
        smiles=req.smiles,
        energy_ev=result.energy_ev,
        energy_kcal_mol=result.energy_kcal_mol,
        optimized_xyz=result.optimized_xyz,
        forces_max_ev_ang=result.forces_max_ev_ang,
        converged=result.converged,
        n_steps=result.n_steps,
        method=result.method,
        n_atoms=result.n_atoms,
        wall_time_ms=result.wall_time_ms,
        engine_used=engine_used,
        note=note,
    )


@app.post("/api/relax-batch", response_model=RelaxBatchResponse)
async def api_relax_batch(req: RelaxBatchRequest, x_api_key: Optional[str] = Header(None)):
    """Relax a whole library in one batched pass.

    engine='alchemi' (default) uses ALCHEMI Toolkit GPU-batched relaxation when the
    backend is available, otherwise falls back to sequential ASE — the response's
    engine_used says which ran. Per-item failures (bad SMILES, charged/open-shell)
    are reported inline as null without failing the batch.
    """
    _check_key(x_api_key)
    t0 = time.time()

    # SMILES → geometry per item; a bad/charged/open-shell input drops that slot
    # to null rather than sinking the whole batch.
    valid = []  # (orig_index, smiles, positions, species)
    for i, smi in enumerate(req.smiles_list):
        try:
            _reject_charged(smi)
            positions, species = _smiles_to_geometry(smi)
            valid.append((i, smi, positions, species))
        except HTTPException:
            continue
        except Exception:
            continue

    opt_results, engine_used, note = relax_batch(
        [(pos, spec) for (_, _, pos, spec) in valid],
        method=req.method, engine=req.engine, fmax=req.fmax, max_steps=req.max_steps,
    )

    results: list[Optional[OptimizeResponse]] = [None] * len(req.smiles_list)
    for slot, (orig_i, smi, _, _) in enumerate(valid):
        r = opt_results[slot]
        if r.success:
            results[orig_i] = OptimizeResponse(
                smiles=smi,
                energy_ev=r.energy_ev,
                energy_kcal_mol=r.energy_kcal_mol,
                optimized_xyz=r.optimized_xyz,
                forces_max_ev_ang=r.forces_max_ev_ang,
                converged=r.converged,
                n_steps=r.n_steps,
                method=r.method,
                n_atoms=r.n_atoms,
                wall_time_ms=r.wall_time_ms,
                engine_used=engine_used,
                note=note,
            )

    return RelaxBatchResponse(
        results=results,
        engine_used=engine_used,
        note=note,
        count=sum(1 for x in results if x is not None),
        elapsed_ms=round((time.time() - t0) * 1000),
    )


@app.post("/api/batch-energy", response_model=BatchEnergyResponse)
async def api_batch_energy(req: BatchEnergyRequest, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    t0 = time.time()

    # SMILES → geometry per item; a bad input drops that slot to null.
    valid = []  # (orig_index, smiles, positions, species)
    for i, smi in enumerate(req.smiles_list):
        try:
            positions, species = _smiles_to_geometry(smi)
            valid.append((i, smi, positions, species))
        except Exception:
            continue

    nnp_results, engine_used, note = compute_energy_batch(
        [(pos, spec) for (_, _, pos, spec) in valid], method=req.method, engine=req.engine)

    results: list[Optional[EnergyResponse]] = [None] * len(req.smiles_list)
    for slot, (orig_i, smi, _, _) in enumerate(valid):
        r = nnp_results[slot]
        if r.success:
            results[orig_i] = EnergyResponse(
                smiles=smi, energy_ev=r.energy_ev, energy_kcal_mol=r.energy_kcal_mol,
                forces_max_ev_ang=r.forces_max_ev_ang, forces_rms_ev_ang=r.forces_rms_ev_ang,
                method=r.method, n_atoms=r.n_atoms, wall_time_ms=r.wall_time_ms,
                engine_used=engine_used, note=note,
            )

    return BatchEnergyResponse(
        results=results, engine_used=engine_used, note=note,
        count=sum(1 for x in results if x is not None),
        elapsed_ms=round((time.time() - t0) * 1000),
    )


@app.post("/api/compare-methods", response_model=CompareResponse)
async def api_compare_methods(req: CompareRequest, x_api_key: Optional[str] = Header(None)):
    """Run all available models on the same molecule and compare."""
    _check_key(x_api_key)

    t0 = time.time()
    positions, species = _smiles_to_geometry(req.smiles)

    results = {}
    for method in ["ani2x", "mace"]:
        r = compute_energy(positions, species, method=method)
        if r.success:
            results[r.method] = {
                "energy_kcal_mol": r.energy_kcal_mol,
                "forces_max_ev_ang": r.forces_max_ev_ang,
                "wall_time_ms": r.wall_time_ms,
            }

    fastest = min(results.items(), key=lambda x: x[1]["wall_time_ms"])[0] if results else "none"

    return CompareResponse(
        smiles=req.smiles,
        results=results,
        fastest=fastest,
        elapsed_ms=round((time.time() - t0) * 1000),
    )


if __name__ == "__main__":
    logger.info(f"Starting NovoMCP Neural Potentials on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
