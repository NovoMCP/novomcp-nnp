"""
Neural Network Potential Model Registry

Manages loading and inference for:
- MACE-MP-0 (Materials Project universal potential)
- ANI-2x (organic molecules, H/C/N/O/F/S/Cl)

All models compute energies and forces on molecular geometries.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logger = logging.getLogger("novomcp-nnp.registry")


@dataclass
class NnpResult:
    success: bool
    energy_ev: Optional[float] = None
    energy_kcal_mol: Optional[float] = None
    forces_max_ev_ang: Optional[float] = None
    forces_rms_ev_ang: Optional[float] = None
    method: str = ""
    wall_time_ms: Optional[float] = None
    error: Optional[str] = None
    n_atoms: int = 0


# Model holders
_models: Dict[str, Any] = {}
_available: Dict[str, bool] = {}

EV_TO_KCAL = 23.0609  # 1 eV = 23.0609 kcal/mol


def initialize():
    """Load all available NNP models."""
    _load_ani2x()
    _load_mace()

    ready = [k for k, v in _available.items() if v]
    logger.info(f"NNP models loaded: {ready} ({len(ready)}/{len(_available)})")
    return len(ready) > 0


def _load_ani2x():
    """Load ANI-2x via TorchANI."""
    try:
        import torchani
        t0 = time.time()
        model = torchani.models.ANI2x(periodic_table_index=True)
        model.eval()
        _models["ani2x"] = model
        _available["ani2x"] = True
        logger.info(f"ANI-2x loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        _available["ani2x"] = False
        logger.warning(f"ANI-2x not available: {e}")


def _load_mace():
    """Load MACE-MP-0 universal potential."""
    try:
        from mace.calculators import mace_mp
        t0 = time.time()
        calc = mace_mp(model="small", device="cpu", default_dtype="float32")
        _models["mace"] = calc
        _available["mace"] = True
        logger.info(f"MACE-MP-0 loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        _available["mace"] = False
        logger.warning(f"MACE-MP-0 not available: {e}")


def compute_energy(
    positions: np.ndarray,
    species: list[int],
    method: str = "ani2x",
) -> NnpResult:
    """
    Compute energy and forces for a molecular geometry.

    Args:
        positions: Atomic positions in Angstrom, shape (n_atoms, 3)
        species: Atomic numbers, length n_atoms
        method: 'ani2x', 'mace', or 'auto' (tries ani2x first)
    """
    if method == "auto":
        # ANI-2x is faster for organic molecules; MACE for anything else
        ani_elements = {1, 6, 7, 8, 9, 16, 17}  # H, C, N, O, F, S, Cl
        if all(z in ani_elements for z in species):
            method = "ani2x"
        elif _available.get("mace"):
            method = "mace"
        elif _available.get("ani2x"):
            method = "ani2x"
        else:
            return NnpResult(success=False, error="No NNP models available")

    if not _available.get(method):
        return NnpResult(success=False, error=f"Model '{method}' not available")

    t0 = time.time()

    try:
        if method == "ani2x":
            result = _compute_ani2x(positions, species)
        elif method == "mace":
            result = _compute_mace(positions, species)
        else:
            return NnpResult(success=False, error=f"Unknown method: {method}")

        result.wall_time_ms = round((time.time() - t0) * 1000, 1)
        result.n_atoms = len(species)
        return result

    except Exception as e:
        return NnpResult(
            success=False,
            error=str(e),
            method=method,
            wall_time_ms=round((time.time() - t0) * 1000, 1),
        )


def _compute_ani2x(positions: np.ndarray, species: list[int]) -> NnpResult:
    """Compute with ANI-2x."""
    model = _models["ani2x"]

    coords = torch.tensor(positions, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    atomic_nums = torch.tensor(species, dtype=torch.long).unsqueeze(0)  # (1, n)

    result = model((atomic_nums, coords))
    energy = result.energies
    energy_hartree = energy.item()
    forces = -torch.autograd.grad(energy, coords, create_graph=False)[0].squeeze(0)

    energy_ev = energy_hartree * 27.2114  # Hartree → eV
    energy_kcal = energy_hartree * 627.509  # Hartree → kcal/mol
    forces_np = forces.detach().numpy()  # eV/Å (after Hartree→eV conversion)
    forces_ev_ang = forces_np * 27.2114  # Hartree/Å → eV/Å

    return NnpResult(
        success=True,
        energy_ev=round(energy_ev, 6),
        energy_kcal_mol=round(energy_kcal, 4),
        forces_max_ev_ang=round(float(np.max(np.abs(forces_ev_ang))), 6),
        forces_rms_ev_ang=round(float(np.sqrt(np.mean(forces_ev_ang ** 2))), 6),
        method="ANI-2x",
    )


def _compute_mace(positions: np.ndarray, species: list[int]) -> NnpResult:
    """Compute with MACE-MP-0 via ASE calculator."""
    from ase import Atoms

    calc = _models["mace"]

    atoms = Atoms(numbers=species, positions=positions)
    atoms.calc = calc

    energy_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()  # eV/Å

    return NnpResult(
        success=True,
        energy_ev=round(float(energy_ev), 6),
        energy_kcal_mol=round(float(energy_ev) * EV_TO_KCAL, 4),
        forces_max_ev_ang=round(float(np.max(np.abs(forces))), 6),
        forces_rms_ev_ang=round(float(np.sqrt(np.mean(forces ** 2))), 6),
        method="MACE-MP-0",
    )


@dataclass
class OptimizeResult:
    success: bool
    energy_ev: Optional[float] = None
    energy_kcal_mol: Optional[float] = None
    optimized_positions: Optional[list[list[float]]] = None
    optimized_xyz: Optional[str] = None
    forces_max_ev_ang: Optional[float] = None
    converged: bool = False
    n_steps: int = 0
    method: str = ""
    wall_time_ms: Optional[float] = None
    error: Optional[str] = None
    n_atoms: int = 0


def optimize_geometry(
    positions: np.ndarray,
    species: list[int],
    method: str = "auto",
    fmax: float = 0.05,
    max_steps: int = 200,
) -> OptimizeResult:
    """Optimize molecular geometry using ASE BFGS with NNP forces.

    ~100x faster than xTB optimization for organic molecules.

    Args:
        positions: Initial positions in Angstrom (n_atoms, 3)
        species: Atomic numbers
        method: 'ani2x', 'mace', or 'auto'
        fmax: Force convergence threshold in eV/Å (default 0.05)
        max_steps: Maximum optimization steps (default 200)
    """
    from ase import Atoms
    from ase.optimize import BFGS
    import io
    import contextlib

    # Resolve method
    if method == "auto":
        ani_elements = {1, 6, 7, 8, 9, 16, 17}
        if all(z in ani_elements for z in species) and _available.get("ani2x"):
            method = "ani2x"
        elif _available.get("mace"):
            method = "mace"
        elif _available.get("ani2x"):
            method = "ani2x"
        else:
            return OptimizeResult(success=False, error="No NNP models available")

    if not _available.get(method):
        return OptimizeResult(success=False, error=f"Model '{method}' not available")

    t0 = time.time()

    try:
        atoms = Atoms(numbers=species, positions=positions)

        if method == "mace":
            atoms.calc = _models["mace"]
        elif method == "ani2x":
            # ASE needs a proper Calculator interface
            from ase.calculators.calculator import Calculator as ASECalc

            class _ANI2xASE(ASECalc):
                implemented_properties = ["energy", "forces"]

                def __init__(self, ani_model):
                    super().__init__()
                    self._model = ani_model

                def calculate(self, atoms=None, properties=None, system_changes=None):
                    super().calculate(atoms, properties, system_changes)
                    pos = torch.tensor(
                        self.atoms.get_positions(), dtype=torch.float32
                    ).unsqueeze(0).requires_grad_(True)
                    spec = torch.tensor(
                        self.atoms.get_atomic_numbers(), dtype=torch.long
                    ).unsqueeze(0)
                    result = self._model((spec, pos))
                    e_hartree = result.energies
                    f_hartree = -torch.autograd.grad(
                        e_hartree, pos, create_graph=False
                    )[0].squeeze(0)
                    self.results = {
                        "energy": e_hartree.item() * 27.2114,
                        "forces": f_hartree.detach().numpy() * 27.2114,
                    }

            atoms.calc = _ANI2xASE(_models["ani2x"])
        else:
            return OptimizeResult(success=False, error=f"Unknown method: {method}")

        # Run BFGS optimization (suppress ASE stdout)
        log_buf = io.StringIO()
        with contextlib.redirect_stdout(log_buf):
            opt = BFGS(atoms, logfile=None)
            converged = opt.run(fmax=fmax, steps=max_steps)

        n_steps = opt.nsteps
        final_energy_ev = float(atoms.get_potential_energy())
        final_forces = atoms.get_forces()

        # Build XYZ string
        n = len(species)
        from ase.data import chemical_symbols
        xyz_lines = [str(n), f"Energy: {final_energy_ev:.6f} eV"]
        for i in range(n):
            sym = chemical_symbols[species[i]]
            x, y, z = atoms.positions[i]
            xyz_lines.append(f"{sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
        xyz_str = "\n".join(xyz_lines) + "\n"

        wall_ms = round((time.time() - t0) * 1000, 1)

        return OptimizeResult(
            success=True,
            energy_ev=round(final_energy_ev, 6),
            energy_kcal_mol=round(final_energy_ev * EV_TO_KCAL, 4),
            optimized_positions=atoms.positions.tolist(),
            optimized_xyz=xyz_str,
            forces_max_ev_ang=round(float(np.max(np.abs(final_forces))), 6),
            converged=bool(converged),
            n_steps=n_steps,
            method=f"{'MACE-MP-0' if method == 'mace' else 'ANI-2x'}-BFGS",
            wall_time_ms=wall_ms,
            n_atoms=n,
        )

    except Exception as e:
        return OptimizeResult(
            success=False,
            error=str(e),
            method=method,
            wall_time_ms=round((time.time() - t0) * 1000, 1),
        )


def _alchemi_available() -> bool:
    """Whether the NVIDIA ALCHEMI Toolkit GPU-batched relaxation path can run.

    Stage 1: the batched implementation is not built yet, so this always returns
    False and engine='alchemi' transparently falls back to the ASE path. Stage 2
    wires the real check (nvalchemi-toolkit import + CUDA) and the batched path.
    """
    return False


def _relax_batch_alchemi(systems, method, fmax, max_steps) -> list["OptimizeResult"]:
    """GPU-batched relaxation via the NVIDIA ALCHEMI Toolkit. Wired in Stage 2.

    Never reached while _alchemi_available() is False.
    """
    raise NotImplementedError("ALCHEMI batched relaxation is not built in this image (Stage 2)")


def relax_batch(
    systems: list,
    method: str = "auto",
    engine: str = "alchemi",
    fmax: float = 0.05,
    max_steps: int = 200,
):
    """Relax a batch of molecular systems in input order.

    Each system is a ``(positions, species)`` pair. ``engine='alchemi'`` (default)
    uses the ALCHEMI Toolkit GPU-batched relaxation when available, otherwise falls
    back to the ASE per-molecule path; ``engine='ase'`` forces the sequential path.

    Returns ``(results, engine_used, note)`` where ``results`` is a list of
    :class:`OptimizeResult` aligned to ``systems``.
    """
    use_alchemi = engine == "alchemi" and _alchemi_available()
    engine_used = "alchemi" if use_alchemi else "ase"
    note = None
    if engine == "alchemi" and not use_alchemi:
        note = "ALCHEMI Toolkit backend unavailable in this image; used ASE fallback."

    if use_alchemi:
        return _relax_batch_alchemi(systems, method, fmax, max_steps), engine_used, note

    # ASE fallback: sequential per-molecule relaxation (correct, not accelerated).
    results = [
        optimize_geometry(pos, spec, method=method, fmax=fmax, max_steps=max_steps)
        for pos, spec in systems
    ]
    return results, engine_used, note


def get_info() -> dict:
    return {
        "models": {k: {"available": v} for k, v in _available.items()},
        "ready": any(_available.values()),
        "capabilities": {
            "single_point": True,
            "optimization": True,
            "charge_spin": False,  # NNPs don't support charge/spin natively
        },
    }
