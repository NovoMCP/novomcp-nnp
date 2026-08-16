"""
GPU smoke test — MACE-MPA-0 (MIT) through the NVIDIA ALCHEMI Toolkit.

Verifies the exact path the merged services now use
(novomcp-nnp/app/models/registry.py:352 and
 novomcp-qm/app/engines/alchemi_conformers.py:50):
load medium-mpa-0 on CUDA -> wrap in the ALCHEMI MACEWrapper -> run a
batched FIRE relax on the GPU. This is the one thing the CPU smoke test
could NOT cover.

Run on any CUDA GPU, in the same environment the GPU image builds:

    pip install --extra-index-url https://pypi.nvidia.com \
        "nvalchemi-toolkit[cu13]" "mace-torch>=0.3.10" rdkit ase numpy
    python gpu_smoke_mace_mpa0.py

PASS  = prints per-conformer relaxed energies with no exception.
FAIL  = any exception in load / wrap / FIRE run (report it; the model swap
        would then need a fix before the deployed GPU path is trusted).
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import numpy as np

import torch
assert torch.cuda.is_available(), "FAIL: no CUDA GPU visible"
print("CUDA device:", torch.cuda.get_device_name(0))

# --- 1) load + wrap MACE-MPA-0 exactly as the service does ---
from nvalchemi.models.mace import MACEWrapper
from mace.calculators.foundations_models import mace_mp

raw = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float32").models[0]
model = MACEWrapper(raw).to("cuda").eval()
print("WRAPPED OK: MACE-MPA-0 loaded + wrapped in the ALCHEMI MACEWrapper on GPU")

# --- 2) build a tiny ETKDG ensemble of ethanol (mirrors _generate_etkdg) ---
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
params = AllChem.ETKDGv3()
params.randomSeed = 42
ids = AllChem.EmbedMultipleConfs(mol, numConfs=4, params=params)
species = [a.GetAtomicNum() for a in mol.GetAtoms()]
n = mol.GetNumAtoms()
positions_list = []
for cid in ids:
    c = mol.GetConformer(cid)
    positions_list.append(np.array(
        [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
         for i in range(n)], dtype=np.float32))
print(f"built {len(positions_list)} conformers of ethanol ({n} atoms)")

# --- 3) batched FIRE relax on GPU (mirrors _batch_relax) ---
try:
    import torch._dynamo as _dynamo
    _dynamo.config.disable = True
except Exception:
    pass
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, ConvergenceHook

hooks = model.make_neighbor_hooks()
znums = torch.tensor(list(species), dtype=torch.long)
datas = [AtomicData(
    atomic_numbers=znums.clone(),
    positions=torch.tensor(np.asarray(p), dtype=torch.float32),
    forces=torch.zeros((n, 3), dtype=torch.float32),
    energy=torch.zeros((1, 1), dtype=torch.float32),
) for p in positions_list]
batch = Batch.from_data_list(datas, device="cuda")

opt = FIRE(model=model, dt=0.1, n_steps=50,
           convergence_hook=ConvergenceHook.from_fmax(0.05), hooks=hooks)
opt.compile_step = False  # eager: toolkit's host-side auto-neighbor method can't run under compile
with opt:
    result = opt.run(batch)

energies = [float(result.get_data(i).energy.reshape(-1)[0]) for i in range(len(positions_list))]
print("RELAXED ENERGIES (eV):", [round(e, 3) for e in energies])
print("lowest:", round(min(energies), 3), "eV")
print("GPU SMOKE TEST PASSED — ALCHEMI + MACE-MPA-0 batch-relaxes on GPU")
