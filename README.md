# novomcp-nnp

Neural network potential service for the [NovoMCP](https://github.com/NovoMCP/novomcp) engine — fast energies, forces, and geometry optimization from pretrained NNPs:

- **ANI-2x** (TorchANI) — organic molecules (H, C, N, O, F, S, Cl)
- **MACE-MP-0** — universal potential (all elements)

Orders of magnitude faster than DFT. Neutral, closed-shell only — use a QM service for charged/open-shell systems.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + which models loaded |
| `POST` | `/api/compute-energy` | single-point energy + forces |
| `POST` | `/api/optimize-geometry` | relax one molecule |
| `POST` | `/api/relax-batch` | relax a whole `smiles_list` in one pass |
| `POST` | `/api/batch-energy` | energies for a list of molecules |
| `POST` | `/api/compare-methods` | run every model on one molecule |

**`engine` field** (`/api/optimize-geometry`, `/api/relax-batch`): `ase` (BFGS) or `alchemi` — GPU-batched relaxation via the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit). Until the ALCHEMI backend is built into the image, `alchemi` transparently falls back to ASE and the response's `engine_used` tells you which actually ran. Per-item failures in a batch (bad SMILES, charged/open-shell) come back as `null` without sinking the batch.

## Run

```bash
docker build -t novomcp-nnp .
docker run -p 8032:8032 novomcp-nnp

curl -s localhost:8032/health
curl -s -X POST localhost:8032/api/relax-batch \
  -H 'Content-Type: application/json' \
  -d '{"smiles_list":["CCO","CCCO"]}'
```

A relaxed molecule comes back with `converged: true`, an `optimized_xyz`, and a real energy.

## Wire it to the engine

```bash
export NOVOMCP_NNP_URL=http://localhost:8032
```

`optimize_geometry_nnp`, `batch_geometry_relaxation`, and `compute_energy` then light up in the engine. See the engine's [deploying-services guide](https://github.com/NovoMCP/novomcp/tree/main/docs/deploying-services) and [verifying-services](https://github.com/NovoMCP/novomcp/blob/main/docs/deploying-services/verifying-services.md).

## Auth

Set `NNP_API_KEY` to require an `X-API-Key` header on requests. Unset (the default) = no inbound auth — fine for localhost or a private network.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
