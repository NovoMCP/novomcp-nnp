# Running the ALCHEMI GPU path

The default `novomcp-nnp` image is **CPU-only** and relaxes molecules one at a
time (ASE BFGS). The optional **ALCHEMI GPU path** relaxes a whole library in a
single batched pass on the GPU, using the [NVIDIA ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit)
(FIRE optimizer + MACE-MP-0 potential).

It activates when **all** of these are true:
- you run the **GPU image** (`Dockerfile.gpu`, which installs the toolkit), and
- an **NVIDIA GPU** is visible in the container, and
- the request uses **`engine="alchemi"`** (the default for `/api/relax-batch`).

If any is missing, the service **transparently falls back to the CPU ASE path**
and the response's `engine_used` tells you which actually ran — so the tool
always works; the GPU just makes the batched path fast.

## What you need

| Requirement | Detail |
|---|---|
| GPU | Any NVIDIA CUDA GPU. Validated on an L40S (48 GB). Smaller cards are fine for typical libraries. |
| Driver | **>= 580** (CUDA 13). Check with `nvidia-smi`. |
| Container runtime | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so `docker run --gpus all` exposes the GPU. |

## Getting a GPU (a few options)

You don't need a cluster — any single CUDA-13 GPU host works:

- **Cloud VM with a GPU + recent driver.** The quickest path is an image that
  already ships the NVIDIA driver + CUDA, e.g. AWS "Deep Learning OSS Nvidia
  Driver AMI" on a `g6e`/`g5` instance, GCP/Azure GPU images, Lambda Cloud,
  RunPod, etc. Confirm `nvidia-smi` shows driver >= 580.
- **Your own workstation** with an NVIDIA GPU, a >= 580 driver, and the NVIDIA
  Container Toolkit installed.
- **NVIDIA-hosted** (NGC / brev.dev) GPU instances.

The service is stateless and short-lived per request, so a scale-to-zero /
spin-up-on-demand GPU is a good fit — bring the GPU up, run your batch, tear it
down.

## Build & run

```bash
# On the GPU host:
docker build -f Dockerfile.gpu -t novomcp-nnp:gpu .
docker run --gpus all -p 8032:8032 novomcp-nnp:gpu
```

First boot downloads the MACE-MP-0 weights (~a few hundred MB) and warms the
model — allow a minute or two before the first request. `/health` reports
readiness.

## Verify it's using the GPU path

```bash
curl -s -X POST localhost:8032/api/relax-batch \
  -H 'Content-Type: application/json' \
  -d '{"smiles_list":["CCO","CCCO","CC(=O)O"],"engine":"alchemi"}'
```

A working GPU path returns `"engine_used": "alchemi"` and, per molecule,
`converged: true` with a real energy and relaxed `optimized_xyz`. Reference
values (MACE-MP-0): ethanol ≈ **−46.6 eV**, propanol ≈ **−63.1 eV**, acetic acid
≈ **−46.3 eV**, all to `fmax < 0.05 eV/Å`.

If you instead see `"engine_used": "ase"` with a `note` about the ALCHEMI
backend being unavailable, the GPU or toolkit isn't visible — check `nvidia-smi`
inside the container (`docker run --gpus all novomcp-nnp:gpu nvidia-smi`).

## Notes & gotchas (from building this)

- **Build into a *fresh* image.** The `Dockerfile.gpu` installs
  `nvalchemi-toolkit[cu13]` (which brings a CUDA-13 build of torch) into a clean
  `python:3.12` base. Layering the toolkit onto an image that already has torch
  + torchvision (e.g. a Deep Learning AMI's bundled env) causes a torch/
  torchvision version clash — start clean, or `pip uninstall torchvision
  torchaudio` first.
- **Eager mode.** The toolkit's host-side automatic neighbor-list selection
  can't run inside `torch.compile`, so the service runs the ALCHEMI path in
  eager mode (`TORCHDYNAMO_DISABLE=1`, set in the image). This is correct and
  still GPU-batched; it just skips graph compilation.
- **Potential.** The ALCHEMI path uses **MACE-MP-0** regardless of the `method`
  field (`method` still selects ANI-2x/MACE for the CPU ASE path). MACE-MP-0's
  absolute energies differ from ANI-2x's — compare like-for-like.
- **Neutral, closed-shell only** — same limitation as the CPU path.
