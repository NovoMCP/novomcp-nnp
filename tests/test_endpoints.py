"""Tests for novomcp-nnp FastAPI endpoints.

Tests charge/uhf rejection, input validation, and health endpoint.
Model-dependent tests are skipped when NNP models aren't installed.
"""

import pytest

try:
    from fastapi.testclient import TestClient
    from main import app
except ImportError:
    pytest.skip("dependencies not installed", allow_module_level=True)


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# --- Charge rejection (compute-energy) ---

def test_compute_energy_rejects_charged_smiles(client):
    """Charged SMILES like [NH4+] should be rejected with 400."""
    resp = client.post("/api/compute-energy", json={
        "smiles": "[NH4+]",
    })
    assert resp.status_code == 400
    assert "charged species" in resp.json()["detail"].lower()


def test_compute_energy_rejects_charge_param(client):
    """Explicit charge=1 parameter should be rejected with 400."""
    resp = client.post("/api/compute-energy", json={
        "smiles": "CCO",
        "charge": 1,
    })
    assert resp.status_code == 400
    assert "charged species" in resp.json()["detail"].lower()


def test_compute_energy_rejects_uhf(client):
    """uhf=1 (open-shell) should be rejected with 400."""
    resp = client.post("/api/compute-energy", json={
        "smiles": "CCO",
        "uhf": 1,
    })
    assert resp.status_code == 400
    assert "open-shell" in resp.json()["detail"].lower()


# --- Charge rejection (optimize-geometry) ---

def test_optimize_rejects_charged_smiles(client):
    resp = client.post("/api/optimize-geometry", json={
        "smiles": "[O-]C(=O)C",  # acetate anion
    })
    assert resp.status_code == 400
    assert "charged species" in resp.json()["detail"].lower()


def test_optimize_rejects_charge_param(client):
    resp = client.post("/api/optimize-geometry", json={
        "smiles": "CCO",
        "charge": 1,
    })
    assert resp.status_code == 400


def test_optimize_rejects_uhf_param(client):
    resp = client.post("/api/optimize-geometry", json={
        "smiles": "CCO",
        "uhf": 2,
    })
    assert resp.status_code == 400


# --- Input validation ---

def test_compute_energy_invalid_smiles(client):
    resp = client.post("/api/compute-energy", json={
        "smiles": "NOT_A_SMILES_XYZ",
    })
    assert resp.status_code == 422


def test_compute_energy_missing_smiles(client):
    resp = client.post("/api/compute-energy", json={})
    assert resp.status_code == 422


# --- Health endpoint ---

def test_health_returns_200_or_503(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert data["service"] == "novomcp-nnp"
    assert "models" in data


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "novomcp-nnp"


# --- Neutral molecule passes (only if models available) ---

def test_compute_energy_neutral_accepted(client):
    """Neutral ethanol should be accepted (not rejected by charge check)."""
    resp = client.post("/api/compute-energy", json={
        "smiles": "CCO",
    })
    # Either 200 (models loaded) or 500 (models not available) — but NOT 400
    assert resp.status_code != 400


# --- relax-batch (Stage 1: engine=alchemi falls back to ASE) ---

def test_relax_batch_requires_smiles_list(client):
    resp = client.post("/api/relax-batch", json={})
    assert resp.status_code == 422


def test_relax_batch_alchemi_falls_back_to_ase(client):
    """engine='alchemi' must transparently fall back to ASE until Stage 2, and say so."""
    resp = client.post("/api/relax-batch", json={
        "smiles_list": ["CCO", "CCCO"],
        "engine": "alchemi",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine_used"] == "ase"
    assert data["note"] and "ASE" in data["note"]
    # order preserved (one slot per input, null when models absent)
    assert len(data["results"]) == 2


def test_relax_batch_ase_no_note(client):
    resp = client.post("/api/relax-batch", json={"smiles_list": ["CCO"], "engine": "ase"})
    assert resp.status_code == 200
    assert resp.json()["engine_used"] == "ase"
    assert resp.json()["note"] is None


def test_relax_batch_bad_item_is_null_not_error(client):
    """A bad SMILES drops its slot to null; the batch still returns 200."""
    resp = client.post("/api/relax-batch", json={
        "smiles_list": ["CCO", "NOT_A_SMILES_XYZ"],
        "engine": "ase",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["results"][1] is None  # bad SMILES → null regardless of model availability


def test_optimize_geometry_accepts_engine_param(client):
    """The single endpoint gains an engine field; alchemi falls back to ASE."""
    resp = client.post("/api/optimize-geometry", json={"smiles": "CCO", "engine": "alchemi"})
    assert resp.status_code != 400  # 200 with models, 500 without — never a validation error
    if resp.status_code == 200:
        assert resp.json()["engine_used"] == "ase"
