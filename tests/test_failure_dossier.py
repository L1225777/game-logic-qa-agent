from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from game_qa_agent.evidence import RunIdentity


DIRECTORY = Path(__file__).resolve().parents[1] / "docs" / "failure_dossier"
spec = importlib.util.spec_from_file_location("failure_dossier_reproduce", DIRECTORY / "reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


def document():
    return json.loads((DIRECTORY / "dossier.json").read_bytes())


def test_dossier_contract_has_three_strong_cases_and_safe_recorded_observations():
    data = document()
    reproduce.validate_dossier(data)
    run = RunIdentity.model_validate(data["run"])
    assert run.source_commit == "120a1db8d19d9548fdf76c4ff54e4f8b120f8612"
    assert data["baseline_suite"] == {"passed": 258, "failed": 0}
    markdown = (DIRECTORY / "README.md").read_text(encoding="utf-8")
    for failure in data["failures"]:
        for key in ("failure_id", "regression_node", "buggy_revision", "fixed_revision", "case_fingerprint"):
            assert failure[key] in markdown
        assert failure["evidence_level"] == "A"
        assert failure["red_evidence_status"] == "historical_red_reproduced_twice"
        assert failure["green_evidence_status"] == "fixed_and_recorded_current_regressions_passed"
        assert failure["affected_functions"] and failure["limitations"]
        assert all(failure[key] for key in ("fixture", "symptom", "root_cause", "minimal_fix", "protected_invariant"))
        for verification in failure["verifications"]:
            observation = verification["observation"]
            assert set(observation) == {"exit_code", "collected", "collection_errors", "reports"}
            for report in observation["reports"]:
                assert set(report) <= {"node", "phase", "outcome", "matched_failure"}
                assert report.get("matched_failure") in (None, reproduce.SIGNATURES[failure["failure_id"]])
    # No pytest traceback, exception payload or full Agent projection is stored.
    serialized = json.dumps(data)
    for canary in ("private-message-marker", "private-evidence-marker", "private-tool-error-marker"):
        assert canary not in serialized


@pytest.mark.parametrize("failure_id", reproduce.FAILURE_IDS)
def test_dossier_revisions_oracle_and_snapshot_hashes_match_real_git_history(failure_id):
    failure = next(item for item in document()["failures"] if item["failure_id"] == failure_id)
    # These are verified immediate before/after pairs, not inferred revision labels.
    assert reproduce.git_bytes("rev-parse", failure["fixed_revision"] + "^").decode().strip() == failure["buggy_revision"]
    assert failure["oracle"]["definition_revision"] == failure["fixed_revision"]
    assert failure["oracle"]["node"] == failure["regression_node"]
    for phase in ("historical_red", "historical_green", "current_green"):
        verification = next(v for v in failure["verifications"] if v["phase"] == phase)
        files = reproduce.snapshot_files(verification["tested_revision"], verification["regression_revision"],
                                           failure["regression_node"])
        assert "game_qa_agent/demo.py" not in files
        assert all(name.endswith(".py") and not Path(name).name.startswith(".env") for name in files)
        hashes = {name: sha256(content).hexdigest() for name, content in files.items()}
        assert reproduce.fingerprint(hashes) == verification["snapshot_fingerprint"]
        if phase == "historical_green":
            assert reproduce.fingerprint({name: digest for name, digest in hashes.items()
                                          if name.startswith("tests/")}) == failure["oracle"]["test_files_sha256"]


@pytest.mark.parametrize("fault", ["collection_error", "wrong_symptom", "missing_red"])
def test_dossier_does_not_upgrade_missing_or_unrelated_failures_to_level_a(fault):
    data = deepcopy(document())
    records = data["failures"][0]["verifications"]
    if fault == "collection_error":
        records[0]["observation"]["collection_errors"] = 1
    elif fault == "wrong_symptom":
        records[0]["observation"]["reports"][1]["matched_failure"] = "unrecognized"
    else:
        records.pop(0)
    with pytest.raises(ValueError, match="dossier_contract_mismatch"):
        reproduce.validate_dossier(data)


@pytest.mark.parametrize("name", ["README.md", "reproduce.py"])
def test_dossier_detects_changed_hashed_artifact(tmp_path, monkeypatch, name):
    for artifact in ("README.md", "reproduce.py"):
        (tmp_path / artifact).write_bytes((DIRECTORY / artifact).read_bytes())
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(reproduce, "HERE", tmp_path)
    with pytest.raises(ValueError, match="dossier_contract_mismatch"):
        reproduce.validate_dossier(document())


def test_dossier_reproduction_guard_blocks_external_access_before_io():
    # Synthetic audit events for env/demo; no actual env file or demo is opened.
    probes = r'''
checks = [
    lambda: socket.getaddrinfo("offline.invalid", 443),
    lambda: sys.audit("open", ".env.DO_NOT_CREATE", "r", 0),
    lambda: exec(compile("pass", "demo.py", "exec")),
]
for check in checks:
    try:
        check()
    except RuntimeError as error:
        assert str(error) == "offline_dossier_boundary"
    else:
        raise AssertionError("External boundary was not blocked")
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", reproduce.OFFLINE_GUARD + probes],
                            capture_output=True, text=True)
    assert result.returncode == 0, "Dossier offline boundary failed"
