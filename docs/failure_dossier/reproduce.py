"""Offline verifier for the three curated historical failures in this directory.

No arbitrary revision/test CLI, checkout, patching, resuming, or Eval execution.
The recorded 'current' revision is pinned; it does not mean today's HEAD.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAILURE_IDS = ("scope_rerun", "empty_choices", "partial_failure_status")
PHASES = {"buggy": "historical_red", "fixed": "historical_green", "current": "current_green"}
SIGNATURES = {
    "scope_rerun": "scope_rerun_missing",
    "empty_choices": "empty_choices_index_error",
    "partial_failure_status": "partial_failure_not_started",
}

# This same guard runs before historical imports in each fresh Python process.
OFFLINE_GUARD = r'''
import os
import socket
import sys
sys.dont_write_bytecode = True
def forbidden(*args, **kwargs):
    raise RuntimeError("offline_dossier_boundary")
def boundary(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        if os.path.basename(os.fsdecode(args[0])).casefold().startswith(".env"):
            forbidden()
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyaddr", "socket.sendto"}:
        forbidden()
    if event == "exec" and os.path.basename(args[0].co_filename).casefold() == "demo.py":
        forbidden()
sys.addaudithook(boundary)
socket.socket.connect = socket.socket.connect_ex = forbidden
socket.create_connection = socket.getaddrinfo = forbidden
socket.gethostbyname = socket.gethostbyname_ex = socket.gethostbyaddr = forbidden
'''

CHILD = OFFLINE_GUARD + r'''
import json
from pathlib import Path
import pytest
sys.path.insert(0, str(Path.cwd()))
import game_qa_agent
assert Path(game_qa_agent.__file__).resolve().is_relative_to(Path.cwd().resolve())

class Observation:
    collected = 0
    collection_errors = 0
    def __init__(self):
        self.reports = []
    def pytest_collection_finish(self, session):
        self.collected = len(session.items)
    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors += 1
    def pytest_runtest_logreport(self, report):
        entry = {"node": report.nodeid, "phase": report.when, "outcome": report.outcome}
        if report.failed:
            text = report.longreprtext
            entry["matched_failure"] = (
                "scope_rerun_missing" if "assert executions ==" in text else
                "empty_choices_index_error" if "IndexError: list index out of range" in text
                    and "response.choices[0]" in text else
                "partial_failure_not_started" if "assert 'start' == 'running'" in text else
                "unrecognized"
            )
        self.reports.append(entry)

observer = Observation()
status = pytest.main(["-q", "-p", "no:cacheprovider", sys.argv[1]], plugins=[observer])
print("DOSSIER_OBSERVATION=" + json.dumps({
    "exit_code": int(status), "collected": observer.collected,
    "collection_errors": observer.collection_errors, "reports": observer.reports,
}, sort_keys=True))
raise SystemExit(status)
'''


def _contracts():
    # Parent-only imports: historical children must import their own snapshot.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from game_qa_agent.evidence import ArtifactDigest, RunIdentity, canonical_json
    return ArtifactDigest, RunIdentity, canonical_json


def fingerprint(value):
    return sha256(_contracts()[2](value)).hexdigest()


def case_definition(failure):
    return {name: failure[name] for name in ("failure_id", "fixture", "oracle")}


def require(condition):
    if not condition:
        raise ValueError("dossier_contract_mismatch")


def git_bytes(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def snapshot_files(revision, test_revision, node):
    names = git_bytes("ls-tree", "-r", "--name-only", revision, "--", "game_qa_agent").decode().splitlines()
    sources = [name for name in names if name.endswith(".py") and name != "game_qa_agent/demo.py"]
    tests = sorted({node.split("::")[0], "tests/conftest.py", "tests/test_action_safety.py"})
    return {path: git_bytes("show", rev + ":" + path)
            for paths, rev in ((sources, revision), (tests, test_revision)) for path in paths}


def expected_observation(observation, node, failure_id, red):
    """A collection/setup/environment error is never historical product red."""
    reports = observation["reports"]
    if (observation["collected"] != 1 or observation["collection_errors"] != 0
            or observation["exit_code"] != (1 if red else 0) or len(reports) != 3):
        return False
    for report, phase in zip(reports, ("setup", "call", "teardown")):
        if report["node"] != node or report["phase"] != phase:
            return False
        expected = "failed" if red and phase == "call" else "passed"
        if report["outcome"] != expected:
            return False
        if expected == "failed" and report.get("matched_failure") != SIGNATURES[failure_id]:
            return False
    return True


def validate_dossier(document):
    ArtifactDigest, RunIdentity, canonical_json = _contracts()
    require(type(document["schema_version"]) is int and document["schema_version"] == 1)
    require(type(document["evidence_package_schema_version"]) is int
            and document["evidence_package_schema_version"] == 1)
    require(document["kind"] == "failure_dossier")
    run = RunIdentity.model_validate_json(canonical_json(document["run"]), strict=True)
    require(not run.source_dirty)
    require([item["failure_id"] for item in document["failures"]] == list(FAILURE_IDS))
    for name, description in document["artifacts"].items():
        require(name in ("README.md", "reproduce.py"))
        digest = ArtifactDigest.model_validate_json(canonical_json(description), strict=True)
        data = (HERE / name).read_bytes()
        require(digest.size == len(data) and digest.sha256 == sha256(data).hexdigest())
    require(set(document["artifacts"]) == {"README.md", "reproduce.py"})
    for failure in document["failures"]:
        require(failure["case_fingerprint"] == fingerprint(case_definition(failure)))
        require(failure["evidence_level"] == "A")
        require([(v["phase"], v["repetition"]) for v in failure["verifications"]] == [
            ("historical_red", 1), ("historical_red", 2), ("historical_green", 1), ("current_green", 1),
        ])
        for verification in failure["verifications"]:
            phase = verification["phase"]
            revision = (failure["buggy_revision"] if phase == "historical_red" else
                        failure["fixed_revision"] if phase == "historical_green" else run.source_commit)
            require(verification["tested_revision"] == revision)
            require(verification["regression_revision"] == (
                run.source_commit if phase == "current_green" else failure["fixed_revision"]
            ))
            require(expected_observation(verification["observation"], failure["regression_node"],
                                         failure["failure_id"], phase == "historical_red"))


def reproduce(failure, phase):
    verification = next(v for v in failure["verifications"] if v["phase"] == PHASES[phase])
    files = snapshot_files(verification["tested_revision"], verification["regression_revision"],
                           failure["regression_node"])
    actual = fingerprint({name: sha256(data).hexdigest() for name, data in files.items()})
    if actual != verification["snapshot_fingerprint"]:
        raise ValueError("snapshot_identity_mismatch")
    with tempfile.TemporaryDirectory(prefix="gameqa_dossier_verify_") as directory:
        destination = Path(directory).resolve()
        require(destination.parent == Path(tempfile.gettempdir()).resolve())
        for name, data in files.items():
            target = (destination / name).resolve()
            require(target.is_relative_to(destination))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", CHILD, failure["regression_node"]],
            cwd=destination, capture_output=True, text=True,
        )
    marker = "DOSSIER_OBSERVATION="
    observation = json.loads(next(line[len(marker):] for line in result.stdout.splitlines()
                                  if line.startswith(marker)))
    matched = result.returncode == observation["exit_code"] and expected_observation(
        observation, failure["regression_node"], failure["failure_id"], phase == "buggy",
    )
    # Never print/archive pytest stdout, provider text, or exception traceback.
    return {"failure_id": failure["failure_id"], "phase": phase,
            "tested_revision": verification["tested_revision"],
            "regression_revision": verification["regression_revision"],
            "snapshot_fingerprint": actual, "matches_recorded_outcome": matched,
            "observation": observation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("failure_id", choices=FAILURE_IDS, nargs="?")
    parser.add_argument("phase", choices=tuple(PHASES), nargs="?")
    parser.add_argument("--check", action="store_true", help="validate dossier metadata and artifact hashes only")
    args = parser.parse_args()
    document = json.loads((HERE / "dossier.json").read_bytes())
    validate_dossier(document)
    if args.check:
        print("Dossier metadata and artifact hashes verified; no tests executed.")
        return 0
    if args.failure_id is None or args.phase is None:
        parser.error("choose one recorded failure and buggy/fixed/current phase")
    failure = next(item for item in document["failures"] if item["failure_id"] == args.failure_id)
    result = reproduce(failure, args.phase)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["matches_recorded_outcome"] else 1


if __name__ == "__main__":
    exec(OFFLINE_GUARD)
    raise SystemExit(main())
