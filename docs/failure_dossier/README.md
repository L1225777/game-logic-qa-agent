# Game QA Agent failure dossier

Three historical product failures, each reproduced on its real buggy revision.

Recorded current source: `120a1db8d19d9548fdf76c4ff54e4f8b120f8612`; dirty: `false`.
Verification run: `2ad64e96194b4cbeb7cc7a54ed5320d4`; mode: `offline`; dossier schema: 1.
The current source identity was captured before adding these dossier files.
The checkpoint containing this document is a later commit; it does not retroactively change the tested revision.

Evidence level | Meaning
--- | ---
A | historical red reproduced
B | historical red evidence exists but not rerun
C | current regression reproducible only

Failure | Layer | Level | Historical red | Fixed green | Recorded current green
--- | --- | --- | --- | --- | ---
scope_rerun | agent_scope_correctness | A | 2 x expected failure | 1 passed | 1 passed
empty_choices | provider_boundary | A | 2 x expected failure | 1 passed | 1 passed
partial_failure_status | evidence_reporting_correctness | A | 2 x expected failure | 1 passed | 1 passed

All 12 focused invocations collected exactly one test with no collection/setup/teardown errors.
The six expected historical failures are not failures of the current deterministic suite.
The clean baseline suite passed 258/258 before curation.

## Identity and reproduction

This document reuses evidence-package RunIdentity, canonical_json and ArtifactDigest.
Run UUID identifies this verification run. Case fingerprints identify the trusted fixture/oracle definition.
Snapshot fingerprints bind the exact Git product and regression/helper files actually tested.
Artifact hashes cover README.md and reproduce.py; dossier.json is the metadata root and does not hash itself.
The local .gitattributes disables newline conversion for the hashed artifacts.

Use an interpreter with the recorded dependencies available (no dependency installation is performed).
The verifier uses a fresh temporary directory and Python process, checks snapshot hashes, blocks
socket/DNS/environment-file/demo access, and never checks out, patches or modifies master.
It prints only structured results. A matching historical red makes the verifier exit 0;
the nested pytest exit code remains 1 and is explicitly retained.

```text
python -B docs/failure_dossier/reproduce.py --check
```

--check validates the stored metadata and artifact hashes only; it does not execute tests.
The current phase is pinned to the source commit above, even when HEAD advances.

## Authorized scope expansion must permit a fresh Tool execution

Failure ID: `scope_rerun`; category: `agent_scope_correctness`; evidence: **A**.

Buggy revision: `d0bae4681064ffb1a129c7783baf4b2271158324`.
Fixed revision: `2af8d80d188422d7680377c80ac23be79c48088f`.
Case fingerprint: `1852bc03e866d7a59697302cb41caa2e582f9c454d9d7ed3ca17db75aaa1a5f2`.

**Affected functions**

- `game_qa_agent/orchestration.py::execute_action`
- `game_qa_agent/models.py::AgentInvestigationState`
- `game_qa_agent/providers.py::DeepSeekProvider.generate_next_action (historical prompt guidance)`

**Exact regression node**

```text
tests/test_tool_scope_version.py::test_same_tool_can_rerun_after_expansion_with_new_trusted_scope
```

**Trigger**: {"actions": ["call_tool", "expand_scope", "call_tool"], "authorized_expansion_task_ids": ["task_b"], "first_execution_references_task_b": true, "initial_scope_task_ids": ["task_a"], "known_task_ids": ["task_a", "task_b"], "tool_name": "npc_static_conflict_checker"}.

**Symptom**: Scope reached version 2, but the same checker executed only for task_a; its authorized task_a/task_b rerun was rejected.

**Root cause**: execute_action rejected any tool name already in called_tool_names, independent of scope_version.

**Minimal fix**: Record called_tool_scope_versions and block only calls at the current version, retaining conservative handling of legacy unversioned history; align the provider's local repeat guidance.

**Protected invariant**: Same Tool/scope repeats remain blocked, while an authorized larger scope permits a new execution with freshly built trusted inputs.

**Observed red facts**: {"actual_tool_input_scopes": [["task_a"]], "expected_tool_input_scopes": [["task_a"], ["task_a", "task_b"]], "scope_version": 2}.

Historical red reproduced twice (one failed test each); fixed and recorded-current regressions each passed.
JSON retains the declared product/test revisions, repetition, snapshot hashes and pytest phase outcomes.

```text
python -B docs/failure_dossier/reproduce.py scope_rerun buggy
python -B docs/failure_dossier/reproduce.py scope_rerun fixed
python -B docs/failure_dossier/reproduce.py scope_rerun current
```

**Limits**

- The focused regression uses a deterministic registered checker to expose dispatch inputs; it does not measure live model decisions.
- The fix commit also contains other hardening; this dossier attributes only the scope/history change to this failure.

## An empty provider choices list must fail at the acceptance boundary

Failure ID: `empty_choices`; category: `provider_boundary`; evidence: **A**.

Buggy revision: `d0bae4681064ffb1a129c7783baf4b2271158324`.
Fixed revision: `2af8d80d188422d7680377c80ac23be79c48088f`.
Case fingerprint: `5638d33aa2f61db0df7b500b6cb334e3d56b346c1d64b3a21ccc9bd4595db70c`.

**Affected functions**

- `game_qa_agent/providers.py::DeepSeekProvider.generate_next_action`
- `game_qa_agent/providers.py::_parse_completion (current location)`

**Exact regression node**

```text
tests/test_provider_completion.py::test_empty_choices_is_rejected_as_controlled_provider_failure
```

**Trigger**: {"client": "in_memory_completion_stub", "completion_choice_count": 0, "network_requests": 0, "sdk_constructor_called": false}.

**Symptom**: The provider boundary indexed choices[0] and raised IndexError instead of a controlled completion rejection.

**Root cause**: The response path selected the first choice before validating that a completion choice existed.

**Minimal fix**: Guard the empty choices collection before indexing and raise a controlled ValueError. The later provider reliability layer preserves this rejection as non-retryable invalid_response.

**Protected invariant**: Malformed completion envelopes cannot escape acceptance as incidental indexing errors or be accepted/retried as valid actions.

**Observed red facts**: {"completion_choice_count": 0, "expected_failure": "controlled_completion_rejection", "failure_signal": "empty_choices_index_error"}.

Historical red reproduced twice (one failed test each); fixed and recorded-current regressions each passed.
JSON retains the declared product/test revisions, repetition, snapshot hashes and pytest phase outcomes.

```text
python -B docs/failure_dossier/reproduce.py empty_choices buggy
python -B docs/failure_dossier/reproduce.py empty_choices fixed
python -B docs/failure_dossier/reproduce.py empty_choices current
```

**Limits**

- The historical fix introduced ValueError, not the later normalized ProviderFailure model.
- Historical tests pass AgentInvestigationState; the recorded current test uses ProviderDecisionContext. Each phase uses tests from its declared revision.
- The SDK constructor is bypassed and the completion client is a local stub; no DeepSeek service behavior is established.

## A failed first Tool execution must not be reported as not started

Failure ID: `partial_failure_status`; category: `evidence_reporting_correctness`; evidence: **A**.

Buggy revision: `4475c3b08e55674b6be393fa4ea0ef0f9fd09472`.
Fixed revision: `ff3f03dfea9b54eabf6792b12008731f495bfdf7`.
Case fingerprint: `d742cdb6c0bb0361d8b088efbe226f0e97af298ae3d95454bba0261b559fc9d3`.

**Affected functions**

- `game_qa_agent/orchestration.py::execute_action`
- `game_qa_agent/report.py::build_qa_report (affected downstream consumer)`
- `game_qa_agent/report.py::render_qa_report_markdown (affected downstream consumer)`

**Exact regression node**

```text
tests/test_tool_execution_evidence.py::test_failed_execution_links_partial_findings_and_preserves_exception_behavior
```

**Trigger**: {"finding_type": "potential_npc_conflict", "finish_action_remains_queued": true, "initial_status": "start", "max_steps": 2, "partial_finding_count": 1, "scope_version": 1, "tool_name": "npc_static_conflict_checker", "tool_raises_after_yield": true}.

**Symptom**: The first Tool produced a finding and a failed execution record, but investigation/report status remained start, selecting the not_started report limitation.

**Root cause**: execute_action assigned running only after successful Tool iteration; the propagated exception bypassed that assignment.

**Minimal fix**: Move the existing running assignment immediately before Tool invocation, after authorization and trusted input construction. Preserve the original exception and partial finding linkage.

**Protected invariant**: Once an authorized Tool actually starts, a failed partial outcome remains started and incomplete; Report must not claim the investigation never started.

**Observed red facts**: {"actual_investigation_status": "start", "actual_report_status": "start", "execution_status": "failed", "expected_status": "running", "finding_count": 1, "issue_indices": [0]}.

Historical red reproduced twice (one failed test each); fixed and recorded-current regressions each passed.
JSON retains the declared product/test revisions, repetition, snapshot hashes and pytest phase outcomes.

```text
python -B docs/failure_dossier/reproduce.py partial_failure_status buggy
python -B docs/failure_dossier/reproduce.py partial_failure_status fixed
python -B docs/failure_dossier/reproduce.py partial_failure_status current
```

**Limits**

- The historical red stops at the status assertion; the later Markdown assertions execute on green. The not_started consequence also follows the unchanged Report status mapping.
- Running is the existing nonterminal status, not evidence of a background worker or an added recovery mechanism.
- No Report implementation change or Tool retry was part of this fix.

## Shared limits

- Historical code was tested with the recorded current Python/dependency versions, not a reconstruction of the original development environment.
- Current means the pinned source_commit in this dossier, not a moving HEAD.
- A snapshot fingerprint hashes trusted non-sensitive source/test definitions; it is not redaction of runtime payloads.
- The case fingerprint binds the fixture description and exact historical regression/helper definitions, conservatively at file granularity.
- This is a curated historical document, not an Agent Eval package accepted by read_evidence_package and not a live-provider success-rate estimate.
- The JSON is the metadata root; it does not hash itself. Git binds its revision, while ArtifactDigest binds the accompanying Markdown and reproduction helper. These are unsigned integrity checks, not authentication.

## README, resume and interview use

Supported statement: Reproduced and documented historical regressions across scope authorization,
provider completion validation, and execution evidence/reporting using commit-pinned offline tests
and safe, hashed verification records.

The interview examples distinguish Tool identity from Tool/scope execution, controlled boundary
failure from incidental Python exceptions, and canonical execution evidence from investigation status.
They establish reproducible engineering fixes, not production readiness or live model quality.
