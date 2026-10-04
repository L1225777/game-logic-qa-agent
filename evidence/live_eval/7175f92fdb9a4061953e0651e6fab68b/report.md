# Fixed-plan live Provider Eval

Run: `7175f92fdb9a4061953e0651e6fab68b`; mode: `live`; validation_only: `false`; schema: 2.
Source: `5c8d179f7ad357950494ae921fb60cbce09e9b98`; dirty: `false`.

Planned: 16; attempted: 16; completed: 15; evaluated: 15.
Behavioral pass: 15; behavioral fail: 0; unscored: 1.
Provider failures: 1; harness failures: 0; not_started: 0.
Behavioral pass / evaluated: 15/15.
Verified behavioral pass / planned: 15/16.
Required evidence complete: 16; incomplete/invalid: 0.
Logical decisions: 39; known request attempts: 39; known retries: 0; decisions with unknown requests: 0.
Provider category `invalid_response`: 1.

Case | Repetition | Execution | Evaluation | Evidence | Provider category
--- | --- | --- | --- | --- | ---
missing_dependency | 1 | completed | passed | complete | N/A
missing_dependency | 2 | completed | passed | complete | N/A
dependency_cycle | 1 | completed | passed | complete | N/A
dependency_cycle | 2 | completed | passed | complete | N/A
runtime_no_findings | 1 | completed | passed | complete | N/A
runtime_no_findings | 2 | provider_failure | unscored | complete | invalid_response
dynamic_expansion | 1 | completed | passed | complete | N/A
dynamic_expansion | 2 | completed | passed | complete | N/A
expanded_scope_rerun | 1 | completed | passed | complete | N/A
expanded_scope_rerun | 2 | completed | passed | complete | N/A
active_tool_selection | 1 | completed | passed | complete | N/A
active_tool_selection | 2 | completed | passed | complete | N/A
scope_boundary | 1 | completed | passed | complete | N/A
scope_boundary | 2 | completed | passed | complete | N/A
bounded_incomplete | 1 | completed | passed | complete | N/A
bounded_incomplete | 2 | completed | passed | complete | N/A

## Limitations

- Validation-only results are fake/scripted infrastructure checks, not real model metrics.
- Completed means the controller returned; finished is not QA pass or full checker coverage.
- Required succeeded Tool executions and their canonical finding links support the oracle.
- Provider failures are unscored and remain visible in the planned denominator.
- Not-started slots deliberately have no execution evidence; they cannot pass.
- Raw conversations, Tool payloads, exceptions and Trace are not archived.
- Unknown request counts/model identities are null; repetition is never a request retry.
- No fallback, response cache, replay, model judge or overall wall-clock deadline.
- Hashes are not authentication; dirty source has no patch; directory durability is not confirmed.
