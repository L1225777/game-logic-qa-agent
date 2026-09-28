# 真实失败记录与面试准备

基准版本：`2324c9e0aabe29d4e5df806316b2293e9ec6f66d`。本文只使用该版本及其祖先的代码、测试和文档；核验日期：2026-09-28。

正文收录 5 个在历史代码上实际复现的缺陷，以及 1 次文档记录的真实 live Eval 失败。离线构造输入触发的缺陷不等于线上事故。只有防御性测试、无法证明历史失败的设计风险不作为事故收录；这不是所有测试或所有改动的清单。

证据强度按本次面试准备的规则评定：有对应缺陷回归测试为**强**，只有 commit 或 issue 记录为**中**，只有文档记述为**弱**。强不代表生产事故，也不代表真实模型质量。对 live 故障，通用分类测试不能替代该次故障的复现证据。

贡献归属采用更严格的标准：Git 作者信息只能证明提交署名，不能证明谁发现问题、谁决定方案或谁编写代码。现有材料没有可靠的人机分工记录，因此逐条标记“待确认”。以下“面试一句话版本”描述项目事实；确认个人贡献前，不应改说成“我独立发现并修复”。

## 1. 动态扩围成功后，同一 Tool 仍被禁止重跑

- **现象**：从 `task_a` 的检查结果获得 `task_b` 的扩围依据，scope 已升到版本 2，但同一 checker 只在旧范围执行过一次；针对 `task_a/task_b` 的合法重跑被拒绝。
- **怎么发现的**：历史 dossier 的 `scope_rerun` 用固定动作序列 `call_tool → expand_scope → call_tool` 捕获 Tool 输入。旧版本实际输入只有 `[[task_a]]`，预期还应有 `[task_a, task_b]`。dossier 保存了两次历史失败和修复后通过记录；本次再次复现相同失败。最初由谁发现、是否来自手工运行，待确认。
- **根本原因**：控制器只按 `called_tool_names` 判断“调用过”，没有把 scope 版本纳入重复调用判断。
- **修复方式**：增加 `called_tool_scope_versions`，同一 Tool 在同一版本仍禁止重复调用；合法扩围后允许在新版本执行，并重新构造可信输入。没有版本的旧调用历史继续保守阻止重跑；同步调整 Provider 提示中的重复调用规则。
- **验证方式（对应的回归测试名）**：`tests/test_tool_scope_version.py::test_same_tool_can_rerun_after_expansion_with_new_trusted_scope` 检查两个输入范围、版本 2 和无决策错误；`tests/test_tool_scope_version.py::test_same_tool_twice_at_same_scope_version_is_rejected` 确认同版本仍只执行一次。
- **证据位置**：缺陷版本 `d0bae4681064ffb1a129c7783baf4b2271158324`；修复 `2af8d80d188422d7680377c80ac23be79c48088f`；[历史记录](failure_dossier/README.md)、[结构化证据](failure_dossier/dossier.json) 中的 `scope_rerun`；[测试](../tests/test_tool_scope_version.py)；实现位于 [orchestration.py](../game_qa_agent/orchestration.py)、[models.py](../game_qa_agent/models.py)。该修复 commit 还包含其他改动，不能全部归因于本问题。
- **贡献边界**：开发者发现问题：待确认；开发者决定以 scope 版本划分调用历史：待确认；AI 实现状态字段、控制器判断、提示调整和回归测试：待确认。
- **面试一句话版本**：项目曾把 Tool 的去重粒度设成“整个调查”，导致扩围后漏执行；改为按 Tool 和 scope 版本判断，既阻止重复调用，又允许新范围接受检查。
- **可能被追问的两个问题**：① 为什么不能直接清空全部调用历史？② 如何证明新范围来自 checker 证据，而不是模型自行扩大权限？

## 2. Provider 的空 choices 响应触发 IndexError

- **现象**：本地 completion stub 返回 `choices=[]` 时，Provider 抛出 `IndexError`，没有在响应验收边界给出受控拒绝。
- **怎么发现的**：dossier 的 `empty_choices` 将修复版本的回归测试放到真实旧产品代码上运行，两次捕获 `empty_choices_index_error`；本次再次复现。该测试绕过 SDK 构造器、不联网，不能说明 DeepSeek 在线上返回过空 choices；首次发现者待确认。
- **根本原因**：代码先读取 `response.choices[0]`，没有先检查集合是否为空。
- **修复方式**：历史最小修复是在索引前判空并抛出受控 `ValueError`。后续 `6a0d9fd3541f52612f97efbc35763b0e070eb333` 引入统一 Provider 失败模型，基准版将这类响应归为不可重试的 `invalid_response`；不能把后续失败分类能力说成最初修复就已具备。
- **验证方式（对应的回归测试名）**：`tests/test_provider_completion.py::test_empty_choices_is_rejected_as_controlled_provider_failure` 检查受控异常及本地错误描述；`tests/test_provider_reliability.py::test_invalid_completion_is_normalized_without_retry` 验证异常响应分类和不重试策略。
- **证据位置**：缺陷版本 `d0bae4681064ffb1a129c7783baf4b2271158324`；修复 `2af8d80d188422d7680377c80ac23be79c48088f`；[dossier.json](failure_dossier/dossier.json) 中的 `empty_choices`；[completion 测试](../tests/test_provider_completion.py)、[可靠性测试](../tests/test_provider_reliability.py)；[providers.py](../game_qa_agent/providers.py) 的基准版 `_parse_completion`。
- **贡献边界**：开发者发现问题：待确认；开发者决定判空及后续不可重试策略：待确认；AI 实现判空、失败分类和相关测试：待确认。
- **面试一句话版本**：项目用离线回归复现了空 choices 引发的越界异常，把响应结构检查前移，使畸形响应在 Provider 边界被受控拒绝。
- **可能被追问的两个问题**：① 为什么格式错误不沿用网络超时的重试策略？② `ValueError` 与后来的 `ProviderError/invalid_response` 分别解决什么问题？

## 3. 第一次 Tool 已产出部分结果并失败，报告却显示未开始

- **现象**：第一次 checker 先 yield 一条 finding 再抛异常，状态中已保留 finding 和 `failed` 执行记录，但调查状态仍为 `start`；Report 因而使用 `not_started` 限制说明。
- **怎么发现的**：dossier 的 `partial_failure_status` 使用会部分产出后失败的生成器 checker，在历史代码上两次得到 `start != running`；本次再次复现。历史红测停在状态断言，Markdown 后续断言只在绿测执行；旧报告文字的后果也可由未变的状态映射确认，不能声称红测逐个执行过所有报告断言。
- **根本原因**：`execute_action` 仅在 Tool 完整迭代成功后才写入 `running`，异常传播跳过了状态赋值。
- **修复方式**：在授权检查和可信输入构造完成后、实际调用 Tool 前设置 `running`。保留原异常、部分 finding 和执行记录中的 finding 索引。不改 Report 实现，不增加 Tool 重试或后台恢复。
- **验证方式（对应的回归测试名）**：`tests/test_tool_execution_evidence.py::test_failed_execution_links_partial_findings_and_preserves_exception_behavior` 验证原异常对象继续传播、剩余 finish 动作未执行、执行状态为 `failed` 且 `issue_indices=(0,)`、调查和报告均为 `running`、报告包含 `still_running` 而非 `not_started`。还验证同 scope 再次调用不会被隐式授权。该异常路径的 Trace 没有 step/final_status，因此证据来自控制器状态与回归断言，不能补称存在完整 Trace。
- **证据位置**：缺陷版本 `4475c3b08e55674b6be393fa4ea0ef0f9fd09472`；修复 `ff3f03dfea9b54eabf6792b12008731f495bfdf7`；[dossier.json](failure_dossier/dossier.json) 中的 `partial_failure_status`；[测试](../tests/test_tool_execution_evidence.py)；[orchestration.py](../game_qa_agent/orchestration.py) 的 `execute_action`；下游 [report.py](../game_qa_agent/report.py)。
- **贡献边界**：开发者发现状态矛盾：待确认；开发者决定保留部分证据并继续传播异常：待确认；AI 实现状态赋值调整和报告回归断言：待确认。
- **面试一句话版本**：项目出现过“Tool 已执行且失败，报告却说未开始”的状态矛盾，通过前移状态迁移，同时保留部分结果和原异常，使报告准确表达已开始但未完成。
- **可能被追问的两个问题**：① 为什么保留 `running`，而不是新增调查级 `failed` 状态？② Trace 缺失时，如何区分 Tool 没运行、运行成功但零 finding、以及部分失败？

## 4. Agent 步数耗尽后仍返回 running

- **现象**：`max_steps=1` 时，Provider 提议未知 Tool，控制器记录一次拒绝，循环结束后调查仍返回 `running`，没有明确表达预算已经耗尽。
- **怎么发现的**：修复 commit 增加了专门回归。本次将该版本测试放到其父版本产品代码上执行，实际失败为 `assert 'running' == 'max_steps_exceeded'`；修复版本同测试通过。这里证明的是旧代码缺陷，不证明当时发生过线上死循环；最初发现过程待确认。
- **根本原因**：循环只处理 `finish/clarify/human_review` 等主动终止，没有在步数预算耗尽后将剩余的 `running` 状态转换为明确结束原因。
- **修复方式**：循环退出后，若状态仍为 `running`，设置 `max_steps_exceeded`；已进入主动终止状态的结果不被覆盖。
- **验证方式（对应的回归测试名）**：`tests/test_action_safety.py::test_max_step_exhaustion_has_explicit_terminal_status` 断言状态为 `max_steps_exceeded`，且保留一次决策错误；本次旧版 1 项失败、修复版 1 项通过，基准版也通过。
- **证据位置**：缺陷版本 `51b6deffac4fc2589cb1087e1cfd7405b2d9902a`；修复 `d0bae4681064ffb1a129c7783baf4b2271158324`；[test_action_safety.py](../tests/test_action_safety.py)；[orchestration.py](../game_qa_agent/orchestration.py) 的 `run_agent_investigation`。本次新增核验的方法和结果见下文，未修改原 dossier。
- **贡献边界**：开发者发现问题：待确认；开发者决定使用 `max_steps_exceeded` 终态：待确认；AI 实现循环后的状态转换和测试：待确认。
- **面试一句话版本**：项目把 Agent 的预算耗尽与主动完成分开记录，修复了循环已退出但状态还显示运行中的问题。
- **可能被追问的两个问题**：① 一次无效决策是否应该消耗 Agent 步数，为什么？② 步数预算与 Provider 请求重试预算有什么区别？

## 5. JSON 合法时，Provider 曾误接收非 stop completion

- **现象**：响应内容是合法的 finish action JSON，但 `finish_reason` 为 `length`、`unexpected` 或 `None` 时，旧 Provider 仍返回动作，没有拒绝响应。
- **怎么发现的**：修复 commit 加入参数化回归。本次使用该测试与旧产品代码组合，三个参数均实际得到 `DID NOT RAISE ValueError`；修复版三个参数全部通过。输入来自本地 stub，不能说模型真实发生过这三种返回，也不能把合法 JSON 描述成已经损坏。
- **根本原因**：旧验收逻辑只读取 content 并做 action schema 校验，未检查 completion 是否按协议正常结束；JSON 可解析并不足以证明响应可接受。
- **修复方式**：解析动作前要求 `finish_reason == "stop"`，其他值受控拒绝。基准版继续将此类响应纳入 `invalid_response`，不通过修补 JSON 或补充模型请求来绕过验收。
- **验证方式（对应的回归测试名）**：`tests/test_provider_completion.py::test_non_stop_completion_is_rejected_even_when_action_json_is_valid`，参数为 `length/unexpected/None`；正向对照为 `tests/test_provider_completion.py::test_stop_completion_is_accepted_before_action_parsing`。
- **证据位置**：缺陷版本 `d0bae4681064ffb1a129c7783baf4b2271158324`；修复 `2af8d80d188422d7680377c80ac23be79c48088f`；[test_provider_completion.py](../tests/test_provider_completion.py)；[providers.py](../game_qa_agent/providers.py)。本次复现独立于 dossier 中的空 choices 问题。
- **贡献边界**：开发者发现问题：待确认；开发者决定仅接受 stop completion：待确认；AI 实现验收门槛和参数化回归：待确认。
- **面试一句话版本**：项目曾只验证 action JSON，后来用历史回归证明还必须检查 completion 结束原因，避免把协议层未正常结束的响应当成有效决策。
- **可能被追问的两个问题**：① 为什么合法 JSON 仍不能接受？② `finish_reason="stop"` 是否就能保证动作有权限、业务结论正确？

## 6. 真实 live Eval 中，一次 invalid_response 使已开始的调查未完成

- **现象**：固定 8 个用例、每例 2 次的正式 DeepSeek Eval 中，`runtime_no_findings` 的第 2 次重复先成功运行 `npc_runtime_checker`，留下 scope 版本 1、零 finding 的成功执行记录；第二次决策被判为 `invalid_response`，最终状态停在 `running`，该 slot 为 `provider_failure / unscored`。
- **怎么发现的**：基准 README 的 “Recorded fixed-plan live run” 记录了 run ID `7175f92fdb9a4061953e0651e6fab68b`、执行版本 `5c8d179f7ad357950494ae921fb60cbce09e9b98`、`source_dirty=false`、`validation_only=false` 及该 slot 的结果。README 记述发布验证和独立 reader 验证均通过，但四个原始产物保存在 Git 工作树外，本次没有独立读取这些产物，也没有重跑在线 Eval。
- **根本原因**：**未确定**。安全记录只保留 `invalid_response` 类别，没有具体拒绝原因或原始响应；不能据此断言为空 choices、截断、非法 JSON、网络超时或产品 bug，也不能认定与第 2、5 条是同一次问题。
- **修复方式**：没有已确认根因，也没有证据表明为这次事件做过专项修复。已有固定策略将其记为 Provider 失败且不评分，保留先前执行证据，不重抽样或重跑该 case。正式 run 共 16/16 attempted、15 completed、15 passed、0 behavioral fail、1 unscored；`pass/evaluated=15/15`，`verified pass/planned=15/16`。这是正确记账与处置，不是“修复模型后全部通过”。
- **验证方式（对应的回归测试名）**：没有精确复现该次真实响应的回归。相关策略测试 `tests/test_live_eval.py::test_provider_failure_categories_preserve_slot_accounting` 验证合成失败不计入 behavioral fail、保留 16 个 planned slots，并按策略处理后续 slot；`tests/test_provider_reliability.py::test_exhausted_request_preserves_prior_investigation_progress` 验证另一种合成请求失败保留先前调查进度。两者只支持机制，不证明本事件根因或恢复成功。
- **证据位置**：结果文档 commit `2324c9e0aabe29d4e5df806316b2293e9ec6f66d`；执行代码 commit `5c8d179f7ad357950494ae921fb60cbce09e9b98`；[README 的正式 live run 记录](../README.md#recorded-fixed-plan-live-run)；[live Eval 策略测试](../tests/test_live_eval.py)、[Provider 进度保留测试](../tests/test_provider_reliability.py)；实现 [live_eval.py](../game_qa_agent/live_eval.py)。文档没有给出外部产物路径，不能虚构可访问的 trace 或 artifact 链接。
- **贡献边界**：开发者启动运行、识别失败和冻结评估口径的具体分工：待确认；AI 实现 runner、核验产物和撰写结果记录的具体分工：待确认；谁决定不重跑该 case：待确认。
- **面试一句话版本**：正式 live Eval 出现过一次 Provider 响应失败，项目保留部分执行证据并如实报告 15/16 planned slots 验证通过，没有用已评分样本的 100% 掩盖未评分失败。
- **可能被追问的两个问题**：① 为什么这个 slot 不是 behavioral fail，却也不能算 pass？② 不保留原始响应会怎样影响根因分析，哪些安全诊断信息足以支持后续定位？

## 核验记录与复现方式

本次仅新增本文，没有改产品代码、测试或原 dossier。核验使用已有 `ai_py` 解释器，版本为 Python 3.10.20、pytest 9.1.1、Pydantic 2.13.5、NetworkX 3.4.2、OpenAI SDK 3.6.0；默认 `python` 命令指向不可用的 WindowsApps 别名，因此实际使用解释器绝对路径。以下命令中的 `python` 指该可用环境。

在基准代码上执行以下范围，结果为 **126 passed**；这是定向测试结果，不是全仓库测试数量，也不是 live Eval 成功率：

```text
python -B -m pytest -q -p no:cacheprovider tests/test_tool_scope_version.py tests/test_provider_completion.py tests/test_tool_execution_evidence.py tests/test_action_safety.py tests/test_live_eval.py tests/test_provider_reliability.py
```

原 dossier 的元数据和产物 hash 校验通过。第 1、2、3 条本次各重跑一次历史缺陷快照，均匹配预期失败，collection/setup/teardown 无错误：

```text
python -B docs/failure_dossier/reproduce.py --check
python -B docs/failure_dossier/reproduce.py scope_rerun buggy
python -B docs/failure_dossier/reproduce.py empty_choices buggy
python -B docs/failure_dossier/reproduce.py partial_failure_status buggy
```

`--check` 只校验元数据，不执行测试。后三条命令中，预期红测的内部 pytest 退出码为 1；复现器确认匹配后自身退出码为 0。原 dossier 已保存每例两次 historical red、一次 fixed green、一次 recorded-current green；其中 `current` 固定指 `120a1db8d19d9548fdf76c4ff54e4f8b120f8612`，**不是本文基准版本**。本次基准绿测来自上述 126 项测试。

第 4、5 条本次另作历史核验：复用 [reproduce.py](failure_dossier/reproduce.py) 的 `snapshot_files` 和离线 `CHILD`，把 Git 产品文件与修复版本测试复制到临时目录，在新 Python 进程运行；没有 checkout、修改或修补历史代码。结果如下，属于本次观察，未写入原 dossier 的三例记录：

| 问题 | 产品缺陷版本 / 测试版本（短 hash） | 历史实际失败 | 修复版本同测试 |
| --- | --- | --- | --- |
| 第 4 条 | `51b6def` / `d0bae46` | 1 failed：`running != max_steps_exceeded` | `d0bae46`：1 passed |
| 第 5 条 | `d0bae46` / `2af8d80` | 3 failed：三个参数均未抛预期 ValueError | `2af8d80`：3 passed |

两组均无 collection/setup/teardown 错误。`CHILD` 的 failure-signature 分类仅认识原 dossier 三例，因此对新增两例标为 `unrecognized`；这里依据实际断言核对，不能宣称通过了原 dossier 的 signature 匹配。历史代码是在上述现有依赖环境运行，未重建当年的完整环境。

下面的离线片段可复现新增两例的红绿对照，直接在仓库根目录的该 Python 环境执行；只写临时目录，不新增仓库代码：

```python
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

h = runpy.run_path("docs/failure_dossier/reproduce.py")
cases = [
    ("51b6deffac4fc2589cb1087e1cfd7405b2d9902a",
     "d0bae4681064ffb1a129c7783baf4b2271158324",
     "tests/test_action_safety.py::test_max_step_exhaustion_has_explicit_terminal_status"),
    ("d0bae4681064ffb1a129c7783baf4b2271158324",
     "2af8d80d188422d7680377c80ac23be79c48088f",
     "tests/test_provider_completion.py::test_non_stop_completion_is_rejected_even_when_action_json_is_valid"),
]
for buggy, fixed, node in cases:
    for revision in (buggy, fixed):
        with tempfile.TemporaryDirectory(prefix="failure_log_") as directory:
            for name, data in h["snapshot_files"](revision, fixed, node).items():
                target = Path(directory) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            print(revision, node, flush=True)
            result = subprocess.run(
                [sys.executable, "-B", "-c", h["CHILD"], node], cwd=directory,
            )
            assert result.returncode == (1 if revision == buggy else 0)
```

退出码断言不能单独证明红测原因；复核时还必须检查输出中的具体断言、收集数量和阶段结果，排除环境错误。

## 总表与面试选择

| 编号 | 标题 | 证据强度 | 面试优先级 | 判断依据 |
| --- | --- | --- | --- | --- |
| 1 | 扩围后同一 Tool 无法重跑 | 强 | 高 | 历史红绿记录、基准回归；能讲清动态范围与重复调用粒度 |
| 2 | 空 choices 触发 IndexError | 强 | 高 | 历史红绿记录、基准回归；边界清楚，容易完整演示 |
| 3 | Tool 部分失败却显示未开始 | 强 | 高 | 历史红绿记录、基准回归；连接状态、部分证据和报告语义 |
| 4 | 步数耗尽仍返回 running | 强 | 中 | 修复 commit、专门回归、本次历史红绿核验；改动较小 |
| 5 | 合法 JSON 掩盖非 stop completion | 强 | 中 | 修复 commit、参数化回归、本次历史红绿核验；与第 2 条同属 Provider 验收 |
| 6 | live Eval 的一次 invalid_response | 弱（事件本身） | 中 | 真实运行仅有仓库文档记述；策略测试不能确定该事件根因 |

最值得优先准备的 **3 条是第 1、3、2 条**，建议按这个顺序讲：

1. **第 1 条：动态扩围与执行身份。** 用旧、新两次 Tool 输入范围展示问题，解释为什么授权、重复调用和 scope 版本必须配合。
2. **第 3 条：异常路径的证据一致性。** 用“一条 finding、一次失败执行、错误的 start 状态”开场，解释为什么修复落在状态迁移而不是报告措辞。
3. **第 2 条：Provider 边界的受控失败。** 用最小空列表输入演示红绿变化，区分最初判空修复与后续统一分类、不重试策略。

这三条均可展示历史失败和现有回归。第 6 条适合作为评估方法的补充，重点讲分母、未评分样本和证据边界，不包装成已定位并修复的模型问题。

## 未发生的已知风险

本节“未发生”仅表示**截至基准版本，所核对的材料中没有找到实际发生证据**，不声称未来不会发生，也不声称所有场景已排除。以下风险不计入上面的失败总表。

| 已知风险或限制 | 已有依据 | 面试时可说的边界 |
| --- | --- | --- |
| Provider context 的条数和字符截断可能遗漏决策所需事实 | [README：Provider decision context and active Tools](../README.md#provider-decision-context-and-active-tools) 说明 omitted context 可能降低决策能力 | 已有有界投影与截断标记；没有真实模型因此做错决策的事故证据，也没有 token 或成本收益测量 |
| Tool 零 finding 或 Agent 自报 finished 被误当作 QA 全面通过 | [README：Report contract and boundaries](../README.md#report-contract-and-boundaries)；`tests/test_live_eval.py::test_self_declared_finish_cannot_pass_zero_finding_oracle` | 有防御性 oracle 测试和明确报告限制；没有证据表明正式结果曾被错误发布为全面通过 |
| 进程中断导致证据发布不完整，且没有自动续跑 | [README：Fixed-plan Provider Eval](../README.md#fixed-plan-provider-eval) 说明中断发布不完整、无 resume，且无整体 run deadline | 文档和故障注入测试支持设计边界；不能说真实 live run 曾因此中断或成功恢复 |
| 无签名的 hash 校验被误当作来源认证 | [failure dossier 的 Shared limits](failure_dossier/README.md#shared-limits)；[README：Local offline evidence package](../README.md#local-offline-evidence-package) | hash 支持完整性核验，不证明产物由谁生成；没有实际篡改事件证据 |
