# High Quality Low-Memory Recognition Optimization Baseline

本文档用于固定默认高质量低内存脱敏线的主体识别优化基准。后续修改应以本文档作为边界和验收依据，避免继续出现局部补丁、样本驱动、旧问题复发、或把问题错误转移到最终模型审查层。

## 1. 适用范围

仅适用于默认高质量低内存脱敏线，不包含大文件模式、旧标准线、旧 full LLM 线、PDF 大文件分组线或历史打包启动逻辑。

核心代码入口：

- `backend/app/api/desensitize.py`
- `backend/app/workers/analysis_worker.py`
- `backend/app/workers/qwen_review_worker.py`
- `backend/app/recognizers/high_quality_lowmem_recognizer.py`
- `backend/app/rules/pipeline.py`
- `backend/app/rules/type_recognizers.py`
- `backend/app/rules/boundary_repair.py`
- `backend/app/rules/false_positive_rules.py`
- `backend/app/rules/subject_ledger.py`
- `backend/app/services/lowmem_entity_utils.py`
- `backend/app/services/chinese_uie_service.py`
- `backend/app/services/chinese_ner_service.py`
- `backend/app/services/risk_snippet_scheduler.py`
- `backend/app/services/qwen_fragment_review_service.py`
- `backend/app/services/recall_first_entity_merge_service.py`
- `backend/app/services/contextual_desensitization_service.py`
- `backend/app/services/coverage_first/`

公开脱敏主体分类只能是：

- `PERSON`
- `ORGANIZATION`
- `LOCATION`
- `GOVERNMENT`

金融机构属于官方机构规则族，最终公开类型应归入 `GOVERNMENT`，不得新增公开类别。数字、日期、金额属于另一条格式处理问题，不应混进主体识别分类设计。

## 2. 当前真实流程

默认线目前是两阶段 worker 隔离流程：

1. API 层 `_run_stage_isolated_analysis_worker_blocking()` 启动 `analysis_worker`，并关闭 Qwen review。
2. `analysis_worker` 执行主识别，产出：
   - `entities`：主识别最终公开结果。
   - `review_entities`：审查面，优先取 recognizer 的 `pre_review_merged`。
   - `review_only_candidates`：规则层 rejected / alias backscan 等只供审查的候选。
   - `analysis_metadata`：主识别元数据。
3. API 层压缩 review payload 后启动 `qwen_review_worker`。
4. `qwen_review_worker` 不再跑 UIE/NER，只基于主识别产物和 review-only 候选调度风险片段。
5. Qwen review 输出新增实体、拒绝项、ledger conflict decisions。
6. 后处理执行 merge、alias propagation、projection、validate/expand、ContextualDesensitizationService refinement、postprocess。
7. 最终导出目录和替换逻辑依赖最终实体及 coverage-first final export bundle。

当前候选生命周期：

1. Seed existing results。
2. Contract structure backfill。
3. DOCX structure backfill。
4. Primary UIE。
5. DOCX unit UIE。
6. Primary NER。
7. DOCX unit NER。
8. TypeRuleRecognizers 规则召回。
9. RuleFirstPipeline canonicalize。
10. BoundaryRepair。
11. FalsePositiveRules。
12. Format validation。
13. Deduplicate。
14. SubjectLedgerBuilder。
15. Merge。
16. RiskSnippetScheduler。
17. QwenFragmentReviewService。
18. Merge + alias propagation。
19. Public projection。
20. Contextual refinement / prune。
21. Directory / replacement / export。

## 3. 当前问题定义

本轮要解决的“漏识别”不是指以下问题：

- 某个 review-only 候选没有被最终模型审到。
- 某个候选被模型审查拒绝。
- 某个主体在目录里展示不全。
- 某个主体切块边界偏左或偏右。

本轮核心问题定义为：

> 文件中存在应脱敏主体，但它在主识别和规则层候选入口中没有稳定形成候选，后续 ledger、Qwen review、目录导出都无从处理。

因此，根因必须从候选入口、结构定位、统一准入 gate、候选生命周期诊断入手，不能先用 prompt、样本词表、或局部正则补丁解决。

## 4. 已确认的结构性风险

### 4.1 召回面窄

`TypeRuleRecognizers._recognize_text_subjects()` 当前主要覆盖：

- 后缀锚定组织和官方机构。
- 表格 label-neighbor。
- 部分字段标签值。
- 别名定义。
- 少数“功能词/并列词 + 短简称 + 右侧动作 cue”的短组织。

这意味着：

- 正常主体如果没有标准后缀、没有标签、没有落入既有动作 cue，并且 UIE/NER 未抓到，就可能完全不进候选。
- 后续 Qwen review 只能围绕已有候选和风险片段审查，并不是全文 discovery，因此无法稳定补回“从未进候选”的主体。

### 4.2 同一个主体准入判断散落在多处

当前至少有以下入口会决定主体能否进入后续流程：

- `TypeRuleRecognizers.ORG_PREFIX_NOISE`
- `TypeRuleRecognizers._clean_organization_match()`
- `BoundaryRepair._LEADING_NOISE`
- `BoundaryRepair._best_org_suffix_anchored_span()`
- `BoundaryRepair._trim_short_org_like_inner()`
- `FalsePositiveRules.assess()`
- `lowmem_entity_utils.is_non_subject_action_or_function_term()`
- `lowmem_entity_utils.subject_noun_gate()`
- `lowmem_entity_utils.subject_left_pollution_reason()`
- `ChineseUIEService._normalize_org_span()`
- `ChineseNERService._normalize_org_span()`
- `QwenFragmentReviewService` 的 deterministic trim / reject / materialize gate

这些入口不是一套统一状态机，而是多层重复判断。结果是：

- 前一层把功能词裁掉，后一层可能把裁剪后的短主体判为弱主体。
- 前一层把污染候选当作可修复，后一层可能直接 reject。
- 官方机构、金融机构、普通公司、短简称共用部分 marker，但准入规则不同，容易互相污染。
- 修一个漏洞时，如果只改其中一处，其他入口仍可能复发同类问题。

### 4.3 `is_weak_function_stripped_org()` 风险过高

当前逻辑把长度 `<= 6` 的裁剪后组织名直接视为弱主体。这会影响：

- `TypeRuleRecognizers._clean_organization_match()`
- `BoundaryRepair._trim_single()`
- `BoundaryRepair._best_org_suffix_anchored_span()`
- `QwenFragmentReviewService` 中多处 deterministic trim / entity decision 校验

该规则会误伤短公司简称、短品牌名、带英文/数字的短组织、以及功能词后紧跟的真实短主体。它不应作为全局硬拒绝规则。

### 4.4 结构单元定位存在重复文本错位风险

`TypeRuleRecognizers._resolve_docx_unit_span()` 和 `_recognize_docx_structure_units()` 在 unit 原始 span 对不上时，会退回 `text.find(unit_text)`。

这在以下场景不稳定：

- 表格中多个单元格内容重复。
- 多个空白/相同标签重复。
- 页眉、页脚、脚注、表格与正文有重复短文本。
- 虚拟文本清洗前后 span 不一致。

该退路会把后出现的结构单元映射到第一次出现的位置，导致：

- 表格主体漏识别。
- 结构候选错位。
- 后续 replacement 或目录只覆盖一部分主体。

### 4.5 模型结果也依赖同一套 gate

UIE/NER 服务并非完全独立召回。它们抽取后仍会调用 `subject_noun_gate()`、`is_non_subject_action_or_function_term()`、`is_org_like_text()` 等工具函数。

因此，一旦 gate 写错，会同时影响：

- 规则层召回。
- UIE fallback。
- NER fallback。
- DOCX unit UIE / NER。
- Qwen 输出 materialize。
- Qwen entity decision trim。

不能把规则层和模型层问题割裂处理。

### 4.6 最终模型当前不是全文兜底

最终 Qwen review 的输入来自风险片段调度：

- ledger conflict snippets。
- rule-first review snippets。
- missing candidate review snippets。
- docx structure snippets。
- header / party / definition / address / signature / narrative hotspot 等关键词片段。

如果某主体既没进入候选，也不在被调度的片段里，Qwen 不会看到它。因此，最后模型兜底能力不足不是单纯 prompt 问题，而是缺少“覆盖式发现片段”。

## 5. 优化总原则

后续优化必须遵守以下原则：

1. 不基于样本内容写规则。
2. 不新增公开主体类别。
3. 不把所有问题都交给 Qwen prompt。
4. 不扩大组织正则来换取召回。
5. 不在多个文件重复增加同类前缀/后缀词表。
6. 不修改大文件模式。
7. 不读取、打印、引用用户上传文件正文。
8. 所有改动必须增加计数级诊断，能回答候选在哪一层消失。
9. 所有修复必须有合成回归测试，不以真实样本作为规则来源。
10. 每一步只解决一个生命周期问题，避免旧问题复发。

## 6. 目标架构

目标不是新增更多零散规则，而是把默认线改成明确的候选生命周期：

```text
source text / source_structure
  -> recognition view normalization
  -> structural unit locator
  -> recall candidates
  -> boundary normalization
  -> subject admission gate
  -> candidate ledger
  -> model adjudication / discovery
  -> subject ledger
  -> merge / alias propagation
  -> public projection
  -> export directory and replacement
```

其中：

- recognition view normalization 只负责生成识别视图和 span map。
- structural unit locator 只负责稳定定位 DOCX 单元，不做主体判断。
- recall candidates 负责“宁可多进入候选账本”，但必须标注证据强弱。
- boundary normalization 只负责裁剪/拆分/扩展边界。
- subject admission gate 是唯一身份判断入口。
- candidate ledger 保留 accepted / review_only / rejected 的计数和理由。
- Qwen review 负责风险裁决和覆盖式 discovery，不替代确定性准入。
- subject ledger 只处理身份合并、冲突、别名、目录一致性。

## 7. 分阶段优化方案

### Phase 0: 建立诊断基线

目的：先能证明主体在哪一层消失。

需要新增或完善的元数据，不含正文：

- `recognition_view_original_length`
- `recognition_view_sanitized_length`
- `recognition_view_removed_inline_space_count`
- `docx_unit_count_by_container`
- `docx_unit_span_exact_count`
- `docx_unit_span_mapped_count`
- `docx_unit_span_unresolved_count`
- `candidate_raw_count_by_source`
- `candidate_raw_count_by_type`
- `candidate_after_boundary_count_by_source`
- `candidate_after_gate_count_by_source`
- `candidate_review_only_count_by_reason`
- `candidate_rejected_count_by_reason`
- `candidate_ledger_total`
- `candidate_ledger_accepted`
- `candidate_ledger_review_only`
- `candidate_ledger_rejected`
- `candidate_ledger_lost_before_subject_ledger`
- `subject_ledger_occurrence_count`
- `subject_ledger_subject_count`
- `qwen_discovery_snippet_count`
- `qwen_discovery_new_entity_count`
- `final_entity_count_by_type_source_layer`

验收标准：

- 每次任务结果元数据可以回答：
  - 主候选总数是多少。
  - 规则候选多少。
  - 结构候选多少。
  - 哪些候选进入 accepted。
  - 哪些进入 review_only。
  - 哪些被 reject。
  - reject 的理由和来源层是什么。
  - 最终模型看到多少 discovery 片段。
  - Qwen 新增实体有多少。
- 不需要查看用户正文即可判断流程是否跑完整。

### Phase 1: 统一 recognition view 和 DOCX unit 定位

目的：先修“识别前文本视图”和“结构单元定位”这两个根基问题。

改动方向：

1. 抽出统一 `RecognitionView`：
   - `original_text`
   - `sanitized_text`
   - `original_to_sanitized`
   - `sanitized_to_original`
   - `removed_inline_space_count`
   - `span_remap_fail_count`

2. 所有规则、UIE、NER、alias exact match、Qwen materialize 都通过同一个 span map。

3. DOCX unit 定位不能再用全局 `text.find(unit_text)` 作为无条件 fallback。

4. DOCX unit fallback 应按以下优先级：
   - 原始 `start/end` 精确匹配。
   - 结合 `unit_id` / `part_name` / `container_type` / `table_index` / `row_index` / `col_index` / `order_index` 的局部顺序定位。
   - 在上一单元结束后向后查找。
   - 如果仍无法唯一定位，标记 `unit_span_unresolved`，该 unit 不直接产生可写实体，只进入 review-only 诊断。

5. 对重复文本必须检测多匹配：
   - 如果同一个 `unit_text` 在全文出现多次，不能默认第一次。
   - 必须用结构顺序约束消歧。

验收标准：

- 表格、页眉、页脚、脚注、正文的 unit span 计数可见。
- `docx_unit_span_unresolved_count` 可见。
- 重复 unit text 不会映射到第一次出现。
- 不再因为结构定位错位导致主体漏进候选。

### Phase 2: 统一主体准入 gate

目的：把散落的“非名词/功能词/主体形态”判断收成一个入口。

新增或重构一个 `SubjectAdmissionGate`，输入为：

- `candidate_text`
- `candidate_type`
- `source`
- `source_layer`
- `trigger`
- `left_context`
- `right_context`
- `boundary_repairs`
- `structure_metadata`

输出为：

- `decision`: `accept` / `review_only` / `reject`
- `public_type`
- `normalized_text`
- `admission_reason`
- `risk_flags`
- `boundary_action`: `none` / `trim_left` / `trim_right` / `split` / `expand`
- `requires_manual_review`

准入策略：

1. 功能词不能作为主体开头保留。
2. 功能词后若存在完整主体或可复核短主体，不能直接丢掉后半段。
3. 短主体不能仅因长度 `<= 6` 被硬拒。
4. 短主体必须依赖证据等级：
   - 强证据：同文已有全称/别名定义/ledger 同主体/结构标签/并列主体/动作谓词。
   - 中证据：左右边界完整但无全称锚点。
   - 弱证据：只有孤立短词，无上下文。
5. 官方机构和普通公司分开准入：
   - 官方机构：必须符合官方/金融机构规则族，或者有正式/常见简称结构证据。
   - 普通公司：后缀、字号、简称、业务动作上下文进入不同证据等级。
6. 泛称、角色词、程序性称呼不进 accepted，可按证据进 reject 或 review_only。

需要迁移的判断：

- `is_non_subject_action_or_function_term`
- `subject_noun_gate`
- `subject_left_pollution_reason`
- `is_weak_function_stripped_org`
- `BoundaryRepair` 中涉及 hard reject 的逻辑
- `FalsePositiveRules` 中与主体准入重复的逻辑
- UIE/NER/Qwen materialize 的主体校验

验收标准：

- 同一候选在规则、UIE、NER、Qwen 后处理里得到一致 gate 结果。
- 不再出现一处放行、一处无理由硬拒的情况。
- `is_weak_function_stripped_org()` 不再作为全局硬拒入口。
- 每个 reject/review_only 都有统一 reason。

### Phase 3: 重写边界标准化为纯边界层

目的：让 `BoundaryRepair` 只做边界，不做身份最终判断。

边界层职责：

- 去字段标签。
- 去开头功能词。
- 去前一主体污染。
- 拆并列主体。
- 补全完整组织后缀。
- 补官方机构区域前缀。
- 对短主体按左右边界切块。

边界层不得：

- 因短主体长度直接 reject。
- 判定官方机构是否成立。
- 判定公司是否是主体。
- 判定泛称是否主体。

边界标准：

1. 左侧功能词强裁剪。
2. 功能词后存在主体形状时，输出裁剪后候选进入 gate。
3. 功能词后不存在主体形状时，进入 reject。
4. 多主体被连接词污染时，拆成多个候选。
5. 公司后缀完整优先，简称补全只在同文有明确全称或后缀邻接时触发。
6. 官方机构不得因为单独出现“银行/法院”等泛称就成立。

验收标准：

- 边界层输出 `boundary_action` 和 `boundary_reason`。
- 边界层不再吞掉可复核主体。
- 并列主体污染不会合成单一主体。
- 公司前缀污染和官方机构外溢不会因其他改动复发。

### Phase 4: CandidateLedger 替代隐式候选流

目的：让所有候选都可追踪，而不是只有 accepted 进入 SubjectLedger。

新增 `CandidateLedger`，记录：

- `candidate_id`
- `original_span`
- `current_span`
- `original_text_hash` 或长度/位置，不记录正文到日志
- `entity_type`
- `public_type`
- `source`
- `source_layer`
- `trigger`
- `recognition_view`
- `boundary_actions`
- `gate_decision`
- `gate_reason`
- `review_only_reason`
- `reject_reason`
- `linked_subject_id`

生命周期状态：

- `raw`
- `boundary_normalized`
- `accepted`
- `review_only`
- `rejected`
- `merged`
- `projected`
- `exported`

与现有 SubjectLedger 的关系：

- CandidateLedger 记录候选生命周期。
- SubjectLedger 只记录被接受或待裁决的身份主体。
- Qwen review 可以读取 CandidateLedger 的 review-only 和 coverage-discovery 区域。

验收标准：

- 任何候选消失都能定位到状态迁移。
- 任务元数据可按 source/type/reason 汇总。
- 不依赖读取用户正文即可判断漏识别发生阶段。

### Phase 5: 补强召回入口，不扩大泛规则

目的：解决“主体完全未看到”。

改动方向：

1. 全称/后缀型组织召回保持严格。
2. 官方机构召回按官方规则族独立维护。
3. 短组织召回不再只依赖少数固定动作词，而是抽象为：
   - 左边界：角色词、连接词、功能词、标点、结构单元起点。
   - 右边界：动作谓词、角色尾词、对象尾词、结构单元终点、并列连接词。
   - 证据：同文全称、别名定义、ledger 同主体、结构标签、重复出现、并列义务。
4. 表格召回要以结构单元为主，而不是正文正则为主。
5. 人名召回要区分角色标签、自然人姓名形态、组织形态冲突。
6. 地名只作为 `LOCATION` 主体或地址，不能吞公司前缀。

不得做：

- 不得简单放宽 `ORG_PATTERN`。
- 不得把动作词、泛称、程序性称呼作为组织候选放出来。
- 不得引入样本词表。

验收标准：

- `candidate_raw_count_by_source` 中规则召回入口增加，但 `rule_organization` 外溢不增加。
- 短主体召回增加主要进入 `review_only` 或带强证据的 `accepted`。
- 官方机构错误外溢不复发。
- 公司前缀污染不复发。

### Phase 6: 最终模型增加 coverage discovery 面

目的：让 Qwen 真正承担最终兜底，但不替代规则层。

新增或重构 discovery 片段调度：

1. 基于 document units 和 candidate coverage，而不是只基于关键词。
2. 对未覆盖但主体密度可能高的 unit 进行片段审查。
3. 对表格行、相邻单元格、标题/正文开头、落款、定义段、角色段建立固定覆盖策略。
4. discovery 片段独立预算，不与 ledger conflict 和普通风险片段抢预算。
5. discovery 输出仍必须通过 `SubjectAdmissionGate`。

预算分层：

- ledger conflict：硬预算，必须审。
- review-only candidates：硬预算，必须审。
- structure risk：优先预算。
- coverage discovery：独立预算。
- ordinary hotspot：剩余预算。

新增元数据：

- `qwen_discovery_snippet_count`
- `qwen_discovery_snippet_selected_count`
- `qwen_discovery_unit_count`
- `qwen_discovery_raw_candidate_count`
- `qwen_discovery_materialized_entity_count`
- `qwen_discovery_new_entity_count`
- `qwen_discovery_rejected_by_gate_count`
- `qwen_discovery_budget_exhausted`

验收标准：

- 最终模型能看到未覆盖结构单元。
- Qwen 新增实体不再只来自已有候选附近。
- discovery 不绕过规则 gate。
- discovery 的新增、拒绝、预算耗尽可计数。

### Phase 7: 合并、别名、目录与导出一致性

目的：保证识别到的主体全部进入最终目录和替换。

改动方向：

1. merge 不应因为 overlap 直接丢弃低优先级但不同身份的候选。
2. alias propagation 应基于 SubjectLedger / CandidateLedger 的 identity，而不是仅基于字符串 exact match。
3. public projection 前后要记录计数。
4. ContextualDesensitizationService prune 不能无诊断删除实体。
5. coverage-first directory rows 和 final entities 要有一致性检查。

新增元数据：

- `merge_input_count`
- `merge_output_count`
- `merge_dropped_overlap_count_by_reason`
- `projection_input_count`
- `projection_output_count`
- `contextual_pruned_count_by_reason`
- `directory_entity_input_count`
- `directory_entity_output_count`
- `directory_missing_desensitized_entity_count`

验收标准：

- 所有实际进行脱敏替换的主体都进入目录。
- 所有目录主体都有 replacement。
- 所有 replacement 对应至少一个 occurrence。
- final entity、directory row、rewrite entry 三者能对账。

## 8. 测试基准

后续每个阶段必须增加合成测试，不能使用真实用户文件内容作为规则来源。

最低测试集合：

1. DOCX unit 定位测试：
   - 重复单元格文本。
   - 相同标签多行。
   - 页眉/正文重复。
   - span 不匹配但结构顺序可定位。
   - span 不匹配且无法唯一定位。

2. 统一 gate 测试：
   - 功能词 + 完整公司。
   - 功能词 + 短组织。
   - 功能词 + 泛称。
   - 公司 + 连接词 + 公司。
   - 公司 + 官方机构。
   - 单独泛称法院/银行。
   - 带正式区域前缀的法院/银行/政府机构。
   - 短组织 + 动作谓词。
   - 短组织 + 对象尾词。

3. 边界测试：
   - 左污染裁剪。
   - 右污染裁剪。
   - 并列拆分。
   - 地区名前缀补全。
   - 分公司后缀补全。
   - 不完整后缀不强补。

4. Qwen discovery 调度测试：
   - 已有候选覆盖区域不重复审。
   - 未覆盖结构单元进入 discovery。
   - ledger conflict 不被 discovery 抢预算。
   - review-only candidates 不被普通 budget 裁掉。

5. 目录一致性测试：
   - final entities 全部进入目录。
   - 替换项和目录项数量可对账。
   - 同一主体不同称呼归并但 occurrence 保留。

建议命令：

```bash
python3 -m compileall -q backend/app
pytest -q backend/tests/test_high_quality_lowmem*.py backend/tests/test_*rule*.py backend/tests/test_*docx*.py
npm run build
git diff --check
```

具体测试文件可按现有测试结构拆分，不要求一次性建立大而全的测试套件，但每个 phase 的核心风险必须有测试。

## 9. 验收指标

每次优化后应至少检查以下元数据，不读取正文：

- `analysis_worker_process = true`
- `analysis_stage_isolation = true`
- `review_started = true`
- `review_completed = true`
- `review_model_used = true`
- `rule_first_input_candidates`
- `rule_first_type_rule_candidates`
- `rule_first_output_candidates`
- `rule_first_rejected_candidates`
- `candidate_ledger_total`
- `candidate_ledger_accepted`
- `candidate_ledger_review_only`
- `candidate_ledger_rejected`
- `docx_unit_span_unresolved_count`
- `qwen_discovery_new_entity_count`
- `post_review_merged`
- `alias_propagation_added`
- `final`
- `directory_entity_output_count`
- `directory_missing_desensitized_entity_count`

质量判断标准：

- 识别入口改动后，候选总数可以增加，但明显外溢错误不能同步增加。
- review-only 增加是可接受的，直接 accepted 的弱主体增加不可接受。
- 官方机构必须保持强规则，不得因为出现“法院/银行”泛称就识别。
- 公司前缀污染不得复发。
- 多主体连接污染不得复发。
- 表格主体遗漏必须能用结构定位指标解释。
- 最终模型新增实体必须有 discovery / review-only / ledger 来源标记。

## 10. 后续实施顺序

严格按以下顺序推进：

1. Phase 0：补诊断计数和候选生命周期元数据。
2. Phase 1：修 RecognitionView 和 DOCX unit 定位。
3. Phase 2：统一 SubjectAdmissionGate。
4. Phase 3：重构 BoundaryRepair 为纯边界层。
5. Phase 4：建立 CandidateLedger。
6. Phase 5：补强规则召回入口。
7. Phase 6：增加 Qwen coverage discovery。
8. Phase 7：目录、替换、导出一致性对账。

不能跳过 Phase 0 和 Phase 1。否则后续所有改动仍然无法证明是否生效，也无法判断漏识别到底发生在哪一层。

## 11. 禁止项

后续优化禁止：

- 根据真实样本内容新增词表。
- 为某个截图或某个具体词写特殊规则。
- 直接放宽 `ORG_PATTERN` 来提高召回。
- 在 `qwen_fragment_review_service.py` 里继续堆 prompt 替代候选入口修复。
- 在多个文件分别添加同一类功能词。
- 把金融机构做成新公开类别。
- 让官方机构因泛称“法院/银行”外溢。
- 让功能词开头的污染候选保留功能词。
- 让结构单元 fallback 到全文第一次 `find()`。
- 修改大文件模式。

## 12. 当前非本轮问题

以下问题不作为本轮主体漏识别优化主线：

- 大文件模式。
- 历史旧线清理。
- 本地启动器。
- GitHub 上传边界。
- PDF/WPS 审查转换功能。
- Ollama 通用配置残留。

如后续要处理这些问题，应另起任务，避免污染默认高质量低内存主体识别优化。

