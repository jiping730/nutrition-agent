# P2-6 实施记录：会话上下文管理与自动压缩

## 阶段目标

解决长对话中把完整历史直接传给模型的问题。在不删除原始消息、不改变用户历史展示的前提下，为模型输入增加 Token 预算、自动摘要、上下文压缩、工具结果截断和可观测状态。

本阶段仍按个人长期使用的小型平台实施，不引入独立上下文服务、消息队列、Temporal 或新的向量数据库。

## 实际实现

### 1. Token 预算与硬限制

新增 `core/context_budget.py`：

- 对中文、英文和混合文本提供确定性 Token 估算；
- 估算结果明确标记为 `conservative_estimate`，通过安全系数控制边界，不伪装成模型精确 Token；
- 将模型上下文窗口拆分为输入预算、系统提示预算、输出预留和安全余量；
- 在软阈值前触发压缩预检查，硬限制使用于模型调用前的最终拦截；
- 保留当前用户问题，按最近消息后缀裁剪历史；
- 工具输出、记忆和检索来源分别设置上限；
- 所有回答、追问、流式和重新生成路径只把预算内的上下文交给模型，原始消息不被修改。

默认配置：

```env
CONTEXT_COMPACTION_ENABLED=true
CONTEXT_WINDOW_TOKENS=32768
CONTEXT_RESERVED_OUTPUT_TOKENS=4096
CONTEXT_RESERVED_SYSTEM_TOKENS=4096
CONTEXT_SAFETY_MARGIN_TOKENS=1024
CONTEXT_ESTIMATE_SAFETY_FACTOR=1.15
CONTEXT_PRECOMPACT_RATIO=0.70
CONTEXT_SOFT_LIMIT_RATIO=0.75
CONTEXT_HARD_LIMIT_RATIO=0.90
CONTEXT_RECENT_TURNS=6
CONTEXT_MIN_RECENT_MESSAGES=2
CONTEXT_MAX_COMPACTION_ROUNDS=3
CONTEXT_SUMMARY_MAX_TOKENS=1200
CONTEXT_SUMMARY_INPUT_TOKENS=8192
CONTEXT_COMPACTION_BATCH_TURNS=8
```

### 2. 自动摘要与确定性降级

新增 `core/context_summary.py`：

- 达到软阈值或未压缩消息达到批次阈值时触发；
- 有可用模型时请求结构化 JSON 摘要；
- 摘要包含目标、已确认事实、限制条件、决策、待确认问题、被否定事实和未解决指代；
- JSON 非法、模型超时、模型不可用时降级到确定性摘要；
- 摘要被明确标记为“不可信，仅作参考”，不会覆盖 Grill 状态、安全规则或当前用户消息；
- 用户的否定和后续修正保留在原始消息与结构化状态中，不依赖摘要推断。

### 3. 上下文组装

新增 `core/context_assembler.py`，组装顺序为：

```text
当前用户问题
  → 安全/系统约束
  → 当前 Grill 结构化状态
  → 最近消息
  → 历史摘要
  → 已授权档案与长期记忆
  → 预算后的工具/RAG 结果
```

`ConversationRuntime` 在调用 LangGraph 前完成上下文准备；`core/workflow.py` 不再依赖固定的 `messages[-5:]`，而是使用上下文包生成预算内 Prompt。Grill 的 `fields/status/target_field` 单独传递，摘要不能伪造任务状态。

- 普通回答和基于历史的追问回答共用同一套上下文组装器；追问分支不会绕过摘要、最近消息保留和硬预算限制。
- 动态压缩按工具结果、图检索、记忆、来源、最近对话、档案和摘要的优先级分层处理；不会直接截断最终 JSON，无法满足硬限制时拒绝模型调用。

### 4. 摘要持久化

新增 `conversation_contexts` 表和迁移 `0011_p2_6_context_compaction`、`0012_context_budget_metadata`，保存：

- `summary_json`、`summary_text`；
- `summary_version`；
- `covered_until_message_id`，用于增量摘要；
- 估算 Token 数、来源模型、内容哈希和降级标记；
- 压缩前后估算值、压缩率、压缩原因、压缩次数、硬限制拦截结果和估算模式；
- `tenant_id/user_id`，沿用现有多租户隔离。

删除会话时同步删除摘要记录；读取会话时依旧返回完整原始消息，摘要只是模型上下文缓存。

### 5. 流式状态和前端提示

新增流式事件：

```text
context.preflight
context.started
context.compacted
context.compaction.started
context.compaction.completed
context.overflow
context.usage
context.completed
```

只有发生实际压缩时才向前端展示简短提示：`已整理较早对话，保留当前目标和最近内容`。不会把内部摘要直接展示给用户。

## 验证结果

- `core/context_budget.py`、`core/context_summary.py`、`core/context_assembler.py` 专项测试覆盖预算、当前问题保留、模型摘要非法 JSON、确定性降级和摘要状态；
- 新增 `evaluation/context_cases.json`（20 条）和 `evaluation/context_runner.py`，可输出上下文 JSON 评测报告；
- 当前确定性上下文评测结果：预算遵循率 `1.00`、当前问题保留率 `1.00`、摘要事实保留率 `1.00`、Grill 状态保留率 `1.00`、上下文溢出率 `0.00`；
- 新增 Prometheus 上下文估算、压缩、溢出拦截、压缩率和摘要降级指标；
- Alembic 已验证从空 SQLite 连续迁移到 `0012_context_budget_metadata`；
- Ruff 检查通过；上下文三个新增模块 Mypy 检查通过；
- 前端 TypeScript 构建检查通过；
- 原有流式事件在未发生压缩时保持兼容，不会无条件增加事件；
- 当前工作树的 `data/` 和系统临时目录受宿主权限限制，相关本地全量 pytest 需要在 CI/Docker 或可写临时目录中执行。

## 交付边界

本阶段解决的是单体小型平台的上下文窗口治理：

- 不删除、不重写历史消息；
- 不保证摘要取代原始事实，结构化 Grill 状态始终优先；
- 不实现模型专用精确 Tokenizer、跨服务共享摘要缓存、分布式摘要队列或计费级 Token 账本；
- 估算值用于应用层预算和压缩门禁，不能等同于 provider 返回的真实 Token usage。
