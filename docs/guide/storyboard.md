# Storyboard (Script-to-Shots) 任务

本文档说明为适配 storyboard 分镜拆分任务所做的改动，以及如何启动训练。

---

## 1. 任务概述

Storyboard 任务：给定一段短剧剧本文本，生成结构化分镜头表（CSV格式），包含镜号、画面内容、景别、拍摄角度、运镜、角色、台词等字段。

训练目标：通过 SkillOpt 循环优化一份 Skill 文档，使 target 模型生成的分镜表在结构与专业度上逼近导演人工标注。

---

## 2. 适配改动一览

### 2.1 新增环境模块 `skillopt/envs/storyboard/`

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包声明 |
| `dataloader.py` | `StoryboardDataLoader` — 从 `items.json` 加载 train/val/test split |
| `adapter.py` | `StoryboardAdapter(EnvAdapter)` — 对接 SkillOpt 训练循环 |
| `rollout.py` | 调 target 模型生成分镜 CSV，再分别计算 hard/soft 分数 |
| `evaluator.py` | LLM-as-judge 评分（5维度百分制），产出 soft score |
| `hard_metrics.py` | 纯算法硬指标：镜头数偏差、切点对齐率、景别编辑距离 |
| `prompts/rollout_system.md` | target 模型的 system prompt 模板 |
| `prompts/analyst_error.md` | 失败样本反思 prompt |
| `prompts/analyst_success.md` | 成功样本反思 prompt |

### 2.2 评分体系

**Hard score（gate 指标，纯算法，无 API 开销）：**

| 子指标 | 含义 | 理想值 |
|--------|------|--------|
| `shot_count_deviation` | \|AI 镜头数 - GT 镜头数\| / GT 镜头数 | 0 |
| `cut_alignment_rate` | GT 切点中被 AI 匹配到的比例（±1句容差） | 1.0 |
| `shot_scale_edit_dist` | 景别序列归一化编辑距离 | 0 |

复合公式：`hard_score = 0.3*(1-count_dev) + 0.4*cut_align + 0.3*(1-scale_edit)`

**Soft score（LLM judge，信号丰富但有成本）：**

5个维度各 0-100：beat_pacing, shot_progression, camera_movement, ai_gen_quality, script_fidelity。overall 百分制映射到 [0,1]。

训练循环中 hard_score 用于 gate（pass/fail），soft score 作为优化信号的辅助参考。

**Gate 逻辑（rollout.py）：**

- `hard_score < 0.5` → 标记为失败（`fail_reason` 被设置），触发 analyst 反思
- CSV 解析失败 → 直接标记失败
- 否则视为通过

注意：`result["hard"]` 为连续值 0-1（非二值），`compute_score` 会对 batch 内所有样本的 hard_score 取均值作为整体 hard accuracy。

**Evaluator 截断容错：**

LLM judge 偶尔因 `max_completion_tokens` 限制导致 JSON 响应被截断。`_extract_json` 增加了 fallback：用正则逐字段提取数值分数，只要能解析出 `overall` 字段即视为有效，避免因截断丢失整条评分。

### 2.3 Rollout 输出结构

每个样本的 rollout 结果包含：

```json
{
  "id": "ep5-p01",
  "task_type": "storyboard",
  "hard": 0.85,
  "soft": 0.92,
  "score": 92.0,
  "dimensions": {"beat_pacing": 95, "shot_progression": 90, ...},
  "reasoning": "...",
  "hard_metrics": {
    "shot_count_deviation": 0.1,
    "cut_alignment_rate": 0.9,
    "shot_scale_edit_dist": 0.15,
    "hard_score": 0.85
  },
  "fail_reason": null
}
```

每个样本的详细输出保存在 `predictions/{id}/`：
- `system_prompt.txt` — 实际发给 target 的 system prompt
- `user_prompt.txt` — user prompt
- `response.txt` — 模型原始输出
- `eval_result.json` — 完整评分（含 hard_metrics + LLM judge）

### 2.4 Claude CLI Backend 适配

`skillopt/model/claude_backend.py` 的关键改动：

| 改动 | 原因 |
|------|------|
| system prompt 改用 `--append-system-prompt-file`（写临时文件） | 原 `--append-system-prompt` 直接传参，长 prompt 超出 Windows 命令行长度限制 |
| user prompt 改用 stdin（`subprocess.run(cmd, input=...)`) | 同上，避免命令行参数过长 |
| `shutil.which("claude")` 自动发现 CLI | Windows 上 Python subprocess 找不到 npm 安装的 `.cmd` 脚本 |
| 添加 `encoding="utf-8", errors="replace"` | Windows 默认编码非 UTF-8，中文内容会乱码 |

### 2.5 环境变量配置

`.env.example` 新增了 claude_chat backend 所需的环境变量说明：

```bash
# Backend 选择
export OPTIMIZER_BACKEND=claude_chat
export TARGET_BACKEND=claude_chat

# 模型部署名（Claude CLI 自动解析为 Bedrock model ID）
export OPTIMIZER_DEPLOYMENT="global.anthropic.claude-opus-4-6-v1[1m]"
export TARGET_DEPLOYMENT="global.anthropic.claude-opus-4-6-v1[1m]"
```

认证方式二选一：
1. **Anthropic API Key**：设置 `ANTHROPIC_API_KEY=sk-ant-...`
2. **AWS Bedrock**：在 Claude Code `settings.json` 中配置 `CLAUDE_CODE_USE_BEDROCK=1`、`AWS_REGION`、`AWS_BEARER_TOKEN_BEDROCK`，无需额外环境变量

### 2.6 训练脚本注册

`scripts/train.py` 的 `_register_builtins()` 中新增 storyboard adapter 注册：

```python
from skillopt.envs.storyboard.adapter import StoryboardAdapter
_ENV_REGISTRY["storyboard"] = StoryboardAdapter
```

配置中 `env.name: storyboard` 即可触发加载。

### 2.7 其他改动

- `skillopt/envs/storyboard/evaluator.py`：`_extract_json` 增加截断 JSON fallback 正则解析
- `skillopt/engine/trainer.py`：所有 `open()` 调用添加 `encoding="utf-8"`
- `skillopt/gradient/reflect.py`、`skillopt/optimizer/slow_update.py`：同上 encoding 修复
- `skillopt/config.py`：YAML 加载添加 `encoding="utf-8"`

### 2.8 数据格式

每条 item 为一个 dict：

```json
{
  "id": "ep5-p01",
  "task_type": "storyboard",
  "script_text": "剧本原文...",
  "ground_truth_csv": "分镜号,场景,画面内容,景别,拍摄角度,运镜,角色,台词\n0,,..."
}
```

数据存放于 `data/storyboard_split/`，含 `train/`、`val/`、`test/` 三个子目录，各自有 `items.json`。

---

## 3. 如何启动训练

### 3.1 前置条件

1. Python 环境已安装 SkillOpt 依赖：`pip install -e .`
2. Claude CLI 已安装且在 PATH 上（`npm install -g @anthropic-ai/claude-code`）
3. 准备好初始 Skill 文件，路径配置在 `configs/storyboard/default.yaml` 的 `env.skill_init`

### 3.2 启动命令

```bash
python scripts/train.py --config configs/storyboard/default.yaml
```

自定义输出目录：

```bash
python scripts/train.py --config configs/storyboard/default.yaml --out_root runs/storyboard_exp01
```

### 3.3 配置说明

`configs/storyboard/default.yaml` 关键参数：

```yaml
model:
  optimizer_backend: claude_chat       # optimizer 用 Claude CLI
  target_backend: claude_chat          # target 也用 Claude CLI
  optimizer: "global.anthropic.claude-opus-4-6-v1[1m]"
  target: "global.anthropic.claude-opus-4-6-v1[1m]"

train:
  num_epochs: 6        # 训练轮数
  batch_size: 2        # 每 step 样本数（数据集小，设为2）
  accumulation: 2      # 梯度累积步数

env:
  name: storyboard
  skill_init: "path/to/initial/SKILL.md"   # 初始 skill 文件
  split_mode: split_dir
  split_dir: data/storyboard_split          # 数据目录
  workers: 2                                # rollout 并发数
  max_completion_tokens: 16384              # target 模型最大输出 token
  exec_timeout: 300                         # 单次调用超时（秒）

optimizer:
  learning_rate: 3          # 每步最大编辑数
  use_slow_update: true     # epoch 边界做 slow update
  use_meta_skill: true      # 跨 epoch 记忆
```

### 3.4 验证 GT baseline

在正式训练前可验证评分体系是否正常工作：

```bash
# 仅跑 hard metrics（无 API 开销，秒级完成）
python scripts/verify_gt_baseline.py

# 含 LLM judge soft score（需要 Claude API）
python scripts/verify_gt_baseline.py --soft
```

预期结果：hard=1.0（GT vs GT 完美匹配），soft≈0.90-0.97（LLM judge 天花板）。

### 3.5 查看训练结果

训练完成后在 `out_root` 目录下：

```
runs/storyboard_exp01/
├── config.json              # 运行时配置快照
├── history.json             # 逐 step 得分记录
├── best_skill.md            # 最优 skill
├── summary.json             # 最终总结
├── skills/                  # 每步 skill 快照
│   ├── skill_v0001.md
│   └── ...
└── steps/                   # 每步详细中间产物
    ├── epoch01_step001/
    │   ├── predictions/     # rollout 输出
    │   ├── patches/         # 反思生成的 patch
    │   ├── step_record.json
    │   └── ...
    └── ...
```
