# Prune4Web: DOM Tree Pruning Programming for Web Agent

**会议**: AAAI 2026  
**arXiv**: [2511.21398](https://arxiv.org/abs/2511.21398)  
**代码**: 无  
**领域**: LLM Agent / Web Agent / DOM处理  
**关键词**: DOM树剪枝, 编程式过滤, Web Agent, 元素定位, 评分函数生成  

## 一句话总结
提出 Prune4Web，通过"LLM 生成评分函数参数 + 固定启发式模板执行"的编程式 DOM 剪枝方法实现 25-50 倍候选元素缩减：三阶段 pipeline（Planner 分解子任务 → Programmatic Filter 生成评分函数剪枝 DOM → Grounder 执行操作），3B 模型在 Multimodal-Mind2Web 上达到 52.4% Step SR（超越所有同参数量基线甚至部分 9.6B/32B 模型），低级 grounding 准确率从 46.8% 提升至 88.28%。

## 研究背景与动机

1. **领域现状**：Web Agent 需要理解网页 DOM 结构来执行操作，但现代网页的 DOM 通常包含 1-10 万 token，直接输入 LLM 会导致 token 截断和注意力稀释。
2. **现有痛点**：现有方案要么直接截断 DOM（丢失关键元素），要么使用独立的排序/过滤模型（训练成本高、泛化差），要么让 LLM 做 Top-N 选择（小模型效果差）。
3. **核心矛盾**：需要在保留关键交互元素的同时大幅减少 DOM 体积。LLM 擅长理解语义但不擅长处理大规模结构化数据；启发式规则快速但缺乏语义理解。如何结合两者的优势？
4. **切入角度**：让 LLM 不直接处理 DOM，而是**生成处理 DOM 的程序参数**——即生成关键词权重字典，由固定的评分函数模板执行多层级、多匹配方式的元素打分。
5. **核心 idea 一句话**：LLM 生成评分函数的关键词参数（可控）+ 固定模板做多层级加权匹配评分（高效）= 编程式 DOM 剪枝。

## 方法详解

### 整体框架
三阶段 pipeline：Planner → Programmatic Element Filter → Action Grounder，统一在 Qwen2.5VL-3B 的双轮对话框架中训练。

### 关键设计

1. **Planner（任务规划器）**:
   - 做什么：将高级任务分解为低级子任务指令
   - 核心思路：输入任务 $T$、截图 $Sc_t$、历史 $H_t$，输出子任务 $S_t$（如"Find the destination field and Type NYC"）。**不访问 HTML**，只做策略分解
   - 设计动机：将规划与元素定位解耦——Planner 只需理解视觉和任务语义，不需要处理庞大的 DOM

2. **Programmatic Element Filter（核心创新）**:
   - 做什么：生成评分函数参数，通过固定模板对 DOM 元素打分并剪枝到 Top-N
   - 核心思路三步：
     - **Step 1 规则初筛**：只保留交互性标签（`<a>`, `<button>`, `<input>` 等）或有 `role` 属性的元素，非交互元素的文本附加到最近的交互元素作为上下文
     - **Step 2 评分函数生成**：LLM 基于子任务 $S_t$ 生成 `keyword_weights` 字典（关键词→权重 1-50），plugged into 固定的评分模板
     - **Step 3 模板执行**：三层级属性匹配（Tier1: visible text → Tier2: aria-label/placeholder → Tier3: class/id），四种匹配方式（exact > phrase > word > fuzzy，用 `rapidfuzz` + `nltk.stem.PorterStemmer`），加权求和后取 Top-20
   - 设计动机：LLM **只生成参数不生成程序**——可控性强（权重范围固定 1-50，关键词基于子任务语义），执行效率高（模板是固定 Python 代码），鲁棒性好（不依赖 LLM 写正确程序）

3. **Action Grounder（操作执行器）**:
   - 做什么：从剪枝后的 Top-N 候选中选择正确元素并生成操作
   - 核心思路：输入子任务 $S_t$ + 候选列表 $C_t$（仅 ~20 个元素而非几千个），输出 CLICK/TYPE/SCROLL 等操作

### 训练策略
- **统一双轮对话**：单模型两轮——第1轮 = Plan + Filter（生成子任务+关键词权重），第2轮 = Ground（选择元素+执行操作）
- **SFT + RFT (GRPO)**：先 SFT 再用分层奖励做 GRPO——$R = R_{\text{format}} + R_{\text{filtering}} + R_{\text{grounding}}$（均为 0/1 二值），鼓励过滤和定位同时正确
- 数据：从 Multimodal-Mind2Web 重标注 ~5503 步

## 实验关键数据

### 主实验（Multimodal-Mind2Web）

| 方法 | 参数量 | Cross-Task Step SR | Cross-Website | Cross-Domain |
|------|-------|-------------------|-------------|-------------|
| GPT-4 | — | 32.3 | 27.0 | 29.7 |
| SeeAct (GPT-4V) | — | 40.2 | 32.4 | 36.8 |
| MindAct Flan-T5XL | — | 52.0 | 38.9 | 39.6 |
| ScribeAgent | 32B | 35.6 | 32.5 | 37.3 |
| SeeClick | 9.6B | 23.7 | 18.8 | 20.2 |
| **Prune4Web (Unified)** | **3B** | **52.4** | **44.9** | **46.1** |

### 低级 Grounding 准确率

| 方法 | Recall@20 | Grounding Acc |
|------|----------|-------------|
| Qwen2.5VL-3B (无剪枝) | — | 46.80% |
| Prune4Web + Qwen2.5VL-3B | **97.46%** | **88.28%** |
| Oracle + GPT-4o | — | 82.83% |
| End-to-End GPT-4o | 85.56% | 70.84% |

### 消融实验

| 训练策略 | 框架 | Step SR |
|---------|-----|---------|
| SFT Only | Separate | 37.9% |
| SFT + RFT | Separate | 42.2% |
| SFT Only | Two-turn Dialogue | 46.5% |
| **SFT + RFT** | **Two-turn Dialogue** | **52.4%** |

### 关键发现
- **3B 模型达到 SOTA 级表现**：52.4% Step SR 超越 GPT-4 (32.3%)、SeeAct (40.2%)、ScribeAgent-32B (35.6%)，接近 MindAct (52.0%) 但用更小模型
- **Grounding 提升 from 46.8% to 88.28%**：剪枝后从几千个候选降到 20 个，小模型也能准确定位
- **编程式过滤比 LLM 直接选择好得多**：对 Qwen2.5VL-3B，LLM Top-N 给 0% 在线完成率，编程式过滤给 5.2%；对 GPT-4o-mini 也显著提升（26.3% → 31.6%）
- **0.5B 模型做下游任务接近 3B**：Prune4Web Filter + 0.5B Grounder 达到 41.3% vs 3B 的 42.2%，说明任务被充分简化
- **双轮对话比分离模型好很多**：52.4% vs 42.2%，说明 Planner 和 Filter/Grounder 之间的信息交互很重要
- **可 plug-and-play 到 UI-tars**：UI-tars + Prune4Web 从 53.6% 升到 54.9%

## 亮点与洞察
- **"LLM 生成参数 + 固定模板执行"**是最精妙的设计哲学——不让 LLM 写程序（不可控），不让 LLM 选元素（大 DOM 下不准），而是让 LLM 告诉模板"用什么关键词、多重要"，由模板做可靠执行。这是 LLM + 传统方法协作的优秀范例
- **三层级四匹配方式的评分模板**设计非常工程化——考虑了属性优先级（可见文本 > aria-label > class）和匹配精度（精确 > 短语 > 词 > 模糊），用 NLP 工具（词干提取、模糊匹配）增强鲁棒性
- **97.5% 的 Recall@20**意味着几乎不会漏掉正确元素，同时将候选从几千减到 20——这是剪枝方法的理想效果

## 局限性 / 可改进方向
- Planning 是主要瓶颈——Planner 产生错误/不进步的计划时下游无法弥补
- 当网页使用非标准 HTML（如用 `<div>` 模拟按钮）时，规则初筛可能遗漏交互元素
- 纯图标元素（无文本/aria-label）无法被关键词匹配机制捕获
- 在线评估只有 30 个任务，规模有限
- 训练数据仅 ~5500 步，可能不够覆盖所有网页类型

## 相关工作与启发
- **vs SeeAct**：SeeAct 用 GPT-4V 端到端处理完整 DOM+截图，Prune4Web 先剪枝再处理——3B 模型超过 GPT-4V 级基线
- **vs MindAct**：MindAct 用单独的排序模型过滤 DOM，Prune4Web 用编程式方法——Recall@20 相当（97.15% vs 97.55%）但 Prune4Web 框架更统一
- **对 Web Agent 开发的启示**："生成参数而非程序"的范式可以推广到所有需要 LLM 与结构化数据交互的场景（如数据库查询、API 调用参数配置）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "编程式DOM剪枝"是非常巧妙的设计，LLM生成参数+模板执行的范式很有启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 离线+在线评估 + 低级grounding + Recall@k + 消融 + plug-and-play验证 + 小模型实验
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，评分模板细节充分，但论文结构层次较多需仔细消化
- 价值: ⭐⭐⭐⭐⭐ 3B模型达SOTA级表现 + 97.5% Recall@20 是实用的工程范式，对Web Agent社区有直接参考价值
