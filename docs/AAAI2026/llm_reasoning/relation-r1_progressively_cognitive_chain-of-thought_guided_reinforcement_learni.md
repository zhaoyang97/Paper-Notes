# Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension

**会议**: AAAI 2026  
**arXiv**: [2504.14642](https://arxiv.org/abs/2504.14642)  
**代码**: https://github.com/HKUST-LongGroup/Relation-R1  
**领域**: LLM推理  
**关键词**: 关系理解, CoT引导, GRPO, 场景图生成, N元关系, 多模态

## 一句话总结

提出 Relation-R1，首个统一二元和 N 元关系理解的框架，通过渐进式认知 CoT 引导的 SFT（模板 CoT → MLLM 生成 CoT）+ GRPO 多奖励优化，在 PSG 数据集上提升 6.84~6.90%，在 SWiG 上也取得 SOTA。

## 研究背景与动机

视觉关系理解是实现类人视觉认知的核心能力。**二元关系**检测判断两个物体之间的交互（如"小孩-喝-杯子"），**N 元关系**则需要识别多个实体在一个活动中的不同语义角色（如"喝"活动中：agent=小孩、liquid=牛奶、container=杯子）。

**现有 MLLM 的关键局限**：
1. 当前多模态模型在物体级别的 grounding 上表现出色，但在关系理解上落后——它们缺乏对多实体间结构性语义依赖的建模
2. 过度依赖语言先验而非视觉语义线索：例如看到人拿着杯子就默认"人在喝杯子"，即使视觉证据并不支持
3. 二元关系和 N 元关系被分别处理，没有统一框架
4. 单纯用 SFT 会过拟合固定训练模式导致泛化差；单纯用 RL 又难以保证输出格式一致性

**核心思路**：结合 SFT 的结构化输出引导 + RL 的探索泛化能力。用认知 CoT 在 SFT 阶段建立多步推理基础（物体识别→空间定位→关系推理），再用 GRPO 通过关系特异性奖励微调泛化。

## 方法详解

### 整体框架

Relation-R1 分两阶段：
- **Stage 1（SFT）**：用认知 CoT 引导模型学会结构化推理过程和规范输出格式
- **Stage 2（RL-GRPO）**：用规则奖励（格式+二元关系+N元关系）通过策略梯度更新优化泛化能力

两种任务统一处理：二元关系用 `<ref>/<box>/<pred>` 标签输出场景图；N 元关系用 `<agent>/<box>` 等角色标签输出 grounded situation frame。

### 关键设计

1. **渐进式认知 CoT 引导（SFT 阶段）**
   - **模板 CoT（特定阶段）**：固定步骤模板，确保学到规范推理模式
     - 二元关系：Object Existence → Object Localization → Relation Existence
     - N 元关系：Activity Recognition → Entities & Roles Recognition → Entity Localization
   - **MLLM 生成 CoT（通用阶段）**：用 Qwen 2.5-VL 基于任务定义 + GT 场景图 + CoT 生成指令自动生成多样化推理路径
   - **渐进过渡**：先在模板 CoT 上充分训练建立基础，再用少量 MLLM CoT 微调引入灵活性
   - 核心洞察：模板 CoT 保证正确性但限制多样性；MLLM CoT 增加探索空间但可能引入噪声——渐进组合二者优势

2. **GRPO 多奖励优化（RL 阶段）**
   - **格式奖励 $r_{\text{form}}$**：输出必须包含 `<think>...</think><answer>...</answer>` 结构，满足 1 分否则 0 分
   - **二元关系奖励 $r_{\text{binary}}$**：$\alpha \cdot R + (1-\alpha) \cdot mR$
     - $R$（Recall）：正确预测的三元组占 GT 三元组的比例
     - $mR$（mean Recall）：各谓词类别上的平均召回率
     - 三元组正确条件：主语/谓语/宾语类别都对 + 边界框 IoU ≥ 0.5
   - **N 元关系奖励 $r_{\text{n-ary}}$**：$\beta \cdot V_e + (1-\beta) \cdot V_{\text{grnd}}$
     - $V_e$（Entity Value）：预测实体类别和语义角色的正确比例
     - $V_{\text{grnd}}$（Grounded Value）：实体边界框 IoU ≥ 0.5 的比例
   - **多任务门控**：根据输出中是否包含 `<ref>` 标签动态选择对应任务的奖励

3. **统一的二元+N 元关系处理**
   - 二元关系：输出场景图描述，用 `<ref>entity</ref><box>[[coords]]</box>` + `<pred>relation</pred>` 格式
   - N 元关系：输出 grounded situation frame，用动词+角色模板（如 "The AGENT drinks a LIQUID from a CONTAINER"）+ `<role>entity</role><box>[coords]</box>` 格式
   - 两种任务在同一模型中联合训练

### 损失函数 / 训练策略

Stage 1 用标准 SFT 交叉熵损失训练。Stage 2 用 GRPO 目标函数，采样 $G$ 个候选响应计算组内优势，通过 KL 散度正则化保持对参考策略的接近。训练基座为 Qwen 2.5-VL。

## 实验关键数据

### 主实验（PSG 场景图生成，Scene Graph Caption 格式）

| 方法 | 模型大小 | Recall | mRecall | Mean |
|------|---------|--------|---------|------|
| IMP (经典方法) | - | 16.50 | 6.50 | 11.50 |
| MOTIFS | - | 20.00 | 9.10 | 14.55 |
| ASMv2 (MLLM方法) | 7B | 35.40 | 22.86 | 29.13 |
| **Relation-R1** | **7B** | **~42** | **~30** | **~36 (+6.84~6.90)** |

### SWiG 数据集（N 元关系，Grounded Situation Recognition）

| 方法 | Value | Grnd-Value |
|------|-------|-----------|
| LEX (zero-shot) | 较低 | 较低 |
| **Relation-R1** | **SOTA** | **SOTA** |

### 消融实验（CoT 策略对比）

| CoT 策略 | 二元 Recall | N 元 Value | 同义关系泛化 |
|---------|-----------|-----------|------------|
| 无 CoT（直接 RL） | 基线 | 基线 | 差 |
| 仅模板 CoT | 较好 | 较好 | 一般 |
| 仅 MLLM CoT | 中等 | 中等 | 较好 |
| **渐进式（模板→MLLM）** | **最优** | **最优** | **最优** |

### 关键发现

- **渐进式 CoT 对同义关系泛化尤为关键**：模型能将"在旁边"和"靠近"、"拿着"和"握着"等同义关系统一理解
- **SFT 和 RL 缺一不可**：仅 SFT 过拟合训练模式，仅 RL 输出格式混乱，两阶段组合效果最佳
- **语言先验偏差被有效缓解**：RL 训练后模型更多依赖视觉证据而非默认关联（如不再看到杯子就默认"喝"）
- **小规模 CoT 数据即有效**：仅需少量 MLLM 生成的 CoT 就能在模板 CoT 基础上显著提升泛化

## 亮点与洞察

- **首个统一二元+N 元关系推理的 CoT+RL 框架**：此前两类任务分开研究，Relation-R1 展示了统一处理的可能性和优势
- **渐进式 CoT 引导策略的通用性**：先用确定性强的模板建立基础→再用多样性强的生成 CoT 扩展泛化，这个模式可迁移到任何需要结构化推理+泛化能力的任务
- **多奖励设计的精细化**：格式、二元关系、N 元关系三种奖励分别约束不同维度，比统一打分更有效
- **语言先验 vs 视觉 grounding 的冲突**被显式建模并通过 RL 缓解，对所有 MLLM 关系推理任务有启发

## 局限性 / 可改进方向

- MLLM 生成的 CoT 质量取决于教师模型（Qwen 2.5-VL），如果教师模型关系理解不够好会引入噪声
- 评估仅在 PSG 和 SWiG 两个数据集上，更多视觉关系理解 benchmark 的验证有待补充
- 二元和 N 元关系的奖励权重（$\alpha$, $\beta$）需要手动调整，能否自适应平衡？
- 目前仅处理图像中的关系，视频场景中的动态关系理解是自然延伸方向

## 相关工作与启发

- **vs ASMv2 (Wang et al. 2024)**：ASMv2 是 MLLM 场景图生成的 SOTA，但仅处理二元关系且泛化受限于 SFT 过拟合；Relation-R1 加入 RL 和 N 元关系显著超越
- **vs DeepSeek-R1**：R1 展示了 RL 能激发推理能力，但直接应用到视觉关系任务时格式不一致；Relation-R1 用 SFT 先建立格式基础再 RL 优化
- **vs LEX (零样本 GSR)**：LEX 依赖 LLM 生成描述做零样本 GSR，效率低且非端到端；Relation-R1 是端到端训练的统一框架

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一二元+N元关系 + 渐进 CoT 引导 + 多奖励 GRPO 的组合是全新设计
- 实验充分度: ⭐⭐⭐⭐ PSG 和 SWiG 双数据集、多种 CoT 策略消融、开放/封闭词汇设置
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，两阶段设计逻辑紧密
- 价值: ⭐⭐⭐⭐ 对 MLLM 关系理解和场景图生成有实际推动作用
