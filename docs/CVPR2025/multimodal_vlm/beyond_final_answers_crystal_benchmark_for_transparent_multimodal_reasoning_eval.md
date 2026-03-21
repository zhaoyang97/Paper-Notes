# Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation

**会议**: CVPR 2025  
**arXiv**: [2603.13099](https://arxiv.org/abs/2603.13099)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: multimodal reasoning, benchmark, step-level evaluation, process reward, GRPO

## 一句话总结
提出 CRYSTAL benchmark（6372 实例），通过 Match F1 和 Ordered Match F1 两个指标在中间推理步骤层面评估 MLLM，揭示了普遍的 cherry-picking 行为和推理顺序混乱问题，并提出 CPR-Curriculum 训练策略改善推理质量。

## 研究背景与动机
1. **领域现状**：MathVista、RealWorldQA 等多模态 benchmark 只看最终答案正确率，无法区分模型是"真正理解"还是"lucky guess"——一个模型可能答对但推理过程完全矛盾。
2. **现有痛点**：(1) 答案正确率评估让模型可以通过 shortcut 获得高分；(2) 现有 CoT 评估缺乏机器可验证的中间步骤检查点；(3) 理论分析表明，以答案为中心的评估在结构上激励幻觉（penalize uncertainty）。
3. **核心矛盾**：模型可以通过生成少量高精度步骤来"cherry-pick"，看起来推理正确但实际上跳过了大量关键推理步。答案正确率和推理忠实度是两个不同的维度。
4. **本文要解决什么？** 如何系统评估 MLLM 的中间推理步质量？如何用步骤级奖励改善模型推理？
5. **切入角度**：借鉴 Delphi 方法（多专家共识构建），用多个 MLLM 独立生成推理步骤，通过语义聚类和人工验证构建参考步骤集。
6. **核心idea一句话**：用语义匹配的 F1 评估推理步骤质量，用乘法式 CPR 奖励替代加法式奖励来训练更忠实的推理。
7. **理论动机**：作者形式化证明了在答案为中心的评估下，对于任何存在不确定性的问题，模型的最优策略是生成最大化答案期望奖励的 CoT（而非忠实反映内部推理），这从结构上激励幻觉。

## 方法详解

### 整体框架
CRYSTAL 包含评估和训练两个部分：评估端提供 6372 个实例（每个含平均 11.6 个参考推理步骤），用 Match F1 和 Ordered Match F1 评分；训练端提出 CPR 奖励用于 GRPO 训练。数据覆盖 MathVista、MMMU、RealWorldQA 三大来源，涵盖数学推理、科学问答、真实场景理解等多种任务类型。

### 关键设计

1. **Delphi-Inspired 参考步骤生成 Pipeline**:
   - 做什么：为每个问题生成高质量的参考推理步骤序列
   - 核心思路：4 个不同架构的 MLLM（Qwen2.5-VL-72B, InternVL3-76B, Gemma3-27B, Llama-4-Maverick）独立生成步骤 → 用 sentence encoder 计算余弦相似度做语义聚类（connected components）→ 选最具代表性的聚类中心 → 第 5 个模型（Molmo-72B）验证逻辑一致性 → 人工质量门控（<5% 需重做）
   - 设计动机：多来源独立生成减少相关错误，语义聚类实现了步骤级的 self-consistency voting

2. **Match F1 & Ordered Match F1**:
   - 做什么：量化预测推理步和参考步的对齐质量
   - 核心思路：用 all-distilroberta-v1 编码所有步骤，余弦相似度 ≥ τ=0.35 的 greedy 1:1 匹配。Match F1 = harmonic mean(Precision, Recall)。Ordered Match F1 在此基础上乘以 LIS ratio 惩罚乱序：$\text{Ordered-F1} = \text{F1} \cdot ((1-\alpha) + \alpha \cdot \text{LIS-ratio})$
   - 设计动机：Match F1 看"有没有"，Ordered F1 看"对不对顺序"，两者互补。α=0.5 为默认值，作者验证了在 0.3-0.7 范围内结论稳健

3. **Causal Process Reward (CPR) + CPR-Curriculum**:
   - 做什么：用于 GRPO 训练的步骤级奖励函数
   - 核心思路：乘法式奖励 — 答案正确时 $R = a_w + s_w \cdot \text{F1}_{step}$，答案错误时 $R = s_w \cdot \text{F1}_{step} \cdot \lambda$（$\lambda=0.3$）。CPR-Curriculum 两阶段训练：Phase 1 只用答案奖励建立基础，Phase 2 引入完整 CPR + 渐进增加推理难度
   - 设计动机：加法式奖励允许模型只靠猜答案获得高分并忽略推理质量；乘法式强制两者耦合。PCGrad 防止准确率和推理的梯度冲突

### 损失函数 / 训练策略
GRPO with CPR-Curriculum，权重 $a_w=0.65, s_w=0.35$（通过 6 组配置消融确定）。Phase 1 建立答案能力后 Phase 2 加入步骤级奖励，训练稳定进行 2800 步（加法式在 1500 步就 NaN 梯度爆炸）。PCGrad（Projected Conflicting Gradients）用于检测并投影冲突梯度，确保准确率目标和推理质量目标不互相干扰。

## 实验关键数据

### 主实验
| 模型 | 参数量 | Accuracy | Match F1 | Precision | Recall | LIS | Ord. F1 |
|------|--------|----------|----------|-----------|--------|-----|---------|
| GPT-5 | - | 57.99% | 0.612 | 0.925 | 0.479 | 0.636 | 0.539 |
| GPT-5-mini | - | 55.59% | **0.773** | 0.978 | 0.669 | 0.560 | **0.670** |
| Gemini 2.5 Flash | - | 53.95% | 0.673 | 0.701 | **0.765** | 0.584 | 0.579 |
| Qwen3-VL-8B | 8B | **57.66%** | 0.659 | 0.827 | 0.590 | 0.624 | 0.572 |
| Gemma3-4B | 4B | 28.65% | 0.618 | 0.878 | 0.506 | 0.668 | 0.547 |
| InternVL3.5-38B | 38B | 51.21% | 0.612 | 0.892 | 0.498 | 0.643 | 0.538 |

### 消融实验（GRPO 训练策略）
| 策略 | Accuracy | Match F1 | Recall | 说明 |
|------|----------|----------|--------|------|
| Baseline (Qwen2.5-VL-3B) | 39.85% | 0.480 | 0.347 | 未训练 |
| Composite (加法式) | 44.92% | 0.426 | 0.284 | 推理反而退化，1500步后NaN |
| Answer-Only | 44.30% | 0.429 | 0.308 | 推理不变 |
| CPR (乘法式) | 41.40% | **0.633** | 0.489 | F1 +32%，但准确率略低 |
| CPR-Curriculum | **47.52%** | **0.633** | 0.493 | 两者都提升，最优 |

### 关键发现
- **Cherry-picking 是普遍现象**：20 个模型中 19 个 Precision >> Recall（1.2x-7.2x），包括未参与 benchmark 构建的商业模型。模型倾向于生成少量高置信步骤而跳过中间推理
- **准确率和推理忠实度分离**：GPT-5 准确率最高（57.99%）但 F1 排第 8；Gemma3-4B（4B）F1 超过 InternVL3.5-38B（38B），说明架构比规模重要
- **推理长度与质量的 trade-off**：更长的推理链不一定带来更高的 Match F1，部分模型（如 Gemini 2.5 Flash）生成更多步骤但顺序混乱导致 Ordered F1 下降
- **没有模型能保持推理顺序**：竞争力模型中 LIS 最高仅 0.636（GPT-5），即 36% 的匹配步骤顺序错误
- **小模型 CPR-Curriculum 提升最显著**：3B 模型通过 CPR-Curriculum 将 F1 从 0.480 提升到 0.633（+32%），同时准确率提升 7.67 个百分点，表明步骤级奖励对小模型的推理改善具有高性价比
- 加法式奖励训练导致梯度爆炸，乘法式 CPR 稳定训练并提升 F1 32%

## 亮点与洞察
- **"Lucky guess" 问题的精准诊断**：Match F1 暴露了答案正确率掩盖的严重推理缺陷。这个发现对 MLLM 评估方法论有深远影响 — 所有只看最终答案的 benchmark 都可能高估模型能力
- **乘法式 vs 加法式奖励的洞察**：加法式奖励允许模型独立最大化各项（靠猜得分 + 忽略推理），乘法式强制耦合，这个设计思路可迁移到其他多目标 RL 场景
- **Delphi pipeline 可复用**：多模型独立生成 + 语义聚类 + 质量门控的参考步骤生成方法可用于构建其他需要中间步骤标注的 benchmark
- **Curriculum 策略的必要性**：直接加入步骤级奖励会损害答案准确率，两阶段 curriculum 解耦了能力建立和推理优化，这对 RL 训练的稳定性设计有参考意义

## 局限性 / 可改进方向
- 参考步骤由 LLM 生成+聚类，可能遗漏某些合理的推理路径（多个正确推理路径只选了一条）
- Match F1 依赖 sentence encoder 的语义匹配质量，对高度抽象的数学推理步骤可能不够准确
- 阈值 τ=0.35 是在验证集上调优的固定值，不同任务领域可能需要不同阈值
- 训练实验只在 Qwen2.5-VL-3B 上做了验证，未验证更大规模模型
- 未探索如何将 CPR 与 DPO 等其他对齐方法结合
- Benchmark 覆盖的视觉推理类型以数学和科学为主，对常识推理、空间推理的覆盖有限

## 相关工作与启发
- **vs MathVista/MMMU**: 它们只评最终答案，CRYSTAL 评中间步骤，是互补关系
- **vs Multimodal-CoT**: 它做了解耦生成但没有步骤级评估指标
- **vs MPBench**: 也做 trajectory-level 评估，但 CRYSTAL 提供了更系统的指标和训练方法
- **vs PRM (Process Reward Model)**: PRM 用训练好的奖励模型打分，CRYSTAL 用参考步骤集+语义匹配，不需要额外的奖励模型，更透明可解释
- **vs Self-Consistency**: Self-Consistency 通过多次采样答案投票，CRYSTAL 通过多模型独立生成步骤做语义聚类，将投票思想从答案层面扩展到步骤层面

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 步骤级评估 + 乘法式奖励是重要创新，发现了普遍的 cherry-picking 现象
- 实验充分度: ⭐⭐⭐⭐⭐ 20 个模型评测 + GRPO 训练 + 详细消融，非常充分
- 写作质量: ⭐⭐⭐⭐⭐ Finding 1-5 结构清晰，结论令人信服
- 价值: ⭐⭐⭐⭐⭐ 对 MLLM 评估范式有重要推动作用
