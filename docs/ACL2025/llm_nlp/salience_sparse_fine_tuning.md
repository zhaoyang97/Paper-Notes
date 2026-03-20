---
title: "Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models"
conference: "ACL 2025"
arxiv: "2412.13488"
code: "https://github.com/0-ml/speft"
domain: "model_compression"
keywords: ["parameter-efficient fine-tuning", "sparse adaptation", "salience metrics", "LoRA", "PEFT"]
---

# Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models

## 一句话总结

首次系统评估 8 种 salience 指标用于稀疏微调（SPEFT）的效果，发现简单的梯度指标 + 静态掩码即可提供最佳性价比，在 GSM8k 上比 LoRA 高出 22.6%，质疑了"复杂方法才能做好 PEFT"的假设。

## 研究背景与动机

1. PEFT 领域主流方法 LoRA 通过低秩分解减少可训练参数，但限制了参数选择的灵活性
2. 稀疏微调（SPEFT）通过添加极稀疏可训练矩阵 θ_sp 来适配下游任务，可自由选择更新哪些参数位置
3. 现有 SPEFT 方法使用了多种不同的 salience 指标来确定非零位置，但缺乏系统性比较
4. 关键问题一：哪种 salience 指标最适合构建稀疏掩码？（一阶梯度 vs 二阶 Fisher 信息）
5. 关键问题二：静态掩码（训练前确定）和动态掩码（训练中更新）哪个更好？
6. 作者从零成本 NAS（Neural Architecture Search）代理中获得启发，将 NAS 领域的 salience 指标引入 SPEFT

## 方法详解

### 整体框架

SPEFT 将每层权重参数化为 θ = θ₀ + θ_sp，其中 θ_sp 是极稀疏矩阵，仅更新其非零位置。关键在于如何确定非零位置（即稀疏掩码 τ 的构建）以及训练中是否更新掩码。

### 8 种 Salience 指标

**一阶指标 (6种)**:
1. **Magnitude**: |θ| — 权重绝对值
2. **Gradient**: ∂ℓ/∂θ — 损失对权重的梯度
3. **SNIP**: |∂ℓ/∂θ ⊙ θ| — 连接灵敏度（Lee et al., 2019）
4. **FORCE**: -∂ℓ/∂θ ⊙ θ — 前瞻连接灵敏度（de Jorge et al., 2021）
5. **Taylor-FO**: (∂ℓ/∂θ ⊙ θ)² — 一阶Taylor展开
6. **SynFlow**: 突触流保留，无需数据（Tanaka et al., 2020）

**二阶指标 (2种)**:
7. **GRaSP**: -(H·∂ℓ/∂θ) ⊙ θ — 梯度信号保留（Wang et al., 2020）
8. **Fisher**: 基于 Fisher 信息矩阵（对角近似）

### 掩码更新策略

- **静态掩码 (Static)**: 训练前根据 salience 指标一次性确定非零位置，训练全程不变
- **动态掩码 (Dynamic)**: 每 K 步重新计算 salience 并更新掩码（替换最不重要的位置）

### 关键设计

- salience 指标在少量校准数据上前向+反向传播一次即可计算
- 稀疏率由预设的非零比例 ρ 控制
- 静态掩码允许编译时优化（稀疏矩阵索引固定），理论上推理也可加速

## 实验关键数据

### RoBERTa-base 在 GLUE 任务上的表现

| 方法 | MRPC | SST-2 | QNLI | 平均 |
|------|------|-------|------|------|
| Full FT | 90.4 | 94.6 | 92.7 | — |
| LoRA | 89.9 | 94.0 | 92.2 | — |
| Gradient SPEFT (static) | **91.4** | 94.3 | 92.6 | — |

- Gradient SPEFT 在 MRPC 上比 LoRA 高 +1.5%，比 Full FT 高 +1.0%

### LLM 在数学推理上的表现（MetaMathQA → GSM8k）

| 方法 | 可训练参数 | GSM8k 准确率 |
|------|-----------|-------------|
| LoRA (r=16) | ~0.3% | ~35% |
| Gradient SPEFT (0.3%) | ~0.3% | **~57.6%** |

- **关键结果**: 同等参数量下，Gradient SPEFT 比 LoRA 高出约 22.6%

### Salience 指标对比（核心消融）

| 指标 | MRPC | SST-2 | 排名 |
|------|------|-------|------|
| Random | 88.7 | 93.0 | 低 |
| Magnitude | 89.2 | 93.5 | 中 |
| **Gradient** | **91.4** | **94.3** | **最高** |
| SNIP | 90.8 | 94.1 | 次高 |
| Fisher | 90.5 | 93.8 | 中 |
| GRaSP | 89.8 | 93.2 | 中低 |

### 静态 vs 动态掩码

- 动态掩码（每 K 步更新）性能与静态掩码持平甚至略低
- 动态掩码额外增加约 15-30% 的训练时间开销（需重新计算 salience）
- 结论：静态掩码足够，无需动态更新

### 关键发现

1. 简单的 Gradient 指标最优——二阶指标（Fisher、GRaSP）并无显著优势，反而计算成本更高
2. 静态掩码足够好——动态掩码不带来实质收益，还增加计算开销
3. SPEFT 在参数效率相同时显著优于 LoRA，特别是在数学推理等需要精细参数选择的任务上
4. "简单即有效"——复杂的 PEFT 方法不一定胜过精心设计的简单 baseline

## 亮点与洞察

- **打破复杂性迷思**: 挑战了"PEFT 需要越来越复杂的方法"的趋势，简单梯度+静态掩码即可
- **NAS 启发 PEFT**: 首次系统性地将零成本 NAS 代理引入 SPEFT 领域，建立了跨领域方法论桥梁
- **SPEFT vs LoRA**: 稀疏微调的灵活性（可选择任意位置更新）在某些任务上远优于低秩适应的约束结构
- **实践指导明确**: 直接给出推荐方案——Gradient + Static，研究者和工程师可以即刻采用

## 局限性

- 实验主要在 RoBERTa-base 和中等规模 LLM 上进行，更大规模模型（70B+）的效果未验证
- 未考虑稀疏矩阵的硬件加速实现（如稀疏矩阵乘法内核）
- 稀疏率 ρ 的选择对性能影响未充分探索
- 仅评估了 NLU 和数学推理任务，生成任务（翻译、摘要等）未覆盖
- 未与 QLoRA 等量化+LoRA 组合方法对比

## 相关工作

- LoRA 及其变体（Hu et al., 2021）
- 稀疏 PEFT: DiffPruning（Guo et al., 2020）、FishMASK（Sung et al., 2021）、Fish-DIP（Das et al., 2023）
- 零成本 NAS 代理: SNIP、GRaSP、SynFlow
- 彩票假说与 LF-SFT（Ansell et al., 2021）
- 混合稀疏专家与多任务稀疏掩码

## 评分

- **新颖性**: ★★★★☆ — 系统性评估本身是贡献，但单个方法论创新有限
- **技术深度**: ★★★★☆ — 8种指标 + 2种策略的组合空间覆盖全面
- **实验充分性**: ★★★★☆ — 在 NLU 和数学推理上有说服力，但任务覆盖可更广
- **实用价值**: ★★★★★ — 提供清晰的实践指南和开源框架，可直接用于 LLM 微调
