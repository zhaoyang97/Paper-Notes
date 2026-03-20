# DiffPO: Diffusion Alignment with Direct Preference Optimization

**会议**: ACL 2025  
**arXiv**: [2503.04240](https://arxiv.org/abs/2503.04240)  
**代码**: 有（论文中提及）  
**领域**: LLM对齐  
**关键词**: diffusion, inference-time alignment, parallel decoding, model-agnostic, preference optimization

## 一句话总结

提出 DiffPO，将 LLM 对齐重新建模为句子级扩散去噪过程，通过 parallel decoding 实现高效推理时对齐，作为即插即用模块可增强任意底座模型的对齐质量。

## 研究背景与动机

1. **领域现状**: RLHF 和 DPO 是主流对齐方法，但需要针对每个策略单独训练，计算资源消耗大。推理时对齐（inference-time alignment）通过直接调整输出分布避免重训练，但仍依赖策略特定的 value function。
2. **现有痛点**: 现有推理时对齐方法（ARGS、BoN）可扩展性有限（需策略特定组件），且推理延迟高（逐 token 生成）。训练时对齐方法（DPO、SimPO）需要为每个底座模型单独训练。
3. **核心矛盾**: 对齐是句子级（关注风格、格式等整体特征），但生成是 token 级（next-token prediction），这种粒度不匹配增加了学习难度。
4. **本文要解决什么**: 实现一种高效、模型无关的对齐方法，一次训练即可应用于多个底座模型，同时减少推理延迟。
5. **切入角度**: 受扩散模型全局可控性启发，将对齐过程类比为句子级去噪：从未对齐的句子 y(0) 逐步修正到对齐的句子 y(T)，每步做句子级预测而非 token 级生成。
6. **核心idea一句话**: LLM 对齐 = 句子级扩散去噪过程，用一个 DiffPO 模型做 plug-and-play 对齐增强。

## 方法详解

### 整体框架

训练阶段：收集对齐轨迹（同一 prompt 的多个不同对齐程度的 response），训练 DiffPO 模型学习将任意对齐程度的输入映射到高对齐输出。推理阶段：底座模型生成初始 response，DiffPO 接收并进行句子级修正，输出对齐后的 response。

### 关键设计

1. **对齐轨迹构建**: 对 UltraFeedback 数据集的每个 prompt，用不同底座模型生成 T=6 个 response，按 ArmoRM reward model 评分排序，最高分为 y(T)（对齐目标），其余为 y(0)~y(T-1)（不同对齐程度的中间状态）。
2. **一致性优化**: 核心训练目标是让 DiffPO 对任意中间状态 y(t) 都能直接预测出 y(T)。使用 Consistency Loss（KL 散度让输入 y(t) 时的输出分布接近输入 y(T) 时的分布，后者用 stop-gradient）+ AR Loss（标准自回归损失保持生成质量）。
3. **模型无关性**: DiffPO 优化的是句子级修正能力，不依赖特定底座模型参数。训练数据来自多个模型的 response，使得 DiffPO 学到通用的对齐修正模式。推理时不需要访问底座模型参数，兼容 API 模型。

### 损失函数 / 训练策略

总损失 L(θ) = L_AR + ω·L_Con，其中 ω=10³。L_Con 使用前向 KL 散度，L_AR 使用标准交叉熵。backbone 使用 Gemma-2-it-2B/9B 或 Llama-3-8B-Instruct，最大生成长度 N=256。

## 实验关键数据

### 主实验

Llama-3-8B-Instruct 作为底座:

| 方法 | MT-bench | AlpacaEval2 LC(%) | AlpacaEval2 WR(%) | HH-RLHF Helpful |
|------|----------|-------------------|-------------------|-----------------|
| Base | 6.78 | 36.83 | 42.12 | 0.67 |
| w. DPO | 6.90 | 47.20 | 53.56 | 0.74 |
| w. SimPO | 7.05 | 52.57 | 58.33 | 0.75 |
| w. DiffPO-9B | **7.40** | **55.84** | **61.88** | 0.72 |

DiffPO-9B 跨模型增强:

| 底座模型 | AlpacaEval2 LC(%) 原始→+DiffPO |
|---------|-------------------------------|
| Llama-3-70B-Instruct | 46.14 → **58.18** (+12.04) |
| Qwen2.5-7B-Instruct | 45.03 → **57.89** (+12.86) |
| Llama-3.2-1B-Instruct | 15.57 → **50.70** (+35.13) |

### 消融实验

不同 DiffPO 规模的影响:

| DiffPO 规模 | Llama-3-8B AlpacaEval2 LC(%) |
|------------|------------------------------|
| DiffPO-8B | 36.44 |
| DiffPO-9B | 55.84 |

9B 模型显著优于 8B，说明 DiffPO 自身的能力对修正质量很重要。结合 DPO/SimPO 预对齐的底座模型，DiffPO 可以进一步提升。

### 关键发现

- DiffPO-9B 可将 Llama-3.2-1B 的 AlpacaEval2 LC 提升 35 个百分点（15.57→50.70），展现强大的 weak-to-strong 增强能力。
- 一次训练多模型部署：同一个 DiffPO 模型可应用于 Llama、Mistral、Qwen 等不同模型族。
- Harmless 分数显著提升（如 Llama-3-SFT 从 0.91 到 0.98），表明 DiffPO 在安全性方面也有正面效果。
- 推理时间优于其他推理时方法（ARGS、BoN），在对齐质量-延迟权衡上达到最优点。

## 亮点与洞察

- 将扩散过程的直觉巧妙迁移到 LLM 对齐：句子级去噪 vs token 级奖励优化，提供全新视角。
- 真正的"一次训练，多模型部署"能力，极大降低了多模型对齐的总成本。
- Consistency 训练使得 DiffPO 可在单步或少步中完成对齐修正，避免多步迭代。
- 可叠加在现有 DPO/SimPO 之上进一步提升，与训练时对齐方法互补。

## 局限性 / 可改进方向

- 最大生成长度限制为 256 token，对长文本 response 的覆盖不足。
- DiffPO 自身需要足够强的语言能力（2B→远弱于 9B），小 DiffPO 效果有限。
- 对齐轨迹构建依赖外部 reward model（ArmoRM），引入了 reward model 的偏差。
- 句子级修正可能改变原始 response 的事实内容，忠实度保证需进一步研究。

## 相关工作与启发

- 与 Aligner/MetaAligner 等 post-hoc 对齐方法相比，DiffPO 的 parallel decoding 策略更高效。
- Consistency model（LCM/CLCM）的思想在 LLM 对齐中的创新应用。
- 可与 RLHF/DPO 流水线互补：RLHF/DPO 做基础对齐，DiffPO 做推理时增强。
- 模型无关性使其特别适合 API-only 模型（如 GPT-4o）的对齐增强。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 扩散视角的对齐方法是全新范式，parallel decoding 应用巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 跨8个底座模型、3个benchmark、多维度消融
- **写作质量**: ⭐⭐⭐⭐ — 框架图清晰，理论推导完整
- **价值**: ⭐⭐⭐⭐⭐ — 模型无关的即插即用对齐增强，极具实用价值
