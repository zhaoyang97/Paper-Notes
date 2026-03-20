# GAPO: Learning Preferential Prompt through Generative Adversarial Policy Optimization

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2503.20194](https://arxiv.org/abs/2503.20194)  
**代码**: [https://github.com/MikeGu721/GAPO](https://github.com/MikeGu721/GAPO) (有)  
**领域**: LLM对齐 / 偏好优化 / 约束生成  
**关键词**: Preferential Prompt, GAN, PPO, Encoder-only Reward Model, 约束遵循  

## 一句话总结

提出 GAPO（Generative Adversarial Policy Optimization）框架，将 GAN 的对抗训练机制与 PPO 结合，使用 encoder-only 奖励模型替代传统 decoder-only 架构，通过"Preferential Prompt"（修改 prompt 中的约束而非 response）的新范式来增强 LLM 对细粒度约束的理解和遵循能力，在 IFEval 和产品描述生成任务上大幅超越 DPO/KTO/SimPO 等基线。

## 背景与动机

LLM 在实际应用中需要严格遵循预定义约束（格式、风格、内容等），尤其在法律文书、医疗记录、工作流自动化等场景中。现有方法主要分两种：(1) 直接合成满足约束的指令-响应对（SFT 范式），但模型只学到"什么是正确回答"而不理解约束本身，容易产生幻觉和走捷径。(2) 偏好响应优化（DPO 等），固定 prompt 通过偏好 response 对来调整输出概率，但也没有教会模型理解约束细节。

两种方法的共同缺陷是：**模型不理解约束**——它只学了给定 prompt 下什么样的 response 是好的，没有学会区分约束之间的微妙差异。当遇到新的或略有变化的约束时，性能会脆弱地崩溃。

## 核心问题

作者提出了一个关键观察：与其在 response 端做偏好学习（preferential response），不如在 prompt 端做偏好学习（preferential prompt）——即固定 response，修改 prompt 中的约束条件，让模型学会区分"满足约束的 prompt"和"不满足约束的 prompt"。这样模型能更深入理解约束本身的语义。

但这个方向面临两大技术挑战：(1) 现有 LLM 均为 decoder-only 架构，单向注意力机制天然不擅长检测 prompt 与 response 之间的不匹配。(2) 不同复杂度的约束之间存在难度差距，需要构造渐进式训练样本来桥接，传统方法需要大量人工干预。

## 方法详解

### 整体框架

GAPO 包含两个阶段：

**阶段一（Warm-up）**：用已有的偏好数据训练一个 encoder-only 的奖励模型（Longformer，0.4B）。这个模型接收 (prompt, response) 对，输出匹配分数。训练数据来自约束感知数据增强：对原始约束集做修改或插入操作，生成不匹配的 rejected prompt，形成 (accepted prompt + response, rejected prompt + response) 偏好对。

**阶段二（Adversarial Training）**：Generator（Qwen-2.5-7B）和 Reward Model 交替训练。Generator 根据 prompt 生成 response，Reward Model 评判生成质量并提供奖励信号，通过 PPO 更新 Generator。同时，Generator 生成的新样本也被加入 Reward Model 的训练集，迫使 Reward Model 不断提升判别能力。这种对抗动态让训练自动产生渐进式难度的样本。

### 关键设计

1. **Encoder-only 奖励模型**：使用 Longformer-Large-4096（仅 0.4B 参数）作为 Reward Model，替代传统的用 LLM 本身（7B+）做奖励模型的做法。Encoder 的双向注意力机制比 decoder 的单向注意力更擅长捕捉 prompt-response 之间的对应关系——这是处理 Preferential Prompt 数据的关键。参数量小一个数量级，计算成本大幅降低。

2. **约束感知数据增强（Constraint-Aware Data Augmentation）**：不需要人工构造偏好对，而是通过两种操作自动生成：(a) **约束修改**——随机选取一个约束并修改使其与原始 response 不兼容；(b) **约束插入**——添加一个与现有约束冲突的新约束。这样每对数据变成 (accepted_prompt, response) 和 (rejected_prompt, response)，核心差异只在 prompt 的约束部分。

3. **GAN-PPO 融合机制**：区别于标准 PPO（Reward Model 训练完后冻结），GAPO 中 Reward Model 和 Generator 迭代对抗训练。Generator 产生越来越好的输出 → Reward Model 需要更精细的判别 → 推动 Generator 进一步提升。实验表明，不同 Reward Model 在约 A12 阶段后稳定收敛，分数层化（0.2~0.95），说明对抗训练成功建立了平衡动态而非退化。

### 损失函数

奖励模型用交叉熵损失：
$$L_R(\theta) = -\mathbb{E}_{(c,t,y)\sim\mathcal{D}'} [y\log R(c,t) + (1-y)\log(1-R(c,t))]$$

Generator 的目标函数是标准 PPO 形式，采用优势函数 $A_n = Q_\pi(c_n, t_n) - V_\pi(c_n)$，其中 $Q_\pi$ 结合了 Reward Model 的即时奖励和折扣未来收益。

## 实验关键数据

| 数据集 | 指标 | GAPO | PPO | SFT | DPO | KTO | SimPO | ORPO |
|--------|------|------|-----|-----|-----|-----|-------|------|
| IFEval | Overall Accuracy | **83.9%** | 75.6% | 78.3% | 33.3% | 54.4% | 30.6% | 33.9% |
| PDD (GPT-4o评) | Score | **90.2%** | 89.7% | 82.6% | 5.4% | - | 2.9% | 7.5% |
| PDD (人工评) | Score | **89%** | 81% | 60% | 0% | - | 0% | 0% |

**Preferential Prompt vs Preferential Response 对比**（6600样本，PDD 数据集）：

| 设置 | PPO | GAPO |
|------|-----|------|
| Preferential Response | 78.5% | 82.9% |
| Preferential Prompt | 89.4% | **95.4%** |
| PP 相对 PR 提升 | +10.9% | **+12.5%** |

### 消融实验要点

- **Preferential Prompt 一致性优于 Preferential Response**：在所有样本量（2k/4k/6.6k）和两种优化方法（PPO/GAPO）下，PP 均显著优于 PR，验证了从 prompt 端学习约束的有效性
- **GAPO 的 scaling 效率优于 PPO**：从 4.2M 到 13.0M token，GAPO 在 PP 设定下提升 24.8 pp，PPO 只提升 20.9 pp
- **Encoder-only RM 的参数效率**：0.4B 的 Longformer 作为 RM 即可超越使用 7B LLM 做 RM 的传统 PPO
- **DPO/SimPO/ORPO 的灾难性失败**：在 Preferential Prompt 场景下，这些方法几乎完全崩溃（GPT-4o 评分均低于 10%），因为 decoder-only 架构无法有效捕捉 prompt 内约束的微妙差异
- **对抗训练动态**：Reward Model 在 A1~A7 阶段快速提升，A12 后稳定收敛，不同模型的最终分数在 0.2~0.95 之间形成层化，表明训练没有退化

## 亮点

- **问题定义的创新**：Preferential Prompt 这个概念非常巧妙——传统方法固定 prompt 比较 response 对，GAPO 固定 response 比较 prompt 对，让模型真正理解约束而非记忆回答。这个视角转换是本文最大的贡献
- **架构选择的洞察**：用小型 encoder-only 模型做 RM 而非 LLM 本身，既解决了 decoder-only 架构不擅长 prompt-response 匹配的根本问题，又大幅降低了计算成本。0.4B vs 7B，效果还更好
- **GAN-PPO 的无缝集成**：传统 PPO 需要先训好 RM 再训 Generator，GAPO 让两者对抗训练，自动产生渐进难度的样本，省去了人工构造中间难度数据的繁琐步骤
- **DPO系列方法的失败模式分析**：实验清晰展示了 DPO/SimPO/ORPO 在 preferential prompt 场景下的灾难性崩溃，为理解这些方法的局限性提供了有价值的实证

## 局限性 / 可改进方向

- **计算开销大**：同时训练 Generator、Reward Model 和 Critic Model 的对抗过程，计算需求远高于 DPO 等直接优化方法，限制了大规模部署
- **依赖基座模型能力**：GAPO 对基座模型的初始生成能力有要求——如果基座模型本身生成质量差，会影响 RM 的训练质量，形成恶性循环。更适合作为已有能力强的模型的增强工具
- **任务多样性有限**：主要在产品描述生成（PDD）和指令遵循（IFEval）两个任务上验证，缺少更多样化的约束场景（如代码生成约束、安全约束等）
- **Longformer 的上下文限制**：使用 4096 token 窗口，对于超长约束的场景可能受限

## 与相关工作的对比

| 方法 | 偏好数据类型 | RM 架构 | 对抗训练 | PP 场景表现 |
|------|------------|---------|---------|-----------|
| DPO/SimPO/ORPO | Preferential Response | 无需RM（隐式） | 否 | 灾难性崩溃（<10%） |
| PPO (标准) | PR 或 PP | Decoder-only（冻结） | 否 | 良好但次于GAPO |
| **GAPO** | **PP（核心）/PR** | **Encoder-only（动态）** | **是** | **最优（95.4%）** |

- **vs DPO/SimPO/ORPO**：这些方法在response端做隐式奖励建模，decoder-only 的单向注意力无法捕捉 prompt 端约束差异，在 PP 场景完全失效
- **vs 标准 PPO**：虽然 PPO 也能处理 PP 数据，但其 RM 训练完后冻结，无法随 Generator 进步而提升判别标准。GAPO 的对抗机制让 RM 持续进化，最终效果更优
- **vs AMaPO (AAAI 2026)**：AMaPO 关注的是 DPO 框架内 margin 的自适应问题，仍停留在 preferential response 范式内。GAPO 从根本上改变了偏好学习的对象（prompt vs response）

## 启发与关联

- **与 RLHF 副作用检测 idea 的关联**：GAPO 的 encoder-only RM 思路可以用于 [RLHF 副作用检测](../../../ideas/llm_nlp/20260317_rlhf_side_effect_detection.md)——用双向注意力模型来检测 RLHF 训练过程中对非目标属性的隐性影响
- **Preferential Prompt 范式的推广潜力**：这个"固定输出比较输入"的思路可能对 prompt engineering 自动化有启发——学习什么样的 prompt 变体会导致输出质量下降，可以反过来指导更好的 prompt 设计
- **小模型做 RM 的趋势**：在 RM 上用领域特化的小模型（0.4B encoder）替代通用大模型（7B+ decoder），符合当前"用专家小模型辅助大模型训练"的趋势。也启发在其他 RL+LLM 场景中探索异构架构
- **对抗训练在 LLM 对齐中的复兴**：GAN 在文本生成领域曾一度沉寂，GAPO 通过巧妙设计（encoder RM + PPO 而非直接文本 GAN）重新激活了对抗训练思路

## 评分
- 新颖性: ⭐⭐⭐⭐ Preferential Prompt + Encoder-only RM + GAN-PPO 融合的三重创新，视角新颖但每个组件都有已知前体
- 实验充分度: ⭐⭐⭐⭐ PR vs PP 对比、不同样本量 scaling、对抗训练动态分析等都做得细致，但任务种类偏少
- 写作质量: ⭐⭐⭐⭐ 结构清晰，motivation 铺陈到位，图表设计合理
- 价值: ⭐⭐⭐⭐ Preferential Prompt 范式和 encoder-only RM 的思路有广泛启发性，但计算开销限制了实际采用
