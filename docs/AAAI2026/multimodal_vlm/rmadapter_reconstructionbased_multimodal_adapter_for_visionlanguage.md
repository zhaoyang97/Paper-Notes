# RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)

**会议**: AAAI 2026
**arXiv**: [2512.06811](https://arxiv.org/abs/2512.06811)
**代码**: 未提及
**领域**: Vision-Language Model / Parameter-Efficient Fine-Tuning
**关键词**: CLIP, Adapter, Reconstruction, Few-shot Learning, Vision-Language Model

## 一句话总结

提出 RMAdapter，一种双分支适配器架构：在标准 adapter 的适应分支旁增加重建分支（类 AutoEncoder），通过共享下投影层和逐层本地重建损失，在 CLIP 少样本微调中实现任务特定适应与通用知识保持的最佳平衡，在 Base-to-Novel 泛化、跨数据集和领域泛化三个任务上全面超越 SOTA（含 Prompt-based 方法）。

## 研究背景与动机

预训练 VLM（如 CLIP）在少样本下游适应中面临核心矛盾——**适应-泛化权衡**：

1. **Prompt Learning 方向**：CoOp → CoCoOp → MaPLe → PromptSRC → CoPrompt 发展迅速，但本质上缺乏显式的知识保持机制，学到的 prompt 对 seen 类高度判别但对 unseen 类偏见严重
2. **Adapter 方向严重不足**：相比 prompt 方向，adapter 方法探索明显不足。现有 adapter（如 MMA）仅有单一分支关注适应，缺乏结构化设计控制判别力和泛化性的平衡
3. **关键观察——Adapter 与 AutoEncoder 的结构同构**：adapter 的下投影→上投影与 AE 的 encoder→decoder 结构同构，自然可以增加一个重建分支来约束特征空间不偏离原始分布

## 方法详解

### 整体框架

RMAdapter 在 CLIP 的视觉和文本编码器的高层（后 k 层）插入双分支 adapter。整个 CLIP 冻结，仅训练 adapter 参数。两个分支共享下投影层，分别进行任务适应和特征重建，通过残差连接与原始 CLIP 输出融合。

### 关键设计

1. **适应分支 (RMAdapter_base)**：标准 adapter 结构，$x_{down} = \sigma(x W_{down} + b_{down})$，$\text{output} = x_{down} W_{up}^{base} + b_{up}^{base}$，注入任务特定知识
2. **重建分支 (RMAdapter_rec)**：两层上投影结构，$\text{output} = \sigma(x_{down} W_{up1}^{rec} + b_{up1}^{rec}) W_{up2}^{rec} + b_{up2}^{rec}$，将隐表示重建回原始特征空间，通过 L2 重建损失约束保持通用知识
3. **共享下投影层**：两个分支共享 $W_{down}$，实现 Pareto 最优的适应-重建权衡。共享下投影让两分支在同一低秩空间工作：适应分支学习任务相关特征，重建分支确保该空间不偏离原始分布——自然互制
4. **逐层本地重建损失**：$\mathcal{L}_{rec}^V = \sum_{i=k}^K \|[c_i, E_i] - \text{RMAdapter}_{rec}([c_i, E_i])\|^2$，在每层独立计算 L2 loss，无需层间反传，计算高效
5. **一致性约束**：$\mathcal{L}_{con} = \lambda_3 \|x^a - x\|_1 + \lambda_4 \|w^a - w\|_1$，约束适应后特征与原始 CLIP 特征的 L1 距离

### 损失函数

$$\mathcal{L} = \mathcal{L}_{ce} + \mathcal{L}_{con} + \mathcal{L}_{rec}$$

交叉熵（分类监督）+ 一致性约束（防止偏离原始特征）+ 重建损失（保持通用知识）。测试了 L2、L1、cosine 三种重建目标，L2 最稳定。

## 实验关键数据

### 主实验：Base-to-Novel 泛化（11 数据集平均）

| 方法 | 类型 | Base Acc | Novel Acc | HM |
|------|------|---------|-----------|-----|
| CLIP (zero-shot) | — | 69.34 | 74.22 | 71.70 |
| CoOp | Prompt | 82.69 | 63.22 | 71.66 |
| CoCoOp | Prompt | 80.47 | 71.69 | 75.83 |
| MaPLe | Prompt | 82.28 | 75.14 | 78.55 |
| PromptSRC | Prompt | 84.26 | 76.10 | 79.97 |
| MMA | Adapter | 83.20 | 76.80 | 79.87 |
| CoPrompt | Prompt | 84.00 | 77.23 | 80.48 |
| **RMAdapter** | **Adapter** | **84.52** | **77.36** | **80.62** |

### 消融实验：关键设计对 HM 的贡献

| 配置 | HM | 说明 |
|------|-----|------|
| 单分支 Adapter (MMA) | 79.87 | 无重建约束 |
| + 重建分支（不共享） | ~80.1 | 参数独立，效果有限 |
| + 共享下投影 | ~80.4 | Pareto 最优权衡 |
| + 一致性约束 | 80.62 | 最终完整版 |
| 重建分支上投影 1 层 | 略低 | 容量不足 |
| 重建分支上投影 2 层 | 最优 | sweet spot |
| 重建分支上投影 3 层 | 下降 | 少样本过拟合 |

### 关键发现

- **跨数据集泛化**（10 数据集平均）：RMAdapter 67.56% vs CoPrompt 67.00% vs MMA 66.61%
- **领域泛化**（4 ImageNet 变体平均）：RMAdapter 60.71% vs PromptSRC 60.65% vs CoPrompt 60.42%
- **重建分支仅增加 ~320K 参数**，+3% GPU 显存、+5% 训练时间，但效果显著
- **Adapter 首次全面超越 Prompt-based 方法**，说明 adapter 方向被严重低估

## 亮点与洞察

- **Adapter vs AutoEncoder 的结构类比**：非常精彩的观察——adapter 的下投影→上投影与 AE 的 encoder→decoder 同构，在此基础上增加重建分支自然且优雅。这种"在已有结构中发现新联系"的思路值得学习
- **共享下投影的直觉**：让两分支在同一低秩空间工作，适应分支学任务特征，重建分支确保低秩空间不偏离——自然互制，Pareto 最优
- **不依赖数据增强或复杂 prompt 设计**：相比 CoPrompt 等方法更简洁
- **重建作为正则化**：利用重建目标作为知识保持，核心思想类似知识蒸馏但实现更轻量——不需要 teacher 模型的前向传播

## 局限性

- 实验基于 ViT-B/16 CLIP，未测试 ViT-L 或 SigLIP、EVA-CLIP 等
- 仅在分类任务上验证，未拓展到检测、分割等下游任务
- 重建分支使用简单 MSE 目标，可尝试更结构化的保持策略（特征方向保持、子空间投影等）
- 未讨论与 LoRA、Prefix Tuning 等其他 PEFT 方法的结合

## 相关工作

| 方法类别 | 代表方法 | 核心机制 | 知识保持策略 |
|----------|---------|---------|-------------|
| Prompt Learning | CoOp, CoCoOp, MaPLe | 可学习 prompt token | 隐式（无显式设计） |
| 正则化 Prompt | PromptSRC, KgCoOp | prompt + 约束 | 自正则化、文本嵌入距离 |
| 混合 Prompt+Adapter | CoPrompt | prompt + adapter | 一致性约束 |
| 单分支 Adapter | CLIP-Adapter, MMA | 仅适应分支 | 无 |
| **双分支 Adapter（本文）** | **RMAdapter** | **适应 + 重建分支** | **显式重建损失 + 共享下投影** |

## 评分

- 新颖性: ⭐⭐⭐⭐ AE-Adapter 类比精彩，双分支设计优雅
- 实验充分度: ⭐⭐⭐⭐ 11 数据集 + 跨数据集 + 领域泛化 + 消融全面
- 写作质量: ⭐⭐⭐⭐ 问题动机和方法推导逻辑清晰
- 价值: ⭐⭐⭐⭐ 证明 Adapter 方向被低估，提供通用 PEFT 设计范式
