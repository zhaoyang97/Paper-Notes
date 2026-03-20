# Learning Robust Intervention Representations with Delta Embeddings

**会议**: ICLR 2026
**arXiv**: [2508.04492](https://arxiv.org/abs/2508.04492)
**代码**: [Project Page](https://palimisis.github.io/Learning-Robust-Intervention-Representations-with-Delta-Embeddings/)
**领域**: 因果表示学习 / OOD 泛化
**关键词**: Causal Representation Learning, Delta Embeddings, Out-of-Distribution, Intervention, Contrastive Learning

## 一句话总结

提出因果 Delta 嵌入（CDE）框架，将干预/动作表示为预干预和后干预状态在潜空间中的向量差，通过独立性、稀疏性和不变性三种约束学习鲁棒的干预表示，在 Causal Triplet 挑战中显著超越基线的 OOD 泛化性能，且能自动发现反义动作的反平行语义结构。

## 研究背景与动机

1. **理解世界如何响应动作和干预是 AI 的核心能力**：特别是在动态环境中操作的智能体，必须恢复生成和转换数据的底层机制，才能实现因果推理和鲁棒泛化。

2. **深度学习模型在分布偏移下泛化失败**：标准模型依赖相关性而非因果机制，当数据分布改变时（如遇到训练中未见的物体-动作组合），性能急剧下降。

3. **因果表示学习关注变量识别，但忽视了干预表示**：大多数 CRL 工作专注于识别潜在因果变量及其关系（如 VAE 框架、score-based 方法），但很少有方法关注学习动作/干预本身的可泛化表示。

4. **两个关键的 CRL 假设为方法设计提供指导**：
   - **独立因果机制（ICM）假设**：数据生成过程由自主且独立的模块组成
   - **稀疏机制偏移（SMS）假设**：一次干预通常只影响少量因果机制

5. **OOD 泛化的两种挑战类型**：
   - **组合偏移**：测试中出现训练未见的物体-动作组合（如训练见过 open(door) 和 close(drawer)，测试需识别 open(drawer)）
   - **系统偏移**：测试中出现全新的物体类别

## 方法详解

### 整体框架

CDE 框架的核心思想：将干预表示为预干预和后干预状态在潜空间中的向量差（Delta），并通过三种属性约束这个差向量，使其成为"因果 Delta 嵌入"。

**Delta 嵌入定义**：给定干预前后的观测对 $(x, \tilde{x})$，Delta 嵌入为 $\delta_a := \phi(\tilde{x}) - \phi(x)$，其中 $\phi$ 是编码器。

在完美反事实假设下，$\delta_a = [0 \cdots \tilde{z}_a - z_a \cdots 0]^T$，即只有被动作 $a$ 影响的维度非零。

### 关键设计

**1. 因果 Delta 嵌入的三种约束**

- **做什么**：定义 CDE 必须满足的三个属性，指导学习目标设计
- **核心思路**：独立性 + 稀疏性 + 不变性 → 可泛化的干预表示
- **设计动机**：这三个属性直接源自 ICM 和 SMS 假设，确保学到的表示具有因果意义

1. **独立性**：动作表示不受场景属性和未被影响物体的影响
2. **稀疏性**：如果 SMS 假设成立，$\delta_a$ 应该是稀疏的（大部分维度为零）
3. **不变性**：同一动作在不同物体上的表示应相似（如 "打开" 的表示不应因对象是门还是抽屉而改变），形式化为 $\text{Var}_{x \sim P(X)}[\delta_a(x)] \approx \mathbf{0}$

**2. 全局 CDE 模型（Model A）**

- **做什么**：从图像对中学习全局级别的因果 Delta 嵌入
- **核心思路**：用 ViT-DINO 提取 CLS token → 因果投影器映射到 $l$ 维潜空间 → 元素减法得到 Delta → 分类器预测动作
- **设计动机**：CLS token 提供全局图像表示，减法操作天然满足独立性

结构方程建模：$\tilde{z}_a = z_a + \delta_a + \epsilon$，其中 $\epsilon$ 是零均值独立噪声（对应 actionable counterfactual 场景）。

**3. Patch-wise CDE 模型（Model B）**

- **做什么**：处理多物体场景中动作只影响局部区域的情况
- **核心思路**：保留 ViT 所有 patch 的输出 → 逐 patch 计算 Delta → 按 L2 范数选 Top-K 最大变化的 patch → 对每个 patch 独立计算损失
- **设计动机**：全局嵌入在复杂场景中可能"平均化"掉局部变化信号

### 损失函数 / 训练策略

三个损失函数的加权组合：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha_{\text{contrast}} \mathcal{L}_{\text{contrast}} + \alpha_{\text{sparsity}} \mathcal{L}_{\text{sparsity}}$$

1. **交叉熵损失** $\mathcal{L}_{\text{CE}}$：确保 Delta 对动作分类任务有用
2. **有监督对比损失** $\mathcal{L}_{\text{contrast}}$：鼓励同类动作的 Delta 聚集，不同动作的 Delta 分离（对应不变性约束）

$$\mathcal{L}_{\text{contrast}} = \sum_{i=1}^{B} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(\delta_i, \delta_p)/\tau)}{\sum_{j \neq i} \exp(\text{sim}(\delta_i, \delta_j)/\tau)}$$

3. **稀疏正则化** $\mathcal{L}_{\text{sparsity}} = \frac{1}{B}\sum_i \|\delta_i\|_1$：L1 惩罚促进稀疏表示

超参数：$\alpha_{\text{contrast}} = 2.0$，$\alpha_{\text{sparsity}} = 1.0$，所有实验统一。端到端训练，编码器也参与更新。

## 实验关键数据

### 主实验

单物体 ProcTHOR 场景（合成数据）：

| 方法 | IID Acc. | OOD Comp. | OOD Syst. | Gap↓ |
|------|----------|-----------|-----------|------|
| Vanilla-R (ResNet) | 0.96 | 0.36 | 0.48 | 0.48 |
| Vanilla-V (ViT-DINO) | 0.95 | 0.34 | 0.47 | 0.48 |
| ICM-R | 0.95 | 0.41 | 0.50 | 0.45 |
| SMS-R | 0.96 | 0.47 | 0.54 | 0.42 |
| **CDE Global** | **0.95** | **显著提升** | **显著提升** | **大幅缩小** |

多物体和真实世界（Epic-Kitchens）场景同样展示了 CDE 在 OOD 泛化上的显著优势。

### 消融实验

| 配置 | 效果 |
|------|------|
| 全部三个损失 | 最佳 OOD 性能 |
| 去掉对比损失 | 不变性下降，OOD 准确率降低 |
| 去掉稀疏正则 | 表示不够紧凑，OOD 略有下降 |
| 只用交叉熵 | 退化为普通分类器，OOD 大幅下降 |
| Global vs Patch-wise | Patch-wise 在多物体场景更优 |

### 关键发现

1. **CDE 在 Causal Triplet 挑战中建立了新的 SOTA**：在合成和真实世界基准上均大幅超越基线
2. **自动发现反义动作的反平行关系**：open vs close 的 Delta 嵌入在潜空间中呈反平行方向，完全无需显式监督
3. **独立性由 Delta 计算自然满足**：不需要专门的损失来约束，减法操作天然消除了场景级变化
4. **对 actionable counterfactual 场景依然有效**：即使噪声 $\epsilon \neq 0$，分类器仍不受影响（理论证明 + 实验验证）
5. **稀疏正则至关重要**：L1 惩罚确保了只有因果相关的维度被激活

## 亮点与洞察

1. **将"学习干预表示"与"学习变量表示"分离**：大多数 CRL 工作关注恢复因果变量，CDE 另辟蹊径关注干预/动作本身的表示，视角独特且实用
2. **Delta = 减法 的极简设计**：不需要复杂的因果发现或结构学习，仅靠编码器输出的减法就能提取因果信息，优雅且有效
3. **三个约束对应三个损失**：独立性（由架构设计保证）→ 稀疏性（L1正则）→ 不变性（对比损失），设计意图清晰
4. **反平行语义结构的自发涌现**：模型自动学到 "open ↔ close" 方向互逆，是因果结构学习的有力证据

## 局限性 / 可改进方向

1. **需要干预前后的图像对**：许多现实场景中只有动作后的观测，无法获得配对数据
2. **动作类别数量有限**：Causal Triplet 基准中动作种类较少（约 10+），对大规模动作空间的可扩展性未验证
3. **静态图像对限制**：不能处理时序动作或持续变化，缺少对视频数据的扩展
4. **ViT-DINO 骨干的依赖**：预训练视觉特征提供了强大的先验，如果骨干更换（如随机初始化），效果可能显著下降
5. **真实世界场景中的遮挡和视角变化**：Epic-Kitchens 中相机运动和遮挡可能引入非因果变化

## 相关工作与启发

- **Causal Triplet (Liu et al., 2023)**：提供了评估框架和 SCM 模型定义，CDE 在此基准上超越了所有先前方法
- **Von Kügelgen et al. (2021)**：从理论证明数据增强对应因果干预时对比学习可以分离因果因素，CDE 将此思想扩展到干预对的对比学习
- **DINO (Caron et al., 2021)**：自监督 ViT 特征为 CDE 提供了强大的视觉先验
- **SMS 正则化 (Lachapelle et al., 2022)**：稀疏机制偏移假设在 CDE 中通过 L1 正则化具体实现
- 启发：Delta Embedding 的思想可能适用于更广泛的场景，如强化学习中的动作效果预测、医学影像中的治疗效果表示

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Delta = 减法的核心思想简洁有力，将 CRL 的焦点从变量识别转向干预表示是新颖的视角；但减法操作本身并不复杂
- **实验充分度**: ⭐⭐⭐⭐ — Causal Triplet 三个难度递增的评估场景覆盖全面，消融实验完整；但缺少更大规模的评估
- **写作质量**: ⭐⭐⭐⭐⭐ — 数学定义严谨，从属性→损失→架构的推导逻辑清晰，可视化（反平行语义结构）非常直观
- **价值**: ⭐⭐⭐⭐ — 为因果表示学习提供了新的研究方向（干预表示），在机器人和 embodied AI 领域有潜在应用价值
