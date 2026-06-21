---
title: >-
  [论文解读] Optimizer Choice Matters for the Emergence of Neural Collapse
description: >-
  [ICLR 2026][优化/理论][Neural Collapse] 通过 3,900+ 次训练实验和理论分析，揭示了优化器选择（特别是权重衰减的耦合方式）对 Neural Collapse 现象涌现起关键决定性作用——AdamW（解耦权重衰减）无法产生 Neural Collapse，而 SGD 和 Adam（耦合权重衰减）可以。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "Neural Collapse"
  - "优化器选择"
  - "权重衰减耦合"
  - "AdamW vs Adam"
  - "隐式偏置"
---

# Optimizer Choice Matters for the Emergence of Neural Collapse

**会议**: ICLR 2026  
**arXiv**: [2602.16642](https://arxiv.org/abs/2602.16642)  
**代码**: 无  
**领域**: 优化理论 / 深度学习理论  
**关键词**: Neural Collapse, 优化器选择, 权重衰减耦合, AdamW vs Adam, 隐式偏置

## 一句话总结

通过 3,900+ 次训练实验和理论分析，揭示了优化器选择（特别是权重衰减的耦合方式）对 Neural Collapse 现象涌现起关键决定性作用——AdamW（解耦权重衰减）无法产生 Neural Collapse，而 SGD 和 Adam（耦合权重衰减）可以。

## 研究背景与动机

Neural Collapse (NC) 是 Papyan et al. (2020) 发现的深度网络训练末期现象：最后一层特征向量和分类器权重会自组织成高度对称的几何结构。NC 包含四个性质：
- **NC1**：类内变异性消失（特征坍缩到类均值）
- **NC2**：类中心收敛到 Simplex ETF（等角紧框架）
- **NC3**：分类器权重与类均值对齐（Self-Duality）
- **NC4**：分类简化为最近类中心分类器

已有的理论分析大多忽略了优化器的角色，暗示 NC 对所有优化方法是普遍的。本文挑战这一假设，证明优化器的选择——特别是权重衰减的耦合方式——对 NC 的涌现至关重要。一个关键发现是：**Adam 能产生 NC，但算法上非常相似的 AdamW 却不能**。

## 方法详解

### 整体框架

本文围绕一个问题展开：为什么算法上几乎相同的 Adam 和 AdamW 在 Neural Collapse 上表现迥异。作者先提出一个可解析追踪的新指标 NC0，证明它收敛到零是 NC 的必要条件，再对 SGD 与 SignGD（Adam/AdamW 的理想化特例）做理论分析，最后用 3,900+ 次训练运行系统验证理论预测。

### 关键设计

**1. NC0 诊断指标：把难追踪的 NC 收敛压缩成一个标量动力学**

原始的 NC1–NC3 指标涉及特征均值、等角紧框架（ETF）对齐、自对偶等结构，既难直接做理论分析，也难看清优化器的作用路径。作者定义 NC0 为最后一层权重矩阵行和的平方范数

$$\alpha_t = \frac{1}{K}\|W_t^\top \mathbf{1}\|_2^2$$

并证明（Proposition 2.1）NC2 与 NC3 成立必然蕴含 NC0 收敛到零，即 NC0 收敛是 NC 的**必要（非充分）条件**。这把一个高维几何问题降到了单个标量的收敛性上：只要 NC0 在训练中发散，就能确定性地断言 NC 不可能发生，无需追踪完整的几何结构。这也是后续所有定理都只针对 NC0 立论、却能直接对 NC 下结论的原因。

**2. 权重衰减的耦合与解耦：Adam 与 AdamW 差异的真正来源**

两类优化器的唯一实质区别在于权重衰减项作用的位置。耦合权重衰减（SGD/Adam）把衰减项放进梯度内部，动量更新写作 $V_{t+1} = \beta V_t + \nabla L_{CE} + \lambda W_t$；解耦权重衰减（SGDW/AdamW）则让衰减项直接乘到参数上，$W_{t+1} = (1-\eta\lambda)W_t - \eta V_{t+1}$。对原始 SGD 这两种写法完全等价，所以以往工作常把"梯度里加 L2 正则"和"直接对参数衰减"混为一谈；但一旦优化器带有自适应/符号化的逐坐标缩放（如 Adam/SignGD），耦合项 $\lambda W_t$ 会和梯度一起先经过缩放再更新，而解耦项绕过缩放直接作用，两者**不再等价**。正是这个被长期忽视的细节，决定了 NC0 是收敛还是发散。

**3. 四种优化器–衰减组合的 NC0 命运：从理论上分出谁能产生 NC**

作者在无约束特征模型（UFM）下分别求解四种组合的 NC0 动力学。关键观察是交叉熵损失梯度的行和恒为零，$\nabla L_{CE}(W_t)^\top \mathbf{1}_K = 0$，因此 NC0 的演化只由权重衰减和动量驱动，与具体损失景观无关——这正是能写出闭式结论的原因。结论上，SGD 配解耦或耦合衰减（Theorem 3.1/3.2）都使 NC0 以指数速率收敛到零；SignGD 配耦合衰减（Theorem 3.4，Adam 特例）轨迹非单调、先增后减，配合学习率衰减到零最终也能收敛；唯独 SignGD 配解耦衰减（Theorem 3.3，AdamW 特例）使 NC0 从零**单调递增**到正常数 $\frac{(K-2)^2}{\lambda^2}$，永远到不了零。直觉上，耦合时随着权重范数增大、衰减项 $\lambda W_t$ 能最终翻转符号梯度的方向，把 NC0 拉回来；解耦时衰减项无法影响符号方向，NC0 只增不减。这就从理论上解释了为何 AdamW 无法产生 NC，而仅一字之差的 Adam 可以。

### 损失函数 / 训练策略

实验统一采用交叉熵损失加 L2 正则化，在 ResNet9 与 VGG9 上跨 MNIST、FashionMNIST、CIFAR10 训练。为系统隔离优化器的影响，作者对 Adam、AdamW、SGD、SGDW、Signum、SignumW 六个优化器各扫 3 个学习率 × 6 个动量 × 6 个权重衰减共 108 种组合，统一训练 200 epochs、batch size 128、并在 1/3 与 2/3 处各将学习率衰减 10 倍——这正是 Theorem 3.4 中 SignGD+耦合衰减能收敛所要求的学习率衰减条件。

## 实验关键数据

### 主实验

ResNet9 在 FashionMNIST 上的最终 NC 指标（越低越好）：

| 优化器 | NC0↓ | NC1↓ | NC2↓ | NC3↓ |
|--------|------|------|------|------|
| SGD | 2.14e-04 (<-99.5%) | 0.05 (-99.3%) | 0.29 (-63.0%) | 0.35 (-75.1%) |
| Adam | 0.34 (-80.6%) | 0.04 (-99.5%) | 0.29 (-63.9%) | 0.29 (-79.5%) |
| AdamW | **5.33 (>100%)** | 0.20 (-97.2%) | 0.54 (-32.4%) | 0.78 (-45.2%) |
| SGDW | 0.55 (-68.9%) | 0.26 (-96.3%) | 0.46 (-42.4%) | 0.80 (-43.5%) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Adam vs AdamW 插值 | 随耦合 WD 增加，NC0/NC2/NC3 平滑改善 | 验证准确率基本不变 |
| 动量对 NC 的加速 | 相同训练损失下，mom=0.9 比 0.7 的 NC 指标明显更低 | 动量的 NC 加速效应超越了对训练损失的加速 |
| 最优 NC3 的超参数 | SGD NC3=0.13, AdamW NC3=0.49 | SGD 在所有优化器中实现最强 NC |

### 关键发现

1. **耦合权重衰减是自适应优化器产生 NC 的必要条件**：AdamW/SignumW 的 NC 指标始终远高于 Adam/Signum，即使权重衰减高几个数量级也无法改善
2. **动量加速 NC 但不仅仅加速收敛**：两个相同训练损失但不同动量的 SGD 运行会到达几何结构截然不同的解
3. **SGD 的 NC 行为对耦合/解耦不敏感**：SGD 和 SGDW 的 NC 指标差距较小，与理论一致
4. **部分 Neural Collapse**：AdamW 可以在 NC1、NC2 上取得最优值，同时 NC0 发散、NC3 不满足——NC 性质不一定同时出现
5. **NC4 是冗余的**：只要训练准确率接近 100%，NC4 总是满足，与其他 NC 指标不相关

## 亮点与洞察

- **提出了 NC0 这一新的诊断指标**：收敛到零是 NC 的必要条件，比原始指标更易追踪和分析
- **挑战了 NC 的普遍性假设**：证明优化器选择决定性地影响 NC 是否涌现
- **揭示了被忽视的细微差异**：Adam 和 AdamW 之间看似微小的权重衰减耦合方式差异，导致截然不同的表征几何
- **NC 不一定意味着更好的泛化**：所有优化器都能达到相似的验证准确率，但 NC 强度差异显著——这限制了用 NC 来理解泛化
- **规模宏大的实验**：3,900+ 训练运行，系统性地控制变量

## 局限与展望

1. **理论分析限于简化设定**：Theorem 3.3/3.4 基于 UFM（无约束特征模型）中的 SignGD，未完全捕捉深度网络和自适应优化器的复杂性
2. **仅分析 NC0**：完整理解 NC1-NC3 在现实优化动力学下的行为仍是开放问题
3. **仅限最后一层**：未分析中间层的 NC 性质（已有工作表明中间层也可能出现 NC）
4. **未覆盖新型优化器**：Lion, MARS, Shampoo, SOAP, Muon 等新优化器的 NC 行为有待探索
5. **需要扩展到更大模型**：ViT 和 DenseNet 等更大架构的实验有限（附录中有初步 ViT 结果）

## 相关工作与启发

- Papyan et al. (2020) 首次发现 NC 现象
- Pan & Cao (2024), Jacot et al. (2024) 研究了权重衰减对 NC 的影响，但未区分耦合/解耦
- Loshchilov & Hutter (2019) 提出 AdamW，但其在 NC 上下文中的影响此前被忽视
- **启发**：优化器不仅影响收敛速度，还决定性地影响学习到的表征的几何结构——优化器选择是一种隐式的归纳偏置

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （首次揭示优化器依赖的 NC 涌现，提出 NC0 指标）
- 实验充分度: ⭐⭐⭐⭐⭐ （3,900+ 运行，系统性变量控制，多数据集多架构）
- 写作质量: ⭐⭐⭐⭐ （理论和实验结合紧密，结构清晰）
- 价值: ⭐⭐⭐⭐ （对理解深度学习优化和表征几何有重要启示）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICLR 2026\] How does the optimizer implicitly bias the model merging loss landscape?](how_does_the_optimizer_implicitly_bias_the_model_merging_loss_landscape.md)
- [\[ICLR 2026\] Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension](towards_efficient_optimizer_design_for_llm_via_structured_fisher_approximation_w.md)
- [\[ICLR 2026\] LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters](lora_meets_riemannion_muon_optimizer_for_parametrization-independent_low-rank_ad.md)

</div>

<!-- RELATED:END -->
