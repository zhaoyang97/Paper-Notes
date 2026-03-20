# Weight Decay may matter more than μP for Learning Rate Transfer in Practice

**会议**: ICLR2026  
**arXiv**: [2510.19093](https://arxiv.org/abs/2510.19093)  
**代码**: 待确认  
**领域**: llm_alignment  
**关键词**: μP, weight decay, learning rate transfer, alignment assumptions, AdamW  

## 一句话总结
大规模实证研究表明 μP 的核心对齐假设在实际 LLM 训练中仅在开始时短暂成立，之后是 independent weight decay（而非 μP）正确稳定了不同宽度模型间的特征学习动态，使得学习率迁移成为可能。μP 的实际作用被重新解释为一种隐式学习率 warmup。

## 背景与动机
1. μP 已成为大规模训练的基石（Llama4/MetaP、Grok-2、GPT-4 等均使用类似方法），通过学习率缩放实现小模型到大模型的 LR 迁移
2. 多项工作发现 μP 必须搭配 independent weight decay 才能实现良好的 LR 迁移，但原因不明
3. μP 的 LR 缩放基于对齐假设：权重更新与输入的对齐度 $\alpha_{\Delta W} = \Theta(1)$，权重对齐度 $\alpha_W = \Theta(1/\sqrt{C})$
4. 这些假设源于无限宽度极限和小 batch 的理论分析，但实际训练中 batch 远大于模型宽度
5. 需要理解为什么 independent WD 是必要的，以及 μP 的真正实际作用是什么

## 方法详解
**相对更新框架**: 统一 μP 和 weight decay 的分析——定义相对表征变化 $\|\Delta Y\|/\|Y\| = (\alpha_{\Delta W}/\alpha_W) \cdot \|\Delta W\|/\|W\|$，其中对齐比率 $\alpha_{\Delta W}/\alpha_W$ 是关键。

**对齐假设验证**: 在 LLaMA 训练中实测权重对齐度和更新对齐度随训练的变化。发现：
- 更新对齐度 $\alpha_{\Delta W}$ 在训练初期后迅速变得宽度相关（违反 μP 假设）
- 对齐比率快速降至约 1（而非 μP 预期的 $\Theta(\sqrt{C})$）

**为何假设崩塌**: 当 batch size $B \gg$ 宽度 $C$ 时（LLM 训练中 $B = 1M$ tokens vs $C \leq 6K$），样本间干扰项主导梯度结构，使更新对齐度变得宽度相关。

**Independent WD 的作用**: 保持 $\eta\lambda$ 不变 → 相对权重更新 $\propto \sqrt{\eta\lambda}$ 跨宽度一致 → 在对齐比率≈1 的条件下正确稳定相对表征变化。

**μP 作为隐式 warmup**: μP + independent WD 本质上是用更低的 LR 和更高的 WD（保持乘积不变），在训练初期产生更小的相对更新，等效于额外的指数型 warmup，初始缩放 $1/m$，逐渐趋近 1。

**替代方案验证**: 不用 μP 缩放，改用显式指数 warmup（$\hat{s}_t = m^{\min(0, t/T_W - 1)}$）可大致复现 μP 的 LR 迁移效果。

## 实验关键数据
- LLaMA 架构在 20B tokens 上训练，宽度从 128 到 2048（1× 到 16×）
- Standard WD + μP: LR 迁移失败（不同宽度的最优 LR 不一致）
- Independent WD + μP: LR 迁移成功（最优 LR 一致）
- 对齐比率在训练开始后快速从 $\sqrt{C}$ 降至 ≈1（Fig.4）
- 50% 指数 warmup + 无 μP 缩放可达到接近 μP + independent WD 的迁移效果（Fig.6）
- ResNet 实验中 μP 的额外 warmup 效果不必要（解释了 μP 在 CNN 时代未出现）

## 亮点
- 从根本上挑战了 μP 的实际工作机制——揭示理论与实践的关键脱节
- 统一框架优雅地将 μP 和 weight decay 纳入同一相对更新视角
- 解释了多个经验性发现的根本原因（为何需要 independent WD、为何 μP 在 CNN 不重要）
- practical implications 直接：可用显式 warmup 替代 μP 缩放

## 局限性 / 可改进方向
- 替代 warmup 方案需要额外调参（warmup 长度/形状），不如 μP + independent WD 简洁
- 仅验证了 LLaMA 和 ResNet 架构，其他架构（MoE、State Space Model）待验证
- 分析基于 Frobenius 范数，谱范数视角可能给出不同结论
- 未分析 μP 在极大规模（>100B 参数）时是否有额外价值
- 假设对齐比率≈1 的经验规律缺乏严格理论保证

## 与相关工作的对比
- **Yang et al. (μP 原作)**: 基于无限宽度理论推导 LR 缩放；本文证明其假设在实际中快速崩塌
- **Wortsman et al. / Blake et al.**: 经验性发现 independent WD 必要；本文首次给出根本解释
- **Everett et al.**: 发现 μP 的输出层/注意力特殊处理非必要；本文沿此方向进一步简化
- **Kosson et al.**: 分析 WD 在 AdamW 中的相对更新调节作用；本文将其与 μP 对接

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (根本性重新理解 μP 的实际机制)
- 实验充分度: ⭐⭐⭐⭐ (LLaMA + ResNet + 详尽对齐测量，但规模有限)
- 写作质量: ⭐⭐⭐⭐⭐ (框架统一优雅，论证层层递进)
- 价值: ⭐⭐⭐⭐⭐ (对大规模训练实践有直接指导意义)
