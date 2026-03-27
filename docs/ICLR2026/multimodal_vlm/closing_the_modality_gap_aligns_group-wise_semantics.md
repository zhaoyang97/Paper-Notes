# Closing the Modality Gap Aligns Group-Wise Semantics

**会议**: ICLR2026  
**arXiv**: [2601.18525](https://arxiv.org/abs/2601.18525)  
**代码**: [https://github.com/ispamm/ModGap](https://github.com/ispamm/ModGap)  
**领域**: 多模态VLM  
**关键词**: modality gap, contrastive learning, CLIP, clustering, multimodal alignment

## 一句话总结
证明 CLIP 中的 modality gap 对实例级任务（检索）无关紧要但严重损害群组级任务（聚类），并提出由 Align True Pairs loss + Centroid Uniformity loss 组成的新目标函数，在双模态和三模态设置中将 gap 几乎降为零，大幅提升聚类 V-Measure（+10-17 分），同时保持检索性能。

## 研究背景与动机

1. **领域现状**：CLIP 及其变体通过 InfoNCE 损失学习跨模态共享空间，但不同模态的嵌入会形成各自的聚类——即"modality gap"。现有工作对此问题态度分裂：有人认为缩小 gap 改善检索，有人认为 gap 和下游性能正相关。
2. **现有痛点**：(a) 现有研究只关注 gap 对检索（实例级任务）的影响，结论互相矛盾；(b) 所有方法只研究双模态（图像+文本），不涉及三模态及以上；(c) gap 的存在导致潜在空间"按模态聚类"而非"按语义聚类"，但这个后果未被系统分析。
3. **核心矛盾**：InfoNCE 优化的是正负对的相对排序（是否最相似），而非绝对距离（是否真正接近）。只要相对排序正确，检索就成功——即使正对的绝对余弦相似度只有 0.34。但聚类依赖绝对距离，gap 使类内散度膨胀 $\|\boldsymbol{\delta}\|^2$。
4. **本文要解决什么？** (a) 理论上阐明 gap 对实例级 vs 群组级任务的不同影响；(b) 提出有效缩小 gap 的方法；(c) 扩展到三模态。
5. **切入角度**：从 within-class scatter 的数学分解出发——gap 向量 $\boldsymbol{\delta}$ 与语义正交，因此等量膨胀所有聚类的散度，这对检索无关但对聚类致命。
6. **核心idea一句话**：modality gap 是检索的无害伪影，但是聚类的系统性障碍——用 true pair 对齐 + 质心均匀性两个损失函数可以同时消除 gap 和改善语义聚类。

## 方法详解

### 整体框架
在标准 InfoNCE 对比学习之上添加两个显式损失：$\mathcal{L}_{\text{ATP}}$（拉近正对）+ $\mathcal{L}_{\text{CU}}$（推开质心），组合为 $\mathcal{L}_{\text{CL}_{\text{gap}}} = \mathcal{L}_{\text{gap}} + \frac{1}{2}(\mathcal{L}^{(m\to n)} + \mathcal{L}^{(n\to m)})$。直接扩展到 $M$ 个模态。

### 关键设计

1. **Align True Pairs Loss ($\mathcal{L}_{\text{ATP}}$)**:
   - 做什么：显式最小化正对（matching pairs）之间的欧氏距离
   - 公式：$\mathcal{L}_{\text{ATP}} = \frac{1}{M-1}\sum_{m\neq a}\frac{1}{N}\sum_i \|\mathbf{z}_i^m - \mathbf{z}_i^a\|_2^2$，其中 $a$ 是锚模态
   - 设计动机：InfoNCE 只优化相对排序，不保证正对的绝对距离接近。$\mathcal{L}_{\text{ATP}}$ 直接拉近正对，从而缩小 gap。但如果只用 $\mathcal{L}_{\text{ATP}}$，整个空间会坍缩到一个点

2. **Centroid Uniformity Loss ($\mathcal{L}_{\text{CU}}$)**:
   - 做什么：确保不同语义样本的跨模态质心在超球面上均匀分布，防止坍缩
   - 公式：$\mathcal{L}_{\text{CU}} = \log\frac{1}{N}\sum_i\sum_{j\neq i}\exp(-2\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2^2)$，其中 $\boldsymbol{\mu}_k = \frac{1}{M}\sum_m \mathbf{z}_k^m$ 是第 $k$ 个样本的跨模态质心
   - 设计动机：(a) 在质心上施加均匀性而非在单模态嵌入上——这保留了已学到的跨模态对齐；(b) RBF 核与单位超球面上的均匀分布关联，确保覆盖整个球面

3. **理论分析（为什么 gap 伤害聚类但不伤害检索）**:
   - 检索只需 $\text{sim}(\mathbf{z}_i^m, \mathbf{z}_i^n) > \max_{j\neq i}\text{sim}(\mathbf{z}_i^m, \mathbf{z}_j^n)$——相对排序不受 gap 影响
   - 聚类的 within-class scatter 分解为：$\mathbb{E}[\|\mathbf{z}_s^m - \boldsymbol{\mu}_s^\delta\|^2] \approx \mathbb{E}[\|\mathbf{z}_s^m - \boldsymbol{\mu}_s^0\|^2] + \|\boldsymbol{\delta}\|^2$——gap 等量膨胀所有聚类
   - 关键数学性质：gap 向量 $\boldsymbol{\delta}$ 与语义向量正交（Zhang et al., 2023），所以它像一个常数偏移，不改变排序但改变绝对距离

### 损失函数 / 训练策略
总损失：$\mathcal{L}_{\text{CL}_{\text{gap}}} = \mathcal{L}_{\text{ATP}} + \mathcal{L}_{\text{CU}} + \frac{1}{2}(\mathcal{L}^{(m\to n)} + \mathcal{L}^{(n\to m)})$。重要发现：随着 gap 缩小，非匹配对的梯度增大（变成更信息丰富的"hard negatives"），匹配对梯度减小——优化自然转向精细化语义结构。

## 实验关键数据

### 主实验（Gap 值 vs 检索 vs 聚类）

| 方法 | 数据集 | Gap ↓ | CM R@1 | V-Measure ↑ | kNN ↑ |
|------|--------|-------|--------|-------------|-------|
| CLIP (LT) | MSCOCO (2-modal) | 0.47 | 74.6 | 12.98 | 26.3 |
| CLIP (FT) | MSCOCO | 0.12 | 73.2 | 12.99 | 31.0 |
| **Ours** | MSCOCO | **0.03** | 70.3 | **23.63** | **36.4** |
| CLIP (LT) | MSR-VTT (3-modal) | 0.29 | 34.2/10.3 | 23.3 | 52.9 |
| **Ours** | MSR-VTT | **0.07** | 32.8/11.8 | **32.1** | **58.0** |
| CLIP (LT) | AV-MNIST (3-modal) | 0.20 | 87.1/84.2 | 77.6 | 87.0 |
| **Ours** | AV-MNIST | **0.09** | 88.7/89.1 | **82.7** | **89.2** |

### 消融实验（Cos True Pairs 提升）

| 方法 | MSCOCO Gap | Cos TP ↑ | MM R@1 | CIDEr (captioning) |
|------|-----------|---------|--------|---------------------|
| CLIP (LT) | 0.47 | 0.34 | 72.5 | 153.2 |
| CLIP (FT) | 0.12 | 0.63 | 73.8 | 155.0 |
| Ours | **0.03** | **0.77** | 76.2 | **158.2** |

### 关键发现
- **检索几乎不受 gap 影响**：CLIP (LT) gap=0.47 和 Ours gap=0.03，MSCOCO 检索 R@1 只差 4.3 分，但 V-Measure 差 10.65 分——证实理论预测
- **聚类与 gap 强相关**：MSR-VTT 上人工控制 gap 从 0.3→0，V-Measure 从 23.3→+17.5 提升
- **正对余弦相似度从 0.34 到 0.77**：CLIP 训练后正对相似度竟然只有 0.34（很远！），说明 InfoNCE 只保证排序不保证接近
- **三模态也有效**：AV-MNIST 和 MSR-VTT（音频+视频+文本）三模态场景均成功缩小 gap 并提升聚类
- **captioning 也受益**：更好的对齐空间让解码器生成更准确的 caption（CIDEr +5）

## 亮点与洞察
- **重新定义 modality gap 的意义**：从"要不要缩 gap"的争论，转向"gap 对什么任务有影响"的精确分析——这个 insight 可以终结该领域的矛盾结论
- **$\mathcal{L}_{\text{ATP}}$ + $\mathcal{L}_{\text{CU}}$ 的互补设计**：拉近匹配对（可能坍缩）+ 在质心层面推开非匹配对（防坍缩）——比直接在嵌入层面做 alignment+uniformity 更优雅
- **梯度视角的机制解释**：gap 缩小 → 非匹配对变成更有效的 hard negatives → 梯度更集中于精细化语义结构——这个发现对理解对比学习动态有重要意义
- **可直接扩展到 N 模态**：方法设计天然支持任意数量模态，不需要架构修改

## 局限性 / 可改进方向
- **检索轻微下降**：MSCOCO 上检索从 74.6 降到 70.3——虽然理论上应该无关，但实际有微小 trade-off。可能是 $\mathcal{L}_{\text{ATP}}$ 的拉近操作轻微扰乱了排序
- **实验规模受限**：最大实验是 MSCOCO（EVA-CLIP ViT-G），未在 LAION-5B 预训练规模验证。大规模对比学习中 gap 的行为可能不同
- **只验证对比学习 pipeline**：CLIP-like 方法。SigLIP、BLIP-2 等非对比方法的 gap 行为未探索
- **改进方向**：将 $\mathcal{L}_{\text{gap}}$ 集成到 VLM 预训练中，可能改善下游群组级理解能力（如 zero-shot 分类、视觉常识推理等）

## 相关工作与启发
- **vs Liang et al. (2022)**：他们首次发现 modality gap 并提出 post-hoc translation 修复。本文提供了更优雅的训练时解决方案，且给出了理论解释"为什么要缩 gap"
- **vs Yaras et al. (2025) 固定温度方案**：固定温度可以部分缩小 gap（0.12），但不如本方法彻底（0.03），且没有理论解释为什么有用
- **vs NotAGap (Fahim et al.)**：NotAGap 缩小 gap 但 CosTP 反而下降（0.11 vs 0.34），说明只缩 gap 不够——还需要同时对齐正对

## 评分
- 新颖性: ⭐⭐⭐⭐ "gap 伤害聚类但不伤害检索"的洞察是核心贡献，loss 设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 4 数据集、2/3 模态、4 类下游任务的全面验证，但缺少大规模预训练实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论→实验→可视化的叙事一气呵成，数学推导清晰
- 价值: ⭐⭐⭐⭐ 为理解和改善多模态潜在空间提供了重要的理论和实践工具
