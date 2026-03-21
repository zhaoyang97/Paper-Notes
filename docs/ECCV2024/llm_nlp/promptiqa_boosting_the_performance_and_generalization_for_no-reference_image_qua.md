# PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts

**会议**: ECCV 2024
**arXiv**: [2403.04993](https://arxiv.org/abs/2403.04993)
**代码**: 无（论文提到将公开但未给链接）
**领域**: LLM/NLP
**关键词**: no-reference IQA, image-score pairs prompt, mixed training, data augmentation, assessment requirement adaptation

## 一句话总结
提出 PromptIQA，通过少量"图像-分数对"（ISP）作为 prompt 的方式，使 NR-IQA 模型训练完成后无需微调即可自适应适配新的质量评估需求，在 12 个数据集、5 类 IQA 任务上均达到 SOTA 性能和泛化能力。

## 研究背景与动机
1. **领域现状**: NR-IQA 模型在训练完成后，评估标准就固化了。然而不同应用场景（自然图像评估、人脸质量、AI 生成图质量、水下图像质量等）的评估需求差异巨大——同一质量分数在不同数据集中可能代表完全不同的主观感受。
2. **现有痛点**:
   - 传统 IQA 模型 $S = \mathcal{R}(\mathcal{V}(I))$ 训练后评估准则不变，面对新需求必须重新训练或微调；
   - 构建 IQA 数据集极其耗时耗力（需要大量人工标注）；
   - 混合训练方法（如 UNIQUE、StairIQA）虽可跨数据集训练，但仍需为每个数据集设计独立回归头或复杂数据变换。
3. **核心矛盾**: 如何让模型以极低数据成本（几张样本）理解新的评估需求，而不需重新训练？
4. **本文解决什么**: 设计 prompt 机制让 IQA 模型像人类标注员一样——先看几个标准样例理解评估标准，再做质量评分。
5. **切入角度**: 受人类标注流程启发——标注员工作前需先看"标准示例"来理解评估要求，这本质上就是 few-shot prompting。
6. **核心 idea**: 用 Image-Score Pairs（ISP）作为 prompt，编码评估需求信息，与待评图像特征融合后输出针对性的质量分数。

## 方法详解

### 整体框架
ISP Prompt（$n$ 对图像-分数对）→ Visual Encoder + Score Expansion → Prompt Encoder（3 ViT blocks）→ ISPP Fusion Module（3 ViT blocks）→ Image-Prompt Fusion Module（8 ViT blocks）→ Quality Regression（2 FC layers）→ 质量分数 $S$

核心公式：$S = \mathcal{R}(\mathcal{FM}_{IP}(\mathcal{V}(I), \mathcal{F}_{AP}))$

### 关键设计

1. **Image-Score Pairs Prompt (ISPP)**:
   - 做什么：用 $n$ 个（默认 10 个）图像-分数对组成 prompt，直观表达评估需求
   - 核心思路：$\mathbf{P} = [\mathcal{ISP}_1, \mathcal{ISP}_2, \ldots, \mathcal{ISP}_n]$，每个 $\mathcal{ISP}_i = (I_i, S_i)$。ISP 采样策略分为：
     - 间隔采样：按分数分布均匀采样，更好反映数据集分布
     - 随机采样：随机选取，更贴近实际应用中的 few-shot 场景
   - 设计动机：相比文本 prompt 可能存在偏差，ISP 直接展示"什么样的图像应得什么分数"，更直观且无歧义。

2. **Prompt Encoder + Fusion Module**:
   - 做什么：先理解每个 ISP 中图像与分数的关系，再整合所有 ISP 形成评估需求特征，最后与待评图像融合
   - 核心思路：
     - 每个 ISP 特征：$\mathcal{F}_{\mathcal{ISP}_i} = \text{CAT}(\mathcal{V}(I_i), \mathcal{E}(S_i)) \in \mathbb{R}^{2 \times N}$
     - Prompt Encoder（3 ViT blocks）：探索图像特征与分数特征的深层注意力关系，得到 $\mathcal{F}_{P_i} \in \mathbb{R}^{1 \times M}$
     - ISPP Fusion Module（3 ViT blocks）：$n$ 个 prompt 特征交互，形成评估需求特征 $\mathcal{F}_{AC} \in \mathbb{R}^{n \times M}$
     - Image-Prompt Fusion Module（8 ViT blocks）：待评图像特征与评估需求特征融合，得到 $\mathcal{F}_{IPF} \in \mathbb{R}^{(n+1) \times M}$
   - 设计动机：分层融合确保模型先理解"标准"再做评估，而非简单拼接。

3. **数据增强策略（Random Scaling + Random Flipping）**:
   - 做什么：打破 GT 标签对 prompt 的"捷径依赖"，强制模型从 ISPP 中学习评估需求
   - 核心思路：
     - Random Scaling（概率 0.5）：$f_{RS}(\mathbf{S}) = \frac{1}{\max(\mathbf{S})} \cdot \mathbf{S}$，同时缩放 ISPP 和 GT 的分数
     - Random Flipping（概率 0.1）：$f_{RF}(\mathbf{S}) = \alpha - \mathbf{S}$，将 MOS 转变为 DMOS 语义（$\alpha=1$）
   - 设计动机：如果 GT 不随 ISPP 变化，模型可能忽略 prompt 直接记忆输入图像→分数的映射（"捷径"）。数据增强使同一张图在不同 prompt 下应产生不同分数，强迫模型"看 prompt 再答题"。

### 损失函数 / 训练策略
- 损失函数：$\mathcal{L}_1$ loss（预测分数与 GT 差的绝对值）
- Visual Encoder：MoNet（预训练 IQA encoder）
- 优化器：Adam，lr=$1 \times 10^{-5}$，weight decay=$1 \times 10^{-5}$，Cosine Annealing 每 50 epochs
- 训练 100 epochs，batch size 66，6 张 NVIDIA 3090
- 12 个数据集混合训练（LIVE, CSIQ, TID2013, Kadid-10k, BID, SPAQ, LIVEC, KonIQ-10K, GFIQA20k, AGIQA3k, AIGCIQA2023, UWIQA），覆盖 5 类 IQA 任务

## 实验关键数据

### 主实验（ADN-IQA 任务典型结果）
| 数据集 | 指标 | PromptIQA | 之前 SOTA (MoNet) | 提升 |
|--------|------|-----------|-------------------|------|
| BID | SROCC↑ | **0.9152** | 0.9012 | +1.53% |
| BID | PLCC↑ | **0.9341** | 0.9152 | +2.07% |
| LIVEC | SROCC↑ | **0.9125** | 0.8998 | +1.41% |
| LIVEC | PLCC↑ | **0.9280** | 0.9169 | +1.21% |
| GFIQA20k | SROCC↑ | **0.9698** | TOPIQ-Face 0.9664 | +0.35% |
| UWIQA | SROCC↑ | **0.8766** | UIQI 0.7423 | +18.1% |

### 泛化实验（用 FR-IQA 模拟新评估需求，10-shot）
| 模型 | 训练方式 | TID2013-SSIM | TID2013-FSIM | TID2013-LPIPS | Kadid-SSIM |
|------|---------|-------------|-------------|---------------|------------|
| MANIQA-SDT | Zero-shot | 0.5391 | 0.8245 | -0.7486 | 0.5553 |
| MANIQA-MDT&FT | Few-shot | 0.4507 | 0.6925 | -0.6202 | 0.5652 |
| MoNet-SDT&FT | Few-shot | 0.5473 | 0.8380 | -0.7478 | 0.5708 |
| **PromptIQA-MDT** | **Few-shot** | **0.5992** | **0.8802** | **0.8064** | **0.5717** |

关键发现：传统模型在 LPIPS（DMOS 类型）上出现负相关（最高 -0.7486），因为训练时用的是 MOS。PromptIQA 通过 prompt 正确识别了 DMOS 语义，得到正相关 0.8064。

### 消融实验
| 配置 | TID2013 | SPAQ | GFIQA20k | UWIQA | 泛化-FSIM | 泛化-LPIPS |
|------|---------|------|----------|-------|-----------|------------|
| w/o mixed training | 0.8849 | 0.9228 | 0.9696 | 0.8781 | 0.8579 | 0.6511 |
| w/o prompt | 0.8929 | 0.9220 | 0.9665 | 0.8602 | 0.7851 | -0.7712 |
| w/o random scaling | 0.9218 | 0.9252 | 0.9683 | 0.8766 | 0.8614 | 0.7797 |
| w/o random flipping | 0.9080 | 0.9245 | 0.9691 | 0.8754 | 0.8646 | 0.6538 |
| **Full PromptIQA** | **0.9223** | **0.9261** | **0.9702** | **0.8839** | **0.8802** | **0.8064** |

### 关键发现
- **Prompt 是泛化的关键**: 去除 prompt 后混合训练性能下降不大，但泛化到新需求时 LPIPS 从 +0.8064 骤降为 -0.7712（负相关！）
- **Random Flipping 对 DMOS 泛化至关重要**: 去除后泛化 LPIPS 从 0.8064 降到 0.6538
- **间隔采样 > 随机采样**: 间隔采样更好地反映数据集分布，但随机采样标准差小、鲁棒性可接受
- **ISP 数量效应**: 从 3 增到 10 时性能单调提升，更多 ISP 提供更多评估标准信息
- **Prompt 有效性验证**: 随机化 ISP 的图像或分数都会导致性能大幅下降；反转分数会使 SROCC/PLCC 变成近似等幅负值（如 0.9698 → -0.9698），证明模型确实在通过 prompt 学习评估需求

## 亮点与洞察
- **In-context Learning 思想迁移到 IQA**: 用 ISP 做 prompt 的设计本质上是 in-context learning 在低级视觉任务的应用，无需文本就能传递复杂的评估语义
- **数据增强巧妙解决"捷径"问题**: Random Scaling 和 Flipping 看似简单，但精准解决了 GT 不随 prompt 变化导致 prompt 失效的核心问题
- **跨任务统一**: 单一模型覆盖 SDN-IQA、ADN-IQA、F-IQA、AIG-IQA、U-IQA 五大任务，无需任何结构改动
- **10-shot 即可适配新需求**: 对比传统方法需要整个数据集微调，PromptIQA 仅需 10 个样本
- **MOS/DMOS 自动识别**: 通过 prompt 中分数的排列模式自动区分 MOS 和 DMOS 语义，无需人工指定

## 局限性 / 可改进方向
- 在 LIVE 和 CSIQ（DMOS 数据集）上性能略低于单数据集训练的模型，说明 DMOS 与 MOS 的分布差异仍未完全弥合
- ISP prompt 需要预先标注几张样本（虽然只要 10 张），在完全无标注场景下不适用
- Visual Encoder（MoNet）冻结了预训练权重，prompt 机制的改进空间可能受限于底层特征质量
- 未探索 prompt 的可解释性——模型从 ISP 中具体学到了什么评估标准？

## 相关工作与启发
- **vs UNIQUE**: UNIQUE 需要每张图的标注方差信息，且实验不完整（多数据集缺失）；PromptIQA 不需任何额外标注
- **vs StairIQA**: StairIQA 为每个数据集设独立回归头，结构冗余且无法泛化到新需求；PromptIQA 统一回归头 + prompt 适配
- **vs MANIQA/MoNet**: 在未见过的评估标准上，即使给 10-shot 微调也无法有效适配（特别是 DMOS→MOS 转换），而 PromptIQA 零微调即可

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ ISP 作为 prompt 的设计极具原创性，将 in-context learning 理念引入 IQA 任务，数据增强策略解决 prompt 失效问题巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个数据集、5 类任务、21 种 SOTA 对比、泛化测试、prompt 影响分析、消融齐全
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，问题定义到位，但表格和实验较多导致阅读量大
- 价值: ⭐⭐⭐⭐⭐ 解决了 IQA 领域"换需求就要重训"的根本痛点，10-shot 适配新需求有极大实用价值
