---
title: >-
  [论文解读] Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM
description: >-
  [CVPR 2025][多模态VLM][测试时适应] FreeTTA 提出一种无需训练、无需存储历史数据的测试时适应方法，通过在线 EM 算法显式建模目标域分布，利用 CLIP 零样本预测作为先验迭代估计每个类别的高斯分布参数，在 15 个数据集上稳定超越现有 TTA 方法。 领域现状：视觉语言模型（如 CLIP）通过大规模…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "测试时适应"
  - "视觉语言模型"
  - "在线EM算法"
  - "高斯混合模型"
  - "无需训练"
---

# Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM

**会议**: CVPR 2025  
**arXiv**: [2507.06973](https://arxiv.org/abs/2507.06973)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 测试时适应, 视觉语言模型, 在线EM算法, 高斯混合模型, 无需训练

## 一句话总结

FreeTTA 提出一种无需训练、无需存储历史数据的测试时适应方法，通过在线 EM 算法显式建模目标域分布，利用 CLIP 零样本预测作为先验迭代估计每个类别的高斯分布参数，在 15 个数据集上稳定超越现有 TTA 方法。

## 研究背景与动机

**领域现状**：视觉语言模型（如 CLIP）通过大规模图文对比预训练获得了强大的零样本泛化能力。但在实际部署中，当测试数据与训练数据存在域偏移（domain shift）时，性能会显著下降。测试时适应（TTA）范式应运而生，旨在利用测试时的无标签数据在线调整模型。

**现有痛点**：当前 TTA 方法存在三大不足：(1) **缺乏目标分布建模**——TPT 等方法逐样本独立处理或仅利用极少量样本关联（TDA），无法利用测试样本之间的内在关系；(2) **可用性限制**——prompt tuning 方法修改模型参数、TDA 存储历史测试特征，在 API 访问和隐私场景下不可行；(3) **效率和稳定性**——TPT/DiffTPT 每个样本都需要多次数据增强 + 梯度反传来优化 prompt，计算开销巨大，且熵最小化可能导致过拟合和过度自信。

**核心矛盾**：理想的 TTA 应同时满足三个特性——显式建模目标分布、无需额外假设的可用性、无需训练的高效性。但现有方法最多满足其中一两项，无法兼得。

**本文目标** 如何设计一种 TTA 方法同时满足：(1) 显式建模目标域分布以利用样本间关系；(2) 不需要访问/存储训练数据或历史测试数据；(3) 无需任何梯度反传，保持高效稳定？

**切入角度**：假设每个类别的测试特征服从高斯分布，整体测试数据是高斯混合分布（GMM）。利用 CLIP 文本嵌入作为初始类别均值，用在线 EM 算法顺序处理每个测试样本——E步计算后验概率、M步更新分布参数——从而无需存储历史数据就能逐步逼近目标域的真实分布。

**核心 idea**：用 CLIP 文本嵌入初始化高斯混合模型参数，通过在线 EM 算法逐样本更新分布参数，结合 CLIP 零样本置信度加权，实现无需训练的在线域适应。

## 方法详解

### 整体框架

给定一个测试样本 $x_t$，使用冻结的 CLIP 图像编码器提取特征，文本编码器用 prompt 模板生成类别特征向量。在线 EM 算法用文本特征初始化均值向量，用单位矩阵初始化共享协方差矩阵。每到一个新样本，E步计算各类后验概率，M步用当前预测更新类别均值和协方差。最终预测 logits 是 CLIP 零样本预测和 GDA 概率的加权组合。

### 关键设计

1. **参数初始化与高斯判别分析（GDA）**:

    - 功能：为在线 EM 提供合理的初始参数估计
    - 核心思路：利用 CLIP 文本编码器为每个类别 $y$ 生成文本特征 $g(t_y)$ 作为初始均值 $\mu_y$，共享协方差矩阵 $\Sigma$ 初始化为单位矩阵 $I$。在此假设下，分类变为比较测试样本到各类均值的马氏距离。后验概率为 $P(y|x_t) \propto \exp\left(-\frac{1}{2}(x_t - \mu_y)^\top \Sigma^{-1}(x_t - \mu_y)\right)$
    - 设计动机：CLIP 文本嵌入和视觉嵌入在同一共享空间，文本特征作为类别均值的初始值既合理又免费。单位矩阵初始化简单无偏，后续通过 EM 迭代会逐步逼近真实分布

2. **在线 EM 算法（Online EM）**:

    - 功能：逐样本在线更新 GMM 参数以适应目标域分布
    - 核心思路：对第 $t$ 个测试样本：
        - **E步**：计算样本属于每个类的后验概率 $\gamma_{y,t} = \frac{\pi_y \cdot \mathcal{N}(x_t|\mu_y, \Sigma)}{\sum_j \pi_j \cdot \mathcal{N}(x_t|\mu_j, \Sigma)}$
        - **M步**：更新先验 $\pi'_y = \frac{N_y + \gamma_{y,t}}{n_t}$，均值 $\mu'_y = \frac{N_y \cdot \mu_y + \gamma_{y,t} \cdot x_t}{N_y + \gamma_{y,t}}$，协方差 $\Sigma' = \frac{(n_t-1)\Sigma + \sum_y \gamma_{y,t}(x_t - \mu'_y)(x_t - \mu'_y)^\top}{n_t - 1}$
        - 关键在于：只需当前样本和累积统计量（各类样本计数 $N_y$、当前均值 $\mu_y$、协方差 $\Sigma$），无需存储任何历史样本
    - 设计动机：传统 EM 需要同时访问所有数据做 batch 更新，不适合 online setting。在线 EM 用增量更新公式，每个样本按后验概率加权地"归入"各类，实现流式适应

3. **VLM 先验整合与置信度加权**:

    - 功能：利用 CLIP 零样本预测的置信度控制每个样本对参数更新的影响力
    - 核心思路：计算 CLIP 零样本预测的自信息熵 $H(x_t) = -\sum_y P_{\text{CLIP}}(z_y=1|x_t) \log P_{\text{CLIP}}(z_y=1|x_t)$。定义权重函数 $w(h) = e^{-\beta h}$——高置信（低熵）样本权重大，低置信（高熵）样本权重小。将 $w(H(x_t))$ 乘到 EM 更新公式中的 $\gamma_{y,t}$ 上。最终 logits 为 CLIP 零样本 logits 与 GDA logits 的线性组合：$\text{logits}_y = FT_y^\top + \alpha(w_y^\top F + b_y)$
    - 设计动机：TTA 初期参数不稳定，高不确定性样本可能引入噪声导致参数漂移。用 CLIP 原始预测置信度做权重，相当于在初期更信任 CLIP 先验、后期随分布估计变准逐步增加 EM 的贡献

### 损失函数 / 训练策略

FreeTTA 完全无需训练——不进行梯度反传、不优化任何参数。所有"更新"都是统计量的增量计算（均值、协方差的在线更新），计算量极小。

## 实验关键数据

### 主实验（Cross-Domain，ViT-B/16）

| 方法 | T.D. | Avail. | T.F. | AIR | CAL | DTD | FLWR | SUN | UCF | AVG |
|------|------|--------|------|-----|-----|-----|------|-----|-----|-----|
| CLIP zero-shot | - | - | - | 23.22 | 93.55 | 45.04 | 66.99 | 65.63 | 65.16 | 64.59 |
| TPT | ✗ | ✔ | ✗ | 24.78 | 94.16 | 47.75 | 68.98 | 65.50 | 68.04 | 65.10 |
| TDA | ✗ | ✗ | ✔ | 23.91 | 94.24 | 47.40 | 71.42 | 67.62 | 70.66 | 67.53 |
| **FreeTTA** | ✔ | ✔ | ✔ | **26.93** | **95.81** | **50.80** | **74.15** | **69.06** | **72.87** | **69.80** |

### 消融实验

| 配置 | ImageNet | AVG(cross-domain) | 说明 |
|------|----------|-------------------|------|
| CLIP zero-shot | 68.34 | 64.59 | 基线 |
| + GDA (无EM) | 68.83 | 65.90 | 静态高斯分类 |
| + Online EM | 69.18 | 67.32 | 加入在线更新 |
| + 置信度加权 | 69.48 | 68.61 | 加入熵加权 |
| + 最终组合 | **69.84** | **69.80** | 完整模型 |

### 关键发现
- FreeTTA 是唯一同时满足"目标分布建模 + 可用性 + 无需训练"三个特性的方法
- 相比 CLIP zero-shot，平均提升约 5 个点（64.59 → 69.80），在个别数据集（EuroSAT）提升超过 20 个点
- 在线 EM 的贡献（+1.42）比静态 GDA（+1.31）更大，说明在线更新确实逐步逼近了真实分布
- 置信度加权带来稳定提升（+1.29），验证了抑制低置信样本的必要性
- FreeTTA 不需要反传梯度，推理速度接近原始 CLIP，而 TPT 每个样本需要 ~35 次前向传播

## 亮点与洞察
- **在线 EM + VLM 先验的组合**是本文最精巧的设计：用文本嵌入初始化均值解决了"无标签"问题，用在线更新解决了"无存储"问题，用熵加权解决了"不稳定"问题——三个挑战被一个统一框架优雅解决
- **统计模型替代神经网络优化**的思路值得学习：复杂的 TTA 问题被建模为简单的 GMM 参数估计，避免了梯度优化的不稳定性和计算开销
- **从数据分布的角度重新思考 TTA** 而非从模型适应角度，提供了一种全新的范式——后续可以探索更复杂的分布假设（如 von Mises-Fisher 分布更适合归一化后的 CLIP 特征）

## 局限与展望
- 高斯分布假设过于简化：CLIP 特征空间中的类条件分布未必是高斯的，特别是类间重叠较大时
- 共享协方差矩阵的假设限制了表达力：不同类别的分布形状可能截然不同
- 在线更新依赖样本到达顺序，如果初期样本分布偏斜，可能导致参数估计偏差
- 超参数 $\alpha$（CLIP 和 GDA logits 的混合权重）和 $\beta$（熵加权强度）需要调节
- 协方差矩阵的在线更新在类别数很多时（如 ImageNet 1000 类）计算量不小

## 相关工作与启发
- **vs TPT/DiffTPT**: 它们通过梯度反传优化 prompt，计算开销大且不稳定。FreeTTA 无需训练，速度快且稳定
- **vs TDA**: TDA 缓存历史测试样本来做 KNN，需要存储数据且不建模分布。FreeTTA 只维护统计量，不存储任何样本
- **vs PromptAlign**: PromptAlign 需要访问源域数据来对齐分布，FreeTTA 不需要任何额外数据

## 评分
- 新颖性: ⭐⭐⭐⭐ 将在线 EM 引入 VLM TTA 的思路新颖，但 GMM+EM 本身是经典方法
- 实验充分度: ⭐⭐⭐⭐⭐ 15 个数据集、跨域和 OOD 两种设置、多个 backbone、详细消融
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，框架图清晰，但部分符号较冗长
- 价值: ⭐⭐⭐⭐ 方法简洁高效，但影响力取决于 TTA 在实际部署中的需求强度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Realistic Test-Time Adaptation of Vision-Language Models](realistic_test-time_adaptation_of_vision-language_models.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](../../ICCV2025/multimodal_vlm/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)
- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)

</div>

<!-- RELATED:END -->
