---
title: >-
  [论文解读] Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated
description: >-
  [AAAI 2026][模型压缩][AI生成图像检测] 提出一种基于贝叶斯决策理论的轻量级后验校准方法，通过在模型输出logit上添加可学习标量偏移α，无需重训练即可显著提升现有AI生成图像检测器在分布偏移下的准确率。 AI生成图像检测是当前数字取证和信息安全的核心任务。现有检测器在训练分布内表现优异，但面对分布外（OOD）…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "AI生成图像检测"
  - "后验校准"
  - "贝叶斯决策理论"
  - "分布偏移"
  - "决策边界"
---

# Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated

**会议**: AAAI 2026  
**arXiv**: [2602.01973](https://arxiv.org/abs/2602.01973)  
**代码**: [github.com/muliyangm/AIGI-Det-Calib](https://github.com/muliyangm/AIGI-Det-Calib)  
**领域**: 模型压缩  
**关键词**: AI生成图像检测, 后验校准, 贝叶斯决策理论, 分布偏移, 决策边界

## 一句话总结

提出一种基于贝叶斯决策理论的轻量级后验校准方法，通过在模型输出logit上添加可学习标量偏移α，无需重训练即可显著提升现有AI生成图像检测器在分布偏移下的准确率。

## 研究背景与动机

AI生成图像检测是当前数字取证和信息安全的核心任务。现有检测器在训练分布内表现优异，但面对分布外（OOD）的伪造图像时性能急剧下降。作者发现一个关键现象：**即使在类别平衡的测试集上，现有检测器也系统性地倾向于将伪造图像错误分类为真实图像**。

这一现象的根本原因是：

**"懒惰"决策机制**：模型在训练时过度依赖表面伪影（如频率噪声、边缘不一致性），这些伪影在未见过的生成方法中不存在

**分布偏移的双重来源**：
   - **标签先验偏移**：训练和测试中真假图像的边缘分布 $P_{\text{tr}}(y) \neq P_{\text{te}}(y)$ 可能不同
   - **类条件输入偏移**：假图像的条件分布 $P_{\text{tr}}(x|y=1) \neq P_{\text{te}}(x|y=1)$ 因生成方法不同而改变

**决策边界错位**：上述偏移导致模型学到的默认阈值 $f(x)=0$ 不再是贝叶斯最优的

作者通过可视化logit分布（图1）清晰展示了这一现象：在ProGAN上训练的CNNSpot，面对StyleGAN2、WFIR和Midjourney生成的图像时，假图像的logit分布整体偏移到了决策边界的错误一侧。

## 方法详解

### 整体框架

本方法是一个**后验（post-hoc）校准框架**，不修改原始检测器的任何参数，仅在推理时对模型输出的logit添加一个标量校准值α。核心流程：
1. 使用预训练的检测器提取测试样本的logit值
2. 用少量验证样本（默认100张，约1%测试集）优化校准参数α
3. 将校准后的logit $\tilde{f}(x) = f(x) - \alpha$ 用于最终预测

### 关键设计

#### 1. **贝叶斯非最优性证明（Proposition 1）**

通过贝叶斯定理推导，在训练分布下模型输出的logit为：

$$f(x) = \log\frac{P_{\text{tr}}(x|y=1) P_{\text{tr}}(y=1)}{P_{\text{tr}}(x|y=0) P_{\text{tr}}(y=0)}$$

而在测试分布下，贝叶斯最优决策边界需要满足：

$$f(x) + \Delta(x) = 0$$

其中偏移量为：

$$\Delta(x) = \log\frac{P_{\text{te}}(x|1)}{P_{\text{tr}}(x|1)} + \log\frac{P_{\text{te}}(1)(1-P_{\text{tr}}(1))}{P_{\text{tr}}(1)(1-P_{\text{te}}(1))}$$

这里假设真实图像分布在训练和测试间保持稳定（$P_{\text{te}}(x|0) \approx P_{\text{tr}}(x|0)$），偏移主要来自假图像类别。只要 $\Delta(x) \neq 0$，默认阈值就不是贝叶斯最优的。

#### 2. **标量校正的理论基础（Proposition 2）**

基于两个关键假设：
- **假设1（系统性条件偏移）**：不同生成模型产生的假图像在视觉特征上表现出一致且系统性的偏差，因此对测试假图像的对数似然比可近似为常数 $c$
- **假设2（一致先验偏移）**：测试集中假样本的先验偏移为常数 $\delta$

在这两个假设下，$\Delta(x) \approx c + \delta' = \text{const}$，因此可以用一个全局标量 $\tilde{\alpha} = -(c+\delta')$ 来校正决策边界。校准后的logit为：

$$\tilde{f}(x) = f(x) - \tilde{\alpha}$$

#### 3. **有监督校准方法**

当有少量标注目标数据时，使用核密度估计（KDE）分别建模两类logit的分布：

$$p_0(z) = p(z|y=0), \quad p_1(z) = p(z|y=1)$$

然后最小化分类错误率来求最优α：

$$\alpha^{\star} = \arg\min_{\alpha} \left[\int_{-\infty}^{\alpha} p_1(z)dz + \int_{\alpha}^{\infty} p_0(z)dz\right]$$

#### 4. **无监督校准方法**

无标签时，利用logit分布的固有双峰结构，通过KDE估计密度 $p(z)$，然后寻找使分布对称的校准点：

$$\Phi(\alpha) = \int_{-\infty}^{\infty} (z-\alpha) \cdot p(z) dz = 0$$

当 $\Phi(\alpha)=0$ 时，分布关于α对称，α即为自然的分割边界。

### 损失函数 / 训练策略

- 方法**不涉及任何训练或微调**，仅需在推理时优化标量α
- 有监督版本通过KDE + 精确度最大化优化α
- 无监督版本通过矩平衡优化法确保鲁棒性
- 默认仅需100张验证样本（约1%数据），甚至10张（<0.1%）也能有效工作

## 实验关键数据

### 主实验（AIGCDetectBenchmark）

在ProGAN训练、16种生成器测试的AIGCDetectBenchmark上：

| 检测器 | 原始准确率 | +有监督校准 | +无监督校准 | 提升(Sup.) |
|--------|-----------|------------|------------|-----------|
| CNNSpot | 70.83% | 78.22% | 77.90% | +7.39% |
| FreDect | 64.03% | 68.88% | 65.09% | +4.85% |
| Fusing | 68.85% | 78.42% | 75.60% | +9.57% |
| LNP | 81.09% | 88.42% | 87.17% | +7.33% |
| UnivFD | 78.43% | 86.35% | 86.20% | +7.92% |
| RINE | 86.77% | 94.56% | 93.79% | +7.80% |
| AIDE | 92.80% | 94.42% | 93.69% | +1.62% |
| Effort | 79.35% | 89.48% | 85.08% | +10.13% |

### GenImage数据集结果

| 检测器 | 原始准确率 | +有监督校准 | 提升 |
|--------|-----------|------------|------|
| RINE | 81.78% | 97.94% | +16.16% |
| Effort | 91.41% | 96.64% | +5.23% |
| AIDE | 88.17% | 93.25% | +5.08% |
| Fusing | 76.16% | 90.20% | +14.03% |

### 消融实验

| 配置 | 说明 | 效果 |
|------|------|------|
| 验证集大小=10 | <0.1%测试数据 | 性能已接近饱和 |
| 验证集大小=100 | 默认设置 | 最优性价比 |
| JPEG压缩(QF=90) | 图像扰动鲁棒性 | AIDE: +15.39% |
| 高斯模糊(σ=1.0) | 图像扰动鲁棒性 | Effort: +8.17% |
| KDE vs 二分搜索 | α估计方法对比 | KDE显著优于其他方法 |

### 关键发现

1. **校准方法对强特征提取器（CLIP-based）效果最显著**：RINE和Effort在GenImage上提升超过16%和5%
2. **无监督方法性能接近有监督方法**：多数情况下差距在1-2%以内
3. **对图像扰动有强鲁棒性**：JPEG压缩和高斯模糊下校准收益更大
4. **极少量数据即可有效校准**：仅需10张图像即可实现大部分性能提升

## 亮点与洞察

1. **问题诊断精准**：从贝叶斯决策理论出发，严格证明了为什么默认阈值在分布偏移下不是最优的，这比纯经验观察更有说服力
2. **解决方案极其轻量**：仅一个标量参数，无需重训练、无需访问训练数据、无需模型内部结构，可即插即用到任何检测器
3. **理论与实践的完美结合**：两个假设（系统性条件偏移和一致先验偏移）虽然是近似，但在实验中被充分验证
4. **无监督版本的实用价值**：在无标注数据的现实部署场景中，无监督校准仍能带来显著提升

## 局限与展望

1. **假设的适用范围**：系统性条件偏移假设要求同一生成器的图像具有一致的偏移，当测试集包含多种差异极大的生成器时，单一标量可能不足
2. **仅针对二分类**：方法目前仅处理真/假二分类，未扩展到生成器归属（attribution）任务
3. **对分布高度重叠的情况**：当真假图像logit分布高度重叠时，标量偏移的效果有限
4. 可以考虑**分区域/分生成器自适应校准**，或将标量扩展为输入依赖的函数

## 相关工作与启发

- **logit调整**方法（Menon et al., 2020）在类别不平衡学习中的应用，为本文提供了理论灵感
- 与**持续学习中的偏差校正**方法类似，但本文处理的是跨生成器的分布偏移
- 启发：许多看似复杂的问题，可能只需要简单但理论驱动的后处理手段即可大幅改善

## 评分

- 新颖性: ⭐⭐⭐⭐ — 简单但有良好理论支撑的后处理方法，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 9个检测器 × 2个benchmark + 鲁棒性 + 消融，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ — 从现象到理论到方法到实验的逻辑清晰流畅
- 价值: ⭐⭐⭐⭐⭐ — 即插即用、零训练成本，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[AAAI 2026\] Can You Tell the Difference? Contrastive Explanations for ABox Entailments](can_you_tell_the_difference_contrastive_explanations_for_abox_entailments.md)
- [\[NeurIPS 2025\] AI-Generated Video Detection via Perceptual Straightening](../../NeurIPS2025/model_compression/ai-generated_video_detection_via_perceptual_straightening.md)
- [\[ICML 2026\] Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images](../../ICML2026/model_compression/images_as_tables_in-context_learning_with_tabpfn_for_low-data_detection_of_ai-ge.md)
- [\[NeurIPS 2025\] FiRA: Can We Achieve Full-Rank Training of LLMs Under Low-Rank Constraint?](../../NeurIPS2025/model_compression/fira_can_we_achieve_full-rank_training_of_llms_under_low-rank_constraint.md)

</div>

<!-- RELATED:END -->
