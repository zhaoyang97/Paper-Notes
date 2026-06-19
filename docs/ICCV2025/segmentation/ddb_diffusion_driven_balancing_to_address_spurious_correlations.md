---
title: >-
  [论文解读] DDB: Diffusion Driven Balancing to Address Spurious Correlations
description: >-
  [ICCV 2025][语义分割][虚假相关性] 提出Diffusion Driven Balancing（DDB）方法，利用Stable Diffusion的文本反演和图像修复能力，自动生成少数组样本来平衡数据集中的虚假相关性，结合基于ERM模型预测概率和积分梯度的双重剪枝策略确保生成质量，在Waterbirds和MetaShift上达到最优最差组准确率。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "虚假相关性"
  - "扩散模型数据增强"
  - "组鲁棒性"
  - "文本反演"
  - "图像修复"
---

# DDB: Diffusion Driven Balancing to Address Spurious Correlations

**会议**: ICCV 2025  
**arXiv**: [2503.17226](https://arxiv.org/abs/2503.17226)  
**代码**: [https://github.com/ArianYp/DDB](https://github.com/ArianYp/DDB)  
**领域**: 图像分割  
**关键词**: 虚假相关性, 扩散模型数据增强, 组鲁棒性, 文本反演, 图像修复

## 一句话总结
提出Diffusion Driven Balancing（DDB）方法，利用Stable Diffusion的文本反演和图像修复能力，自动生成少数组样本来平衡数据集中的虚假相关性，结合基于ERM模型预测概率和积分梯度的双重剪枝策略确保生成质量，在Waterbirds和MetaShift上达到最优最差组准确率。

## 研究背景与动机

深度网络使用经验风险最小化（ERM）训练时，容易依赖数据中与标签虚假相关的特征而非真正的因果特征，导致分布外泛化失败。例如，训练集中大多数牛出现在绿色牧场，大多数骆驼出现在沙漠，模型可能学会仅根据背景分类动物。

**现有方案及局限**：

**重新加权方法（JTT、AFR）**：假设少数组损失高，上调其权重。但ERM中高损失可能有其他原因，且少数组样本可能太少难以有效重加权

**数据混合方法（DaC、DISC、LISA）**：通过混合/拼接不同组样本增强少数组。但DaC缺乏语义控制、生成质量低；LISA需要组标签进行训练

**基于扩散模型的方法（FFR）**：用扩散模型生成平衡数据集。但对prompt敏感、可能生成有害样本、需要预知数据偏差

**核心矛盾**：如何在不需要训练集组标签的前提下，自动、精准地生成高质量少数组样本来打破虚假相关性？

**核心idea**：从图像的组合性视角出发——每张图像由因果部分（core features）和虚假部分（spurious features）组成。通过文本反演学习每类的因果特征token，用语言分割定位因果区域，用Stable Diffusion修复来替换因果部分生成新类样本，再用ERM模型的预测概率和积分梯度进行双重剪枝。

## 方法详解

### 整体框架
DDB是三阶段方法：(1) 生成新数据——学习因果token + 语言分割 + 扩散修复；(2) 剪枝——双重条件过滤低质量样本；(3) 重训练——在平衡数据集上重训练ERM模型。

### 关键设计

1. **因果特征学习（Textual Inversion）**:

    - 功能：为每个类别学习一个可学习的token嵌入 $[C_i]$，编码该类因果特征的视觉语义
    - 核心思路：使用模板句 "A photo of a $[C_i]$ bird" 作为prompt，冻结Stable Diffusion参数，仅优化文本编码器中 $[C_i]$ 的嵌入，最小化去噪损失：
    $C_i^* = \arg\min_{C_i} \mathbb{E}_{z,I,\epsilon,t} [\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(I))\|_2^2]$
    - 从每类20-40个样本（优先选少数组样本）中学习
    - 设计动机：通过文本反演获得对因果特征的精确控制能力，而非依赖手动设计的prompt

2. **因果部分替换（Diffusion Inpainting）**:

    - 功能：将多数组样本的因果部分替换为另一类的因果特征，保留虚假背景，生成少数组样本
    - 核心思路：
        - 用LangSAM（GroundingDINO + SAM）自动分割因果区域：$M = m(x_j)$
        - 仅对掩码区域加噪，使用学到的token引导去噪生成新的因果部分：
       $$z_t = (1-M) \odot z_0 + M \odot (\sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon)$$
        - 多数组识别：选取ERM模型上损失最低的K个样本作为多数组
    - 设计动机：保留原图背景（虚假特征不变），精确替换因果部分（改变标签），自然地生成少数组样本

3. **双重剪枝策略（Bicephalous Pruning）**:

    - 功能：过滤扩散模型生成的低质量或无效样本
    - 核心思路：两个互补的剪枝条件——
        - **条件1：ERM预测概率**。计算生成样本在ERM模型上的softmax输出 $\psi_j$，若 $\psi_j \geq \Psi_i$（类别平均概率），说明图像实际未产生有效变化（仍被当作原类），剪枝。
        - **条件2：积分梯度归因分数**。计算修改前后的积分梯度差异：
       $$\text{IG}_k(x') = (x_k' - x_k) \times \int_0^1 \frac{\partial f_{i'}(x + \alpha(x'-x))}{\partial x_k'} d\alpha$$
       累积掩码区域的归因分数 $\rho = \sum_k M_k \cdot \text{IG}_k$，若 $\rho \leq P_{i'}$（阈值），说明修改区域对标签改变贡献不足，剪枝。
    - 设计动机：单一条件不够——ERM概率变化可能是因为无关噪声而非有效因果替换（图3(a)），归因分数高但概率未变说明修改不充分（图3(c)）。双重条件确保生成样本既有效改变了因果特征，又对模型预测产生了正确影响。

### 损失函数 / 训练策略
- 重训练损失：$L_{total} = L_{train} + \gamma_1 L_{gen1} + \gamma_2 L_{gen2}$
- $L_{gen1}$ 和 $L_{gen2}$ 分别对两个类别的生成样本使用交叉熵损失
- $\gamma_1, \gamma_2$ 用于上调新增样本权重
- 使用ResNet-50（ImageNet预训练）作为分类器
- Stable Diffusion v2 用于图像生成
- 文本反演和Inpainting均冻结扩散模型参数

## 实验关键数据

### 主实验
三个标准虚假相关基准上的最差组准确率（Worst-Group Accuracy, WGA）：

| 方法 | 组标签 | Waterbirds WGA | CelebA WGA | MetaShift WGA |
|------|--------|---------------|------------|---------------|
| Base (ERM) | ✗/✗ | 74.6 | 42.2 | 67.0 |
| JTT | ✗/✓ | 86.7 | 81.1 | 64.6 |
| DFR | ✗/✓✓ | 92.9 | 88.3 | 72.8 |
| DaC | ✗/✓ | 92.3 | 81.9 | 78.3 |
| LISA | ✓/✓ | 89.2 | **89.3** | 59.8 |
| DISC | ✗/✗ | 88.7 | - | 73.5 |
| **DDB (ours)** | ✗/✓ | **93.0** | 85.8 | **81.2** |

### 消融实验
剪枝效果：

| 数据集 | 类别 | 生成样本数 | 被剪枝数 | 无剪枝WGA | 有剪枝WGA |
|--------|------|-----------|---------|----------|----------|
| Waterbirds | Landbird | 1300 | 531 | 91.28 | **93.0** |
| Waterbirds | Waterbird | 1112 | 393 | - | - |
| CelebA | NotBlond | 120000 | 7439 | 81.7 | **85.8** |
| MetaShift | Dog | 400 | 299 (75%) | 80.6 | **81.2** |

文本反演设置对剪枝率和性能的影响（Waterbirds）：

| 反演样本数 | 0 | 10 | 20 | 30 | 40 |
|-----------|---|----|----|----|----|
| 被剪枝数 | 1018 | 818 | 804 | 737 | 782 |
| WGA | 88.9 | 90.5 | 89.9 | **93.0** | 92.1 |

### 关键发现
- DDB在Waterbirds和MetaShift上均取得最优WGA，且不需要训练集组标签
- **剪枝至关重要**：MetaShift中Dog类75%的生成样本被剪枝（因原图中狗太小或不存在），无剪枝性能下降明显
- 文本反演显著提升生成质量：无反演时剪枝率高达1018/2412，有反演（30样本）降至737/2412
- CelebA上性能略低于LISA（85.8 vs 89.3），因CelebA的虚假特征（发长、性别）对扩散模型修改更困难——LangSAM在头发区域识别不稳定
- DDB对虚假物体（spurious objects，如Waterbirds的背景）和虚假特征（spurious features，如CelebA的性别）均有效

## 亮点与洞察
- **组合性视角切入精准**：将虚假相关性问题转化为图像组合问题——保留虚假背景+替换因果前景，自然且高效
- **文本反演+Inpainting的巧妙组合**：文本反演提供语义控制，Inpainting提供空间控制，两者互补
- **双重剪枝设计务实**：认识到生成pipeline中每个组件都可能失效，通过ERM概率+归因分数双重过滤确保质量
- **不需要训练集组标签**：通过ERM损失排序自动识别多数/少数组，降低了方法的使用门槛
- 整个pipeline高度自动化，从因果特征学习到样本生成到质量控制全链路无需人工干预

## 局限与展望
- 依赖LangSAM的分割质量——当因果部分（如头发）难以用简单文本描述时效果下降
- Stable Diffusion在某些场景下的修复质量不稳定（如小物体、复杂纹理）
- 仅验证了二分类任务，多分类场景下需要为每对类别进行因果替换，复杂度增加
- 文本反演需要20-40个样本，在极端少样本场景下可能不足
- CelebA上未达最优，说明对"特征级"虚假相关（vs "物体级"）的处理仍有改进空间
- 需要验证集组标签来调超参数

## 相关工作与启发
- 组合视角（core + spurious features）为虚假相关性问题提供了清晰的形式化框架
- 文本反演是获取细粒度视觉概念控制的有效手段，可用于其他需要精确语义控制的数据增强
- 扩散模型的Inpainting能力在数据增强中有广阔的应用前景，不仅限于虚假相关性问题
- 积分梯度作为生成样本质量评估工具是一个新颖的应用

## 评分
- 新颖性: ⭐⭐⭐⭐ 文本反演+Inpainting用于虚假相关性是新颖组合，但框架仍基于已有组件
- 实验充分度: ⭐⭐⭐⭐ 三个标准基准+完整消融，但缺乏更大规模或更多类别的验证
- 写作质量: ⭐⭐⭐⭐ 方法阐述清晰，pipeline图易懂，但问题设置部分略显冗长
- 价值: ⭐⭐⭐⭐ 实用的数据增强范式，对分布外泛化问题有明确贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness](../../CVPR2026/segmentation/concept-guided_fine-tuning_steering_vits_away_from_spurious_correlations_to_impr.md)
- [\[NeurIPS 2025\] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation](../../NeurIPS2025/segmentation/diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICCV 2025\] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP](know_no_better_a_data-driven_approach_for_enhancing_negation_awareness_in_clip.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)

</div>

<!-- RELATED:END -->
