---
title: >-
  [论文解读] SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks
description: >-
  [ECCV2024][AI安全][联邦学习] 提出 SkyMask，利用参数级可学习二值掩码在服务器端检测恶意客户端模型更新，实现攻击无关的鲁棒联邦学习，在恶意客户端占比高达 80% 时仍能有效防御。 联邦学习 (FL) 通过分布式训练保护数据隐私，但其分布式特性使系统容易受到拜占庭攻击——被攻破的客户端上传恶意模型更新来破…
tags:
  - "ECCV2024"
  - "AI安全"
  - "联邦学习"
  - "Byzantine Attack"
  - "Learnable Mask"
  - "Robust Aggregation"
---

# SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks

**会议**: ECCV2024  
**arXiv**: [2312.12484](https://arxiv.org/abs/2312.12484)  
**代码**: [KoalaYan/SkyMask](https://github.com/KoalaYan/SkyMask)  
**领域**: 优化  
**关键词**: federated learning, Byzantine Attack, Learnable Mask, Robust Aggregation  
**机构**: 上海交通大学、Stevens Institute of Technology、Queen's University Belfast、Intel

## 一句话总结

提出 SkyMask，利用参数级可学习二值掩码在服务器端检测恶意客户端模型更新，实现攻击无关的鲁棒联邦学习，在恶意客户端占比高达 80% 时仍能有效防御。

## 背景与动机

联邦学习 (FL) 通过分布式训练保护数据隐私，但其分布式特性使系统容易受到拜占庭攻击——被攻破的客户端上传恶意模型更新来破坏全局模型。随着细粒度攻击（如 Fang attack、AGR-agnostic attack）的出现，攻击变得更加隐蔽：它们利用不同层和参数的敏感度差异，精准地在少量参数上投毒，使得模型级别的异常检测难以发现。

现有防御策略主要分两类：

1. **模型级防御**（如 FLTrust、Krum、FLAME）：基于模型更新的整体统计量（欧氏距离、余弦相似度）检测异常，面对细粒度攻击时要么检测失败，要么过度反应误伤正常客户端
2. **贪心参数过滤**（如 Trimmed-Mean）：逐坐标排序裁剪，但细粒度攻击只修改特定参数，能轻松绕过

作者通过 PCA 可视化实验发现：虽然细粒度攻击在模型级别伪装成正常更新，但为每个模型更新训练可学习掩码后，恶意客户端的掩码能在高维空间中被明显区分出来。这一观察启发了 SkyMask 的设计。

## 核心问题

如何在不预知攻击类型的情况下，以参数级粒度检测联邦学习中的恶意模型更新，有效防御包括细粒度攻击在内的各种拜占庭攻击？

## 方法详解

### 整体流程

SkyMask 在每轮通信中包含六步：

1. 服务器分发全局模型给各客户端
2. 客户端本地训练后上传模型更新
3. 服务器冻结所有模型更新，为每个客户端分配一个同尺寸的可学习掩码
4. 在干净的 root 数据集上训练所有掩码至收敛
5. 通过聚类分析掩码，检测并移除恶意客户端
6. 仅聚合剩余良性模型更新为新全局模型

### 掩码初始化与训练

服务器为每个客户端 $i$ 创建掩码 $m_i$（初始化为全1），冻结所有本地模型参数。聚合后的掩码模型为：

$$\tilde{W}_{t+1} = \frac{\sum_{i=1}^{n} \tilde{m}_i \odot W_{t+1}^i}{\sum_{i=1}^{n} \tilde{m}_i}$$

其中 $\tilde{m}_i = \sigma(m_i)$ 通过 sigmoid 函数近似二值掩码，范围在 $(0, 1)$。这里用 sigmoid 替代硬阈值是为了保证梯度可反向传播。

掩码在 root 数据集 $D_r$ 上采用标准交叉熵损失训练：

$$m_i := m_i - \gamma \cdot \nabla_{m_i} f(\tilde{W}_{t+1}, D_r)$$

收敛后，通过阈值 $\tau$ 将实值掩码二值化：$\hat{m}_i[k] = 1$ 当 $\tilde{m}_i[k] > \tau$，否则为 $0$。

### 掩码聚类与分类

训练好的二值掩码代表了各客户端模型更新的参数级特征。使用高斯混合模型 (GMM) 对掩码进行聚类：

- **无攻击时**聚类结果仅一个簇，所有客户端被保留
- **有攻击时**形成两个簇，需判断哪个是良性簇

为确定良性簇，引入可信 root 模型：服务器在 root 数据集上训练一个额外模型，为其分配掩码并一起训练。与 root 模型掩码同簇的客户端被认定为良性。

### 为什么有效

恶意模型的投毒往往集中在特定层/参数。训练过程中，掩码学会将有毒参数的权重降为 0 以优化全局模型性能，因此恶意客户端的掩码呈现出与良性客户端显著不同的 0-1 模式。这种机制不依赖对具体攻击方式的了解，因此是攻击无关的。

### 计算复杂度

- 时间：$O(Tt_m + nVt_m)$，其中 $t_m$ 为掩码训练迭代数，额外开销相当于做几轮额外的本地训练
- 空间：$O(nV)$，约为其他方法的两倍（需存储 $n$ 个掩码），但在服务器端是可接受的
- 掩码参数间无数据依赖，可利用并行训练加速

## 实验关键数据

### 基础设置

- 数据集：Fashion-MNIST、CIFAR-10、CIFAR-100（non-IID, bias $p=0.5$）
- 模型：四层 CNN（Fashion-MNIST）、ResNet20（CIFAR-10/100）
- 100 个客户端，默认 20% 恶意，root 数据集仅 100 个样本

### 防御效果（测试精度，20% 恶意客户端）

| 攻击 | FedAvg | FLTrust | SkyMask |
|------|--------|---------|---------|
| CIFAR-10 Min-Max | 0.58 | 0.68 | **0.77** |
| CIFAR-10 Fang-Trim | 0.10 | 0.68 | **0.76** |
| CIFAR-10 Fang-Krum | 0.58 | 0.75 | **0.77** |
| CIFAR-100 Min-Max | 0.16 | 0.30 | **0.44** |
| CIFAR-100 Fang-Trim | 0.01 | 0.34 | **0.44** |
| CIFAR-100 Fang-Krum | 0.03 | 0.37 | **0.44** |

SkyMask 在所有攻击下的精度达到或接近无攻击水平，最高领先 SOTA 防御 14%。

### 恶意检测质量（CIFAR-10）

| 方法 | Min-Max FPR/FNR | Min-Sum FPR/FNR | Fang-Krum FPR/FNR |
|------|------|------|------|
| Tolpegin | 36.5%/88.0% | 38.8%/78.0% | 37.4%/84.0% |
| FLDetector | 100%/100% | 100%/100% | 87.2%/100% |
| SkyMask | **0%/0%** | **0%/0%** | **0%/0%** |

### 高恶意比例

恶意客户端占比 40%、60%、80% 时，SkyMask 是唯一维持有效防御的方法。当恶意占比超过 50%，FLTrust 也开始失效（精度波动、收敛方向错误），而 SkyMask 仍然稳定收敛。

### 可扩展性

客户端数扩展至 200 和 500 时，在所有细粒度攻击和 targeted 攻击下 FPR 和 FNR 均保持 0%。

## 亮点

1. **参数级检测粒度**：首次将可学习掩码引入拜占庭防御，在参数层面捕获恶意更新特征，跳出了传统模型级检测的局限
2. **攻击无关**：无需针对特定攻击定制防御，对 7 种 SOTA 攻击（含 untargeted 和 targeted）均有效
3. **极端场景鲁棒**：恶意占比高达 80% 仍能防御，是唯一在高恶意比例下维持性能的方法
4. **对良性客户端友好**：无攻击时不影响模型性能（与 FedAvg 差距 <1%），不会误伤正常客户端
5. **模块化设计**：可作为插件集成到各种聚合算法中，兼容性强

## 局限与展望

1. **root 数据集依赖**：服务器需要一个干净的小数据集（100 样本），在某些隐私严格场景下获取可能受限
2. **计算开销**：每轮需要额外的掩码训练迭代（$t_m$ 轮前向/反向传播），当客户端数量大或模型复杂时，服务器计算压力增大
3. **空间开销翻倍**：需为每个客户端存储与模型同尺寸的掩码，当模型参数量很大时（如 LLM）内存需求显著
4. **聚类假设**：假设攻击者行为在掩码空间中形成可分簇，如果攻击者能够适应掩码机制进行自适应攻击，有效性可能下降
5. **仅在 CNN 和 ResNet20 上验证**：缺少更大模型（如 ViT、大规模 ResNet）上的实验

## 与相关工作的对比

| 方法 | 检测粒度 | 攻击无关 | 高恶意比例 | 无攻击影响 |
|------|---------|---------|----------|-----------|
| FLTrust | 模型级 | 部分 | ≤50% | 精度下降 2-5% |
| Krum | 模型级 | 否 | 差 | 精度下降 6-27% |
| Trimmed-Mean | 参数级（贪心） | 否 | 差 | 轻微影响 |
| DeepSight | 模型级 | 部分 | 差 | 轻微影响 |
| FLAME | 模型级 | 部分 | 差 | 轻微影响 |
| FLDetector | 模型级 | 否 | 未测试 | FPR 18.2% |
| **SkyMask** | **参数级** | **是** | **≤80%** | **<1%** |

与最强基线 FLTrust 相比，SkyMask 的核心优势在于：(1) 参数级而非模型级检测，能应对细粒度攻击；(2) 恶意比例超过 50% 时仍有效。

## 启发与关联

- 掩码学习的思路可推广到其他分布式系统的异常检测场景，不限于联邦学习
- 与 FedMask 中掩码用于个性化不同，SkyMask 将掩码用于安全检测，展示了掩码机制的多面性
- root 数据集的使用与 FLTrust 一脉相承，但 SkyMask 更充分地利用了 root 数据的信息——不仅用于计算相似度，还用于驱动掩码训练
- 未来可探索：自适应攻击者如何对抗掩码机制、无 root 数据场景下的变体、掩码训练效率优化

## 评分
- 新颖性: ⭐⭐⭐⭐ — 参数级可学习掩码用于拜占庭防御是新颖的切入点
- 实验充分度: ⭐⭐⭐⭐ — 3 数据集 × 7 攻击 × 多种基线，含高恶意比例和可扩展性实验
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法描述完整，可视化分析有说服力
- 价值: ⭐⭐⭐⭐ — 实用性强，在极端场景下大幅超越现有方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)
- [\[ECCV 2024\] Towards Multi-modal Transformers in Federated Learning](towards_multi-modal_transformers_in_federated_learning.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](../../CVPR2025/ai_safety/infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[ICML 2026\] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection](../../ICML2026/ai_safety/vpd-100k_towards_generalizable_and_fine-grained_visual_privacy_protection.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)

</div>

<!-- RELATED:END -->
