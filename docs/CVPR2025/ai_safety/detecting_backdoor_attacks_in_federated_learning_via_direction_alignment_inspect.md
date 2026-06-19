---
title: >-
  [论文解读] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection
description: >-
  [CVPR 2025][AI安全][联邦学习] 提出 AlignIns 防御方法，通过双粒度方向对齐检测（全局方向 + 细粒度符号分析）识别联邦学习中的恶意模型更新，在 IID 和 non-IID 设置下均优于现有防御方法。 领域现状：联邦学习（FL）因分布式训练特性天然容易受到后门攻击，恶意客户端可提交带毒的模型更新以操纵…
tags:
  - "CVPR 2025"
  - "AI安全"
  - "联邦学习"
  - "backdoor attack"
  - "defense"
  - "direction alignment"
  - "sign analysis"
  - "anomaly detection"
---

# Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection

**会议**: CVPR 2025  
**arXiv**: [2503.07978](https://arxiv.org/abs/2503.07978)  
**代码**: [JiiahaoXU/AlignIns](https://github.com/JiiahaoXU/AlignIns)  
**机构**: University of Nevada, Reno
**领域**: AI 安全 / 联邦学习  
**关键词**: federated learning, backdoor attack, defense, direction alignment, sign analysis, anomaly detection

## 一句话总结
提出 AlignIns 防御方法，通过双粒度方向对齐检测（全局方向 + 细粒度符号分析）识别联邦学习中的恶意模型更新，在 IID 和 non-IID 设置下均优于现有防御方法。

## 研究背景与动机

**领域现状**：联邦学习（FL）因分布式训练特性天然容易受到后门攻击，恶意客户端可提交带毒的模型更新以操纵全局模型。已有多种攻击方法（Badnet、DBA、Scaling、PGD、Neurotoxin）威胁 FL 安全。

**现有痛点**：
   - **基于幅度的防御**（Manhattan/Euclidean 距离）：当模型趋近收敛时，所有更新幅度都很小，恶意更新与正常更新在幅度上难以区分
   - **基于 Cosine 相似度的防御**（FoolsGold 等）：仅捕获全局方向相似性，忽略细粒度信息（如参数符号分布）
   - 在 non-IID 数据场景下，正常客户端的更新方向本身就很多样，使得异常检测更加困难
   - 缺乏对非 IID 数据下过滤型防御方法的理论分析

**核心矛盾**：后门攻击的双重目标（保持主任务精度 + 最大化后门精度）使得恶意更新在幅度上必须模仿正常更新，但在方向的细粒度特征上可能暴露异常。

**切入角度**：从两个粒度检查方向对齐——全局时序方向对齐和细粒度重要参数符号对齐。

**核心 idea**：时序方向对齐（TDA） + 重要参数符号对齐（MPSA） + MZ-score 异常检测 + 后过滤裁剪 = 鲁棒后门防御。

## 方法详解

### AlignIns 整体流程
接收所有客户端模型更新 → 方向对齐检测（两步） → 过滤恶意更新 → 裁剪 → 聚合

### 关键设计

1. **时序方向对齐（Temporal Direction Alignment, TDA）**

    - 功能：评估每个模型更新与最新全局模型方向的 Cosine 相似度
    - 核心思路：正常更新应与全局收敛方向大致一致，恶意更新可能有异常对齐模式
    - 计算：$\text{TDA}_i = \cos(\Delta_i^t, \theta^t)$
    - 用 MZ-score 进行异常检测，超出 $\lambda_c$ 半径的标记为可疑

2. **重要参数符号对齐（Major Parameter Sign Alignment, MPSA）**

    - 功能：分析模型更新中重要参数的符号分布
    - 核心思路：提取每个更新中幅度 Top-$k$（$k = 0.3 \times d$）的参数，统计其符号与所有更新的主导符号（principle sign）的对齐比例
    - 主导符号：跨所有更新的多数投票符号
    - 效果：捕获全局 Cosine 相似度无法发现的细粒度异常

3. **MZ-score 异常检测**

    - 使用鲁棒的 Modified Z-score（基于中位数而非均值）
    - 超参数最少：仅需 $\lambda_c$ 和 $\lambda_s$ 两个过滤半径
    - 默认值：$\lambda_c = 1.0$，$\lambda_s = 1.0$

4. **后过滤模型裁剪（Post-filtering Clipping）**

    - 对通过方向检测的更新，进一步裁剪异常大的幅度
    - 防御可能绕过方向检测的幅度攻击

### 理论贡献
- 提供 AlignIns 鲁棒性的理论分析
- 证明 AlignIns 在 FL 训练中的传播误差有界
- 首个对非 IID 数据下过滤型防御的理论鲁棒性分析

## 实验关键数据

### IID CIFAR-10 主实验（ResNet9，20% 攻击者，50% 下毒率）

| 方法 | 干净 MA↑ | Badnet BA↓ | DBA BA↓ | Neurotoxin BA↓ | 平均 RA↑ |
|------|---------|-----------|---------|---------------|---------|
| FedAvg（无防御） | 89.47 | 67.61 | 70.42 | 79.40 | — |
| FoolsGold | — | — | — | — | 较低 |
| Multi-Metrics | — | — | — | — | 中等 |
| **AlignIns** | **最优** | **最低** | **最低** | **最低** | **最优** |

### 跨设备 FL 设置（100 客户端，CIFAR-10）

| 方法 | IID RA↑ | Non-IID RA↑ |
|------|---------|------------|
| FoolsGold | 82.99 | 较低 |
| **AlignIns** | **最优** | **最优** |

AlignIns 在 cross-device（大规模客户端）设置下同样有效。

### 消融实验（CIFAR-10）

| 配置 | IID 平均 RA↑ | IID BA↓ | Non-IID 平均 RA↑ |
|------|-------------|---------|-----------------|
| 仅 MPSA (30%) | 88.55 | 2.88 | — |
| TDA + MPSA（完整） | **最优** | **最低** | **最优** |

### 关键发现
- TDA 和 MPSA 互补：TDA 捕获全局方向异常，MPSA 捕获细粒度符号异常
- 在 non-IID 场景下优势更大，因为 MPSA 不受正常更新多样性的影响
- 对 5 种 SOTA 攻击（Badnet、DBA、Scaling、PGD、Neurotoxin）均有效

## 实验设置补充

### 数据集与FL配置

| 参数 | 默认值 |
|------|-------|
| 客户端数 | 20（cross-silo）/ 100（cross-device） |
| 攻击比例 | 20%（4/20 恶意客户端） |
| 下毒率 | 50% |
| Non-IID 程度 | Dirichlet β=0.5 |
| 本地训练 epoch | 2 |
| CIFAR-10 训练轮数 | 150 |
| CIFAR-100 训练轮数 | 100 |
| MPSA 参数 k | 0.3×d（Top-30%参数） |

## 亮点与洞察
- **双粒度检测**逻辑清晰：全局方向 + 参数符号分布，从粗到细覆盖不同层面的异常
- **MPSA 指标**新颖：利用重要参数的符号分布而非幅度，在模型收敛时仍有区分力
- **MZ-score** 比标准 Z-score 更鲁棒，对异常值不敏感
- 理论分析完备：首个证明非 IID 下过滤型防御鲁棒性有界的工作
- 完全兼容现有 FL 框架，无需修改客户端训练过程
- 附录验证了对 trigger-optimization 等自适应攻击的鲁棒性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](../../ICML2025/ai_safety/adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[CVPR 2025\] DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders](dede_detecting_backdoor_samples_for_ssl_encoders_via_decoders.md)

</div>

<!-- RELATED:END -->
