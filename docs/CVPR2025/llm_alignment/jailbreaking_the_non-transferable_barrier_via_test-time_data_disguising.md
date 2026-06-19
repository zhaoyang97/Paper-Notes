---
title: >-
  [论文解读] Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising
description: >-
  [CVPR 2025][LLM对齐][non-transferable learning] 提出 JailNTL，首个针对 Non-Transferable Learning (NTL) 模型的黑盒攻击方法，通过测试时数据伪装将未授权域的数据"变装"为授权域的数据，仅用 1% 授权样本即可将未授权域准确率提升最高 55.7%，无需修改模型。
tags:
  - "CVPR 2025"
  - "LLM对齐"
  - "non-transferable learning"
  - "model IP protection"
  - "adversarial attack"
  - "域适应"
  - "data disguising"
---

# Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising

**会议**: CVPR 2025  
**arXiv**: [2503.17198](https://arxiv.org/abs/2503.17198)  
**代码**: [https://github.com/tmllab/2025_CVPR_JailNTL](https://github.com/tmllab/2025_CVPR_JailNTL)  
**领域**: LLM对齐  
**关键词**: non-transferable learning, model IP protection, adversarial attack, domain adaptation, data disguising

## 一句话总结
提出 JailNTL，首个针对 Non-Transferable Learning (NTL) 模型的黑盒攻击方法，通过测试时数据伪装将未授权域的数据"变装"为授权域的数据，仅用 1% 授权样本即可将未授权域准确率提升最高 55.7%，无需修改模型。

## 研究背景与动机

**领域现状**：Non-Transferable Learning (NTL) 是一种模型知识产权保护技术，通过训练使模型在授权域上表现良好但在未授权域上性能严重下降（"非迁移屏障"）。现有攻击方法要求白盒访问模型（参数可见）。

**现有痛点**：(1) 白盒攻击（如 RTAL、FTAL、TransNTL）需要完整模型参数访问，在实际 API 部署场景中不可行。(2) 现有白盒攻击效果有限——TransNTL 在 CIFAR10→STL10 上仅恢复 27.2% 准确率。(3) 无黑盒攻击方法存在。

**核心矛盾**：NTL 模型的非迁移屏障本质上是一种域间的"语义断裂"——模型学会了区分授权和未授权域的数据分布差异。如果能在测试时把未授权域数据"伪装"成授权域数据，就能绕过屏障。

**本文目标** 在只能获得模型 logits（黑盒）、极少量授权样本（1%）和未标注未授权测试数据的条件下，恢复模型在未授权域上的性能。

**切入角度**：不修改模型，而是修改输入——训练一个数据伪装网络将未授权域数据变换到授权域分布，同时保持内容（语义）不变。

**核心 idea**：训练一个轻量伪装网络，让未授权域的数据在外观上像授权域、在内容上保持原样，从而"欺骗" NTL 模型跨过非迁移屏障。

## 方法详解

### 整体框架
JailNTL 包含两个核心组件：Data-Intrinsic Disguising (DID) 负责外观伪装，Model-Guided Disguising (MGD) 利用 NTL 模型的 logits 进一步优化伪装效果。推理时只需将未授权域数据通过伪装网络变换后再送入 NTL 模型。

### 关键设计

1. **Data-Intrinsic Disguising (DID)**

    - 功能：将未授权域数据的外观变换到授权域分布
    - 核心思路：
        - 正向伪装：$f_d(x_u) \approx$ 授权域外观，对抗损失 $L_{adv}$ 让判别器无法区分
        - 反向重建：$\hat{f}_d(f_d(x_u)) \approx x_u$，一致性损失 $L_{cs}$ 保持内容
        - 双向结构：同时训练反向伪装 $\hat{f}_d(x_a) \approx$ 未授权域外观
        - 伪装网络 $f_d$：ResNet 架构（2 下采样 + 9 残差块 + 2 上采样）
    - 设计动机：GAN 式训练确保伪装后数据的分布与授权域一致

2. **Model-Guided Disguising (MGD)**

    - 功能：利用 NTL 模型的黑盒输出进一步优化伪装
    - 核心思路：
        - 置信度匹配：$L_{cf} = |E_{cf}(x_a) - E_{cf}(f_d(x_u))|$，让伪装数据和授权数据在模型上的输出熵一致
        - 类别平衡：$L_{ba} = |E_{ba}(P_{disg}) - E_{ba}(P_{auth})|$，确保伪装数据的类别分布与授权域一致
        - 零阶梯度估计：用有限差分近似梯度，避免需要模型内部参数
    - 设计动机：DID 只做分布层面的伪装，MGD 进一步从模型行为层面确保伪装效果

### 损失函数 / 训练策略
- 总损失：$L_{total} = L_{adv} + L_{advr} + \lambda_{cs}(L_{cs} + L_{csr}) + \lambda_{cf}L_{cf} + \lambda_{ba}L_{ba}$
- 优化器：SGD（降质）/ Adam（增强），学习率 $\gamma_e = 10^{-4}$，$\gamma_g = 2 \times 10^{-4}$
- 判别器：PatchGAN，Batch size 5，RTX 4090 GPU

## 实验关键数据

### 主实验

| 域对 | NTL方法 | 未授权域Acc(攻击前) | 未授权域Acc(JailNTL) | 提升 |
|-----|---------|-------------------|---------------------|------|
| CIFAR10→STL10 | CUTI | 9.0 | 64.7 | **+55.7** |
| CIFAR10→STL10 | NTL | 9.8 | 61.4 | +51.6 |
| STL10→CIFAR10 | CUTI | 9.9 | 43.5 | +33.6 |
| VisDA-T→V | CUTI | 10.0 | 25.4 | +15.4 |

**vs 白盒 SOTA TransNTL：** CIFAR10→STL10 27.2% → JailNTL 61.4%（+34.2%，且 JailNTL 是黑盒！）

### 消融实验

| 配置 | CIFAR10→STL10 (CUTI) |
|------|---------------------|
| DID only (JailNTL*) | 64.0% |
| + Lcf (置信度) | 64.3% |
| + Lba (类别平衡) | 64.0% |
| Full JailNTL | **64.7%** |

| 联合攻击 | STL10→CIFAR10 |
|---------|---------------|
| TransNTL alone | 37.7% |
| TransNTL + JailNTL | **44.1%** (+6.4%) |

### 关键发现
- **黑盒超越白盒**：JailNTL（黑盒）61.4% vs TransNTL（白盒）27.2%，说明数据层面的攻击比模型层面更有效
- DID 贡献了绝大部分效果（64.0%），MGD 提供增量改善
- t-SNE 可视化：伪装后的数据（蓝色）与授权域（绿色）聚类，远离未授权域（红色）
- Grad-CAM 分析：伪装数据的注意力图与授权域数据匹配
- 与白盒方法互补：TransNTL + JailNTL 组合比单独使用更好

## 亮点与洞察
- **数据攻击 vs 模型攻击的范式转换**：不修改模型而修改输入，在黑盒场景下天然更适用
- **1% 授权样本即可**：极低的数据需求使攻击变得非常实际
- **黑盒超越白盒**：打破了"白盒一定更强"的直觉，揭示了 NTL 屏障的脆弱性
- 对 NTL 领域是重要的安全警告——当前的非迁移屏障可能不够安全

## 局限与展望
- VisDA 上提升相对有限（+15.4%），大域间差距下伪装效果有限
- 伪装网络本身需要训练资源和少量授权样本
- 未对抗更强的 NTL 防御方法
- 伪装后的图像在像素层面已经改变，可能被简单的输入检测发现

## 相关工作与启发
- **vs TransNTL**: 白盒方法，直接微调模型参数，效果远不如 JailNTL
- **vs RTAL/FTAL**: 白盒微调方法，几乎无法恢复未授权域性能

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个黑盒攻击 NTL 的方法，数据伪装思路新颖
- 实验充分度: ⭐⭐⭐⭐ 多个域对、消融、可视化分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐⭐ 对 NTL 安全性的重要警示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] On the Rejection Criterion for Proxy-Based Test-Time Alignment](../../ACL2026/llm_alignment/on_the_rejection_criterion_for_proxy-based_test-time_alignment.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](../../ACL2025/llm_alignment/queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[CVPR 2025\] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)
- [\[ACL 2025\] Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction](../../ACL2025/llm_alignment/rewrite_to_jailbreak_discover_learnable_and_transferable_implicit_harmfulness_in.md)

</div>

<!-- RELATED:END -->
