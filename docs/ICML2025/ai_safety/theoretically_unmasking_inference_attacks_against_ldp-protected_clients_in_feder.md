---
title: >-
  [论文解读] Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models
description: >-
  [ICML 2025][AI安全][联邦学习] 首次为联邦学习中基于全连接层和自注意力层的主动成员推断攻击（AMI）：在LDP保护下：推导出理论成功率的上下界，揭示即使在LDP保护下，隐恓风险仍依赖于隐私预算 $\varepsilon$，且要有效缓解攻击所需的噪声会严重损害模型效用。 联邦学习中的隐私风险 - 联邦学习虽然不…
tags:
  - "ICML 2025"
  - "AI安全"
  - "联邦学习"
  - "本地差分隐私"
  - "成员推断攻击"
  - "全连接层"
  - "自注意力机制"
  - "Transformer"
---

# Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models

**会议**: ICML 2025  
**arXiv**: [2506.17292](https://arxiv.org/abs/2506.17292)  
**代码**: 无  
**领域**: AI安全 / 联邦学习隐私  
**关键词**: 联邦学习, 本地差分隐私, 成员推断攻击, 全连接层, 自注意力机制, Vision Transformer  

## 一句话总结

首次为联邦学习中基于全连接层和自注意力层的**主动成员推断攻击（AMI）**在**LDP保护下**推导出理论成功率的上下界，揭示即使在LDP保护下，隐恓风险仍依赖于隐私预算 $\varepsilon$，且要有效缓解攻击所需的噪声会严重损害模型效用。

## 研究背景与动机

### 联邦学习中的隐私风险
- 联邦学习虽然不直接共享数据，但模型更新（梯度或权重）仍可泄露训练数据的敏感信息
- **成员推断攻击（MIA）**：判断某条记录是否在模型训练集中
- **主动MIA（AMI）**：不诚实的服务器主动篡改全局模型参数来增强推断能力

### 现有研究不足
1. 大多数MIA研究要么忽略LDP，要么缺乏理论保证
2. [Vu et al., 2024] 提出了低多项式时间复杂度的AMI攻击，但理论分析仅适用于**无LDP保护**的场景
3. LDP引入的随机噪声跨迭代和客户端变化，使得理论分析极具挑战性

### 研究目标
在LDP保护下，从理论和实践角度证明客户端数据对AMI攻击的**根本脆弱性**。

## 方法详解

### 整体框架

论文在安全博弈框架 $\mathsf{Exp}^{\text{AMI}}_{\text{LDP}}$ 下分析两类攻击：

1. **FC-based AMI**：利用全连接层的结构漏洞
2. **Attention-based AMI**：利用自注意力层的记忆机制

### 关键设计

#### 1. **FC-based 攻击 ($\mathcal{A}^{\mathcal{D}}_{\mathsf{FC}}$)**
- **功能**：通过精心设置两层FC的权重和偏置，检测目标样本 $T$ 是否在训练数据中
- **核心思路**：第一层计算 $z_0 = \max\{\tau^{\mathcal{D}} - \|\mathcal{M}^\varepsilon(X) - T\|_{L_1}, 0\}$
    - 若 $\mathcal{M}^\varepsilon(X)$ 落在以 $T$ 为中心的 $L_1$ 球内 → 梯度非零 → 表明 $T$ 存在
    - 若超出球外 → 梯度为零
- **设计动机**：通过设置 $\tau^{\mathcal{D}} = \Delta^{\mathcal{X}}$（数据字母表中最小 $L_1$ 距离的一半），区分目标和非目标样本

#### 2. **Attention-based 攻击 ($\mathcal{A}^{\mathcal{D}}_{\mathsf{Attn}}$)**
- **功能**：利用自注意力的记忆能力，配置注意力头以记忆输入batch并排除目标pattern
- **核心思路**：
    - 头1过滤目标pattern $v$ → 若 $v$ 在数据中，输出偏向全局均值 $\bar{X}^\varepsilon$
    - 头2正常记忆 → 输出接近输入 $x_i^\varepsilon$
    - 通过两个头输出的差异 $|z_i^1 - z_i^2|$ 是否超过阈值 $\gamma$ 来推断
- **扩展到ViT**：将离散token域的攻击扩展到连续图像domain

### 理论结果

#### Theorem 1（FC攻击下界）
$$\mathbf{Adv}^{\mathsf{AMI}}_{\text{LDP}}(\mathcal{A}^{\mathcal{D}}_{\mathsf{FC}}) \geq 1 - \frac{n + |\mathcal{X}| - 1}{|\mathcal{X}| - 1} P_{\mathcal{M}^\varepsilon}$$

- $P_{\mathcal{M}^\varepsilon}$：LDP机制使保护后的数据跳出邻域球的概率
- 当 $|\mathcal{X}|$ 大（如BitRand），下界约为 $1 - P_{\mathcal{M}^\varepsilon}$

#### Theorem 2（FC攻击上界）
$$\mathbf{Adv}^{\mathsf{AMI}}_{\text{LDP}}(\mathcal{A}^{\mathcal{D}}_{\mathsf{FC}}) \leq \frac{e^\epsilon - 1}{e^\epsilon + 1}$$

#### Theorem 3（Attention攻击下界）
下界由三项组成：$P_{\text{proj}}$（投影分量小于阈值的概率，控制假阳性）、$P_{\text{box}}$（模式落入中心区域的概率，控制假阴性）和数据分离度 $\Delta^\varepsilon$。

### 攻击失败场景
- FC攻击失败：(a) 目标的保护版本跳出 $B_1(T, \Delta^{\mathcal{X}})$；(b) 非目标的保护版本落入球内
- Attention攻击失败：当噪声过大时，所有embedding聚集在中心 → $P_{\text{box}} \approx 1$

## 实验关键数据

### 主实验：FC攻击成功率 vs 隐私预算（Fig. 7-8）

| LDP机制 | 数据集 | ε=3时攻击成功率 | ε=6时攻击成功率 | 准确率损失@80%防护 |
|---------|-------|---------------|---------------|-----------------|
| BitRand | CIFAR10 | ~70% | ~100% | ≥20% |
| GRR | CIFAR10 | ~65% | ~100% | ≥25% |
| RAPPOR | CIFAR10 | ~60% | ~100% | ≥20% |
| dBitFlipPM | CIFAR10 | ~55% | ~100% | ≥30% |

### Attention攻击实验（Fig. 9）

| 模型 | 数据集 | ε=3时成功率 | 批大小影响 |
|------|--------|-----------|----------|
| ViT-B-32-224 | CIFAR10 | ~100% | 稳健（批大小10-100） |
| ViT-B-32-384 | ImageNet-1k | ~100% | 稳健 |

### 关键发现

1. **隐私-效用困境**：将推断率降到80%以下所需的噪声会导致≥20%的模型准确率损失
2. **理论-实验一致性**：理论下界在 $\varepsilon=8$ 时与实验的 $\approx 100\%$ 成功率一致
3. **Attention攻击更强**：ε=3即达到接近100%成功率（FC攻击需ε=5-6）
4. **高维数据更脆弱**（Remark 5）：$d_X \to \infty$ 时 $P_{\text{proj}} \to 1$，攻击优势趋于最大

## 亮点与洞察

1. **首个LDP下AMI的理论框架**：填补了联邦学习隐私保护理论分析的空白
2. **上下界兼备**：Theorem 1+2为FC攻击提供了完整的理论刻画
3. **跨模态验证**：从视觉（ResNet, ViT）扩展到NLP（BERT, GPT-1, RoBERTa），证明隐私风险无处不在
4. **实用警示**：LDP作为"金标准"隐私保护，在面对主动对手时可能需要重新审视

## 局限与展望

1. **Attention攻击的理论仅适用于连续域**：NLP（离散token）场景需要独立理论框架
2. **依赖数据分布先验**：$P_{\mathcal{M}^\varepsilon}$和$P_{\text{proj}}$的具体值依赖于特定LDP机制和数据分布
3. **单轮攻击**：仅考虑单次FL迭代，多轮攻击可能更强
4. **防御策略不足**：主要揭示风险，未提出有效防御方案

## 相关工作与启发

- **AMI攻击起源**：[Nasr et al., 2019] 首次在FL中引入AMI，[Vu et al., 2024] 提出低复杂度变体
- **LDP机制**：BitRand, GRR, RAPPOR, dBitFlipPM 等经典本地隐私算法
- **自注意力记忆性**：[Ramsauer et al., 2021] 证明注意力等价于Hopfield层→为攻击提供理论基础
- **启发**：隐私保护需要综合考虑LDP+安全聚合+差分隐私的多层防御

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次在LDP下给出AMI攻击的理论界
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖CIFAR10/100/ImageNet、4种LDP机制、FC/Attention两类攻击、视觉+NLP
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ — 对联邦学习隐私保护实践具有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models](privacy-shielded_image_compression_defending_against_exploitation_from_vision-la.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[ICML 2025\] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks](de-antifake_rethinking_the_protective_perturbations_against_voice_cloning_attack.md)
- [\[ICML 2025\] Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)

</div>

<!-- RELATED:END -->
