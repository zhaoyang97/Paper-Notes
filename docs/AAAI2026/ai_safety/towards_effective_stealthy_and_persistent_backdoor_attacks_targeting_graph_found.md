---
title: >-
  [论文解读] Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models
description: >-
  [AAAI2026][AI安全][backdoor attack] 提出 GFM-BA，首个系统性地针对 Graph Foundation Models (GFMs) 预训练阶段的后门攻击方法，通过 label-free trigger 关联、node-adaptive trigger 生成和 persistent backdoor anchoring 三个模块，同时解决有效性、隐蔽性和持久性三大挑战。
tags:
  - "AAAI2026"
  - "AI安全"
  - "backdoor attack"
  - "graph foundation model"
  - "图神经网络"
  - "trigger generation"
  - "adversarial ML"
---

# Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models

**会议**: AAAI2026  
**arXiv**: [2511.17982](https://arxiv.org/abs/2511.17982)  
**代码**: [RingBDStack/GFM-BA](https://github.com/RingBDStack/GFM-BA)  
**领域**: AI安全  
**关键词**: backdoor attack, graph foundation model, GNN security, trigger generation, adversarial ML

## 一句话总结
提出 GFM-BA，首个系统性地针对 Graph Foundation Models (GFMs) 预训练阶段的后门攻击方法，通过 label-free trigger 关联、node-adaptive trigger 生成和 persistent backdoor anchoring 三个模块，同时解决有效性、隐蔽性和持久性三大挑战。

## 研究背景与动机
GFMs 在多域图数据上预训练后适配下游任务，用户常直接使用开源预训练模型。这创造了后门攻击的现实威胁面：攻击者控制预训练阶段，注入后门后发布模型。

传统 GNN 后门攻击与 GFM 场景的根本差异：

| 条件 | 传统 GNN | GFM |
|---|---|---|
| 下游标签可用 | ✓ | ✗ |
| 同域训练/推理 | ✓ | ✗（跨域） |
| 模型参数不变 | ✓ | ✗（下游 fine-tune） |

由此引出三大挑战：

**有效性**：预训练时无法获知下游标签，如何确保 trigger 导致预期的分类错误？

**隐蔽性**：不同域的节点特征分布差异大，固定 trigger 易被异常检测发现

**持久性**：下游 fine-tuning 可能擦除后门行为（backdoor forgetting）

现有方法局限：GCBA 需要下游标签；CrossBA 无法控制目标标签，退化为对抗逃逸攻击。

## 方法详解

### 模块1：Label-Free Trigger Association
- 用预训练 GNN 提取预训练图的节点嵌入，通过 **Farthest Point Sampling (FPS)** 选取 $k$ 个 prototype embeddings
- FPS 的贪心策略确保 prototype 分散覆盖嵌入空间，理论证明（Proposition 1）当类间分离度足够大时，FPS 更可能覆盖多个下游类别
- 下游注入时，攻击者通过少量试探查询找到对应目标标签的 prototype

### 模块2：Node-Adaptive Trigger Generator
- 用 MLP 根据目标节点特征 $\mathbf{x}_i$ 和目标嵌入 $\mathbf{e}_j$ 动态生成 trigger 特征：$\mathbf{x}_{ij}^{tri} = \text{MLP}([\mathbf{x}_i \| \mathbf{e}_j])$
- Trigger 设计为 3 节点全连接子图，插入目标节点邻域
- 双目标优化：$\mathcal{L}_{eff}$ 确保 triggered 节点嵌入对齐目标 prototype；$\mathcal{L}_{ste}$ 确保 trigger 特征与目标节点特征相似（保持 graph homophily）
- 关键：不修改预训练模型参数，利用编码器中已有的 latent backdoor logic

### 模块3：Persistent Backdoor Anchoring
- 实验观察：大部分预训练参数在下游 fine-tuning 中变化极小
- 用 graph mixup 合成跨域图模拟潜在下游分布
- 基于 model pruning 的重要性估计识别 fine-tuning 敏感参数
- 对敏感参数施加随机扰动 $\theta_k \leftarrow \theta_k + \epsilon|\theta_k|$，训练 trigger generator 在扰动下仍保持有效性
- 持久性损失：$\mathcal{L}_{per} = \text{Var}(\{\mathcal{L}_{eff}^j\}) + \text{Mean}(\{\mathcal{L}_{eff}^j\})$

## 实验关键数据

### 攻击有效性（ASR %，Target-Controlled 场景）

| 方法 | Cora | CiteSeer | PubMed | Photo | Computers |
|---|---|---|---|---|---|
| GCBA_M (GCOPE) | 4.77 | 5.98 | 21.65 | 3.48 | 4.62 |
| CrossBA (GCOPE) | 14.29 | 16.67 | 33.33 | 9.25 | 7.98 |
| **GFM-BA (GCOPE)** | **90.40** | **89.06** | **100.00** | **84.53** | **78.54** |
| CrossBA (SAMGPT) | 13.61 | 16.67 | 33.33 | 12.10 | 9.20 |
| **GFM-BA (SAMGPT)** | **100.00** | **100.00** | **100.00** | **99.80** | **100.00** |

Target-Controlled ASR 比最强基线 CrossBA 提升 **66-91%**。

### 隐蔽性（边净化后 ASR）
GFM-BA 在边净化防御后仍保持高 ASR（GCOPE 上 100%），平均超越基线 **36.81%**（GCOPE）、**19.98%**（MDGPT）、**36.73%**（SAMGPT）。清洁准确率不下降。

### 持久性（fine-tuning 后 ASR 下降）

| 方法 | Cora Drop | Photo Drop | Computers Drop |
|---|---|---|---|
| CrossBA (SAMGPT) | ↓4.74 | ↓9.40 | ↓0.60 |
| **GFM-BA (SAMGPT)** | **↓1.34** | **↓4.00** | **↓1.40** |
| CrossBA (MDGPT) | ↓1.36 | ↓4.60 | ↓2.40 |
| **GFM-BA (MDGPT)** | **↓0.68** | **↓0.60** | **↓0.80** |

Fine-tuning 后 ASR 下降极小（多数 <2%），持久性显著优于基线。

## 亮点
- **Label-free 攻击范式**：通过 FPS 选取 prototype embeddings 绕过对下游标签的依赖，是 GFM 后门攻击的关键突破
- **自适应 trigger 生成**：node-adaptive 设计保持 graph homophily，显著提升隐蔽性
- **无需修改模型参数**：利用预训练编码器的 latent logic，清洁准确率不受影响
- **理论支撑**：Proposition 1/2 分别为 FPS 覆盖性和参数不敏感性锚定提供理论基础

## 局限性
- 仅验证节点分类任务，图分类/链接预测场景未涉及
- FPS 的 prototype 覆盖性在类别极不均衡时可能失效
- 攻击假设需要下游少量试探查询来匹配 prototype 与标签，部分场景下可能不可行
- 防御评估仅用简单边净化，缺少对 spectral filtering、model pruning 等更强防御的测试
- 3 节点固定 trigger 结构的最优性未探讨

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统解决 GFM 后门攻击三大挑战，label-free 设计有突破性
- 实验充分度: ⭐⭐⭐⭐ — 5 数据集 x 3 victim GFMs x 3 基线，含消融和超参分析
- 写作质量: ⭐⭐⭐⭐ — 问题动机和挑战分析清晰，方法描述严谨
- 价值: ⭐⭐⭐⭐ — 揭示 GFM 安全隐患，推动可信 AI 研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Stealthy and Effective Backdoor Attacks on Lane Detection: A Naturalistic Data Poisoning Approach](../../CVPR2026/ai_safety/towards_stealthy_and_effective_backdoor_attacks_on_lane_detection_a_naturalistic.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[CVPR 2026\] DASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples](../../CVPR2026/ai_safety/dash_a_meta-attack_framework_for_synthesizing_effective_and_stealthy_adversarial.md)
- [\[ICLR 2026\] Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning](../../ICLR2026/ai_safety/hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)
- [\[CVPR 2026\] Towards Human-Imperceptible Backdoor Attacks on Text-to-Image Diffusion Models](../../CVPR2026/ai_safety/towards_human-imperceptible_backdoor_attacks_on_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
