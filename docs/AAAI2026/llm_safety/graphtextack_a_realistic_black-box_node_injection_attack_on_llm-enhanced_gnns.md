---
title: >-
  [论文解读] GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs
description: >-
  [AAAI 2026][LLM安全][文本属性图] 提出 GraphTextack——首个针对 LLM 增强 GNN 的黑盒多模态节点注入投毒攻击，通过进化优化框架联合优化注入节点的图结构连接和语义特征，不依赖模型内部信息或代理模型，在5个数据集和2类LLM-GNN模型上显著优于12种基线方法。 - 领域现状：文本属性图（T…
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "文本属性图"
  - "对抗攻击"
  - "节点注入"
  - "图神经网络"
  - "进化优化"
  - "黑盒攻击"
  - "多模态攻击"
---

# GraphTextack: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs

**会议**: AAAI 2026  
**arXiv**: [2511.12423](https://arxiv.org/abs/2511.12423)  
**代码**: 待确认  
**领域**: AI安全/对抗攻击  
**关键词**: 文本属性图, 对抗攻击, 节点注入, LLM增强GNN, 进化优化, 黑盒攻击, 多模态攻击

## 一句话总结

提出 GraphTextack——首个针对 LLM 增强 GNN 的黑盒多模态节点注入投毒攻击，通过进化优化框架联合优化注入节点的图结构连接和语义特征，不依赖模型内部信息或代理模型，在5个数据集和2类LLM-GNN模型上显著优于12种基线方法。

## 研究背景与动机

- **领域现状**：文本属性图（TAG）中节点同时拥有结构关系和文本描述，近年LLM增强GNN方法（One-for-all、E5+GCN等）通过LLM提取语义嵌入再用GNN做结构聚合，在节点分类等任务上达到SOTA。但这种集成继承了GNN对结构扰动和LLM对文本对抗的双重脆弱性。
- **核心痛点**：(1) 现有攻击方法大多为**单模态**（仅扰动结构或仅扰动文本），对LLM增强GNN效果有限——文本攻击通常仅降低<5%准确率，结构攻击需修改>10%边或白盒访问；(2) 多数攻击假设不切实际——白盒访问、直接修改已有节点/边，或依赖代理模型；(3) 实际场景中攻击者只能创建新实体（如虚假用户、产品），不能修改已有数据。
- **核心矛盾**：LLM增强GNN的脆弱性分布在结构和语义两个相互依赖的模态中，单一模态攻击难以充分利用；但联合优化面临离散、高维、非可微的组合搜索空间。
- **切入角度**：节点注入攻击——注入新节点而非修改已有数据，模拟真实世界中创建虚假账号/产品的场景，采用无需梯度的进化算法导航联合搜索空间。

## 方法详解

### 整体框架

GraphTextack 采用迭代式进化优化框架：对于每个需注入的节点，维护一组候选注入策略的种群，通过选择、交叉、变异和适应度评估迭代优化。核心创新在于：(1) 联合编码结构连接与语义特征的候选表示；(2) 多模态交叉/变异操作；(3) 融合局部预测扰动和全局图影响力的多目标适应度函数。

### 关键设计

1. **候选表示与类条件语义特征生成（C1+C2）**

    - 每个候选 $s_i$ 编码：注入节点的边连接 $E_i' \subseteq V' \times V$ + 特征生成策略（类标签 $c \in \mathcal{Y}$）
    - 边数从原图度分布采样，确保注入节点的连接模式与真实节点相似（隐蔽性）
    - 特征生成：指定类标签 $c$ 后，从属于该类的已有节点的经验分布中采样 $X'(v') \sim p(X(v) | Y(v) = c)$——通过查询目标模型获取伪标签来构建分布
    - 设计动机：避免直接优化连续嵌入（高维、不现实），类条件采样使注入节点在语义空间中自然地融入现有节点分布

2. **多模态多目标适应度函数（C3）**

    - $\text{Fitness}(s_i) = \alpha \cdot \Delta_{\text{conf}}(s_i) + \beta \cdot \text{PR}(s_i)$
    - **局部预测偏移** $\Delta_{\text{conf}}$：注入前后两跳邻域内节点最大置信度的平均变化 $\frac{1}{|\mathcal{N}_2(v')|}\sum_{v \in \mathcal{N}_2(v')}|C_v - C_v'|$——衡量语义层面的攻击效果
    - **全局PageRank影响力** $\text{PR}$：注入后节点的PageRank得分——高中心性意味着通过消息传递影响更多节点
    - 设计动机：单一目标（仅预测偏移或仅中心性）不足以捕捉投毒攻击的效果，因为投毒攻击的影响需要在模型重训后才能体现

3. **多模态进化操作（C4）**

    - **选择**：按适应度排序，保留 top-$N_e$ 精英个体
    - **交叉**：随机选择交叉点 $j$，新候选继承 $s_1$ 的前 $j$ 条边和 $s_2$ 的后续边，类标签从某一父代随机继承——实现结构与语义交互探索
    - **变异**：以概率 $p_{\text{mut}}$ 随机修改边连接或特征分配——维持种群多样性
    - 每个注入步骤独立运行进化优化，逐步构建被投毒的图

### 复杂度分析

搜索空间为 $O(|V|^{r \cdot d_{\max}} \times |\mathcal{F}|^r)$，指数级不可穷举。GraphTextack 在固定种群规模 $N_p$ 和代数 $T_{\text{gen}}$ 下，每步复杂度为 $O(N_p \cdot T_{\text{gen}} \cdot (r \cdot d_{\max}^2 + |E|))$，有效线性于图规模。

## 实验

### 主实验结果（Representation-level Enhancer，One-for-all 模型）

| 数据集 (Clean Acc) | 方法 | r=0.01 | r=0.03 | r=0.05 |
|:--|:--|:--|:--|:--|
| Cora (80.95) | Best text* | — | 74.36 | — |
| | GANI | 77.18 | 70.98 | 66.06 |
| | WTGIA | 76.35 | 69.63 | 65.80 |
| | **GraphTextack** | **73.99** | **65.75** | **62.02** |
| PubMed (71.65) | WTGIA | 65.89 | 48.96 | 43.10 |
| | G²A²C | 62.10 | 54.96 | 43.78 |
| | **GraphTextack** | **60.78** | **48.43** | **42.05** |
| WikiCS (76.31) | WTGIA | 71.87 | **64.17** | **60.95** |
| | **GraphTextack** | **71.69** | 64.58 | 61.35 |
| ogbn-arxiv (75.44) | G²A²C | 72.63 | 69.50 | 66.67 |
| | **GraphTextack** | **71.95** | **68.23** | **66.61** |
| ogbn-products (83.51) | GANI | 78.93 | 74.83 | 69.51 |
| | **GraphTextack** | 78.99 | **74.26** | **69.05** |

### 消融实验（WikiCS，Representation-level Enhancer）

| 变体 | r=0.01 | r=0.05 |
|:--|:--|:--|
| GraphTextack（完整） | **71.69** | **61.35** |
| 无交叉 | 73.50 | 64.42 |
| 无变异 | 73.39 | 65.19 |
| 无预测偏移 | 73.77 | 66.83 |
| 无PageRank | 71.92 | 61.97 |

### 关键发现

- **多模态攻击的优势**：即使最强的文本攻击修改了所有节点特征，其效果仍不如仅注入1-5%节点的 GraphTextack，证实了联合结构+语义攻击的必要性
- **无代理模型优势**：GraphTextack 直接查询目标模型，避免了代理模型与目标模型之间架构差异导致的近似误差，在表示级增强器模型上尤为突出
- **运行效率最优**：通过避免梯度计算和代理模型训练，GraphTextack 在所有数据集上实现了最低的平均注入运行时间
- **消融验证**：交叉/变异/预测偏移/PageRank 四个组件均不可缺少，其中预测偏移对攻击效果影响最大

## 亮点

- ⭐ **首个黑盒多模态节点注入攻击**：不需要模型梯度、参数或代理模型，同时优化结构和语义两个模态
- ⭐ **威胁模型高度现实**：节点注入（创建新实体）比修改已有数据更符合真实攻击场景——如电商注入虚假产品、学术网络注入劣质论文
- ⭐ **提供理论分析**：包括搜索空间复杂度、多模态对抗协同效应（Lemma B.1 的交叉项 $\gamma \cdot d_E d_X$）、局部预测偏移界（Lemma B.3）
- ⭐ 在12种基线中取得整体最优或接近最优的攻击效果，同时运行效率最高

## 局限性

- **类条件特征采样策略较简单**：直接从同类节点的经验分布采样，未学习更具适应性的特征生成策略
- **假设攻击者可以自由创建连接**：实际中注入节点到已有节点的连接可能受限（如社交网络需对方确认）
- **进化超参数敏感性**：种群大小、交叉/变异概率、$\alpha$/$\beta$ 平衡系数需要调优
- **未讨论防御**：仅提到"为未来防御工作奠定基础"，未评估任何防御手段下的攻击鲁棒性

## 相关工作

- **图对抗攻击（结构）**：Nettack (Zügner et al. 2018)——贪心修改; PRBCD (Geisler et al. 2021)——梯度投影; Meta-attack (Zügner & Günnemann 2020)
- **节点注入攻击**：AFGSM (Wang et al. 2020)——近似快速梯度符号; TDGIA (Zou et al. 2021)——拓扑缺陷; GANI (Fang et al. 2024)——遗传算法+代理模型; G²A²C (Ju et al. 2023)——强化学习
- **文本对抗攻击**：BertAttack (Li et al. 2020), HotFlip (Ebrahimi et al. 2018), VIPER (Eger et al. 2019)
- **LLM-GNN攻击**：WTGIA (Lei et al. 2024)——文本级注入攻击（单模态）; Guo et al. (2024)——白盒修改攻击

## 评分

⭐⭐⭐⭐ — 问题定义新颖（首个黑盒多模态注入），实验全面（5数据集×2模型×12基线），理论与实验兼顾。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[CVPR 2026\] Omni-Attack: Adversarial Attacks on Open-Ended VQA in Black-Box Multimodal LLMs](../../CVPR2026/llm_safety/omni-attack_adversarial_attacks_on_open-ended_vqa_in_black-box_multimodal_llms.md)
- [\[ICLR 2026\] Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test](../../ICLR2026/llm_safety/auditing_black-box_llm_apis_with_a_rank-based_uniformity_test.md)
- [\[ACL 2026\] Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](../../ACL2026/llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)
- [\[ACL 2025\] Defense Against Prompt Injection Attack by Leveraging Attack Techniques](../../ACL2025/llm_safety/defense_prompt_injection.md)

</div>

<!-- RELATED:END -->
