---
title: >-
  [论文解读] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion
description: >-
  [ICML 2025][计算生物][治疗性肽设计] PepTune 结合 Masked Discrete Language Model (MDLM) 和蒙特卡罗树搜索 (MCTS) 多目标引导策略，在离散肽 SMILES 空间中同时优化多种治疗属性（结合亲和力、溶解性、膜通透性等），生成含非天然氨基酸和环化修饰的从头设计肽药物。
tags:
  - "ICML 2025"
  - "计算生物"
  - "治疗性肽设计"
  - "离散扩散"
  - "多目标优化"
  - "MCTS"
  - "SMILES"
---

# PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion

**会议**: ICML 2025  
**arXiv**: [2412.17780](https://arxiv.org/abs/2412.17780)  
**代码**: [https://huggingface.co/ChatterjeeLab/PepTune](https://huggingface.co/ChatterjeeLab/PepTune)  
**领域**: 计算生物  
**关键词**: 治疗性肽设计, 离散扩散, 多目标优化, MCTS, SMILES

## 一句话总结
PepTune 结合 Masked Discrete Language Model (MDLM) 和蒙特卡罗树搜索 (MCTS) 多目标引导策略，在离散肽 SMILES 空间中同时优化多种治疗属性（结合亲和力、溶解性、膜通透性等），生成含非天然氨基酸和环化修饰的从头设计肽药物。

## 研究背景与动机

**领域现状**：肽疗法（如 GLP-1 受体激动剂 semaglutide/liraglutide）在糖尿病、肥胖等疾病治疗中取得里程碑式成功。肽能结合多样的蛋白表面、打断蛋白-蛋白相互作用，自 2000 年以来已有 33 种 FDA 批准的治疗性肽。

**现有痛点**：设计同时满足多个冲突目标（如结合亲和力、溶解性、膜通透性）的肽是重大挑战。现有方法限于(1)连续空间、(2)无条件生成、(3)单目标引导。传统方法依赖筛选 10^12 量级的随机组合文库。

**核心矛盾**：治疗性肽需要含非天然氨基酸(nAAs)和环化修饰（提高稳定性和通透性），但现有深度学习模型只能处理 20 种标准氨基酸；同时，多目标引导在离散空间极为困难——梯度无法直接计算。

**本文目标**：构建一个能在离散肽 SMILES 空间中进行多目标条件生成的扩散模型。

**切入角度**：(1)用 SMILES 而非氨基酸序列表示肽→支持 nAAs 和环化；(2)用 MCTS 而非梯度引导→解决离散空间引导难题。

**核心 idea**：MDLM 负责探索离散肽空间的有效结构，MCTS 负责引导生成向多个治疗目标的 Pareto 最优方向演化。

## 方法详解

### 整体框架

- **输入**：目标蛋白序列 + 待优化的治疗属性列表
- **第一阶段**：在 1100 万条肽 SMILES 上预训练 PepMDLM（无条件生成器）
- **第二阶段**：用属性分类器引导的 MCTS 策略对生成过程进行多目标条件引导
- **输出**：一组 Pareto 最优的肽 SMILES + 各项属性分数

### 关键设计

1. **状态依赖的 Masking Schedule**:

    - 核心洞察：肽键(peptide bond)是所有有效肽的基础结构
    - 设计多项式masking schedule：$\alpha_t(\mathbf{x}_0) = 1 - t^w$ 对肽键token，$\alpha_t = 1 - t$ 对非肽键token
    - 肽键token在前向过程中被更晚mask，在逆过程中被更早unmask
    - 训练损失中肽键token权重放大 $w$ 倍：$\frac{w}{t} \log \langle \mathbf{x}_0, \mathbf{x}_\theta \rangle$
    - **为什么**：肽的绝大多数 SMILES 不是有效肽——让模型先"搭好骨架"再填充侧链

2. **全局序列无效性损失 (Invalid Loss)**:

    - 对预测概率取 argmax 得到离散序列，检查是否为有效肽
    - 无效序列通过 softmax 概率加权的惩罚传播梯度
    - $\mathcal{L}_{\text{invalid}} = \sum_\ell \text{SM}(x_{\theta,k}^{(\ell)}) \cdot \mathbf{1}[\tilde{\mathbf{x}}_0 \text{ is Invalid}]$
    - **为什么**：argmax 不可微，用 softmax 概率作为 surrogate 来绕过

3. **MCTS 多目标引导**:

    - **Selection**：从根节点（全mask）开始，根据累积奖励选择非支配的子节点路径
    - **Expansion**：用 Gumbel noise 采样 $M$ 个不同的 unmask 方案
    - **Rollout**：贪心 unmask 到完整序列，用分类器计算 $K$ 个目标的分数
    - **Backpropagation**：奖励向量回传更新路径上所有节点
    - 奖励定义：$r_k(\mathbf{x}) = \frac{1}{|\mathcal{P}^*|} \sum_n \mathbb{I}[s_k(\mathbf{x}) \geq s_k(\tilde{\mathbf{x}}_n)]$（击败 Pareto 集中的比例）
    - 无效肽惩罚：从所有维度扣减与无效比例成正比的分数
    - **为什么**：离散空间无梯度，MCTS 通过搜索+奖励信号实现不依赖梯度的多目标引导

### 损失函数 / 训练策略

- 总损失：$\mathcal{L} = \mathcal{L}_{\text{NELBO}}^\infty + \mathcal{L}_{\text{invalid}}$
- RoFormer 骨架：8层，768 隐藏维度，8 注意力头
- 训练数据：1100 万肽 SMILES（SmProt + CycloPs 合成数据）
- PeptideCLM SPE 分词：581 tokens，平均 4 字符/token
- 8×A6000 GPU，1600 GPU 小时，AdamW，lr=3e-4
- MCTS：128 iterations，50 children/expansion

## 实验关键数据

### 主实验

| 目标蛋白 | 属性 | 最佳 Docking Score | 对照 |
|---------|------|-------------------|------|
| TfR (血脑屏障) | 结合+溶解+非溶血 | -8.4 kcal/mol | T7肽: -8.4（但PepTune更短） |
| GLP-1R (糖尿病) | 结合+溶解+非溶血 | -7.4 kcal/mol | semaglutide: -5.7 (更长) |
| GFAP (亚历山大病) | 结合+通透+溶解+非溶血 | -8.5 kcal/mol | 无已知肽 binder |
| TfR+GLAST (双靶) | 双靶结合+溶解 | TfR:-10.5, GLAST:-9.2 | 首次双靶肽设计 |

### 消融实验

| 配置 | Validity↑ | Diversity | 说明 |
|------|----------|-----------|------|
| PepMDLM (无引导) | 45% (len=100) | 0.705 | 基线无条件 |
| PepTune (MCTS引导) | **100%** (after 20 iter) | 0.677 | MCTS 提升有效性到100% |
| HELM-GPT (对照) | 83.9% | 0.595 | HELM 表示，不支持 nAAs |
| 无 state-dependent mask | ~30% | - | 有效性显著下降 |
| 无 invalid loss | ~35% | - | 有效性有下降 |

### 关键发现

- MCTS 引导在 20 迭代后即达到 100% 有效肽生成率
- 生成的肽含丰富的非天然氨基酸（平均 2.94/肽）和环化结构
- 多目标引导不显著牺牲多样性（相比无条件生成仅差 0.03）
- 双靶 peptide 实验证明 PepTune 可同时优化对两个蛋白的结合亲和力
- GLP-1R binder docking 分数优于 semaglutide，且序列更短

## 亮点与洞察

1. **首个离散空间多目标引导扩散**：MCTS + masked diffusion 的组合是对该领域的重要贡献
2. **SMILES 表示的优势**：支持 nAAs 和环化，大幅扩展了可设计的肽空间
3. **状态依赖 masking 的物理直觉**：先骨架后侧链的生成顺序与化学直觉一致
4. **临床相关性**：多个真实治疗靶点的案例研究，有下游实验验证前景

## 局限与展望

1. 依赖合成肽数据（CycloPs），稀有 nAAs 可能增加合成难度和成本
2. 属性分类器的质量是瓶颈——膜通透性等属性缺乏外部验证
3. MCTS 采样速度较慢（128 迭代 × 50 children × rollout）
4. 尚无湿实验验证（仅计算 docking）

## 相关工作与启发

- MDLM (Sahoo et al.) 提供了 masked discrete diffusion 的基础
- RFpeptides 等结构基础模型需要目标 3D 结构，PepTune 只需序列
- 启发：MCTS 引导策略可扩展到蛋白质设计、DNA 序列优化等其他离散生物序列生成

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ MCTS + 离散扩散 + 肽 SMILES 的组合非常创新
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个单靶 + 2 个双靶案例，详细消融
- 写作质量: ⭐⭐⭐⭐ 内容丰富，但篇幅较长
- 价值: ⭐⭐⭐⭐⭐ 对药物设计有重要实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[ICML 2025\] Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing](latent_imputation_before_prediction_a_new_computational_paradigm_for_de_novo_pep.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/computational_biology/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[ICML 2025\] Kinetic Langevin Diffusion for Crystalline Materials Generation](kinetic_langevin_diffusion_for_crystalline_materials_generation.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)

</div>

<!-- RELATED:END -->
