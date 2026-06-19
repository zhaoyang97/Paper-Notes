---
title: >-
  [论文解读] Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization
description: >-
  [ACL 2025][计算生物][蛋白质语言模型] 提出KPO框架，通过构建蛋白质安全知识图谱(PSKG)并结合加权图剪枝策略识别"相似但安全"的蛋白质对，用DPO微调蛋白质语言模型使其远离有害序列空间，同时保持功能性。 领域现状：蛋白质语言模型（PLM）如ProtGPT2、ProGen2等在蛋白质序列生成上取得了巨大成功…
tags:
  - "ACL 2025"
  - "计算生物"
  - "蛋白质语言模型"
  - "知识图谱"
  - "DPO"
  - "生物安全"
  - "偏好优化"
---

# Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization

**会议**: ACL 2025  
**arXiv**: [2507.10923](https://arxiv.org/abs/2507.10923)  
**代码**: [GitHub](https://github.com/HICAI-ZJU/KPO)  
**领域**: 计算生物  
**关键词**: 蛋白质语言模型, 知识图谱, DPO, 生物安全, 偏好优化

## 一句话总结

提出KPO框架，通过构建蛋白质安全知识图谱(PSKG)并结合加权图剪枝策略识别"相似但安全"的蛋白质对，用DPO微调蛋白质语言模型使其远离有害序列空间，同时保持功能性。

## 研究背景与动机

**领域现状**：蛋白质语言模型（PLM）如ProtGPT2、ProGen2等在蛋白质序列生成上取得了巨大成功，能够进行功能优化和从头设计。这些模型通过学习大量蛋白质序列数据来捕获序列-结构-功能的隐含关系。

**现有痛点**：与文本LLM的安全问题（伦理/社会层面）不同，PLM的安全风险有直接的物理后果——可能无意中生成增强病毒传播性、逃避免疫应答、产生耐药性的有害蛋白质序列，这可能导致公共卫生危机甚至被利用制造生物武器。然而，当前PLM几乎完全专注于功能性和生成性能，对安全性重视不足。

**核心矛盾**：既有的蛋白质安全相关工作主要集中在对已知蛋白质序列引入安全性突变，但未解决生成阶段的更高风险——模型在生成全新序列时如何避免产生有害蛋白质？

**本文目标** 如何在微调PLM时引入安全性约束，使其在保持功能性的同时最小化生成有害蛋白质的概率？

**切入角度**：从NLP领域的LLM对齐（RLHF/DPO）获取灵感，关键创新在于用蛋白质安全知识图谱提供领域专家知识来构造偏好对。有害蛋白质和安全蛋白质之间通过共享的Gene Ontology功能注释存在关联，利用这种关联可以找到"功能相似但安全"的替代蛋白质。

**核心 idea**：构建蛋白质安全知识图谱(PSKG)编码有害/安全蛋白质的生化关系，通过图结构+嵌入相似性挖掘高质量偏好对，用DPO引导PLM远离有害序列空间。

## 方法详解

### 整体框架

KPO框架包含三个阶段：(1) 构建蛋白质安全知识图谱(PSKG)，编码有害蛋白质、安全蛋白质与Gene Ontology功能注释之间的关系；(2) 加权图剪枝，保留最有信息量的安全蛋白质节点以降低计算复杂度；(3) 在图结构和嵌入空间中识别与有害蛋白质最相似的安全蛋白质，构造偏好对用DPO微调PLM。

### 关键设计

1. **蛋白质安全知识图谱(PSKG)构建**:

    - 功能：构建编码有害蛋白质 $P_H$ 和安全蛋白质 $P_B$ 关系的知识图谱
    - 核心思路：从UniProt数据库搜集"toxin"和"antigen"关键词标注的有害蛋白质（~18,000条），从Swiss-Prot排除有害蛋白质后获取安全蛋白质。通过Gene Ontology (GO)功能注释建立间接关联——若有害蛋白质 $p_H^i$ 和安全蛋白质 $p_B^j$ 共享GO term $g_z$，则形成三元组 $(p_H^i, g_z, p_B^j)$
    - 设计动机：GO层次结构能捕获从粗粒度（如"binding activity"）到细粒度（如"DNA-binding transcription factor activity"）的功能关联，使PSKG不仅是注释集合而是编码了生物学专家知识的图结构

2. **加权指标图剪枝**:

    - 功能：将大规模PSKG裁剪为保留最关键节点的紧凑子图，显著降低计算开销
    - 核心思路：为每个安全蛋白质节点计算重要性得分 $S(p_B^j) = \alpha \cdot C_{GO}(p_B^j) + \beta \cdot C_{Deg}(p_B^j)$，其中 $C_{GO}$ 衡量与高分GO节点的连接数，$C_{Deg}$ 衡量度中心性。GO节点的评分也综合了桥接度 $R(g_z)$（连接多少有害-安全蛋白质对）和邻居广度 $O(g_z)$（连接多少安全蛋白质）
    - 设计动机：保留top-50% GO节点和top-50% 安全蛋白质节点。实验表明评分分布呈长尾分布，低分节点贡献边际信息极少，裁剪后计算时间减半而性能不损失

3. **基于图+嵌入的偏好对构造与DPO微调**:

    - 功能：从裁剪后的PSKG中找出与每个有害蛋白质最"相似但安全"的蛋白质，构造偏好对进行DPO微调
    - 核心思路：综合图结构距离和TransE嵌入余弦相似度寻找匹配：$s(p_H^i, p_B^j) = \mu \cdot \frac{1}{\text{dis}(p_H^i, p_B^j)} + (1-\mu) \cdot \cos(e_{p_H^i}, e_{p_B^j})$。对每个有害蛋白质选top-M安全蛋白质构造偏好对 $(p_B^j, p_H^i)$，用DPO损失微调：$L_{KPO} = -\log \sigma(\varphi \cdot [\log P_\theta(p_B^j|x) - \log P_\theta(p_H^i|x)])$
    - 设计动机：与随机配对（普通DPO）不同，PSKG引导的配对确保安全蛋白质和有害蛋白质在功能上相关，这样模型能学到"在功能相似空间中偏向安全方向"的细微区别

### 损失函数 / 训练策略

使用TRL库实现DPO训练，学习率5e-5，在8×A100 GPU上训练约2小时/epoch。有害蛋白质数据集按8:2分为训练/测试。DPO损失中 $\varphi$ 作为缩放因子控制优化强度。

## 实验关键数据

### 主实验

三个PLM基座模型上的安全性与功能性评估：

| 模型 | BLAST↓ | MMseq2↓ | ToxinPred3↓ | GB1↑ | GFP↑ |
|------|--------|---------|-------------|------|------|
| ProtGPT2 | 0.269 | 0.325 | 0.070 | 0.030 | 1.526 |
| ProtGPT2+KPO | **0.138** | **0.149** | **0.024** | **0.041** | **2.204** |
| ProGen2 | 0.155 | 0.170 | 0.029 | 0.144 | 1.683 |
| ProGen2+KPO | **0.128** | **0.117** | **0.007** | 0.024 | 1.562 |
| InstructProtein | 0.410 | 0.285 | 0.031 | 0.030 | 1.983 |
| InstructProtein+KPO | **0.086** | **0.079** | **0.003** | **0.191** | **2.319** |

### 消融实验

ProtGPT2上不同偏好对构造方法对比：

| 方法 | BLAST↓ | MMseq2↓ | ToxinPred3↓ | 说明 |
|------|--------|---------|-------------|------|
| DPO (随机配对) | ~0.18 | ~0.20 | ~0.04 | 不用PSKG |
| KPO-random | ~0.16 | ~0.17 | ~0.03 | 随机剪枝PSKG |
| KPO-community | ~0.15 | ~0.16 | ~0.03 | 社区检测剪枝 |
| **KPO (本文)** | **0.138** | **0.149** | **0.024** | 加权指标剪枝 |

### 关键发现

- **安全性大幅提升**：KPO在BLAST/MMseq2相似度上降低50-80%，ToxinPred3毒性预测降低66-90%
- **功能性未受损甚至提升**：GFP fitness提升44%（ProtGPT2），17%（InstructProtein）。原因是引导模型远离有害空间后，能更好地探索功能性更优的安全区域
- **嵌入空间分析**：t-SNE可视化显示KPO微调后的生成序列嵌入与有害蛋白质嵌入明显分离
- **3D结构验证**：ColabFold预测显示KPO生成的蛋白质与有害蛋白质的RMSD从~1.4Å升至~8.0Å，结构差异显著增大
- **图剪枝策略有效**：KPO优于KPO-random和KPO-community，证明加权指标剪枝保留了最有信息量的节点

## 亮点与洞察

- **首个蛋白质安全知识图谱(PSKG)**：将NLP的安全对齐思路迁移到蛋白质领域，概念上有创新性。通过GO注释建立有害/安全蛋白质间的间接关联，比简单的序列相似性配对提供了更丰富的生物学先验知识
- **安全与功能的双赢**：通常认为安全约束会牺牲性能，但KPO展示了引导模型远离有害空间反而有利于探索功能更优的区域。这个洞察可以迁移到其他领域的安全对齐研究中
- **从LLM到PLM的对齐迁移**：证明了DPO这类文本LLM对齐技术可以有效迁移到蛋白质序列生成，为跨模态的安全对齐提供了实践路径

## 局限与展望

- **仅关注序列级安全**：未直接约束3D结构层面的有害构象，依赖下游评估工具。可整合AlphaFold预测的结构信息作为额外奖励信号
- **有害蛋白质定义局限**：仅通过"toxin"和"antigen"关键词检索，可能遗漏其他类型的有害蛋白质（如朊病毒、过敏原）
- **计算开销**：大规模PSKG的构建和TransE嵌入训练仍需要较大计算资源
- **数据安全风险**：论文使用的有害蛋白质数据集本身存在安全隐患，作者已表示限制公开访问

## 相关工作与启发

- **vs 蛋白质突变安全方法**: 如Li et al. (2024)的unlearning方法专注于对已知蛋白质引入安全突变，而KPO直接在生成阶段施加安全约束，覆盖更广
- **vs 标准DPO**: 消融实验表明用PSKG引导的偏好对比随机配对效果显著更好，说明领域知识对生物安全对齐至关重要
- **启发**：这种"知识图谱引导偏好优化"的范式可以迁移到其他需要领域安全约束的生成任务，如药物分子生成、化学反应预测等

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地解决PLM生成阶段的安全问题，PSKG构建有创新
- 实验充分度: ⭐⭐⭐⭐ 3个PLM基座+多维度安全/功能评估+消融+超参敏感性+结构分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题严肃且解决方案完整
- 价值: ⭐⭐⭐⭐ 蛋白质AI安全是重要且尚未被充分研究的方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](../../NeurIPS2025/computational_biology/g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](../../AAAI2026/computational_biology/epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](concept_bottleneck_language_models_for_protein_design.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/computational_biology/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[ICML 2025\] Aligning Protein Conformation Ensemble Generation with Physical Feedback](../../ICML2025/computational_biology/aligning_protein_conformation_ensemble_generation_with_physical_feedback.md)

</div>

<!-- RELATED:END -->
