---
title: >-
  [论文解读] Online Iterative Self-Alignment for Radiology Report Generation
description: >-
  [ACL 2025][医疗NLP][放射学报告生成] 提出在线迭代自对齐（OISA）方法：通过自生成→自评估→自对齐→自迭代的四阶段循环，利用多目标偏好优化（MODPO）让轻量级 RRG 模型在无需外部大模型或人工标注的条件下，持续提升放射学报告质量，在 MIMIC-CXR 和 IU-Xray 上达到 SOTA。
tags:
  - "ACL 2025"
  - "医疗NLP"
  - "放射学报告生成"
  - "在线迭代自对齐"
  - "多目标偏好优化"
  - "MODPO"
  - "DPO"
---

# Online Iterative Self-Alignment for Radiology Report Generation

**会议**: ACL 2025  
**arXiv**: [2505.11983](https://arxiv.org/abs/2505.11983)  
**代码**: 无  
**领域**: 医学NLP  
**关键词**: 放射学报告生成, 在线迭代自对齐, 多目标偏好优化, MODPO, DPO  

## 一句话总结

提出在线迭代自对齐（OISA）方法：通过自生成→自评估→自对齐→自迭代的四阶段循环，利用多目标偏好优化（MODPO）让轻量级 RRG 模型在无需外部大模型或人工标注的条件下，持续提升放射学报告质量，在 MIMIC-CXR 和 IU-Xray 上达到 SOTA。

## 研究背景与动机

**领域现状**：放射学报告生成（RRG）旨在自动为放射影像生成自由文本描述。现有方法主要通过 SFT 在影像-报告数据对上训练，近期研究开始用强化学习（RL）做后训练对齐，将模型输出与放射科医生偏好对齐。

**现有痛点**：(a) 高质量标注数据规模有限，SFT 模型容易过拟合、泛化性差；(b) 传统 RL 对齐（如 CMN+RL、MPO）仍受限于训练集的数据覆盖范围；(c) Hein et al. (2024) 的方案虽然性能好，但依赖 8B 大模型（CheXagent）生成偏好数据和 LLM 评分模型（GREEN），成本过高且只能做离线对齐。

**核心矛盾**：偏好对齐需要大量高质量偏好数据，但医疗领域的专家标注极其昂贵且不可扩展；依赖外部大模型又违背了轻量级部署的初衷。

**本文目标** 让轻量级 RRG 模型仅用自身生成的数据就能实现持续的多目标偏好对齐，摆脱对固定数据集和外部大模型的依赖。

**切入角度**：用 one-hot 权重向量作为条件，让模型针对不同临床目标生成多样化报告，再利用现有放射学评估指标自动构建多目标偏好数据集，通过 MODPO 优化并迭代。

**核心 idea**：四阶段自循环（生成-评估-对齐-迭代）+ 多目标偏好优化（MODPO），实现轻量级 RRG 模型的持续自我改进。

## 方法详解

### 整体框架

OISA 包含两大模块和四个步骤的迭代循环：

- **偏好数据构建模块（PDC）**：包含自生成（Self-Generation）和自评估（Self-Evaluation），负责自动构建多目标偏好数据集
- **多目标对齐模块（MOA）**：包含自对齐（Self-Alignment）和自迭代（Self-Iteration），负责用 MODPO 优化模型并启动下一轮迭代

整体流程为：$\pi_{\text{ref}}^{(i)} \xrightarrow{\text{PDC}} \mathcal{D}^{(i)} \xrightarrow{\text{MODPO}} \pi_{\theta_\mathbf{w}}^{(i)} \rightarrow \pi_{\text{ref}}^{(i+1)} \rightarrow \cdots$，每轮迭代都用更新后的模型生成更高质量的偏好数据。

### 关键设计1：条件式多目标自生成（Self-Generation）

- **核心机制**：引入 one-hot 权重向量 $\hat{\mathbf{w}}_k = [w_1, \ldots, w_N]$（其中 $w_k=1$）作为模型的条件输入，让 RRG 模型针对第 $k$ 个目标生成专门化的报告。通过切换权重向量，同一张影像可以生成偏向不同临床目标的多份报告
- **去重策略**：两级去重确保数据多样性——(1) 患者级别：同一患者不同视角的报告只保留 BERTScore 最高的一份（22.7 万→13 万）；(2) 疾病标签级别：用 CheXbert 提取 14 种疾病标签分组（共 579 组），组内丢弃 BERTScore<0.5 的报告，相似度>0.8 的报告对只保留质量更高的一份（13 万→9.8 万）
- **设计动机**：轻量级 SFT 模型本身难以为同一 prompt 生成多样化响应，通过条件式生成+去重解决多样性不足问题

### 关键设计2：分层采样自评估（Self-Evaluation）

- **构建流程**：对每个目标维度 $k$，用对应的评估指标 $M_k$（RadCliQ / RadGraphF1 / GREEN）对候选报告打分，然后通过分层采样构建偏好对——(1) 按疾病标签分组；(2) 每组中选评估得分最高的作为 chosen 响应 $y^w$；(3) 从同组剩余报告中随机选一份作为 rejected 响应 $y^l$
- **分层采样机制**：计算每组应采集的样本数 $K_c$，确保各疾病类别按比例均匀覆盖，最终构建 $K=10000$ 对的偏好数据集
- **结果**：对 N=3 个目标重复上述过程，得到多目标偏好数据集 $\mathcal{D} = [\mathcal{D}_{\text{RadCliQ}}, \mathcal{D}_{\text{RadGraphF1}}, \mathcal{D}_{\text{GREEN}}]$

### 关键设计3：基于 MODPO 的多目标对齐（Self-Alignment）

- **算法选择**：采用 Multi-Objective DPO（MODPO），以最小额外成本在 DPO 基础上实现多目标对齐
- **两步训练**：(1) 对每个偏好数据集 $\mathcal{D}_k$ 用标准 DPO 损失训练边际奖励模型 $\mathcal{R}_k$；(2) 在 MODPO 损失中加入边际奖励作为 margin 项，权重向量 $\mathbf{w}$ 作为 prompt 进行训练
- **权重采样**：训练时 $\mathbf{w}$ 的每个维度从 $\{0.2, 0.4, 0.6, 0.8, 1.0\}$ 中采样，产生分布均匀的 Pareto 前沿
- **权重融合**：权重向量 $\mathbf{w}$ 通过多头注意力机制融合到图像特征中（$\mathbf{w}$ 作为 query，图像特征作为 key 和 value）
- **自迭代**：对齐后令 $\pi_{\text{ref}} \leftarrow \pi_{\theta_\mathbf{w}}$ 启动新一轮循环，共迭代 3 轮

### 损失函数 / 训练策略

- 基线模型：PromptMRG（219.9M 参数），OISA 模型 230.1M 参数（多出约 10M 用于权重条件融合）
- 迭代 3 轮，每轮 60 个 epoch，每 epoch 约 15 分钟（NVIDIA 4090, 24GB）
- batch size=16，学习率 1e-5，Adam 优化器，$\beta=0.5$
- 推理用 beam search（beam width=3），MIMIC-CXR/IU-Xray 最大报告长度分别为 150/110

## 实验关键数据

### MIMIC-CXR 与现有方法对比（Table 3）

| 方法 | 参数量 | B1 | B4 | BERTScore | RadCliQ(↓) | RadGraphF1 | CheXbertF1 | GREEN |
|------|--------|------|------|-----------|------------|------------|------------|-------|
| R2Gen | 78.5M | 0.353 | 0.103 | 0.866 | 2.89 | 0.195 | 0.276 | 0.306 |
| CMN+RL | 60.8M | 0.381 | 0.109 | 0.871 | 2.83 | 0.214 | 0.292 | 0.315 |
| PromptMRG | 219.9M | 0.398 | 0.112 | 0.857 | 2.77 | 0.227 | 0.476 | 0.289 |
| MPO | 63.3M | 0.416 | 0.139 | 0.878 | 2.63 | 0.257 | 0.353 | 0.324 |
| **OISA (iter3)** | **230.1M** | **0.428** | **0.129** | **0.885** | **2.54** | **0.273** | **0.516** | **0.341** |
| MedVersa | 7B | 0.280 | 0.090 | 0.711 | 2.45 | 0.289 | 0.471 | 0.381 |
| CheXagent | 8B | 0.172 | 0.021 | 0.669 | 2.88 | 0.190 | 0.265 | 0.268 |

### IU-Xray 与现有方法对比（Table 4）

| 方法 | 参数量 | B1 | B4 | BERTScore | RadCliQ(↓) | RadGraphF1 | CheXbertF1 | GREEN |
|------|--------|------|------|-----------|------------|------------|------------|-------|
| PromptMRG | 219.9M | 0.401 | 0.098 | 0.871 | 2.60 | 0.274 | 0.211 | 0.457 |
| **OISA (iter3)** | **230.1M** | **0.431** | **0.131** | **0.889** | **2.51** | **0.308** | **0.232** | **0.527** |
| MedVersa | 7B | 0.247 | 0.047 | 0.884 | 2.71 | 0.209 | 0.217 | 0.516 |
| CheXagent | 8B | 0.191 | 0.036 | 0.876 | 2.81 | 0.184 | 0.097 | 0.407 |

### 迭代效果分析（MIMIC-CXR, 均权 w=1/3）

| 阶段 | RadCliQ(↓) | RadGraphF1 | GREEN | BERTScore |
|------|------------|------------|-------|-----------|
| SFT baseline | 2.77 | 0.227 | 0.289 | 0.857 |
| Iteration 1 | 2.65 | 0.244 | 0.323 | 0.865 |
| Iteration 2 | 2.63 | 0.251 | 0.325 | 0.874 |
| Iteration 3 | 2.61 | 0.254 | 0.327 | 0.879 |

### 关键发现

- 每轮迭代偏好数据质量持续提升：RadGraphF1 和 GREEN 的各分位数逐轮上升，RadCliQ 逐轮下降
- 当某个目标权重设为 1 时，对应指标达到最优；均权时各指标均接近次优，证明多目标对齐有效
- OISA（230M）在 NLG 指标上全面超过 7B 级别的 VLM 模型（MedVersa、CheXagent），在放射学指标上与 MedVersa 可比
- 推理速度 0.905s/报告，与基线 PromptMRG（0.874s）接近，远快于 MedVersa（5.11s）和 CheXagent（2.3s）

## 亮点与洞察

- **完全自主的改进闭环**：不依赖外部大模型或人工标注，利用成熟的放射学评估指标（RadCliQ、RadGraphF1、GREEN）作为偏好信号的代理，成本极低
- **多目标 Pareto 前沿**：通过权重条件实现了连续、平滑的 Pareto 前沿，用户可在推理时通过调整权重来控制报告风格（偏向临床准确性 vs 语言流畅度）
- **理论保障**：在线性奖励假设下证明了子最优性上界会随迭代收紧——每轮新生成的偏好数据更好地覆盖了目标策略的分布
- **极致的效率**：偏好学习每轮仅用 10K 对数据、每 epoch 仅 0.14 小时（对比 SFT 阶段 227K 数据、每 epoch 2.39 小时），3 轮迭代总训练成本约 25 GPU 小时

## 局限与展望

- 仅在 PromptMRG 一种基线模型上验证，未测试其他架构/规模的 RRG 模型
- 用现有评估指标代替真实放射科医生偏好，指标与临床实际需求的一致性有待验证
- 仅在胸部 X 光数据集上实验，CT/MRI 等其他模态未测试
- 迭代次数有限（3 轮），更多轮次是否会出现性能饱和或退化未充分探讨

## 相关工作与启发

- **vs MPO (Xiao et al., 2025)**：同一团队的前作，用 RL 优化多维度奖励但受限于固定训练数据；OISA 通过在线迭代生成扩展了数据覆盖范围
- **vs Hein et al. (2024)**：用 CheXagent (8B) 生成偏好数据 + GREEN 评分做离线 DPO；OISA 不需要大模型，用轻量模型自己生成自己评估
- **vs SPIN / Self-Play**：理念相通——通过与自身过去版本竞争来提升，但 OISA 增加了多目标维度和医疗领域的评估指标
- **vs Constitutional AI**：类似的自评估-自改进范式，但 OISA 用领域特定指标而非通用原则

## 评分

- 新颖性: ⭐⭐⭐⭐ 条件式多目标自生成 + MODPO 迭代优化的组合在 RRG 领域是新的
- 实验充分度: ⭐⭐⭐⭐ 两个数据集 + 7 种指标 + 多权重配置 + 3 轮迭代 + Pareto 前沿可视化
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，含理论分析
- 价值: ⭐⭐⭐⭐ 以极低成本实现了轻量级模型的持续改进，对医疗 AI 部署有实际意义
---
title: >-
  [论文解读] Online Iterative Self-Alignment for Radiology Report Generation
description: >-
  [ACL 2025][医学图像][放射学报告生成] 提出在线迭代自对齐（OISA）方法用于放射学报告生成——四阶段循环（自生成多样数据→自评估多目标偏好→自对齐多目标优化→自迭代进一步提升），无需额外人工标注即可迭代提升报告质量，在多个评估指标上达到 SOTA。
tags:
  - ACL 2025
  - 医学图像
  - 放射学报告生成
  - 自对齐
  - 迭代优化
  - 多目标偏好
  - RLHF
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)
- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)
- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_nlp/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ACL 2025\] VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare](vital_pluralistic_alignment_healthcare.md)

</div>

<!-- RELATED:END -->
