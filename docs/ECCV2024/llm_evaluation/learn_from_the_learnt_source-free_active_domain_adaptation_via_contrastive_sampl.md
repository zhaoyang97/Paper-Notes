---
title: >-
  [论文解读] Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence
description: >-
  [ECCV 2024][LLM评测][无源域适应] 提出 LFTL（Learn from the Learnt）框架，通过对比主动采样（CAS）和视觉持久性引导适应（VPA）两个核心模块，在无源数据、极少量目标标注（≤5%）的条件下实现高效域适应，在 VisDA-C 上仅用 1% 标注即达到 87.4% 准确率。
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "无源域适应"
  - "主动学习"
  - "对比采样"
  - "视觉持久性"
  - "域迁移"
---

# Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence

**会议**: ECCV 2024  
**arXiv**: [2407.18899](https://arxiv.org/abs/2407.18899)  
**代码**: 有 ([https://github.com/lyumengyao/lftl](https://github.com/lyumengyao/lftl))  
**领域**: LLM评测  
**关键词**: 无源域适应, 主动学习, 对比采样, 视觉持久性, 域迁移

## 一句话总结

提出 LFTL（Learn from the Learnt）框架，通过对比主动采样（CAS）和视觉持久性引导适应（VPA）两个核心模块，在无源数据、极少量目标标注（≤5%）的条件下实现高效域适应，在 VisDA-C 上仅用 1% 标注即达到 87.4% 准确率。

## 研究背景与动机

### 域适应的现实困境

深度神经网络在训练数据与测试数据分布一致时表现优异，但面对域偏移（domain shift）时性能急剧下降。域适应（DA）旨在将源域知识迁移到目标域。然而，现有 DA 方法面临两个严峻的现实约束：

**源数据不可获取**：由于严格的数据保护法规和存储/计算资源限制，适应阶段往往无法访问源域数据。无源域无监督适应（SFUDA）方法虽然不需要源数据，但由于缺乏任何域的确定性监督信号，**适应的病态性**（ill-posedness）被加剧，性能存在天花板。

**少量标注可行但未被利用**：目标域样本在适应阶段是完全可获取的，投入少量标注即可带来巨大性能收益。因此，Source-Free Active DA（SFADA）是一个更实际的设定——不用源数据，同时用最小标注预算主动查询目标域标签。

### SFADA 的三重新挑战

将"无源数据"和"主动学习"结合带来独特挑战：

**查询选择难**：一般的主动学习准则（entropy、margin 等）在域偏移下失效；ADA 方法通常依赖源数据来识别差异性目标样本，在无源设定下不可用

**跨域对齐难**：没有源数据，无法通过最小化分布散度进行流形对齐,但标注成本提高了性能预期

**持续改进难**：迭代式查询-适应过程需要保证每轮都有性能提升

### 已有方法的不足

- **MHPL**：集成三种主动学习策略，但依赖源不相似度指标，**仅能做一轮采样**，无法保证持续改进
- **SALAD**：引入额外的 Guided Attention Transfer Network，计算开销大且标注利用效率低
- **SFUDA 方法（SHOT++等）**：不使用任何标注，VisDA-C 上需 21.6K 迭代（5.8h）达到 87.3%；而 LFTL 仅需 780 迭代（0.3h+1.83h标注）即达 87.4%

### 核心洞察："从已学中学"

源域知识封装在预训练模型 $\mathcal{M}_s$ 中；每轮主动学习后，前一轮模型 $\mathcal{M}_t^{(r-1)}$ 也封装了已学到的目标知识。**利用中间结果——前一轮的假设（hypothesis）和特征表示——是零额外开销的信息源**。

## 方法详解

### 整体框架

LFTL 交替执行两个阶段：
1. **对比主动采样（CAS）**：利用前一轮模型的假设，查询当前最具信息量的目标样本
2. **视觉持久性引导适应（VPA）**：利用前几轮模型提供的特征表示，引导目标域分布对齐

给定源预训练模型 $\mathcal{M}_s$ 和总标注预算 $B$，进行 $R$ 轮迭代，每轮查询 $b = B/R$ 个样本并适应模型。

### 关键设计

#### 1. **对比主动采样（Contrastive Active Sampling, CAS）**

**功能**：在每轮主动学习中，从未标注目标样本池中选择最具信息量的 $b$ 个样本请求标注。

**核心思路**：受对比解码（contrastive decoding）启发，比较当前模型 $\mathcal{M}_t^{(r)}$ 和前一轮模型 $\mathcal{M}_t^{(r-1)}$ 的预测分布差异，以此识别"仍然困难"的样本：

$$\tilde{\mathbf{p}}^{(r)}(\cdot|x_{tu}^i) = \begin{cases} \log \mathbf{p}^{(r)} & \text{if } r=0 \\ \log \mathbf{p}^{(r)} + \alpha(\log \mathbf{p}^{(r)} - \log \mathbf{p}^{(r-1)}) & \text{if } r>0 \end{cases}$$

- 当 $r=0$（首轮）：直接使用源模型预测
- 当 $r>0$：对比当前模型与前一轮模型的预测，**放大两轮间的差异**

在对比解码后的预测 $\tilde{\mathbf{p}}$ 上，采用 **Best-versus-Second-Best (BvSB)** 作为不确定性指标：

$$u_{cm}(x_{tu}^i) = \tilde{p}(y_a^i | x_{tu}^i) - \tilde{p}(y_b^i | x_{tu}^i)$$

$u_{cm}$ 越小 → 样本越不确定、越新颖 → 优先查询标注。

**类别平衡因子**：为避免某些"容易迁移"的类别被过度采样，引入类别可迁移性估计 $u_{ct}$：统计高置信样本中各类别的频率，频率高的类别表示迁移容易，权重被降低。最终混合准则：

$$u^i = u_{cm}^i + \lambda \cdot u_{ct}(y_a^i)$$

**设计动机**：(1) 利用前一轮模型作为"对照"是零成本的，无需额外计算；(2) 对比解码同时考虑了个体不确定性和时间维度的进展停滞性；(3) 类别平衡因子避免了域偏移下的类别采样偏差。

#### 2. **视觉持久性引导适应（Visual Persistence-guided Adaptation, VPA）**

**功能**：利用主动标注样本作为锚点，引导未标注样本的特征分布对齐。

**核心思路**：将主动标注的样本作为"代表锚点"，通过软相似度最小化鼓励未标注样本聚集到最近锚点周围：

$$\mathcal{L}_{ac} = -\mathbb{E}_{x_{tu} \sim \mathcal{T}_u} \mathbf{d}_{tu}^T \log(\mathbf{d}_{tu})$$

其中 $\mathbf{d}_{tu} = \delta[\mathcal{D}(f_e(x_{tu}), \mathbf{f}(\mathcal{T}_l))]$ 是未标注样本与所有锚点间的归一化余弦距离向量。

**持久性记忆库**：关键创新在于，锚点的特征表示不是仅用当前模型计算，而是通过指数移动平均（EMA）融合历史模型的判断：

$$\tilde{\mathbf{f}}(x_{tl}) \leftarrow \gamma \mathbf{f}(x_{tl}) + (1-\gamma) \tilde{\mathbf{f}}(x_{tl}), \quad \gamma = 0.9$$

这确保了源域知识和中间轮学到的域不变知识不会在迭代中遗忘。

**设计动机**：(1) 无源数据时无法直接做分布对齐，用标注锚点近似源分布结构；(2) EMA 持久性记忆库解决了迭代式适应中的"灾难性遗忘"——每一轮的域不变信息被有效保留；(3) 不使用伪标签，避免误差累积。

#### 3. **熵最小化正则**

引入标准的熵最小化损失促进判别性特征：

$$\mathcal{L}_{ent} = -\mathbb{E}_{x_{tu} \sim \mathcal{T}_u} \mathbf{p}(\cdot|x_{tu})^T \log[\mathbf{p}(\cdot|x_{tu})]$$

### 损失函数 / 训练策略

总损失由三部分组成：

$$\mathcal{L} = \mathcal{L}_{ce} + \beta_1 \mathcal{L}_{vpa} + \beta_2 \mathcal{L}_{ent}$$

- $\mathcal{L}_{ce}$：标注样本上的标准交叉熵
- $\mathcal{L}_{vpa}$：视觉持久性引导的未标注样本聚类损失
- $\mathcal{L}_{ent}$：未标注样本的熵最小化

**训练极其高效**：VisDA-C 上整个查询-适应过程仅需 780 次训练迭代（0.3h），加上估计的标注时间（1.83h），总时间远低于 SFUDA 方法 SHOT++ 的 5.8h。

## 实验关键数据

### 主实验：VisDA-C（ResNet101）

| 方法 | 设定 | 标注% | 平均准确率(%) |
|------|------|-------|-------------|
| SHOT++ | SFUDA | 0% | 87.3 |
| MHPL | SFADA | 5% | 85.9 |
| SALAD | SFADA | 5% | 86.1 |
| LADA | ADA（需源数据） | 5% | 87.5 |
| **LFTL（本文）** | **SFADA** | **1%** | **87.4** |
| **LFTL（本文）** | **SFADA** | **5%** | **89.0** |

LFTL 仅用 **1% 标注即超越 SHOT++**（0% 标注但需 27 倍训练迭代），5% 标注时超越需要源数据的 LADA。

### 消融实验：组件贡献（VisDA-C, 5% 标注）

| 配置 | 准确率(%) | 说明 |
|------|----------|------|
| 源模型（无适应） | 52.3 | 基线 |
| + 标准主动学习（Entropy） | 82.1 | 域偏移下一般主动学习准则效果有限 |
| + CAS（不含类别平衡） | 85.7 | 对比采样带来显著提升 |
| + CAS（含类别平衡） | 86.3 | 类别平衡因子进一步改善 |
| + CAS + VPA（无EMA） | 87.5 | 锚点聚类引导有效 |
| + CAS + VPA（含EMA持久记忆） | **89.0** | EMA 防止遗忘，全组件最优 |

### 关键发现

1. **效率惊人**：在 VisDA-C 上，LFTL 的整个查询+适应流程比 SHOT++ 快 **17 倍**（780 vs 21.6K 迭代），且无需源数据
2. **持续改进**：随着标注预算从 1% 增加到 5%，性能持续稳步提升（87.4→89.0），MHPL 等方法缺乏这种保证
3. **主动采样时间节省 25%**：相比 ADA SOTA（LADA），采样阶段速度提升 25%，适应速度提升约 17 倍
4. **通用性强**：在 Office-31、DomainNet、VisDA-C 三个不同规模基准上均达到 SFADA SOTA

## 亮点与洞察

1. **"对比解码"思想的迁移**：将 NLP 中对比解码应用到主动学习场景，用前后模型的假设差异来识别新颖样本，思路简洁优美
2. **零额外开销的关键思想**：CAS 和 VPA 利用的都是迭代过程中自然产生的中间结果（前一轮模型、前几轮特征），不引入任何额外计算
3. **EMA 持久记忆库的优雅设计**：用动量更新融合历史和当前的特征判断，一行公式解决了多轮迭代中的遗忘问题
4. **不使用伪标签**：在无源、少标注的极端条件下避免了伪标签带来的错误累积

## 局限与展望

1. **仅验证了闭集DA**：未探索开放集、部分集、通用DA等更复杂的语义偏移场景
2. **对比采样的超参敏感性**：$\alpha$, $\lambda$, $\kappa$ 等超参需要调优，不同数据集可能需要不同配置
3. **未探索检测/分割任务**：实验聚焦分类任务，域适应在目标检测和语义分割上的适用性未验证
4. **标注预算分配策略**：固定每轮标注 $b = B/R$ 个样本，未探索自适应预算分配

## 相关工作与启发

- **SHOT++ [NeurIPS 2021]**：SFUDA 强基线，本文证明少量标注可以大幅超越纯无标注方法
- **LADA [CVPR 2022]**：ADA SOTA，依赖源数据；LFTL 在不用源数据的条件下接近甚至超越其性能
- **对比解码 [ACL 2023]**：NLP 中的技术，启发了 CAS 的设计

## 评分

- **新颖性**: ⭐⭐⭐⭐ — SFADA设定虽非首创，但CAS和VPA的设计思路新颖自然
- **实验充分度**: ⭐⭐⭐⭐ — 三个基准全面验证，消融详尽，效率对比充分
- **写作质量**: ⭐⭐⭐⭐ — 问题动机层层递进，方法符号系统清晰
- **价值**: ⭐⭐⭐⭐ — 对实际部署场景有较强参考价值，效率优势突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ECCV 2024\] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)
- [\[ECCV 2024\] VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding](visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](../../ACL2026/llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](../../ICCV2025/llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)

</div>

<!-- RELATED:END -->
