---
title: >-
  [论文解读] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks
description: >-
  [ICLR 2026][时间序列][ECG] 对8个ECG基础模型在12个数据集、26个临床任务上进行"现实检验"式全面基准评测，发现紧凑的结构化状态空间模型（SSM）ECG-CPC在7个任务类别中的5个上超越了大规模Transformer，证明架构设计比模型规模更重要。 领域现状：12导联心电图（ECG）是最广泛使用的心脏…
tags:
  - "ICLR 2026"
  - "时间序列"
  - "ECG"
  - "基础模型"
  - "结构化状态空间模型"
  - "心电图"
  - "基准评测"
---

# Benchmarking ECG FMs: A Reality Check Across Clinical Tasks

**会议**: ICLR 2026  
**arXiv**: [2509.25095](https://arxiv.org/abs/2509.25095)  
**代码**: [https://github.com/AI4HealthUOL/ecg-fm-benchmarking](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)  
**领域**: 时间序列  
**关键词**: ECG, 基础模型, 结构化状态空间模型, 心电图, 基准评测  

## 一句话总结
对8个ECG基础模型在12个数据集、26个临床任务上进行"现实检验"式全面基准评测，发现紧凑的结构化状态空间模型（SSM）ECG-CPC在7个任务类别中的5个上超越了大规模Transformer，证明架构设计比模型规模更重要。

## 研究背景与动机

**领域现状**：12导联心电图（ECG）是最广泛使用的心脏诊断工具。近年来多个ECG基础模型（FM）相继发布，包括基于CNN的ECGFounder、基于Transformer的ECG-JEPA/ST-MEM/HuBERT-ECG/ECG-FM，以及基于对比学习的MERL/ECGFM-KED等。这些模型使用了不同的预训练策略（监督、自监督、对比学习）和不同规模的数据集。

**现有痛点**：
   - 已有研究往往只在有限的数据集或单一任务类别上评估，无法得出可泛化的结论
   - 对比baseline常常选择较弱的模型，导致FM的优势被高估
   - 缺少对"模型规模 vs 架构选择"的系统性分析——更大的模型是否一定更好？

**核心矛盾**：FM领域默认"规模即质量"的假设是否在ECG领域成立？不同架构（CNN/Transformer/SSM）在不同临床任务上的泛化能力差异有多大？

**本文要解决的问题**：三个核心研究问题——(1) 哪种架构在多样化ECG任务上泛化最好？(2) FM如何随标注数据量缩放？(3) 是什么导致了不同模型间的性能差异？

**核心idea**：搭建覆盖7个任务类别的全面评测框架，并引入自训练的轻量级SSM模型ECG-CPC作为对照，揭示ECG FM的真实能力边界。

## 方法详解

### 整体框架
这是一篇评测论文，核心不是提出新模块，而是把8个预训练ECG基础模型连同2个从头训练的监督baseline，放到12个公开数据集、26个临床任务（含分类与回归）上做一次"现实检验"。每个模型在 fine-tuning、frozen、linear 三种评估模式下统一评测，再叠加标注效率缩放分析与表示相似性分析（CKA），从而回答"哪种架构泛化最好、FM如何随数据缩放、性能差异从何而来"三个问题。

### 关键设计

**1. 评测矩阵：8 个预训练 FM + 2 个从头训练 baseline × 7 类临床任务**

已有 ECG FM 研究往往只在有限数据集或单一任务上评估，又常拿较弱的模型当对比 baseline，于是 FM 的优势被系统性高估。为了让"架构 vs 规模"的结论站得住脚，作者在模型和任务两个维度上都把评测面铺满。模型侧覆盖三大架构家族与监督/自监督/对比三类预训练策略：CNN 系收 ECGFounder（RegNet，33.8M，监督）、MERL（ResNet18，4.6M，对比学习）、ECGFM-KED（ResNet，9.7M，对比学习）；Transformer 系收 ECG-JEPA（87.2M，JEPA）、ST-MEM（90.3M，MAE）、HuBERT-ECG（97.2M，MLM）、ECG-FM（93.9M，MLM+对比）；SSM 系是本文自训练的 ECG-CPC（S4 骨干，仅 3.8M），再加两个从头训练的监督锚点 Net1D（33.8M，CNN）与 S4（2.2M，SSM）。任务侧铺到 7 个类别——成人 ECG 解释（9 数据集 11 任务）、儿科 ECG 解释、心脏结构与功能（超声心动图指标回归）、心脏/非心脏出院诊断、急性护理预测（病情恶化/死亡率/ICU 入院）、患者特征预测（年龄/性别/生物与实验室指标），合计约 1650 个回归与分类标签。不同类别需要模型捕获 ECG 信号中不同层面的信息：诊断任务靠波形形态，急性护理预测靠更隐含的预后信号——只有把它们一起测，才能看出谁是真正的"通才"，也才能把"3.8M 的 ECG-CPC 能否赢过约 25 倍参数量的 Transformer"这个问题摆到台面上。

**2. 评估协议：三种模式 + 片段化推理 + bootstrap 检验**

每个模型跑三种模式。Fine-tuning 是全模型微调，并对 Transformer/SSM 启用层级学习率（backbone 比预测头低10–100倍）——这一步并非可有可无，HuBERT-ECG、ECG-FM 等模型不用层级学习率甚至无法收敛。Frozen 冻结 encoder，用可学习的 query-attention head 做池化；Linear 冻结 encoder 只接线性头，用来探测表示的线性可分性。输入侧统一用2.5秒片段训练、推理时对4个片段取平均，而非直接喂10秒完整录音，实测这样反而更稳。统计上用 bootstrap 置信区间（$n=1000$）做显著性检验，分类报 macro AUROC，回归报 z-normalized MAE。

**3. ECG-CPC：把"好归纳偏置胜过大参数"做成可验证的对照组**

ECG-CPC 是作者为了检验假设而专门训练的轻量模型：S4 结构化状态空间模型作骨干，用对比预测编码（CPC）做自监督预训练，在 HEEDB 数据集（1070万样本）上训练，仅3.8M参数，单块 NVIDIA L40 GPU 训练三周即可完成。它存在的意义就是当一个"反例"——如果 S4 自带的长程稳定记忆、光谱滤波和全局参数化卷积这些归纳偏置真的契合ECG信号结构，那么它应该能在不堆参数的情况下追平甚至超过90+M的Transformer。

**4. 标注效率缩放分析：把"学得快"和"学得高"拆开看**

为了量化 FM 相对监督训练到底省了多少标注，作者在 EchoNext 数据集上做受控缩放实验：训练集按2的幂次逐级缩减到最小1/128，对每条学习曲线拟合 $CN^{-\alpha} + L_0$，其中 $L_0$ 反映性能上限、$\alpha$ 反映收敛速度。再定义标注效率比 $r = N^*/N$，即 FM 达到监督 baseline 同等性能所需的数据比例，$r$ 越小越省标注。这个分解让"低数据量该选谁、高数据量该选谁"变成可读的曲线，而不是一个笼统的"FM更省数据"。

## 实验关键数据

### 主实验：Fine-tuning模式下跨7类任务的排名

| 任务类别 | 第1名 | 第2名 | 第3名 | S4 baseline |
|---------|------|------|------|------------|
| 成人ECG解释 | ECGFounder/ECG-JEPA/ECG-CPC | ECG-FM | MERL | 被超越 |
| 儿科ECG解释 | ECG-JEPA | ECGFounder | ST-MEM | 第6 |
| 心脏结构功能 | **ECG-CPC** | ECGFounder | ECG-JEPA | 第6 |
| 心脏/非心脏诊断 | **ECG-CPC** | ECG-FM | S4 | 第3 |
| 急性护理预测 | **ECG-CPC**/ECG-FM | ECGFounder | ECG-JEPA | 未被显著超越 |
| 患者特征 | **ECG-CPC** (5/6任务第1) | MERL/ECG-FM | - | 被超越3/6 |

ECG-CPC在7类中的5类排名第一，尽管参数量仅3.8M——不到最大Transformer的1/25。

### 标注效率实验

| 模型 | 标注效率比r (N=250-1000) | 含义 |
|------|------------------------|------|
| ECG-JEPA | 0.11-0.42 | 最高标注效率（低数据量时最优） |
| ECG-CPC | 0.21-0.40 | 接近JEPA，且性能上限更高 |
| ECGFounder | 0.30-0.62 | 标注效率较低 |
| 总体 | 3.3-9× | FM相比监督baseline的标注效率提升 |

关键发现：ECG-JEPA学习速度快但性能上限低（"快但矮"），ECG-CPC学习稍慢但上限更高（"慢但高"）——选择应依据数据量：<1000样本选ECG-JEPA，>1000选ECG-CPC。

### 表示相似性分析（CKA）
- ECG-CPC的表示演化最清晰、最结构化：早期CNN层冗余，后续S4层逐层特化
- ECGFounder的中层高度冗余（S0-S4层几乎相同），仅最终层特化
- ECG-JEPA的中间Transformer block几乎相同（Blk1-10），仅最终block差异化
- **性能相近的模型学到了截然不同的内部表示**，说明通往有效ECG表示的路径不止一条

### 关键发现
- SSM超越Transformer的核心原因：S4的归纳偏置（稳定长程记忆、光谱滤波、全局参数化卷积）天然匹配ECG信号的结构，无需大量参数即可高效学习
- 层级学习率对Transformer/SSM至关重要：某些模型（HuBERT-ECG, ECG-FM）不用层级学习率甚至无法训练
- 2.5秒裁剪+测试时平均优于直接使用完整10秒录音
- 没有一个模型在所有任务上一致最优，但ECG-CPC最接近这一目标

## 亮点与洞察
- **"架构 > 规模"的有力证据**：ECG-CPC以3.8M参数（约1/25的Transformer参数量）在大多数任务上超越90+M参数的Transformer模型。这挑战了FM领域"越大越好"的假设，表明对于ECG这类结构化时间序列，好的归纳偏置远比参数规模重要。
- **缩放曲线的"斜率 vs 上限"分析**：将FM的标注效率分解为"学习速度"和"性能上限"两个独立维度，提供了依据数据量选择模型的实用指南——这个分析框架可以推广到其他FM基准评测。
- **CKA揭示的"殊途同归"现象**：性能相近的模型内部表示差异巨大，说明当前仅基于任务性能的评估可能不足以全面评价FM的质量。
- **极低资源训练的可能性**：ECG-CPC仅用单GPU训练三周即达到顶级性能，为资源有限的医学AI实验室提供了实用路径。

## 局限与展望
- **仅限域内评测**：所有测试均为域内测试，缺少跨设备/跨人群的域外泛化评估（虽然作者承认标签不兼容是主要困难）
- **多任务训练的混淆**：部分任务使用多任务联合训练以节约计算，这可能使某些任务的性能高于或低于专门训练的模型
- **预训练数据不统一**：各FM在不同数据集上预训练，无法完全剥离预训练数据的影响；虽然理想的做法是在统一数据集上重训练所有模型，但计算成本太高
- **缺少单导联/可穿戴设备评估**：所有评估基于12导联标准ECG，而可穿戴设备通常只有单导联
- **改进方向**：结合token级和序列级预训练目标（如ECG-FM的做法）、统一预训练数据集的受控消融、扩展到域外泛化评估

## 相关工作与启发
- **vs ECGFounder (Li et al., 2025)**：ECGFounder用RegNet+监督预训练在成人ECG解释上表现优异，但在非诊断任务上不及ECG-CPC，可能因为监督标签集与下游任务的覆盖不足
- **vs ECG-JEPA (Kim, 2024)**：JEPA的联合嵌入预测架构提供了最佳标注效率（尤其小样本），但性能上限低于CPC方法的ECG-CPC
- **vs Mamba/现代SSM (Gu & Dao, 2024)**：作者内部实验表明，新型SSM如Mamba在连续医学信号上不一定优于S4，S4对ECG的归纳偏置更合适
- **启发**：SSM在医学时间序列上的潜力可能被低估，值得在其他生理信号（EEG、PPG）上进一步探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 评测框架全面系统，ECG-CPC虽非全新架构但其出色表现提供了重要洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 8个FM+2个baseline、12个数据集、26个任务、3种评估模式+缩放分析+CKA，极为充分
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、分析深入、结论明确且有实用指导价值
- 价值: ⭐⭐⭐⭐⭐ 对ECG FM社区有重要参考价值，"架构>规模"的发现对更广泛的医学AI社区也有启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICLR 2026\] TimeRecipe: A Time-Series Forecasting Recipe via Benchmarking Module Level Effectiveness](timerecipe_a_time-series_forecasting_recipe_via_benchmarking_module_level_effect.md)
- [\[ICLR 2026\] Can we generate portable representations for clinical time series data using LLMs?](can_we_generate_portable_representations_for_clinical_time_series_data_using_llm.md)
- [\[ICML 2025\] Foundation Models for Clinical Records at Health System Scale](../../ICML2025/time_series/foundation_models_for_clinical_records_at_health_system_scale.md)
- [\[ICML 2026\] Position: Current Benchmarking Hinders Real Progress in Deep Learning for Time Series](../../ICML2026/time_series/position_current_benchmarking_hinders_real_progress_in_deep_learning_for_time_se.md)

</div>

<!-- RELATED:END -->
