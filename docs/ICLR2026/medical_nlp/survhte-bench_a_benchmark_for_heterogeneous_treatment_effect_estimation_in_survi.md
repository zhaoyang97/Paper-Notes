---
title: >-
  [论文解读] SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis
description: >-
  [ICLR2026][医疗 LLM][heterogeneous treatment effects] 提出 SurvHTE-Bench，首个面向右删失生存数据的异质处理效应（HTE）估计综合基准，涵盖 40 个合成数据集、10 个半合成数据集和 2 个真实数据集，系统评估了 53 种估计方法在不同因果假设违反和删失水平下的表现，发现没有单一方法占主导地位，生存 meta-learner（特别是 S-Learner-Survival 和 Matching-Survival）在高删失和假设违反场景下表现最为稳健。
tags:
  - "ICLR2026"
  - "医疗 LLM"
  - "heterogeneous treatment effects"
  - "survival analysis"
  - "right-censored data"
  - "causal inference"
  - "benchmark"
  - "CATE"
  - "meta-learners"
  - "precision medicine"
---

# SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis

**会议**: ICLR2026  
**arXiv**: [2603.05483](https://arxiv.org/abs/2603.05483)  
**代码**: [GitHub](https://github.com/Shahriarnz14/SurvHTE-Bench)  
**领域**: 医学图像  
**关键词**: heterogeneous treatment effects, survival analysis, right-censored data, causal inference, benchmark, CATE, meta-learners, precision medicine

## 一句话总结

提出 SurvHTE-Bench，首个面向右删失生存数据的异质处理效应（HTE）估计综合基准，涵盖 40 个合成数据集、10 个半合成数据集和 2 个真实数据集，系统评估了 53 种估计方法在不同因果假设违反和删失水平下的表现，发现没有单一方法占主导地位，生存 meta-learner（特别是 S-Learner-Survival 和 Matching-Survival）在高删失和假设违反场景下表现最为稳健。

## 研究背景与动机

### 问题定义

异质处理效应（HTE）估计旨在量化同一治疗对不同个体的差异化疗效，在精准医疗和个性化政策制定中具有核心价值。在生存分析场景中，观测时间受右删失影响（即部分个体在研究结束前未发生目标事件），这使得 HTE 估计面临三重挑战：

**反事实不可观测**：每个个体只能观察到接受或未接受治疗时的一个结果

**混杂因素**：观察性研究中治疗分配受协变量影响

**删失机制**：删失可能与事件时间相关（信息性删失），违反标准假设

### 现有评估实践的缺陷

尽管近年来提出了众多生存 HTE 估计方法（因果生存森林、生存 meta-learner、结果插补方法 等），评估实践仍高度碎片化：

- 各研究使用自定义模拟数据，假设设定和删失水平各不相同
- 缺乏统一的已知真实值（ground truth）基准
- 不同方法之间无法进行公平比较
- 估计器在**同时多假设违反**下的鲁棒性未知

已有的因果推断基准（如 CausalBench）仅针对完全观测结果，而生存 ATE 基准不覆盖个体级异质效应。**生存 HTE 估计领域至今缺少标准化基准**——这正是本文的核心动机。

### 因果识别假设

估计条件平均处理效应（CATE）依赖五个关键假设：

- **(A1) 一致性**：观测结果等于潜在结果 $T_i = T_i(W_i)$
- **(A2) 可忽略性**：潜在结果独立于治疗分配（给定协变量）
- **(A3) 正性**：所有协变量水平下治疗概率有界远离 0 和 1
- **(A4) 可忽略删失**：删失时间独立于事件时间（给定协变量和治疗）
- **(A5) 删失正性**：删失概率不为 1

在实际应用中，这些假设经常被违反——未观测的预后因素破坏可忽略性、治疗指南破坏正性、与预后相关的退出导致信息性删失。SurvHTE-Bench 的核心目标就是衡量估计器在这些违反情况下的行为。

## 方法详解

### 整体框架

SurvHTE-Bench 想回答一个看似简单却一直没被系统验证的问题：当生存数据带右删失、又叠加各种因果假设违反时，到底哪类 HTE 估计器还靠得住。整个基准沿"数据 → 方法 → 指标"三层搭起来：先造一批**已知 CATE ground truth** 的数据集（从纯合成到半合成再到真实），覆盖从理想 RCT 到多重假设违反、从低删失到 >90% 删失的极端组合；再把现有 53 种估计方法统一收编进三大家族同台竞技；最后用统一的 CATE RMSE 和 ATE 偏差打分，跨数据集做 Borda Count 排名。这样任何一种方法的强弱都能落到"在什么删失水平、什么假设违反下"这个坐标系里，而不再是各说各话。

### 关键设计

**1. 三层数据梯度：从纯合成到真实，让 ground truth 的"可算性"和数据的"真实性"逐层换挡**

整个基准最难的事是"既要有 CATE 真值能算误差，又要尽量像真实临床数据"——这两件天生矛盾，于是用三层数据把它们逐层权衡过去。

第一层**纯合成数据**是唯一能拿到精确 CATE ground truth 的地方，因为两个潜在结果 $T_i(0)$ 和 $T_i(1)$ 都由已知机制生成。这里刻意把两条轴解耦正交：一条轴管**治疗分配与因果假设**，从随机化的 RCT-50 / RCT-5 一路退化到破坏可忽略性（UConf）、破坏正性（NoPos）、再叠加信息性删失（InfC），共 8 种配置；另一条轴管**事件时间分布与删失率**，用 Cox / AFT / Poisson 三种生存时间分布配上低（<30%）/ 中（30–70%）/ 高（>70%）三档删失，凑成 5 种场景。

| 配置 | 随机化 | 可忽略性 | 正性 | 可忽略删失 |
|------|:---:|:---:|:---:|:---:|
| RCT-50 | ✓ | ✓ | ✓ | ✓ |
| RCT-5 | ✓ | ✓ | ✓ | ✓ |
| OBS-CPS | ✗ | ✓ | ✓ | ✓ |
| OBS-UConf | ✗ | ✗ | ✓ | ✓ |
| OBS-NoPos | ✗ | ✓ | ✗ | ✓ |
| OBS-CPS-InfC | ✗ | ✓ | ✓ | ✗ |
| OBS-UConf-InfC | ✗ | ✗ | ✓ | ✗ |
| OBS-NoPos-InfC | ✗ | ✓ | ✗ | ✗ |

| 场景 | 生存时间分布 | 删失率 |
|------|:---:|:---:|
| A | Cox | 低（<30%） |
| B | AFT | 低（<30%） |
| C | Poisson | 中（30-70%） |
| D | AFT | 高（>70%） |
| E | Poisson | 高（>70%） |

$8 \times 5 = 40$ 个合成数据集，每个含 50,000 样本、5 维均匀分布协变量、二元治疗。正因为两轴解耦，后面才能干净地把"某方法掉链子"归因到底是被假设违反害的，还是被高删失害的。

第二层**半合成数据**嫌纯合成的协变量太"干净"，换上真实临床数据的协变量、只在其上模拟治疗分配和生存结局——既保住 ground truth 可算，又把真实数据里协变量依赖的治疗、非线性交互重新引进来。共 10 个：1 个 ACTG 半合成（基于 HIV 临床试验的 23 维协变量，删失 51%），9 个 MIMIC 半合成（基于 MIMIC-IV ICU 数据库的 36 维协变量，删失横跨 53%–88%）。

第三层**真实数据**彻底放手 ground truth、贴近实战。Twins（双胞胎出生数据，11,400 对，删失率 84.8%）靠同卵双胞胎天然提供近似反事实，难得带有可用 ground truth；ACTG 175（HIV 抗逆转录病毒治疗临床试验，2,139 名患者，基线删失仅 13.7%）则被人工加删失到 >90%，专门压测极端删失。三层叠起来，就能检验合成数据上的排名能不能迁移到真实场景。

**2. 53 种方法的统一分类：把零散估计器收编进三大家族，连此前未发表的自然扩展也补齐**

基准的另一半价值，在于第一次把零散的生存 HTE 方法摆到同一张分类表上同台竞技。家族一是**结果插补方法**（42 种变体，数量最多）：先用 Pseudo-obs / Margin / IPCW-T 把删失时间补成可用结果，再套标准 CATE 估计器（S-/T-/X-/DR-Learner 各配 Lasso / 随机森林 / XGBoost，外加 Double-ML 和 Causal Forest）。家族二是**直接生存因果方法**（2 种），不绕插补、直接在时间-事件结果上做因果推断，包括基于神经网络平衡表示学习的 SurvITE 和把广义随机森林扩到生存数据的 Causal Survival Forests。家族三是**生存 meta-learner**（9 种变体），把 meta-learner 的基学习器换成生存模型（S-/T-/Matching-learner 搭配 Random Survival Forests / DeepSurv / DeepHit），其中不少是本文补上的、此前未发表的自然扩展——正是它们在后面的高删失场景里成了赢家。

**3. 统一评估口径：以个体级 CATE RMSE 为主、群体级 ATE 偏差为辅，再用 Borda Count 跨数据集聚合**

有了数据和方法，还需要一把不被单个数据集带偏的尺子。主指标是 **CATE RMSE**，直接衡量个体级处理效应估得准不准；**ATE 偏差**作为辅助，看群体平均处理效应有没有系统性偏移；另有插补 MAE、回归/生存模型拟合度（C-index、AUC）等诊断量帮助定位误差来源。每个数据集上各方法按 CATE RMSE 排名，再跨全部数据集做 **Borda Count**（把各数据集的名次相加再排序）聚合，得到不被任何单一数据集主导的总排名——这也是后文"没有单一方法占主导"这一核心结论的统计基础。

## 实验结果

### 合成数据总体排名（Borda Count）

每个数据集上按 CATE RMSE 排名，跨 40 个数据集 × 10 次随机划分取平均：

| 排名 | 方法 | 平均排名 | 方法家族 |
|:---:|------|:---:|:---:|
| 1 | S-Learner-Survival (DeepSurv) | 5.17 | 生存 meta-learner |
| 2 | Matching-Survival (DeepSurv) | 5.42 | 生存 meta-learner |
| 3 | Double-ML + Margin | 6.65 | 结果插补 |
| — | Causal Survival Forests | 5.10* | 直接生存因果 |

*注：5.10 为方法家族级别（11 个家族中选最优变体后）的排名。

方法家族级别排名（每个数据集选该家族最优变体）：S-Learner-Survival（3.30）> Matching-Survival（3.48）> Double-ML（3.98）> Causal Survival Forests（5.10）。

### 假设违反对性能的影响

| 场景 | 最优方法趋势 | 关键发现 |
|------|------|------|
| RCT-50（理想条件） | 结果插补方法占优 | Double-ML（3.60）和 Causal Forest（5.60）与生存 meta-learner 持平 |
| RCT-5（治疗极不平衡） | Double-ML 领先 | T-Learner-Survival 跌至末位（9.00），因治疗组样本稀疏 |
| OBS-UConf（可忽略性违反） | 生存 meta-learner 稳定 | 生存 meta-learner 和 CSF 的 ATE 偏差一致，结果插补方法偏差增大 |
| OBS-NoPos（正性违反） | Double-ML/X-Learner 强势 | CSF 排名大幅下降，对确定性治疗分配区域敏感 |
| 多重违反 | 生存 meta-learner 重获优势 | 在正性 + 其他假设同时违反时，生存 meta-learner 最鲁棒 |
| InfC（信息性删失） | 生存方法持续领先 | 所有方法性能均下降，CATE RMSE 方差显著增大 |

### 删失率的影响

| 删失水平 | 最优家族 | 代表方法 |
|------|------|------|
| 低（场景 A, B） | 结果插补 | Double-ML 排名第一 |
| 中（场景 C） | 竞争均衡 | 各家族接近 |
| 高（场景 D, E） | 生存 meta-learner | S-Learner-Survival（1.6）、Matching-Survival（2.4），对其他方法形成压倒性优势 |

场景 D（高删失 + AFT 分布）下几乎所有估计器 ATE 偏差均大幅发散，表明高删失率下基于 RMST 的处理效应估计仍是极具挑战的任务。

### 半合成数据

MIMIC-ii–v 系列（删失 53%–88%）CATE RMSE 比较（10 次重复均值 ± 标准差）：

| 方法 | ACTG (51%) | MIMIC-v (53%) | MIMIC-ii (88%) |
|------|:---:|:---:|:---:|
| Double-ML | **10.651±0.24** | **7.891±0.05** | 7.954±0.05 |
| S-Learner-Survival | 11.713±0.24 | **7.897±0.04** | **7.921±0.04** |
| Matching-Survival | 12.523±0.29 | 7.912±0.04 | 7.949±0.04 |
| SurvITE | 12.714±0.56 | 7.906±0.07 | **7.931±0.05** |
| CSF | 11.674±0.17 | 7.893±0.04 | 7.963±0.06 |

关键发现：(1) 在中等维度的 ACTG 数据中 Double-ML 最优；(2) 在高删失 MIMIC 数据中生存方法（SurvITE 和 S-Learner-Survival）最稳定；(3) 各方法 RMSE 差距在真实协变量空间中被压缩。

### 真实数据

- **Twins 数据集**：S-Learner 和 DR-Learner（含插补）以及 S-Learner-Survival 表现最佳（RMSE ≈ 7.2 天），Double-ML 表现最差（与合成排名不一致，暗示数据特有模式）
- **ACTG 175 数据集**：在人工高删失条件下，CSF 估计最稳定，生存 meta-learner（T-/Matching-learner）表现出显著不稳定性

## 亮点与创新

- **首个生存 HTE 基准**：填补了右删失生存数据 HTE 评估的空白，建立了可复现、可扩展的标准化评估平台
- **系统化方法分类**：首次将 53 种方法统一到三大家族框架中，包含多个此前未发表的自然扩展变体
- **全面的假设违反分析**：不仅测试单一假设违反，还考察多重同时违反，揭示了方法鲁棒性的真实边界
- **实用选型指南**：为从业者提供了清晰的方法选择路线——低删失用 Double-ML，高删失用 S-Learner-Survival，多重违反用生存 meta-learner

## 局限性

- 假设违反为二元（有/无），未建模**渐进式违反严重度**（如 Rosenbaum Γ 敏感性分析）
- 仅考虑**静态二元治疗**和固定基线协变量，未覆盖时变治疗、工具变量和动态协变量
- 目标估量主要关注 RMST，虽附录补充了生存概率结果，但**中位生存时间、时变风险比等临床常用指标**未涉及
- 合成数据的协变量结构（5 维均匀分布）可能**不充分代表真实高维医疗数据**
- 部分真实数据集（MIMIC-IV）需凭证访问，影响可复现性

## 相关工作

- **非删失 HTE 基准**：Shimoni et al. (2018)、Crabbé et al. (2022)、CausalBench (2024) 等针对完全观测结果
- **生存 ATE 基准**：Voinot et al. (2025) 针对群体平均效应，不涉及个体异质效应
- **因果生存森林**：Cui et al. (2023) 扩展广义随机森林到生存数据，但评估范围有限
- **SurvITE**：Curth et al. (2021) 基于平衡表示的神经网络方法
- **生存 meta-learner**：Bo et al. (2024)、Noroozizadeh et al. (2025) 将 meta-learner 适配生存模型
- **结果插补**：Qi et al. (2023) 提出 IPCW-T 等删失时间替代策略
- **双重去偏机器学习**：Chernozhukov et al. (2018) Double-ML 框架

## 评分

| 维度 | 分数 |
|------|:---:|
| 新颖性 | ⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐ |
| 意义 | ⭐⭐⭐⭐ |
| 清晰度 | ⭐⭐⭐⭐ |

**总评**：⭐⭐⭐⭐ — 作为首个生存 HTE 综合基准，实验设计严谨（40 合成 + 10 半合成 + 2 真实，53 种方法），填补了重要空白。核心发现（无单一最优方法、删失率和假设违反决定方法选择）具有显著实践价值。但在假设违反渐进建模和估量多样性上仍有改进空间。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark](medarabench_large-scale_arabic_medical_question_answering_dataset_and_benchmark.md)
- [\[ICLR 2026\] HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)
- [\[ACL 2026\] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](../../ACL2026/medical_nlp/heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)
- [\[NeurIPS 2025\] HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](../../NeurIPS2025/medical_nlp/healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_nlp/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)

</div>

<!-- RELATED:END -->
