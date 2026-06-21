---
title: >-
  [论文解读] Beyond Classification: A Cough Regression Benchmark for Respiratory Acoustic Foundation Models
description: >-
  [ICML2026][音频/语音][咳嗽声学] 现有呼吸声学基础模型（FM）几乎只在咳嗽**分类**上被评过，本文首次系统地把它们放到**连续值回归**任务上（从咳嗽音频被动估计年龄、BMI、疾病概率），用统一的冻结编码器 + 三种回归头、5 个 FM × 6 个目标 × 3 个数据集的协议做了一个多模型多目标基准，揭示了"数据规模 × 头部容量"权衡、生成式预训练优势、以及强烈不对称的跨数据集迁移等一系列被分类评测掩盖的结论。
tags:
  - "ICML2026"
  - "音频/语音"
  - "咳嗽声学"
  - "呼吸基础模型"
  - "回归基准"
  - "线性探针"
  - "跨数据集泛化"
---

# Beyond Classification: A Cough Regression Benchmark for Respiratory Acoustic Foundation Models

**会议**: ICML2026  
**arXiv**: [2606.15436](https://arxiv.org/abs/2606.15436)  
**代码**: 待确认  
**领域**: 音频语音 / 呼吸声学 / 基础模型评测  
**关键词**: 咳嗽声学, 呼吸基础模型, 回归基准, 线性探针, 跨数据集泛化  

## 一句话总结
现有呼吸声学基础模型（FM）几乎只在咳嗽**分类**上被评过，本文首次系统地把它们放到**连续值回归**任务上（从咳嗽音频被动估计年龄、BMI、疾病概率），用统一的冻结编码器 + 三种回归头、5 个 FM × 6 个目标 × 3 个数据集的协议做了一个多模型多目标基准，揭示了"数据规模 × 头部容量"权衡、生成式预训练优势、以及强烈不对称的跨数据集迁移等一系列被分类评测掩盖的结论。

## 研究背景与动机

**领域现状**：咳嗽声音里其实编码了远超"有病/没病"标签的生理信息——频谱时序特征反映气道几何、呼吸肌力量、黏膜黏度，这些都与年龄、体成分、疾病严重度**定量协变**。在呼吸病负担最重的中低收入国家（LMIC），出生记录、体重秤、影像设备常常都没有，于是"用一段咳嗽被动估出年龄/BMI"就成了分诊的可行替代指标。与此同时，在大规模无标注音频上预训练的基础模型（OPERA、HeAR、M2D+Resp）能通过线性探针高效迁移，大幅降低临床音频 AI 的标注负担。

**现有痛点**：这些 FM 被密集地在**分类**上 benchmark 过，但**回归能力几乎无人刻画**。HeAR 原文虽评过咳嗽的年龄/BMI 回归，却只用单一模型 + 固定线性探针 + 按设备切分，没有多模型对比、没有非线性头、没有跨数据集泛化；OPERA 的回归任务（肺活量估计）只来自深呼吸和元音，不来自咳嗽；M2D+Resp 则从未在咳嗽回归上被评过。

**核心矛盾**：分类指标会**掩盖**回归任务里真正要紧的东西——一个把所有样本都预测成人群均值的模型，在某些分类设定下也能显得"有效"，但它对患者个体毫无预测力。要诚实地评回归，必须始终和"均值预测基线（MAD）"对比，只有 MAE < MAD 才说明学到了均值之上的信号。

**本文目标**：给出一个统一、可比、subject-disjoint 的咳嗽回归基准，回答四个被忽略的问题——哪种回归头在什么数据规模下最优？生成式 vs 对比式预训练谁更适合回归？跨数据集能不能迁移、方向是否对称？低数据时哪些 FM 更省标注？

**切入角度**：把所有 FM **冻结**当特征提取器，只换回归头（线性 / MLP-small / 全宽 MLP），把变量收敛到"预训练范式 × 头部容量 × 数据规模"三个维度，从而干净地拆出每个因素的贡献。

**核心 idea**：用"多模型 × 多目标 × 多数据集 + 始终对照 MAD 基线"的回归基准，把分类评测看不到的预训练范式差异、容量-规模权衡、迁移不对称性暴露出来。

## 方法详解

### 整体框架
基准的输入是三个咳嗽数据集的原始音频，统一重采样到 16 kHz 单声道、裁/补到 2 秒；交给 5 个**冻结**的呼吸/健康音频编码器各抽一次嵌入（之后所有头和评测复用同一份嵌入）；嵌入再喂给三种回归探针（线性 / MLP-small / 全宽 MLP），在三类评测机制下产出 MAE——数据集内回归（6 个目标）、回归头对比（90 个 model×task×head 组合）、跨数据集迁移（6 个年龄迁移方向）。所有结果都和均值预测基线 MAD 并列报告，并用 best MAE / MAD 量化"信号强度"。这是一个评测管线而非新模型，核心贡献在"协议设计 + 受控对比"，因此不画方法框架图。

### 关键设计

**1. 始终对照 MAD 的回归协议：把"看似有效"和"真有信号"分开**

回归任务最容易自欺的地方是：一个塌缩到人群均值的模型也能给出不错的 MAE。为此本文对每个标签分布都计算其**平均绝对偏差** MAD（即朴素均值预测基线的 MAE），所有模型 MAE 都和 MAD 并列，并定义信号强度比 $\text{best MAE}/\text{MAD}$——只有显著 $<0.90$ 才算"学到了高于随机的、患者个体级的信号"。这个设计直接戳破了一个假象：在 Zambia 临床队列 CIDRZ 上，四个目标的 best/MAD 都在 0.92–0.99 之间（年龄甚至落在基线 1% 内），说明 FM 嵌入在该临床队列**根本没有可用的个体级信号**；而 Coswara 年龄是唯一信号清晰的目标（best/MAD = 0.81）。临床分数目标（X 光异常、TB 概率）的模型间差距近乎为零（TB ≤ 0.004、X 光 ≤ 0.012 MAE），指向一个共享的表征天花板而非某个模型真的恢复了临床变化。

**2. 三档回归头 × 数据规模：拆出"容量-规模"权衡**

为厘清是预训练表征不行还是探针容量不匹配，本文统一对比三种头：**Linear**（$d_{\text{feat}}\to1$，标准线性探针）、**MLP-small**（$d_{\text{feat}}\to256\to1$，256 维瓶颈 + ReLU + 0.3 dropout，故意做成嵌入维度无关、可跨 FM 公平比较）、**全宽 MLP**（$d_{\text{feat}}\to d_{\text{feat}}\to1$，对 M2D+Resp 的 3840 维会产生约 1500 万隐藏参数）。结论是 MLP-small 在 30 个 model×task 里赢了 23 个，最多比线性探针好 0.38 yr（HeAR / Coswara）；而全宽 MLP 在小临床集 CIDRZ（$N_{\text{train}}=669$）上过拟合（M2D+Resp 退化 +0.53 yr，源于约 22000:1 的参数-样本比），却在更大的 CoughVID（$N_{\text{train}}=3050$）上恢复并让 Opera-GT 拿到最佳 9.53 yr。这就是一条可作部署指南的**数据规模 × 头部容量权衡**：小数据用 MLP-small，大数据才敢上全宽 MLP。所有头统一用 Adam（lr $=10^{-4}$、L2 $=10^{-5}$、batch 64）、每轮 LR 衰减 0.97、MSE 损失、按验证 MAE 早停（patience 10），5 seed 取均值±std。

**3. 受控的预训练范式对照：生成式 vs 对比式**

把 OPERA 家族里架构相近、只差预训练目标的三个编码器拉来对照——Opera-CT（对比式 Transformer）、Opera-CE（对比式 CNN）、Opera-GT（生成式 MAE），它们都在同样的 136K 呼吸片段上预训练。结果是 Opera-GT 在三个数据集的年龄回归上**全部优于** Opera-CT（CIDRZ 10.49 vs 10.52、Coswara 10.16 vs 10.25、CoughVID 9.62 vs 9.79 yr）。其中 CIDRZ 的 0.03 yr 差落在一个 seed std 内不算数，但 Coswara/CoughVID 的 0.09/0.17 yr 超过 seed 方差，所以趋势由两个更大的数据集驱动。这把 OPERA 原文在**呼吸音**上发现的"生成式预训练利于回归"延伸到了**咳嗽**。

**4. 跨数据集迁移不对称 + 低数据机制：揭示泛化方向与标注效率**

在六个年龄迁移方向上训一处测它处（不做适配），发现迁移**强烈不对称**：只有当 CIDRZ 作为目标时才成功——CoughVID→CIDRZ 甚至是负 gap（−0.17 yr）、Coswara→CIDRZ 基本持平（+0.03 yr），都由 Opera-CE 领跑，说明大规模网络采集数据能替代稀缺的临床训练数据；反向则失败——CIDRZ→Coswara 退化 +2.43 yr（+26.6%）、CIDRZ→CoughVID +0.94 yr，小临床人群泛化不到大众包人群。低数据机制上，HeAR 和 M2D+Resp 在 $N=50$ 就逼近满量性能（HeAR 距其 $N=669$ 仅差 0.02 yr），而 OPERA 模型在 $N=50$ 方差很大（std 达 ±0.22 yr）、要到 $N=400$ 才稳——指向**预训练语料的多样性决定低数据回归表现**（HeAR 313M 健康音频 / M2D+Resp 含 AudioSet，OPERA 仅 136K 呼吸片段）。

### 损失函数 / 训练策略
所有回归头一律用 MSE 损失 + Adam（lr $10^{-4}$、weight decay $10^{-5}$、batch 64）、每 epoch LR ×0.97、最多 64 epoch、按验证集 MAE 早停（patience 10）；CIDRZ/Coswara 用 64/16/20% subject-disjoint 划分，CoughVID 沿用官方 UUID 级划分（train 3050 / val 1019 / test 2789）；每个配置 5 seed 取均值±std。

## 实验关键数据

### 主实验

数据集内年龄回归 MAE（MLP-small，5 seed 均值，单位 yr；MAD 为均值预测基线；best/MAD 越低越好）：

| 任务 | MAD | Opera-CT | Opera-GT | HeAR | M2D+Resp | best/MAD |
|------|-----|----------|----------|------|----------|----------|
| CIDRZ 年龄 | 10.35 | 10.52 | 10.49 | 10.29† | 10.40 | 0.99 |
| Coswara 年龄 | 11.31 | 10.25 | 10.16 | **9.12** | 9.58 | 0.81 |
| CoughVID 年龄 | 10.29 | 9.79 | 9.62 | 9.61 | 9.79 | 0.93 |
| CIDRZ BMI (kg/m²) | 3.74 | 3.60 | 3.67 | 3.60† | 3.63 | 0.96 |
| CIDRZ TB (prob) | 0.205 | 0.189 | 0.190 | 0.188† | 0.192 | 0.92 |

> † HeAR 在 CIDRZ 上的结果可能受预训练污染（CIDRZ 或出现在 HeAR 预训练语料），因此被排除在 headline 主张之外。HeAR 在 Coswara 年龄上以 9.12 yr MAE 领跑（best/MAD = 0.81，是全表唯一有清晰信号的任务）。

### 跨数据集年龄迁移（MLP-small，每行取最佳模型；Gap = 跨域 − 同域 MAE）

| 训练 → 测试 | 最佳模型 | 跨域 | 同域 | Gap |
|------|------|------|------|------|
| CoughVID → CIDRZ | Opera-CE | 10.34 | 10.51 | **−0.17** |
| Coswara → CIDRZ | Opera-CE | 10.54 | 10.51 | +0.03 |
| Coswara → CoughVID | Opera-CT | 10.42 | 9.79 | +0.63 |
| CoughVID → Coswara | HeAR | 10.05 | 9.12 | +0.94 |
| CIDRZ → CoughVID | HeAR | 10.54 | 9.61 | +0.94 |
| CIDRZ → Coswara | HeAR | 11.55 | 9.12 | +2.43 |

### 关键发现
- **CIDRZ 临床队列触及随机底板**：四个目标 best/MAD 在 0.92–0.99，年龄落在基线 1% 内，FM 嵌入对该临床队列无可用个体级信号；临床分数目标模型间几乎无差，是共享表征天花板而非真信号。
- **容量-规模权衡可作部署指南**：MLP-small 赢 23/30；全宽 MLP 在 $N_{\text{train}}=669$ 小集上过拟合（约 22000:1 参数样本比），在 $N_{\text{train}}=3050$ 上恢复并刷新最佳。
- **迁移单向成立**：大众包/网络数据 → 小临床人群可行（甚至负 gap），反向必败（最差 +26.6%）；HeAR 只在三个退化方向"退得最少"，作者明确不把它读作迁移优势——唯二不退化的方向都由 Opera-CE 领跑。
- **预训练多样性决定低数据效率**：HeAR / M2D+Resp 在 $N=50$ 即近满量，OPERA 需 $N=400$。

## 亮点与洞察
- 最让人"啊哈"的是 **best/MAD 这个简单比值**直接把"看似有效"的回归打回原形——它揭示 CIDRZ 上所有模型其实都在均值附近打转，这种诚实的负面结论在只看 MAE 的评测里会被完全埋掉。
- 把"预训练范式/头部容量/数据规模"三个变量用冻结编码器干净拆开，是一个可复用的评测范式：换个临床声学任务也能照搬这套受控对比。
- 主动标注并排除 HeAR–CIDRZ 可能的预训练污染（用 † 标注、踢出 headline），这种"数据泄漏自查"的诚实做法值得迁移到所有基础模型 benchmark。
- "生成式预训练在回归上占优"从呼吸音延伸到咳嗽，提示做被动健康估计时优先考虑生成式（MAE）而非对比式预训练。

## 局限与展望
- **正面信号稀薄**：六个目标里只有 Coswara 年龄（best/MAD 0.81）有清晰信号，CIDRZ 全部触底，整体更像"FM 在咳嗽回归上还不行"的诚实负面基准而非性能突破。
- **临床分数 ≠ 临床终点**：X 光异常、TB 概率是连续派生分数而非二元放射/微生物学诊断，作者明确把它们当"分数复现"而非临床终点预测，可临床价值有限。
- **数据集规模偏小**：CIDRZ 仅 $N=1049$、subject-disjoint 后训练集只有数百，小样本让全宽 MLP 必然过拟合，部分结论受规模而非模型本质限制。
- **可能的预训练污染**：HeAR–CIDRZ 重叠虽被排除出 headline，但低数据机制里 HeAR 的"$N=50$ 即近满量"仍不能完全排除污染贡献，需要更干净的留出集复核。
- 迁移和范式对照都只在**年龄**目标上充分展开，BMI/疾病概率的迁移性几乎没碰，泛化结论的覆盖面有限。

## 相关工作与启发
- **vs HeAR 原文（Baur 2024）**：HeAR 评过咳嗽年龄/BMI 回归，但单模型 + 固定线性探针 + 按设备切分；本文扩成多模型、多头、跨数据集，并补上 MAD 对照与污染自查。
- **vs OPERA benchmark（Zhang 2024）**：OPERA 的回归任务（肺活量）只来自深呼吸/元音，本文专做咳嗽，并把其"生成式预训练利于回归"的发现从呼吸音验证到咳嗽。
- **vs M2D+Resp（Niizumi 2025）**：该模型此前从未在咳嗽回归上被评过，本文首次纳入并发现它在低数据机制下与 HeAR 一样高效（$N=50$ 近满量）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个多模型多目标咳嗽回归基准，问题切口实在，但属评测而非新方法。
- 实验充分度: ⭐⭐⭐⭐ 5 FM × 6 目标 × 3 数据集、三档头、跨域与低数据机制覆盖全面，5 seed 报方差。
- 写作质量: ⭐⭐⭐⭐ 结论诚实、对照清晰、主动标注污染；负面结论占多数但表述克制到位。
- 价值: ⭐⭐⭐⭐ 为 LMIC 被动健康估计与呼吸 FM 选型提供了可信的受控证据与部署指南。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [\[CVPR 2026\] BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models](../../CVPR2026/audio_speech/babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio.md)
- [\[ICML 2026\] Attend to Anything: Foundation Model for Unified Human Attention Modeling](attend_to_anything_foundation_model_for_unified_human_attention_modeling.md)
- [\[CVPR 2026\] TAPE: Task-Adaptive Prototype Evolution in Audio-Language Models for Fully Few-shot Class-incremental Audio Classification](../../CVPR2026/audio_speech/tape_task-adaptive_prototype_evolution_in_audio-language_models_for_fully_few-sh.md)
- [\[ACL 2026\] Beyond Transcripts: A Renewed Perspective on Audio Chaptering](../../ACL2026/audio_speech/beyond_transcripts_a_renewed_perspective_on_audio_chaptering.md)

</div>

<!-- RELATED:END -->
