---
title: >-
  [论文解读] Beyond Hearing: Learning Task-Agnostic ExG Representations from Earphones via Physiology-Informed Tokenization
description: >-
  [ICLR 2026][自监督学习][ExG] 用耳机形态的轻量硬件采集 50 小时自由生活态 ExG 数据，并提出"生理学先验的多频带 tokenization (PiMT)"把信号拆成 12 个物理意义明确的子频带 token，配合重建式自监督预训练，学到一套跨视/听/味/触/嗅五感任务都能用的任务无关 ExG 表示。
tags:
  - "ICLR 2026"
  - "自监督学习"
  - "ExG"
  - "可穿戴传感"
  - "耳机脑电"
  - "多频带 tokenization"
  - "自监督预训练"
  - "任务无关表示"
---

# Beyond Hearing: Learning Task-Agnostic ExG Representations from Earphones via Physiology-Informed Tokenization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=s79tJrxDmt](https://openreview.net/forum?id=s79tJrxDmt)  
**代码**: 待确认  
**领域**: 自监督表示学习 / 生理信号基础模型  
**关键词**: ExG、可穿戴传感、耳机脑电、多频带 tokenization、自监督预训练、任务无关表示  

## 一句话总结
用耳机形态的轻量硬件采集 50 小时自由生活态 ExG 数据，并提出"生理学先验的多频带 tokenization (PiMT)"把信号拆成 12 个物理意义明确的子频带 token，配合重建式自监督预训练，学到一套跨视/听/味/触/嗅五感任务都能用的任务无关 ExG 表示。

## 研究背景与动机
- **领域现状**：ExG 信号（EEG/EMG/EOG/ECG）能反映神经、肌肉、眼动、心脏活动，是注视追踪、情绪识别、睡眠分期等一大批应用的基础。深度学习近年把单任务 ExG 分析做得很好，而基础模型（foundation model）范式则有望把"大规模数据→通用表示"的成功复制到日常 ExG 分析上。
- **现有痛点**：ExG 基础模型几乎没人做成，卡在两点。其一，**数据多样性不足**——绝大多数 ExG 数据是在实验室受控环境、用笨重昂贵设备（一套 EEG 头戴动辄 $10,000–50,000）采的，规模小、场景单一，自由生活态数据几乎是空白。其二，**模型设计高度任务专用**——不同任务绑死不同频带（注视追踪靠低频 0.1–15 Hz，情绪识别靠高频 8–30 Hz），于是处理管线和架构都按某个固定频带定制，注视追踪的模型没法直接迁移去做情绪识别。
- **核心矛盾**：想要通用，就得覆盖全频谱；但单一宽带滤波（如 0–100 Hz）会把各模态的生理特征糊在一起、丢失细粒度信息且难以适配具体任务——**"宽带覆盖广但特征糊"与"窄带特征清晰但不通用"之间存在根本张力**。此外不同硬件对各频段的增益/衰减不同，更不该预设"哪个频带/电极才重要"。
- **本文目标**：做一套**可规模化、任务无关、能在野外运行**的 ExG 监测方案，硬件、数据集、训练方法三者协同设计。
- **核心 idea**：**【硬件破数据瓶颈】** 把 ExG 传感塞进耳机形态（NeuroBuds），低成本、可长期佩戴，先把自由生活态数据采起来；**【物理先验破任务专用】** 不预设任务频带，而是按公认生理学知识把信号固定拆成 12 个子频带 token，让模型自己在全频谱里挑任务相关特征；**【重建式自监督】** 用无标注自由生活数据做多目标重建预训练，学到可迁移的通用表示。

## 方法详解

### 整体框架
PiMT 是一条"分频带 token 化 → 双向 Mamba 编码 → 重建预训练 → 下游微调"的流水线：原始多通道 ExG 信号先被一个固定的生理学滤波器组拆成 12 个子频带，每个频带切 patch 并经共享线性层投影成 token；这些按"频率→通道→时间"排好序的 token 喂进双向 Mamba 编码器得到上下文表示；预训练阶段用 6 个重建任务（含时域/频域、原始/掩码版本）联合优化编码器；下游再接任务头微调。

```mermaid
flowchart LR
    A[原始多通道 ExG 信号] --> B[ExG 滤波器组<br/>12 个生理频带]
    B --> C[Patch 切分 + 共享线性 tokenizer<br/>频率×通道×时间 三维 token]
    C --> D[双向 Mamba 编码器<br/>f→c→l 扫描顺序]
    D --> E[预训练: 6 个重建 decoder<br/>时域/频域 × 原始/掩码]
    D --> F[微调: 分类头 / 回归头]
```

### 关键设计
**1. 生理学先验的多频带 tokenization（PiMT）：把"该看哪个频带"从超参变成结构。** 不同于按任务定制窄带或硬塞一个宽带，PiMT 依据公认生理学知识预定义 12 个 canonical 子频带滤波器，覆盖跨模态的关键节律——EEG 的 delta(0.5–4 Hz)/theta(4–8 Hz)/alpha(8–13 Hz)/beta(13–30 Hz)/gamma(30–100 Hz)、EMG 的低/中/高频(15–45 / 45–95 / 95–100 Hz)、EOG 整体(0.1–20 Hz)、ECG 的低/高频(0.03–0.12 / 0.12–0.488 Hz)以及 QRS 复合波(8–50 Hz)。给定通道信号 $X_c \in \mathbb{R}^T$，并行套用全部 $N_F$ 个滤波器得到逐频带信号 $X_{f,c} \in \mathbb{R}^T$，每个只保留频带 $f$ 内的成分。这样模型拿到的不是一团糊在一起的宽带信号，而是 12 路"物理意义明确的同源视图"，从结构上保证了对全频谱的细粒度访问。

**2. 三维 patch token 化 + 频率优先的扫描顺序。** 每路频带信号 $X_{f,c}$ 被切成不重叠 patch $p_{f,c,l} \in \mathbb{R}^w$，于是每个 patch 同时被频率 $f$、通道 $c$、时间 $l$ 三个维度定位，形成结构化的 3D 表示；再经一个跨所有 token 共享的线性 tokenizer 投影成 $e_{f,c,l} \in \mathbb{R}^d$。把多频带显式拆开会让序列变长，因此编码器选**双向 Mamba**：它对序列建模是线性复杂度，而 Transformer 是平方复杂度，正好适配这条加长的序列。embedding 按经验最优的 $(f \times c \times l)$ 顺序（频率优先、通道其次、时间最后）展平送入编码器，输出上下文表示 $z$ 供下游任务头使用。

**3. 六任务重建式自监督预训练：从无标注自由生活数据榨取鲁棒表示。** 作者沿用"重建优于对比学习"的生理基础模型经验，设计 6 个各带独立轻量 MLP decoder、共享同一编码器的重建任务：(i) **自编码**——直接重建原始 patch，逼编码器抓时域特征并降噪；(ii) **掩码重建**——沿时间/通道/频率三个维度部分遮盖输入 $p_{\text{mask}}$ 再恢复，强迫上下文推断；(iii–iv) **频域幅度/相位重建**——对 $z$ 经 FFT 得到的幅度 $p_A$、相位 $p_P$ 分别重建；(v–vi) **掩码频域重建**——对掩码输入再做一遍幅度/相位重建，提升从残缺输入推断频谱的能力。各任务用 MAE 损失，按系数 $\lambda$ 加权求和成总重建损失 $\mathcal{L} = \sum_t \lambda_t \mathcal{L}_t$。微调时编码器当特征提取器：分类任务对 patch 输出做均值池化后接全连接 + 交叉熵，回归任务（注视追踪）用 patch 级线性 decoder 产出序列再聚合。

**4. NeuroBuds 硬件 + DailySense 数据集的协同设计。** 方法之所以成立，离不开一个能采到野外数据的硬件：NeuroBuds 是耳挂式 ExG 原型，把放大、数字化、板载存储、无线传输集成进一块 4.2 cm × 2.2 cm、20 g、$80 的轻量 PCB，耳周电极同时拿到近耳 EEG（T7–T10 等位点）、耳廓 EMG、外侧 EOG，覆盖认知/肌肉/眼动。基于它，作者采了 DailySense：22 名被试、50 小时无标注自由生活录音 + 20 小时跨五感（视/听/味/触/嗅，含注视追踪、视频/音频兴趣推断、纹理/味觉/气味分类共六个基准任务）的有标注数据。预处理刻意保持极简（50/60 Hz 陷波 + Butterworth 带通 + 重采样到 200 Hz + 归一化 + 4 秒窗），避免引入任务相关假设而破坏"任务无关"目标。硬件、数据、训练方法三者互为支撑，是这套方案能跑通的根本。

## 实验关键数据

### 主实验表格（DailySense，分类用 F1↑，注视追踪用角误差↓）

| 方法 | Video | Audio | Taste | Touch | Smell | 分类 Avg. | Gaze |
|---|---|---|---|---|---|---|---|
| SVM | 0.665 | 0.610 | 0.556 | 0.554 | 0.510 | 0.579 | 6.60° |
| EEGNet | 0.753 | 0.712 | 0.709 | 0.643 | 0.669 | 0.697 | 6.52° |
| DeepConvNet | 0.680 | 0.706 | 0.633 | 0.638 | 0.636 | 0.659 | 7.04° |
| TST | 0.773 | 0.705 | 0.731 | 0.669 | 0.667 | 0.709 | 6.54° |
| PatchTST | 0.771 | 0.749 | 0.731 | 0.686 | 0.681 | 0.724 | 6.47° |
| EEGConformer | 0.738 | 0.752 | 0.688 | 0.678 | 0.670 | 0.705 | 6.53° |
| Bidirectional-Mamba | 0.820 | 0.858 | 0.733 | 0.762 | 0.722 | 0.779 | 6.53° |
| **PiMT（无预训练）** | 0.858 | 0.885 | 0.790 | 0.807 | 0.753 | **0.819** | **6.11°** |
| PatchTST（带预训练） | 0.807 | 0.786 | 0.697 | 0.700 | 0.670 | 0.732 | 6.42° |
| **PiMT（带预训练）** | **0.964** | **0.961** | **0.801** | **0.860** | **0.793** | **0.876** | **6.00°** |

即使不预训练，PiMT 也以 81.9% 平均 F1 / 6.11° 注视误差超过所有 baseline（较最强 baseline +4% F1、−0.41° 误差）；加上自由生活数据预训练后平均 F1 从 81.9% 提到 87.6%，而 PatchTST 同样预训练只从 72.4% 微涨到 73.2%，说明 PiMT 与重建式预训练的组合才能真正吃到野外数据的红利。

### 消融实验表格（频带数量，DailySense）

| Tokenization 策略 | 分类 F1 趋势 | 注视误差 |
|---|---|---|
| 1-band (0.1–75 Hz) | 最低 | 最高 |
| 2-band | ↑ | ↓ |
| 4-band | ↑↑ | ↓↓ |
| **12-band（本文）** | **最高（较少频带 +4.6% F1）** | **最低** |

性能随频带数单调上升，12 频带相比更粗的切分平均涨 4.6% F1 且注视误差最低，验证"细粒度物理频带分解能挖出微弱但有生理意义的频谱线索"。

### 关键发现
- **公开基准也成立**：在 DREAMER(0.910)、SEED(0.820)、Sleep-EDF(0.822)、BCI Competition IV 2b(0.693) 四个公开数据集上，PiMT 全面超过 PatchTST 与 Bidirectional-Mamba，证明泛化不止于自采数据集。
- **显著性分析印证物理先验**：注视/视频任务在低频带强激活（贴合眼动），味/触/嗅/听觉兴趣任务在高频带强激活（贴合体感 beta–低 gamma 与近耳 EMG），且这种任务相关的频带聚焦是**无监督自动出现**的。
- **数据规模可扩展**：预训练数据越多测试损失越低；下游分类性能在约 30% 预训练数据处趋于饱和，呈现合理的边际递减。

## 亮点与洞察
- **把领域知识当结构而非超参**：与其让模型从零学"该看哪个频带"，不如直接按生理学拆成 12 个物理频带 token，既保留全频谱细粒度、又天然支持任务无关——这是一种很优雅的"先验注入"方式。
- **硬件—数据—算法三位一体**：论文的真正贡献不是单点算法，而是用一个 $80 的耳机原型把"野外 ExG 数据"这个长期空白补上，再用 PiMT + 自监督把数据红利兑现，三者缺一不可。
- **Mamba 用在刀刃上**：多频带显式展开会让序列变长，正好用 Mamba 的线性复杂度抵消，这个动机比"为新而新"地换 backbone 更扎实。
- **DailySense 填补"五感"基准空白**：首个能跨视/听/味/触/嗅做 ExG 分析的数据集，对社区有独立价值。

## 局限与展望
- **数据集尚未公开**：DailySense 还在走 IRB / 法务流程，能否公开或有条件共享未定，复现门槛高。
- **被试与时长规模仍偏小**：22 人、单任务最多 7 人参与，跨被试泛化、长尾人群（年龄/生理差异）覆盖有限。
- **12 频带是人工固定的**：虽有生理学依据，但频带边界仍是手工设定，是否对所有任务/硬件最优、能否做成可学习的自适应频带划分值得探索。
- **任务仍是二分类为主**：味/触/嗅都是 2 类（甜vs酸、粗vs滑、花vs酸），离真实复杂日常场景还有距离。
- **野外鲁棒性细节**：自由生活态下运动伪迹、佩戴漂移、电极接触变化对长期稳定性的影响，论文未充分压力测试。

## 相关工作与启发
- **三类 ExG 方法谱系**：传统深度框架（EEGNet、DeepConvNet 用时空卷积）→ Transformer 系（EEGConformer、PatchTST、Medformer 抓长程依赖）→ 自监督系（BrainBERT 把 BERT 式掩码用到颅内 EEG，BIOT 跨数据集，BrainWave 扩到大规模临床基础模型）。本文站在第三类肩上，但把战场从"实验室受控"挪到"自由生活态"。
- **对可穿戴/生理基础模型的启发**：当数据多样性是真正瓶颈时，"换更聪明的算法"不如"换能采到野外数据的硬件"；而把领域物理先验编码进 token 结构，是在数据有限时拿到强泛化的高性价比手段。
- **重建 vs 对比**：本文沿用并强化了"生理信号上重建优于对比学习"的经验，6 个时域+频域重建目标的组合可作为同类信号自监督设计的参考模板。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 生理学先验的 12 频带 tokenization + 耳机硬件 + 自由生活态数据集三者协同，是 ExG 基础模型方向少见的系统性原创，单点技术虽不算颠覆但组合很新。
- **实验充分度**: ⭐⭐⭐⭐ — 自采六任务 + 四个公开基准 + 频带数消融 + 显著性分析 + 数据规模曲线，覆盖全面；扣分在数据集未公开、规模偏小、任务多为二分类。
- **写作质量**: ⭐⭐⭐⭐ — 动机—矛盾—方法逻辑清晰，图1/图2把流水线与数据集讲得明白，物理频带与任务对应关系交代到位。
- **价值**: ⭐⭐⭐⭐ — 把 ExG 推向"野外、任务无关、可穿戴"，DailySense 与 NeuroBuds 对社区有长期价值，落地潜力大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis](carl_camera-agnostic_representation_learning_for_spectral_image_analysis.md)
- [\[CVPR 2025\] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning](../../CVPR2025/self_supervised/task-agnostic_guided_feature_expansion_for_class-incremental_learning.md)
- [\[ICLR 2026\] Spatially Informed Autoencoders for Interpretable Visual Representation Learning](spatially_informed_autoencoders_for_interpretable_visual_representation_learning.md)
- [\[ICLR 2026\] A Bayesian Nonparametric Framework for Learning Disentangled Representations](a_bayesian_nonparametric_framework_for_learning_disentangled_representations.md)
- [\[ICLR 2026\] Midway Network: Learning Representations for Recognition and Motion from Latent Dynamics](midway_network_learning_representations_for_recognition_and_motion_from_latent_d.md)

</div>

<!-- RELATED:END -->
