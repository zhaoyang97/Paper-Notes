---
title: >-
  [论文解读] Extreme Weather Nowcasting via Local Precipitation Pattern Prediction
description: >-
  [ICLR 2026][时间序列][降水临近预报] 提出确定性临近预报框架 exPreCast，用局部时空注意力 + 立方双路上采样(CDU) + 时间提取器(TE) 在 SEVIR/MeteoNet 以及新构建的均衡 KMA 雷达数据集上以 1/30 的计算量逼近扩散集成模型的极端降水预报精度。 领域现状：随着气候变化…
tags:
  - "ICLR 2026"
  - "时间序列"
  - "降水临近预报"
  - "极端天气"
  - "Transformer"
  - "上采样"
  - "雷达数据集"
---

# Extreme Weather Nowcasting via Local Precipitation Pattern Prediction

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=fDknsQhSgm](https://openreview.net/forum?id=fDknsQhSgm)  
**代码**: [https://github.com/tony890048/exPreCast](https://github.com/tony890048/exPreCast)  
**领域**: 时空序列预测 / 雷达降水临近预报  
**关键词**: 降水临近预报, 极端天气, Video Swin Transformer, 上采样, 雷达数据集  

## 一句话总结
提出确定性临近预报框架 exPreCast，用局部时空注意力 + 立方双路上采样(CDU) + 时间提取器(TE) 在 SEVIR/MeteoNet 以及新构建的均衡 KMA 雷达数据集上以 1/30 的计算量逼近扩散集成模型的极端降水预报精度。

## 研究背景与动机
**领域现状**：随着气候变化，暴雨、台风等极端降水事件日益频繁，准确的降水临近预报（nowcasting）对防灾减灾至关重要。雷达观测提供高分辨率、实时的降水场，催生了大量数据驱动的临近预报模型，从 ConvLSTM、PhyDNet、SimVP 到 EarthFormer，再到近期基于扩散的生成式集成方法（CasCast 等）。

**现有痛点**：当前两类方法各有硬伤。扩散生成集成虽然能预测精细结构、刷新 SOTA，但推理成本极高（CasCast 在 SEVIR 上需要 4567 GFLOPs、近 400M 参数），无法满足实时业务需求；确定性模型计算高效，却普遍偏向"正常降水"，在小尺度高强度的极端降水上预测被平滑掉。此外常用上采样方式都不理想——线性插值会把高强度小区域平滑成噪声抹掉，pixel-shuffle 又会产生棋盘格伪影。

**核心矛盾**：极端降水恰恰是小区域、高强度、高频细节，但它既被"高效确定性模型"平滑掉，又难以用"高精度扩散模型"实时算出来——精度与效率难以兼得，而且评测基准本身就偏（SEVIR 全是风暴、MeteoNet 几乎都是普通雨），无法检验模型在全谱降水强度上的泛化。

**本文目标**：构建一个既高效又能保留极端降水细节、且能灵活调整预报时长的确定性框架，并提供一个正常与极端降水均衡分布的真实雷达基准。

**核心 idea**：以 Video Swin Transformer 的**局部移窗注意力**契合"降水由局地气象现象决定"的先验，配上**融合插值与像素重排的 CDU 上采样**保住高频极端信号，再用**时间提取器 TE** 把时间维度从预报时长中解耦，实现一次架构覆盖短期到长期预报。

## 方法详解

### 整体框架
exPreCast 是一个编码器-解码器的 3D Swin Transformer。编码器把雷达体数据切成不重叠的 3D patch，经多级 3D Swin 块 + Patch Merging 逐级下采样并提取局部时空特征；瓶颈层后，解码器镜像结构上采样，但把标准上采样换成自研的 CDU 块以保留高频纹理，跳连传递多尺度特征；最后 Patch Expanding 投影到目标分辨率，由 TE 块把时间维度调整到所需的预报提前量。

```mermaid
flowchart LR
    A[过去雷达序列] --> B[3D Patch 划分]
    B --> C[编码器: 3D Swin + Patch Merging<br/>局部移窗注意力下采样]
    C --> D[瓶颈: 2×3D Swin]
    D --> E[解码器: 3D Swin + CDU 块<br/>立方双路上采样]
    C -. 跳连 .-> E
    E --> F[Patch Expanding]
    F --> G[TE 时间提取器<br/>调整预报时长]
    G --> H[未来降水场预测]
```

### 关键设计

**1. 局部移窗时空注意力骨干：用 Video Swin 把"降水是局地现象"写进归纳偏置。** 短时降水由局部气象特征主导，作者因此用 Video Swin Transformer 替代全局注意力，把自注意力限制在移位窗口内，让特征学习局部模式而非全局关联，移窗机制在保持计算效率的同时引入有限的跨窗上下文。编码器-解码器加跳连的结构保证了多尺度特征流动，使小尺度强降水结构在下采样后仍可被恢复。

**2. CDU 立方双路上采样：双分支融合插值与像素重排，既去伪影又保高频。** 这是全文最关键的模块，专治极端降水被上采样抹掉的问题。CDU 并联两条分支：插值分支先用通道混合的 3D 卷积保持通道数，经 PReLU 激活与三线性插值上采样得到 $z_{ti}$；像素重排分支则先用 3D 卷积把通道扩张，激活后做 3D pixel-shuffle 上采样得到 $z_{ps}$。给定输入 $z_{in}\in\mathbb{R}^{b\times t\times h\times w\times c}$，两支均输出 $\mathbb{R}^{b\times t^*\times h^*\times w^*\times \frac{c}{2}}$，再拼接后用一层 3D 卷积融合：
$$z_{out}=\mathrm{Conv3D}(z_{ti}\oplus z_{ps})\in\mathbb{R}^{b\times t^*\times h^*\times w^*\times \frac{c}{2}}$$
其中 $(t^*,h^*,w^*)=(s_t t, s_h h, s_w w)$。三线性分支负责平滑连贯、抑制 pixel-shuffle 的棋盘格伪影，像素重排分支负责重建高频细节、避免插值带来的过度平滑混叠，两者互补，从而在小尺度高强度区域同时做到"细节不丢、伪影不生"。

**3. 时间提取器 TE：把时间维度与预报时长解耦，一套架构覆盖短/长期。** 临近预报的需求时长跨度大——即时预警要超短期，防灾准备要长时程。TE 接在解码器之后，用沿 $H,W,C$ 维滑动的时空 3D 卷积把解码器输出的时间维 $T$ 变换到目标时长 $T^*$：
$$Y=\mathrm{Conv3D}_{(T)}(Z_{decoder})\in\mathbb{R}^{B\times T^*\times H\times W\times C}$$
短期预报时 CDU 解码器在时间方向用较小放大因子、TE 提取最小有效特征；长期预报时 CDU 用较大时间放大因子让 transformer 学到更丰富的时序动态，TE 再压到目标帧数。由于长短期预报共享同一段历史输入，编码器可复用——先在短期任务上训练后冻结编码器，长期模型只微调解码器与 TE，构成高效的迁移学习训练范式，大幅降低开发长时程模型的成本。

## 实验关键数据

### 主实验表格
在三个分布迥异的数据集（SEVIR 偏极端、MeteoNet 偏正常、KMA 均衡）上以 CSI/HSS 评测，CSI 带池化（POOL4/16）更能反映局部模式保真度。

| 数据集 | 模型 | 参数(M) | FLOPs(G) | CSI-M(POOL16) | 极端阈值 CSI(POOL16) | HSS |
|--------|------|--------|---------|---------------|---------------------|-----|
| KMA | CasCast | 391.0 | 1,729 | 0.4837 | CSI-80: **0.1695** | 0.3806 |
| KMA | **exPreCast** | 32.0 | **55** | **0.4841** | CSI-80: 0.1488 | **0.4042** |
| SEVIR | CasCast | 392.9 | 4,567 | **0.5525** | CSI-219: **0.2841** | **0.5602** |
| SEVIR | **exPreCast** | 32.0 | **208** | 0.5427 | CSI-219: 0.2910 | 0.5430 |
| MeteoNet | EarthFormer | 15.1 | 309 | 0.2155 | CSI-47: 0.0472 | 0.3748 |
| MeteoNet | **exPreCast** | 32.0 | **199** | **0.4446** | CSI-47: **0.2525** | **0.4116** |

exPreCast 在 KMA 上 CSI-M 反超 CasCast 并拿下最高 HSS，却用约 1/30 的 FLOPs、约 1/10 的参数；在 SEVIR 极端阈值的 POOL16 上甚至超过 CasCast；MeteoNet 上则全面领先所有基线（CasCast 因结果极不稳定被排除）。

### 消融实验表格
KMA 1 小时预报上对比上采样策略（PS=像素重排, TI=三线性, CDU=本文）：

| 上采样 | CSI-M POOL16 | CSI-80 POOL16 | CSI-M(末帧) POOL16 | CSI-80(末帧) POOL16 |
|--------|-------------|---------------|--------------------|---------------------|
| PS | 0.4632 | 0.1379 | 0.3633 | 0.0771 |
| TI | 0.4740 | 0.1436 | 0.3884 | 0.1023 |
| **CDU** | **更优** | **更优** | **最稳健** | **最稳健** |

CDU 在带池化指标与末帧（长期）预报上均最优，单一 PS/TI 在长期预报上无法提供可靠性能。

### 关键发现
- **效率/精度权衡是核心卖点**：相比扩散集成 CasCast，exPreCast 用一个量级更低的算力换来几乎相同甚至更高的极端降水精度，且 HSS 更高，更适合实时业务。
- **CDU 是极端信号保真的关键**：双路融合显著缓解棋盘格伪影、抑制平滑，CSI 池化指标一致提升，尤其利好长期预报。
- **长期预报优势明显**：6 小时(36 帧)预报中，迁移学习版 exPreCast† 各项 CSI 全面领先，且是唯一能捕捉到强降水事件的模型。

## 亮点与洞察
- **把领域先验直接编码进架构**：移窗局部注意力对应"降水局地性"、CDU 对应"极端降水是小尺度高频"、TE 对应"预报时长可变"，三个模块各自回应一条物理/任务先验，设计动机清晰。
- **CDU 的"双路互补"思路通用性强**：插值保平滑、像素重排保高频，这种拼接融合可迁移到其他需要兼顾结构与细节的稠密预测/超分任务。
- **均衡数据集填补评测空白**：KMA(2014–2023, 10 分钟间隔) 借助韩国季风+台风的气候特性天然覆盖从正常到极端的全谱降水，比偏置的 SEVIR/MeteoNet 更能检验泛化。

## 局限与展望
- **仍是确定性模型**：单点预测无法表达降水的不确定性，缺少扩散集成的概率/集合预报能力，对风险决策的置信区间支持有限。
- **极端最高阈值仍略逊扩散**：KMA 的 CSI-80(POOL16) 上 CasCast 仍小幅领先，最极端尾部事件的精度还有差距。
- **依赖特定区域气候**：KMA 的"均衡性"来自韩国独特气候，迁移到其他气候带是否仍均衡、模型是否需重训未充分验证。
- **TE 的时长灵活性边界**：超长时程(远超 6 小时)预报的退化、以及 TE 卷积能学到的时序动态上限尚待进一步评估。

## 相关工作与启发
- **时空序列预测骨干**：ConvLSTM 把卷积塞进循环、FourCastNet/AFNO 走傅里叶算子、Video Swin 走移窗注意力——本文选择后者并加领域模块，体现"局部性优先"在降水任务上的合理性。
- **雷达临近预报**：EarthFormer(时空注意力)、NowcastNet、以及 CasCast/DiffCast 等扩散方法构成性能天花板，本文证明确定性模型经过针对性设计可在远低成本下逼近它们。
- **上采样研究**：从线性/pixel-shuffle 到本文受 dual upsample 启发的 CDU，提示稠密预测中上采样模块对高频保真的重要性常被低估。
- **启发**：对"精度-效率"二难的任务，与其堆生成式大模型，不如把领域先验拆成可插拔模块注入轻量确定性骨干，往往能拿到更实用的帕累托点。

## 评分
- 新颖性: ⭐⭐⭐⭐ CDU 双路上采样与 TE 时长解耦设计巧妙、契合领域先验，虽非颠覆性但组合创新扎实
- 实验充分度: ⭐⭐⭐⭐ 三数据集 + 短/长期 + 上采样消融 + 效率对比，覆盖全面；可再补不确定性/概率评测
- 写作质量: ⭐⭐⭐⭐ 结构清晰、动机与模块一一对应，个别表述有笔误但不影响理解
- 价值: ⭐⭐⭐⭐ 高效逼近扩散精度 + 均衡 KMA 数据集，对实时降水预报业务与社区基准都有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Probabilistic Precipitation Nowcasting with Rectified Flow Transformers](../../CVPR2026/time_series/probabilistic_precipitation_nowcasting_with_rectified_flow_transformers.md)
- [\[NeurIPS 2025\] Probability Calibration for Precipitation Nowcasting](../../NeurIPS2025/time_series/probability_calibration_for_precipitation_nowcasting.md)
- [\[ICLR 2026\] Improving Extreme Wind Prediction with Frequency-Informed Learning](improving_extreme_wind_prediction_with_frequency-informed_learning.md)
- [\[ICLR 2026\] Local Geometry Attention for Time Series Forecasting under Realistic Corruptions](local_geometry_attention_for_time_series_forecasting_under_realistic_corruptions.md)
- [\[ICLR 2026\] MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters](mixlinear_extreme_low_resource_multivariate_time_series_forecasting_with_01k_par.md)

</div>

<!-- RELATED:END -->
