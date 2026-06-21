---
title: >-
  [论文解读] SuperF: Neural Implicit Fields for Multi-Image Super-Resolution
description: >-
  [ICLR 2026][图像恢复][多图超分辨率] SuperF 把多帧低分辨率图像当成"重建目标"而不是网络输入，用一个跨帧共享的坐标 MLP（隐式神经表示）在高分辨率连续网格上拟合场景，并同时优化每帧的仿射对齐参数，从而在**完全不需要高分辨率训练数据**的测试时优化（TTO）框架下实现卫星与手持相机 burst 的多图超分，放大倍率最高到 ×8。
tags:
  - "ICLR 2026"
  - "图像恢复"
  - "多图超分辨率"
  - "隐式神经表示"
  - "测试时优化"
  - "子像素对齐"
  - "神经场"
---

# SuperF: Neural Implicit Fields for Multi-Image Super-Resolution

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FiiItlSqqL](https://openreview.net/forum?id=FiiItlSqqL)  
**代码**: https://sjyhne.github.io/superf (项目页 + demo + 数据集)  
**领域**: 图像恢复 / 超分辨率  
**关键词**: 多图超分辨率, 隐式神经表示, 测试时优化, 子像素对齐, 神经场

## 一句话总结
SuperF 把多帧低分辨率图像当成"重建目标"而不是网络输入，用一个跨帧共享的坐标 MLP（隐式神经表示）在高分辨率连续网格上拟合场景，并同时优化每帧的仿射对齐参数，从而在**完全不需要高分辨率训练数据**的测试时优化（TTO）框架下实现卫星与手持相机 burst 的多图超分，放大倍率最高到 ×8。

## 研究背景与动机
**领域现状**：超分辨率分两条路。单图超分（SISR）本质是个欠定逆问题，必须靠强先验——要么从大量高分辨率数据里学，要么靠另一模态的高分辨率引导。多图超分（MISR）则换了个思路：同一场景拍多张带**亚像素位移**的低分辨率帧，每帧因为采样栅格不同会产生不同的混叠（aliasing）伪影，这些"看似噪声"的差异其实携带了互补的高频信息，把它们融合就能重建出共享的高分辨率图。

**现有痛点**：SISR 靠学到的先验容易"幻觉"出现实中不存在的结构——这对手机拍照尚可接受，但对医学、遥感等科学应用是致命的。MISR 这边，有监督方法需要成对的 LR-HR 训练集，而高分辨率数据采集昂贵、配对又非平凡；TTO 方法（如 Wronski 的可调向核回归）不需要训练数据，但把 LR 帧当作模型输入直接回归出 HR 图，表达受限。已有的把隐式神经表示（INR）用于 burst 融合的工作（如 Nam 的 NIR，原本为图层分离设计）虽然思路相近，但**没有为 MISR 精确求解子像素对齐而设计**，作者证明这一点恰恰是 MISR 的关键。

**核心矛盾**：MISR 的成败系于能否把多帧的亚像素位移**精确对齐**到同一连续坐标系——位移估不准，多帧信息不仅帮不上忙，反而会互相模糊（实验里"多帧但不对齐"的 PSNR 比单帧还低）。而把 LR 帧当输入的传统范式，天然难以在连续坐标空间里精修这种小于一个像素的位移。

**本文目标**：在不依赖任何高分辨率训练数据的 TTO 框架下，(1) 用一个连续场表示底层 HR 信号；(2) 把每帧的子像素对齐做到足够精确；(3) 对真实数据里的云、噪声等异常像素保持鲁棒。

**切入角度**：借鉴 De Lutio 的引导超分思路，把问题**反转**——不把 LR 帧喂给模型，而是把它们当作监督**目标**；模型本身是一个定义在连续高分辨率坐标上的 INR。INR 的连续性恰好能同时服务于两件事：在像素坐标空间里做亚像素对齐，以及表示底层 HR 信号。

**核心 idea**：用一个跨帧共享的坐标 MLP 表示 HR 场，并把每帧的仿射变换参数直接作为可优化变量，**联合优化"对齐 + 表示"**——简单，但正是这一联合优化让 INR 真正适配了 MISR 任务。

## 方法详解

### 整体框架
SuperF 的核心是一个共享的隐式神经表示 $f_\theta$（坐标 MLP）：它输入高分辨率网格上的连续坐标 $v\in[0,1)^d$，输出该位置的 RGB 强度（以及可选的不确定性）。要让这同一个场服务于 $T$ 帧带位移的低分辨率图，关键是先把每帧对齐到同一参考系。

整条管线这样转：取**输出分辨率**对应的高分辨率坐标网格 $v$；对第 $t$ 帧，用该帧的仿射变换 $\hat A^{(t)}$ 把坐标变换到该帧视角（基准帧固定 $\hat A^{(1)}=I$）；经 Fourier 位置编码后送入共享 MLP，再过一个逐帧的谱投影 $\hat\rho^{(t)}$（校正亮度/对比度），得到该帧视角下的 HR 估计 $\hat y^{(t)}_\theta(v)=\hat\rho^{(t)}(f_\theta(\hat A^{(t)}v))$；由于监督信号只有 LR 帧，再用一个 boxcar 滤波（即平均池化）把 HR 输出下采样到 LR 分辨率 $\hat y^{(t)}_{LR,\theta}$，和真实 LR 帧 $y^{(t)}_{LR}$ 比对算 loss。训练就是对 $\theta$ 和所有 $\hat A^{(t)}$ 联合做梯度下降，逐步把对齐和共享表示一起优化好。推理时直接在 HR 网格上 query $f_\theta$ 即得超分结果。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["HR 坐标网格 v"] --> B["仿射对齐<br/>逐帧 Â(t)·v<br/>基准帧固定为 I"]
    B --> C["Fourier 位置编码<br/>对抗谱偏置"]
    C --> D["共享 INR + 反转问题<br/>坐标 MLP fθ + 谱投影 ρ(t)<br/>输出 HR RGB"]
    D --> E["超采样优化<br/>avg pool 下采样到 LR"]
    E -->|与 LR 帧比对| F["GNLL 不确定性损失<br/>逐帧方差·忽略噪声像素"]
    F -->|反传联合更新 θ 与 Â(t)| B
    D -.推理时直接 query.-> G["HR RGB 输出"]
```

### 关键设计

**1. 共享 INR + 反转问题：把多帧 LR 当重建目标而非输入**

传统 TTO-MISR 把一串 LR 帧作为模型**输入**直接回归 HR 图，表达能力受栅格离散性限制。SuperF 反转了问题定义：模型是一个定义在连续 HR 坐标上的坐标 MLP $f_\theta$，所有 $T$ 帧**共享**同一个 $f_\theta$，而 LR 帧退居为监督**目标**。形式化地，作者假设成像过程满足 $y^{(t)}_{LR}(v)\approx \phi * y_{HR}(A^{(t)}v)$，即 LR 帧是 HR 信号经仿射变换 $A^{(t)}$、再被 boxcar 滤波 $\phi$（空间平均池化）下采样得到。于是优化目标是逐帧重建损失在 $T$ 帧上的平均：

$$\arg\min L(\theta,\hat A^{(1)},\dots,\hat A^{(T)})=\frac{1}{T}\sum_{t=1}^{T}\sum_{v\in W}\ell\big(\hat y^{(t)}_{LR,\theta}(v),\,y^{(t)}_{LR}(v)\big)$$

由于 $f_\theta$ 把所有帧的信息都灌进同一组权重，它被迫去拟合一个能同时解释所有帧的**底层连续 HR 信号**——这正是多帧互补信息被融合的地方。此外每帧还配一个谱投影 $\rho^{(t)}$（每个光谱波段一个缩放+平移参数），吸收帧间亮度/对比度差异，基准帧的 $\rho^{(1)}$ 固定为缩放 1、平移 0。

**2. 直接参数化仿射变换：让子像素对齐精确可优**

这是 SuperF 相对最近邻 baseline（NIR）最关键的改进。NIR 用**另一个 ReLU MLP** $g(t)$ 以帧索引为条件去估计变换矩阵 $T_t=g(t)$，这种间接参数化对亚像素级别的精修并不友好。SuperF 直接把每帧的变换矩阵 $\hat A^{(t)}$ 当作模型参数的一部分，仅用三个标量显式刻画：两个平移 $\Delta x^{(t)},\Delta y^{(t)}$ 和一个旋转角 $\alpha^{(t)}$，并随 $\theta$ 一起做梯度下降。同时沿用 MISR 惯例**固定基准帧** $\hat A^{(1)}=I$，让所有其他帧相对它对齐——这既减少了自由度，又因为基准帧与 HR 参考天然对齐，便于用 HR 数据做评测而不引入整体错位。消融（Table 3）显示这一"直接参数化"是三个组件里最关键的：单加它就把对齐误差从 0.650 砍到 0.012、PSNR 从 24.63 升到 26.14。

**3. 超采样优化：在 HR 网格上算、再池化下采样去监督**

坐标 MLP 有谱偏置（spectral bias）——优先拟合低频、高频细节收敛极慢；而 SuperF 只有 LR"视图"作监督，更容易让 MLP 只输出低频。为此作者采用超采样策略：优化时让 INR 直接在**超分输出对应的高分辨率网格** $v$ 上运行，得到 HR 估计后，再用平均池化下采样到 LR 分辨率去和 LR 目标比对（即式 2 中的 $\hat y^{(t)}_{LR,\theta}(v)=\phi*\hat\rho^{(t)}(f_\theta(\hat A^{(t)}v))$）。相比直接在 LR 网格上算，这让梯度回传携带了 HR 网格的细粒度位置信息，显著改善亚像素对齐与高频恢复。消融里它是仅次于直接参数化的第二关键组件，且与直接参数化叠加后 PSNR 跳到 31.30。这里还需配合 Fourier features 位置编码：$\gamma(v)=[\cos(2\pi b_1^Tv),\dots,\sin(2\pi b_m^Tv)]^T$，其中 $b_i\sim N(0,\sigma^2 I)$，尺度 $\sigma$ 控制采样频率范围、是对抗谱偏置的关键超参（卫星域取 10、地面域取 3）。

**4. GNLL 异方差不确定性：让模型主动"忽略"云和噪声像素**

真实卫星时序里某些帧会被云遮挡，强行最小化这些像素的重建误差反而污染表示。SuperF 让 MLP 解码器**额外为每帧、每波段输出一个不确定性估计** $\hat s^{(t)}_{LR}(v)$（最终层输出数从 $n_c$ 增到 $(T+1)\times n_c$），把它解释为以预测 HR 信号为均值的高斯分布的对数方差，并用高斯负对数似然（GNLL）替代 MSE：

$$L_{GNLL}=\frac{1}{2}\sum_{t=1}^{T}\Big(\hat s^{(t)}_{LR}(v)+\frac{\big(\hat y^{(t)}_{LR,\theta}(v)-y^{(t)}_{LR}(v)\big)^2}{\exp(\hat s^{(t)}_{LR}(v))}\Big)$$

对那些模型怎么也拟合不好的位置（如云），GNLL 可以通过**调高方差**来降低损失，而不必硬去最小化重建误差；第一项 $\hat s$ 充当正则防止方差被无脑放大。MSE 是 GNLL 在常方差下的特例。实验显示在含云的 WorldStrat-bitter 上 GNLL 明显优于 MSE，在干净时序上两者持平。

### 损失函数 / 训练策略
逐帧重建损失在 $T$ 帧上取平均（式 3），其中 $\ell$ 可取 MSE 或 GNLL。优化器 AdamW，基础学习率 $2\times10^{-3}$，用余弦退火在 2000 次迭代内衰减到 $1\times10^{-6}$，batch size = 1 帧，ReLU MLP，单张 H100 训练。评测时裁掉四周 16 像素边界、并按 Bhat 的做法做颜色匹配后处理校正全局色偏。

## 实验关键数据

### 主实验
两个域的数据集：自建的 **SatSynthBurst**（从 WorldStrat 的 20 张高分卫星图合成，每张生成 16 帧 LR，含 ×2/×4/×8、亚像素位移、MTF 模拟与高斯噪声）和 **SyntheticBurst**（Bhat 提供的手持地面 burst，选 50 个有结构的场景，每个 14 帧）。指标 PSNR / SSIM / LPIPS。

PSNR 对比（数值越高越好，括号内为样本标准差，方括号为迭代数）：

| 方法 | SatSynth ×2 | ×4 | ×8 | Synth ×2 | ×4 | ×8 |
|------|------|------|------|------|------|------|
| Bilinear | 34.69 | 29.71 | 26.62 | 27.66 | 26.12 | 25.44 |
| Lafenetre 2023 (核回归 TTO) | 33.46 | 27.70 | 24.88 | 27.02 | 26.46 | 25.19 |
| NIR (Nam 2022) [5k] | 25.65 | 24.99 | 23.61 | 24.46 | 23.39 | 22.93 |
| **SuperF MSE (ours)** [2k] | 36.73 | 32.94 | 28.87 | 29.38 | 27.90 | 27.08 |
| **SuperF GNLL (ours)** [2k] | **37.26** | **34.03** | **29.28** | **29.48** | 27.47 | 26.58 |

两个已有方法甚至打不过 bilinear（PSNR 维度），而 SuperF 用 MSE 和 GNLL 都全面超越。在卫星域 GNLL 稳定优于 MSE（光谱变化更明显时更鲁棒），在地面域两者持平。SuperF 仅用 2k 迭代就超过 NIR 跑满 5k 的结果。

### 消融实验
逐步打开各组件（×4，16 帧，SatSynthBurst / SyntheticBurst）：

| FF | 多帧 | 对齐 | SatSynth PSNR | Synth PSNR | 说明 |
|----|------|------|------|------|------|
| ✗ | ✗ | ✗ | 20.33 | 16.63 | 裸 INR 单帧 |
| ✓ | ✗ | ✗ | 30.42 | 24.69 | 加 Fourier 编码（单帧，≈bilinear） |
| ✓ | ✓ | ✗ | 28.11 | 22.83 | 多帧但不对齐 → 反而掉点 |
| ✓ | ✓ | ✓ | **32.94** | **27.87** | 联合对齐才真正用上多帧 |

组件级拆解（在 NIR 基础上逐项加，SatSynthBurst ×4）：

| 配置 | PSNR | 对齐误差 ↓ | 说明 |
|------|------|------|------|
| NIR baseline | 24.63 | 0.650 | 无任何本文组件 |
| + 直接参数化 T | 26.14 | 0.012 | 最关键，对齐误差骤降 |
| + 超采样 SS | 24.76 | 0.079 | 单独加效果有限 |
| + 固定基准帧 FBF | 26.39 | 0.319 | 有益但依赖直接参数化 |
| 直接 T + 超采样 | 31.30 | 0.012 | 两者叠加大涨 |
| **SuperF (全部)** | **32.94** | **0.012** | — |

### 关键发现
- **对齐是 MISR 的命门**：多帧不对齐（PSNR 28.11）反而比单帧（30.42）更差，因为亚像素位移直接糊掉信号；只有联合优化对齐才把多帧信息变成增益。
- **直接参数化 > 间接 MLP 估计**：相对 NIR 用辅助 MLP 估变换，直接把仿射三参数当模型参数优化，是把对齐误差从 0.65 降到 0.012 的最大功臣。
- **GNLL 的价值在脏数据**：含云时序（WorldStrat-bitter）上 GNLL 能主动调高云像素方差从而忽略它们，明显优于 MSE；干净时序上两者无差。
- **Fourier 尺度 $\sigma$ 域相关**：卫星图最优 10、地面 burst 最优 3，但同域内跨样本可共用，且与所选 loss 无关。

## 亮点与洞察
- **"把目标当输入反过来用"是个可迁移的范式**：不喂 LR 进网络、而让连续场去解释一组 LR 观测，使得亚像素对齐可以在连续坐标空间里用梯度精修——这套"联合优化表示与几何变换"的思路对任何多观测融合任务（多视角、时序、多模态）都有借鉴。
- **简单到位**：核心创新只是把变换矩阵从"另一个 MLP 估"改成"三个标量直接优化"，却带来最大增益，说明对 INR-MISR 而言归纳偏置的选择比模型容量更重要。
- **超采样治谱偏置**：在 HR 网格算、池化下采样监督，等于给低频偏置的 MLP 注入了高频位置梯度，这个 trick 对所有"只有低分辨率监督却想恢复高频"的 INR 任务通用。
- **零 HR 训练数据 = 零幻觉**：纯测试时优化，从根上避免了 SISR 学习先验带来的虚构结构，对遥感/医学等不容幻觉的场景特别契合。

## 局限与展望
- 作者承认在**噪声主导**的真实场景下（如剧烈光照变化、地表变化、季节积雪），"重复观测同一场景"的假设会失效，需要进一步处理极端噪声。
- $\sigma$ 等关键超参**域相关需调**，跨域迁移时仍要重新选；虽然同域内泛化良好，但缺一个自动选 $\sigma$ 的机制。
- TTO 范式每个场景都要单独优化（2000 迭代/样本），相比一次训练、处处推理的有监督方法在大批量部署上更慢——文中报告了 H100 单卡的时间/显存/FLOPs（附录）。
- SyntheticBurst 缺对齐的 LR 基准帧，需要暴力后处理对齐才能算指标，评测流程稍重；合成卫星数据虽尽力模拟 MTF 与噪声，与真实 Sentinel-2 仍有差距。

## 相关工作与启发
- **vs Lafenetre 2023 / Wronski 2019（可调向核回归 TTO）**：他们直接用 LR 帧做核回归重建 RGB，为 ≤×2 的卫星 burst 设计；SuperF 改用连续 INR 表示并联合优化对齐，在 ×4/×8 大倍率上优势显著，且跨卫星/地面两域泛化。
- **vs NIR (Nam 2022)（INR burst 融合）**：NIR 原为图层分离设计，用辅助 MLP 估变换、不固定基准帧、无超采样；SuperF 三处改动（直接参数化仿射、超采样、固定基准帧）专门面向 MISR 的子像素对齐，把 PSNR 从 24.63 抬到 32.94。
- **vs LIIF / Thera（INR 做 SISR）**：那些是有监督、学嵌入的任意尺度单图超分；SuperF 是无监督 TTO 多图超分，不学先验、靠多帧互补信息约束重建，从而规避幻觉。

## 评分
- 新颖性: ⭐⭐⭐⭐ "反转问题 + 直接参数化仿射 + 超采样"组合简洁而切中 MISR 要害，虽是已有 idea 的精巧重组。
- 实验充分度: ⭐⭐⭐⭐ 双域、多倍率、组件级消融 + 真实 Sentinel-2 + 含云鲁棒性都覆盖，自建数据集补齐了对齐基准帧的缺失。
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，方法的数学刻画（成像模型、GNLL）严谨易懂。
- 价值: ⭐⭐⭐⭐ 无需 HR 训练数据、零幻觉，对遥感/科学成像实用，附 demo 与开源数据集。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations](../../CVPR2026/image_restoration/the_surprising_effectiveness_of_noise_pretraining_for_implicit_neural_representa.md)
- [\[ICLR 2026\] Continuous Space-Time Video Super-Resolution with 3D Fourier Fields](continuous_space-time_video_super-resolution_with_3d_fourier_fields.md)
- [\[NeurIPS 2025\] Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](../../NeurIPS2025/image_restoration/implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)
- [\[ICML 2026\] Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](../../ICML2026/image_restoration/semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)
- [\[ICLR 2026\] Test-Time Domain Generalization for Image Super-Resolution](test-time_domain_generalization_for_image_super-resolution.md)

</div>

<!-- RELATED:END -->
