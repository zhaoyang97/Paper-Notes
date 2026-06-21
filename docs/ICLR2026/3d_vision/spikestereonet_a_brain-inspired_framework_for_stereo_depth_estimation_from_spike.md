---
title: >-
  [论文解读] SpikeStereoNet: 面向 Spike 流的类脑双目深度估计框架
description: >-
  [ICLR 2026][3D视觉][Spike 相机] 本文提出 SpikeStereoNet，直接从一对原始 spike 流（脉冲相机输出的二值高频流）估计双目深度，用一个三层循环脉冲神经网络（RSNN）作为迭代细化算子反复更新视差，并配套发布了大规模合成与真实 spike 双目数据集，在两个数据集上都超过现有 frame-based / event-based 立体匹配方法，且在仅用 10% 训练数据时仍保持高精度。
tags:
  - "ICLR 2026"
  - "3D视觉"
  - "Spike 相机"
  - "双目深度估计"
  - "脉冲神经网络"
  - "迭代细化"
  - "类脑计算"
---

# SpikeStereoNet: 面向 Spike 流的类脑双目深度估计框架

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lPMPFeioCZ](https://openreview.net/forum?id=lPMPFeioCZ)  
**代码**: https://github.com/Criticality-Cognitive-Computation-Lab/SpikeStereoNet  
**领域**: 3D视觉  
**关键词**: Spike 相机, 双目深度估计, 脉冲神经网络, 迭代细化, 类脑计算

## 一句话总结
本文提出 SpikeStereoNet，直接从一对原始 spike 流（脉冲相机输出的二值高频流）估计双目深度，用一个三层循环脉冲神经网络（RSNN）作为迭代细化算子反复更新视差，并配套发布了大规模合成与真实 spike 双目数据集，在两个数据集上都超过现有 frame-based / event-based 立体匹配方法，且在仅用 10% 训练数据时仍保持高精度。

## 研究背景与动机
**领域现状**：双目深度估计长期建立在普通帧相机之上，主流是先构造代价体（cost volume）再用 3D CNN 正则化，或像 RAFT-Stereo 那样用循环单元（ConvGRU）迭代细化视差。这些方法在静态、慢速场景下已经做得很好。

**现有痛点**：帧相机在高速、动态场景里会出现运动模糊和延迟，深度直接糊掉。受生物视网膜启发的 **spike 相机**能以高达 40,000 Hz 的时间分辨率异步发放二值脉冲、捕捉丰富的亮度信息，天然适合这类极端场景；但目前**没有专门面向 spike 流的双目算法，也没有配套的 benchmark**。直接把 spike 流转成帧再喂给现有方法，会引入时间量化误差、运动模糊或巨大的计算开销，把传感器本来的优势抹掉。

**核心矛盾**：spike 相机输出的是**异步、二值、高吞吐**的脉冲流，而 frame-based 算法期待的是**同步、带强度值**的图像对——两种数据形态根本不匹配。event 相机的立体方法虽然存在，但 event 走的是差分采样（只记录亮度变化），spike 走的是积分采样（累积光子到阈值发放），二者处理方式不能照搬。

**本文目标**：(1) 设计一个能直接吃原始 spike 流、不经过转帧的端到端双目深度网络；(2) 补上这个新方向缺失的数据 benchmark。

**切入角度**：作者注意到生物神经元的关键属性（发放阈值、静息电位、膜时间常数）并非固定，而是随神经元状态和上下文动态变化。如果把这种"自适应"机制搬进迭代细化算子，或许既能匹配 spike 数据的时序特性，又能让迭代过程稳定收敛。

**核心 idea**：用一个**自适应循环脉冲神经网络（ALIF-RSNN）替代 RAFT-Stereo 里的 ConvGRU**，作为视差的迭代更新算子，让网络在 spike 域内逐步细化深度。

## 方法详解

### 整体框架
SpikeStereoNet 接收来自左右两个 spike 相机的原始脉冲流 $S_l, S_r \in \{0,1\}^{N\times H\times W}$，输出一张全分辨率深度图。整体仍沿用"特征提取 → 构造相关性金字塔 → 迭代细化视差 → 上采样"的 RAFT-Stereo 范式，但把核心的迭代更新算子换成了类脑的 RSNN，并在前端用专门的网络从 spike 流里抽多尺度特征。训练上分两步：先在合成数据上监督预训练，再用域自适应微调迁移到真实 spike 数据。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["左/右原始 spike 流<br/>S ∈ {0,1}^(N×H×W)"] --> B["Spike 特征提取<br/>多尺度特征 + 上下文特征"]
    B --> C["相关性金字塔<br/>4 级 1D 池化代价体"]
    C --> D["RSNN 迭代更新算子<br/>ALIF 神经元逐步细化视差"]
    D -->|残差更新 d_t = d_(t-1)+Δd_t| D
    D --> E["凸组合上采样<br/>1/4 → 全分辨率"]
    E --> F["视差 → 深度"]
```

### 关键设计

**1. 直接吃 spike 流的多尺度特征提取：把二值脉冲变成可匹配的稠密特征**

spike 相机的输出是高频二值帧序列，直接做立体匹配既稀疏又含噪。本文用两个 feature network 和一个 context network 来处理：feature network 接收左右 spike 流 $S_{l(r)}\in\mathbb{R}^{N\times H\times W}$，先用 $7\times7$ 卷积降到半分辨率，再经一串残差块抽特征并下采样到 1/4 分辨率，进一步生成 1/4、1/8、1/16 三个尺度的特征 $\{f_{l,i}, f_{r,i}\}$。其中 1/4 尺度的 $f_{l,4}, f_{r,4}$ 用来构造代价体，所有尺度则作为 3D 正则化网络的引导。context network 共享类似骨干，产出多尺度上下文特征，用于在每一轮迭代时初始化并向 RSNN 注入上下文。这样设计的好处是把 spike 流在网络早期就转成多尺度的稠密表示，让后续迭代有足够丰富的空间细节可用，而无需经过"转帧"这一损失信息的步骤。

**2. 1D 相关性金字塔：把全配对代价体压成沿视差维度的多级结构**

给定左右特征图 $f_l, f_r$，先构造所有像素对的相关性代价体 $C_{ijk}=\sum_h f_{l,hij}\cdot f_{r,hik}$，然后沿最后一维（视差维）做 1D 平均池化，得到 4 级相关性金字塔 $\{C_i\}_{i=1}^4$，每级用 kernel size 2、stride 2 逐步下采样视差维度。这一步沿用 RAFT-Stereo 把 2D 光流限制到 1D 视差维的思路：不用建完整的 4D 体，而是用金字塔在多个尺度上提供从局部到全局的匹配线索，在迭代时按当前视差去金字塔里查询局部代价，既保留大范围感受野又控制了计算量。

**3. ALIF-RSNN 迭代更新算子：用自适应脉冲神经元替代 ConvGRU 做视差细化（核心贡献）**

这是全文最关键的设计。本文用一个三层 RSNN 作为更新模块，每层都含层内的循环连接（recurrent）和层间的前馈连接（feedforward），形成层级化的时序处理。1/8、1/16 分辨率层接收上下文特征和来自同层/前层 spike 状态经权重得到的突触后电流；1/4 分辨率层额外接入当前视差估计和从相关性金字塔取出的局部代价体。每个 RSNN 单元先用 sigmoid 卷积算出三个自适应变量

$$\alpha_t=\sigma(\mathrm{Conv}([s_{t-1},x_t],W_\alpha)+c_\alpha),\quad \beta_t=\sigma(\cdots W_\beta\cdots),\quad \gamma_t=\sigma(\cdots W_\gamma\cdots)$$

其中 $c_\alpha,c_\beta,c_\gamma$ 是来自 context network 的上下文嵌入。随后送入作者提出的自适应漏积分发放（ALIF）神经元：

$$h_t=\alpha_t\cdot v_{t-1}+(1-\alpha_t)\cdot(W_{rec}s_{t-1}+W_f s_t^{(l-1)}),\quad v_t^{th}=\beta_t\cdot v_{peak}$$
$$s_t=\theta(h_t-v_t^{th}),\quad v_t=h_t-\gamma_t\cdot s_t\cdot v_t^{th}$$

这里 $v_t$ 是膜电位，$v_t^{th}$ 是发放阈值，$\theta(\cdot)$ 是 Heaviside 阶跃函数，$W_{rec},W_f$ 是用卷积核实现的循环/前馈突触权重。三个自适应变量 $\alpha,\beta,\gamma\in[0,1]$ 分别控制**膜电位保留、发放阈值、软重置强度**——这正是把"神经元属性随状态和上下文动态变化"这一生物事实搬进了模型。与固定参数的经典 LIF 相比，自适应让神经元能根据 spike 数据的时序特性动态调节，从而更好地匹配异步脉冲流。最高分辨率层的 RSNN 负责输出残差视差，逐轮更新 $d_t=d_{t-1}+\Delta d_t$。作者还对网络动力学做了分析：隐状态差随时间递减（收敛），权重 Jacobian 的特征值全落在单位圆内（稳定），PCA 显示隐状态随时间发散（表达力强）——这给迭代细化为什么有效提供了一个动力学层面的解释。

**4. 凸组合上采样：从 1/4 分辨率恢复全分辨率深度**

迭代在 1/4 分辨率上进行以省算力，最后基于末层 RSNN 隐状态，用两个卷积层预测残差视差并更新当前视差，再用凸组合（convex combination）策略把 1/4 视差上采到全分辨率，最终把视差转换成深度。这一步保证迭代细化的精度不在简单上采样中丢失，得到边界锐利的稠密深度图。

### 损失函数 / 训练策略
总损失由三项组成：

$$L=L_{stereo}+\lambda_f L_{rate\_reg}+\lambda_v L_{v\_reg}=\sum_{t=1}^T \eta^{T-t}\|d_{gt}-d_t\|_1+\lambda_f\sum_{i=1}^N(r_i-r_0)^2+\lambda_v\sum_{i=1}^N\sum_{t=1}^T v_i(t)^2,\quad \eta=0.9$$

主损失 $L_{stereo}$ 是所有迭代步预测视差 $\{d_t\}$ 与真值 $d_{gt}$ 的 L1 距离，权重随迭代步递增（越靠后越重）。第二项是发放率正则，让第 $i$ 个神经元平均发放率 $r_i$ 逼近目标 $r_0$，促进时序稀疏；第三项是电压正则，约束膜电位幅度。两个正则项都鼓励稀疏并提升性能。训练用 AdamW + one-cycle 学习率（初始 $2\times10^{-4}$），梯度裁剪到 $[-1,1]$，每个样本 16 步迭代，batch size 8，训练 300k 步，配合水平/垂直随机翻转增强。

## 实验关键数据

### 主实验
合成数据集上，所有方法都以 spike 流为输入，SpikeStereoNet 在几乎所有指标上最优：

| 数据集 | 方法 | bad 2.0 (%)↓ | bad 3.0 (%)↓ | AvgErr (px)↓ | FLOPs (B) |
|--------|------|------|------|------|------|
| 合成 | RAFT-Stereo | 4.64 | 2.76 | 0.48 | 798 |
| 合成 | Selective-Stereo | 4.57 | 2.66 | 0.45 | 957 |
| 合成 | MonSter | 4.64 | 2.72 | 0.46 | 1567 |
| 合成 | **SpikeStereoNet (Ours)** | **4.13** | **2.38** | **0.42** | **473** |

真实 spike 数据集（先合成预训练再域自适应微调）上同样领先：

| 数据集 | 方法 | bad 2.0 (%)↓ | bad 3.0 (%)↓ | AvgErr↓ |
|--------|------|------|------|------|
| 真实 | DLNR | 5.64 | 3.38 | 0.61 |
| 真实 | Selective-Stereo | 5.50 | 3.43 | 0.58 |
| 真实 | **SpikeStereoNet (Ours)** | **5.33** | **3.19** | **0.56** |

值得注意的是 SpikeStereoNet 的 FLOPs（473B）是表中最低之一，参数量（12.15M）也优于多数模型，在性能和效率之间取得了好的平衡。

### 消融实验
网络结构与正则化消融（合成数据，AvgErr 越低越好）：

| 配置 | bad 2.0 (%)↓ | AvgErr (px)↓ | 说明 |
|------|------|------|------|
| Ours (full) | 4.13 | 0.42 | 完整模型 |
| w/o RC & FFC | 12.07 | 1.29 | 同时去掉循环+前馈连接，崩得最惨 |
| w/o RC | 11.05 | 0.83 | 去层内循环连接 |
| w/o FFC | 5.86 | 0.68 | 去层间前馈连接 |
| w/o GN module | 7.49 | 0.58 | 去自适应变量上的 group norm |
| w/o regularization | 7.77 | 0.61 | 去全部正则项 |

更新算子替换消融（同样规模神经元）：

| 更新算子 | bad 2.0 (%)↓ | AvgErr (px)↓ | 说明 |
|------|------|------|------|
| Vanilla RNN | 7.28 | 0.66 | 普通 RNN |
| GRU | 4.53 | 0.48 | 即 RAFT-Stereo 用的单元 |
| LSTM | 4.77 | 0.49 | |
| Raw SNN | 11.05 | 0.83 | 朴素脉冲网络 |
| LIF (固定 α,β,γ) | 7.05 | 0.69 | 非自适应脉冲神经元 |
| **ALIF RSNN (Ours)** | **4.13** | **0.42** | 自适应脉冲神经元 |

### 关键发现
- **循环+前馈连接是骨架**：同时去掉 RC 和 FFC 时 AvgErr 从 0.42 飙到 1.29，是掉点最严重的消融，说明跨时间、跨层的信息流动对捕捉 spike 数据时序动态至关重要；其中循环连接（RC）比前馈连接（FFC）更关键。
- **自适应是脉冲方案能赢的核心**：把 ALIF 换成固定参数的 LIF，AvgErr 从 0.42 退到 0.69，甚至不如普通 GRU（0.48）；只有引入自适应阈值/重置的 ALIF-RSNN 才超过 GRU，证明"让神经元属性动态变化"才是脉冲方案优于经典循环单元的关键。
- **数据效率突出**：仅用 10%–50% 训练数据时本文在所有比例上都优于 RAFT-Stereo / Selective-Stereo 等，且训练数据越少差距越大，显示在数据稀缺场景下泛化更强。
- **迭代过程稳定收敛**：隐状态差随迭代递减、权重特征值全在单位圆内，从动力学上解释了迭代细化为何有效。

## 亮点与洞察
- **把"生物神经元的自适应性"落成可训练的 ALIF 算子**：$\alpha,\beta,\gamma$ 分别对应膜电位保留、发放阈值、软重置，且由上下文动态生成，而非固定常数——这是从神经科学事实到工程模块的一次干净映射，消融也证明它就是赢 GRU 的关键。
- **数据 benchmark 本身是重要贡献**：用 Blender + 视频插帧（每对相邻帧插 50 帧）+ 类脑脉冲生成机制造出合成 spike 流，再用双 spike 相机 + Kinect 采了 3000+ 真实立体 spike 对（30 万+ 帧），填补了这个方向无标准数据集的空白。
- **可迁移思路**：把帧式迭代细化框架里的循环单元换成自适应脉冲单元，这套"自适应 RSNN 做迭代更新算子"的范式可以迁移到光流、event 立体等其他需要时序细化的低层视觉任务。

## 局限与展望
- **真实数据规模与多样性有限**：真实集只在室内采集、分辨率 $400\times250$，户外/远距离/复杂光照下的泛化未充分验证。
- **依赖域自适应迁移**：真实数据上需要先合成预训练再微调，合成-真实之间的域差仍靠 domain adaptation 弥合，端到端纯真实训练的效果未知。
- **效率优势未到神经形态硬件层面**：虽然 FLOPs 较低，但论文是在 RTX 4090 上用 PyTorch 训练评测，脉冲网络在神经形态芯片上的真实能耗/延迟收益尚未展示。
- **改进思路**：可探索更大规模真实 spike 数据、自适应变量的可解释性分析，以及把 ALIF-RSNN 部署到 neuromorphic 硬件以兑现脉冲计算的低功耗承诺。

## 相关工作与启发
- **vs RAFT-Stereo**：两者都用迭代细化 + 相关性金字塔，但 RAFT-Stereo 的更新算子是 ConvGRU、面向帧输入；本文换成 ALIF-RSNN 并直接吃 spike 流，消融里 ALIF（0.42）优于 GRU（0.48），且数据效率更好。
- **vs 固定参数 LIF / Raw SNN**：朴素脉冲网络（Raw SNN，0.83）和固定 LIF（0.69）都远不如本文，说明把脉冲网络做进立体匹配，关键不在"用脉冲"而在"让脉冲神经元自适应"。
- **vs event-based 立体方法（如 ZEST）**：event 相机走差分采样、捕捉亮度变化，本文针对 spike 相机的积分采样设计专门处理，合成集上 AvgErr 0.42 远优于 ZEST 的 0.62。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个直接从原始 spike 流做双目深度的类脑框架，ALIF-RSNN 更新算子设计扎实。
- 实验充分度: ⭐⭐⭐⭐ 合成+真实双数据集、丰富消融、数据效率与动力学分析齐全，真实集规模偏小。
- 写作质量: ⭐⭐⭐⭐ 方法与动力学分析清晰，部分模块（如 GN、context 注入）细节略简。
- 价值: ⭐⭐⭐⭐⭐ 同时贡献算法和稀缺的 spike 立体 benchmark，对神经形态视觉社区推动明显。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding](../../ICML2026/3d_vision/svl_spike-based_vision-language_pretraining_for_efficient_3d_open-world_understa.md)
- [\[ICLR 2026\] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [\[ICLR 2026\] Spiking Discrepancy Transformer for Point Cloud Analysis](spiking_discrepancy_transformer_for_point_cloud_analysis.md)
- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [\[ICLR 2026\] RayI2P: Learning Rays for Image-to-Point Cloud Registration](rayi2p_learning_rays_for_image-to-point_cloud_registration.md)

</div>

<!-- RELATED:END -->
