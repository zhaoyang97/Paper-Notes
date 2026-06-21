---
title: >-
  [论文解读] Frequency-Aware Dynamic Gaussian Splatting
description: >-
  [ICLR 2026][3D视觉][动态高斯泼溅] 本文从频率视角揭示动态 3DGS 运动模糊的根因——"高频渲染细节"与"高频运动"在固定高斯核上互相争抢表达力，提出频率分化高斯核（FDGK）+ 傅里叶形变网络（FDN），把细节表达与运动建模解耦，在合成/真实 4D 基准上显著减少模糊并刷新 SOTA。
tags:
  - "ICLR 2026"
  - "3D视觉"
  - "动态高斯泼溅"
  - "4D 重建"
  - "运动模糊"
  - "频率感知"
  - "形变场"
  - "傅里叶嵌入"
---

# Frequency-Aware Dynamic Gaussian Splatting

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=UZ00ac4eqA](https://openreview.net/forum?id=UZ00ac4eqA)  
**代码**: 待确认  
**领域**: 3D 视觉 / 4D 动态重建  
**关键词**: 动态高斯泼溅, 4D 重建, 运动模糊, 频率感知, 形变场, 傅里叶嵌入  

## 一句话总结
本文从频率视角揭示动态 3DGS 运动模糊的根因——"高频渲染细节"与"高频运动"在固定高斯核上互相争抢表达力，提出频率分化高斯核（FDGK）+ 傅里叶形变网络（FDN），把细节表达与运动建模解耦，在合成/真实 4D 基准上显著减少模糊并刷新 SOTA。

## 研究背景与动机
- **领域现状**：基于形变场的动态 3DGS（DeformGS、4D-GS、SC-GS、Grid4D 等）已能实时重建 4D 场景，主流做法是维护一套 canonical 高斯，用 MLP 形变网络预测随时间的残差位移 $(\Delta x, \Delta R, \Delta S)$。
- **现有痛点**：动态重建在**新视角**下普遍出现严重运动模糊，尤其在物体边界和快速形变处。以往工作只在"细化形变场"层面打转，忽视了运动本身的频谱特性。
- **核心矛盾**：vanilla 3DGS 给每个高斯固定的不透明度分布（中心实、边缘渐透），静态场景靠高斯堆叠就能恢复高频细节；但在动态场景下，这逼着形变网络**同时**干两件矛盾的事——既要编排密集高斯堆叠去恢复每帧的高频外观，又要让这堆高斯跨帧协调一致地表达高频运动且不散架。两难之下网络倾向折中为"均匀低频运动"，结果就是新视角下的运动模糊。
- **本文目标**：把"高频细节表达"和"高频运动建模"这两份职责从形变网络上拆开，分别交给更有表达力的高斯核与频率感知的形变网络。
- **核心 idea**：**【频谱解耦】** 让高斯核自己分化成高频核（管尖锐边界）/低频核（管平滑区域），把细节表达从形变网络卸下来；**【傅里叶运动】** 用高频傅里叶嵌入把每个点的运动表示成多频周期叠加，配合频率感知门控只放大真正动的点。

## 方法详解

### 整体框架
FAGS 在 Grid4D 形变骨干基础上做两处升级：表达侧用**频率分化高斯核（FDGK）**改造 alpha-blending，让高斯按频率特性自适应分工；运动侧用**傅里叶形变网络（FDN）**把哈希编码的低频时空特征与高频傅里叶嵌入融合，经多头解码器预测形变，并由频率感知门控调节每个高斯的运动强度；最后加一个频域的傅里叶频率损失把整套机制拉向高频细节。

```mermaid
flowchart LR
    A[Canonical 高斯<br/>λ,β 可学习] --> B[FDGK<br/>自适应 alpha 调制 ψ(g)]
    C["4D 坐标 (x,y,z,t)<br/>四组哈希编码"] --> D[低频时空特征]
    E[高频傅里叶嵌入<br/>多频 sin/cos] --> F[高低频融合特征]
    D --> F
    F --> G[多头解码器]
    G --> H[频率感知门控 η<br/>调节运动强度]
    B --> I[频率分化高斯]
    H --> I
    I --> J[渲染 + 傅里叶频率损失]
```

### 关键设计

**1. 频率分化高斯核（FDGK）：让每个高斯自选频率身份。** 回看 alpha 计算 $\alpha_i = o_i\exp[-\frac{1}{2}(p-\mu_{2D_i})^T\Sigma_{2D_i}^{-1}(p-\mu_{2D_i})]$，作者把指数项记为 $g$，再用一个可学习的分段调制函数把它重写为 $\alpha_i = \min(o_i\psi(g), 0.99)$。$\psi(g)$ 由两个可学习参数控制：斜率参数 $\lambda\in[0,1]$（令 $r=0.5+\lambda$、$b=0.25-0.5\lambda$）决定中段映射的陡峭程度——$\lambda=0.5$ 时退化为标准高斯，$\lambda<0.5$ 时 $\psi$ 缓变得到平滑低频核，$\lambda>0.5$ 时 $\psi$ 急变得到尖锐高频核；边界参数 $\beta$ 则独立缩放分化区间的左右端点 $p_l=0.5-\beta d_{g_0}$、$p_r=0.5+\beta d_{g_0}$。相比 DRK 只调 $r$、导致同斜率高斯被迫共享同一分化区间，FDGK 用独立的 $\beta$ 解耦了"频率特性"与"分化跨度"，让每个高斯既能选频率又能选作用范围。$\lambda,\beta$ 通过反向传播联合优化（作者给出了 $\partial\psi/\partial\lambda$、$\partial\psi/\partial\beta$ 的闭式分段梯度），同时把 $\alpha$ 约束在可控区间内，稳定形变网络训练。这样高频细节交给少量尖锐高斯，平滑区交给低频高斯，大幅减少靠密集堆叠救场的依赖。

**2. 傅里叶形变网络（FDN）：把运动表示成多频周期叠加。** 直接对 10 万级高斯逐时刻优化轨迹不可行，作者改为给每个点编码高频运动再与低频空间特征融合。先把 4D 坐标拆成四组 3D 哈希编码 $(x,y,z)$、$(x,y,t)$、$(y,z,t)$、$(x,z,t)$，经 MLP 得到空间特征 $f_{spa}$ 与时间特征 $f_{tem}$。关键是为每个高斯设计**高频傅里叶嵌入** $f_{fre}=[w_1\sin(\pi\gamma_1 t), w_1\cos(\pi\gamma_1 t),\dots]$，频率 $\gamma_i=2^{\frac{3i-3}{m-1}}$ 做更密的多尺度采样，而振幅 $[w_1,\dots,w_m]=\text{MLP}(f_{spa})$ 与时间无关、对每个高斯特定——相当于把该点的运动当作一段无限循环周期信号做傅里叶分解，用振幅分布刻画它在各频率上动多少。融合后的高低频特征既能稳住全局低频趋势，又能捕捉短时高频位移。

**3. 频率感知门控（FG）：只让真正动的点动起来。** 形变网络会无差别地给所有高斯预测更新，可能误改本该静止的点。作者把高频傅里叶特征 $f_{fre}$ 与低频时间嵌入 $f_{tem}$ 融合后送入解码器 $D_\theta$，除常规的旋转 $R_x$、平移 $T_x$、$\Delta r,\Delta s$ 外额外输出一个门控分数 $\eta$，并以 $\mu'=\eta R_x\mu+\eta T_x$、$S'=S+\eta\Delta s$、$R'=R+\eta\Delta r$ 的形式调节形变强度。高频运动的高斯拿到大 $\eta$ 允许属性快速变化，近静止的低频高斯拿到小 $\eta$ 被抑制——相比硬性 clamp 静态边界，$\eta$ 平滑可变，让网络在每个时刻自适应地调控各高斯的动态行为。

**4. 傅里叶频率损失：在频域显式督促高频。** 光有上面三件套还不够，需要一个频域目标把它们的潜力激发出来。对渲染图 $I'$ 与目标图 $I$ 做 FFT 取幅度谱，定义 $L_{fre}=\|I'_{amp}-I_{amp}\|_1$（因相位主要编码结构、两图几何高度相似故只比幅度）。总损失 $L=\sigma_c L_{L1}+(1-\sigma_c)L_{MISS}+\sigma_r L_r+\sigma_{fre}L_{fre}$ 在 Grid4D 重建损失上叠加该项，强调难优化的高频区域，把高斯与形变网络一起拉向更精细的局部拟合。

## 实验关键数据

### 主实验表格
D-NeRF 合成数据集 7 场景平均（与 Grid4D 同骨干对比）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 4D-GS | 36.30 | 0.986 | 0.019 |
| SC-GS | 41.59 | 0.994 | 0.015 |
| Grid4D | 41.99 | 0.993 | 0.008 |
| Grid4D+DRK | 39.43 | 0.990 | 0.015 |
| **Ours (FAGS)** | **42.76** | **0.995** | **0.007** |

真实场景数据集平均：

| 数据集 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------|--------|--------|---------|
| Neu3D | Grid4D | 31.63 | 0.937 | 0.149 |
| Neu3D | **Ours** | **32.18** | **0.946** | **0.146** |
| HyperNeRF (Interp.) | Grid4D | 28.59 | 0.844 | 0.199 |
| HyperNeRF (Interp.) | **Ours** | **29.02** | **0.850** | **0.195** |
| HyperNeRF (Rig) | Grid4D | 25.24 | 0.685 | 0.319 |
| HyperNeRF (Rig) | **Ours** | **25.63** | **0.719** | **0.269** |

### 消融实验表格
D-NeRF 平均，逐组件去除：

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Full | **42.76** | **0.995** | **0.007** |
| w/o FDGK | 42.11 | 0.993 | 0.009 |
| w/o HFE | 42.38 | 0.994 | 0.009 |
| w/o FG | 42.70 | 0.994 | 0.008 |
| w/o $L_{fre}$ | 42.50 | 0.994 | 0.008 |
| w/o (FG + $L_{fre}$) | 42.43 | 0.994 | 0.008 |
| w/o FDGK.λ | 42.30 | 0.994 | 0.008 |
| w/o FDGK.β | 42.26 | 0.994 | 0.008 |

### 关键发现
- **FDGK 贡献最大**：去掉后 PSNR 掉 0.65、LPIPS 翻倍到 0.009，验证"高斯频率分化卸载细节表达"是核心增益来源；其中 $\lambda$ 和 $\beta$ 各自有效（去任一都掉点），说明独立边界参数确有必要。
- **高斯自发分化**：初始全部 $\lambda=\beta=0.5$（等价标准高斯），约 5000 步后明显分裂成低频/高频两簇，比例约 3:2，证明频率分工是优化自然学出来的而非人为指派。
- **真实场景增益显著**：HyperNeRF Rig 子集 SSIM 从 0.685 提到 0.719、LPIPS 从 0.319 降到 0.269，定性上修复了 Grid4D 的刀尖消失等伪影。

## 亮点与洞察
- **问题诊断漂亮**：把困扰动态 3DGS 的运动模糊归因为"固定高斯核上高频细节与高频运动的频谱争抢"，比单纯归咎于形变场不够精细更深刻，且给出了清晰的"职责解耦"解法。
- **可学习频率身份**：用 $\lambda$（频率）+ $\beta$（跨度）两个解耦参数让每个高斯自适应选择频率特性，且分化是优化涌现的，工程上只是对 alpha-blending 的轻量改造，易于插入现有 3DGS 管线。
- **运动的频谱视角**：傅里叶嵌入把逐点运动表示成多频振幅叠加，避开了逐时刻轨迹优化的算力墙，同时门控 $\eta$ 平滑抑制静态点，比硬 clamp 更优雅。

## 局限与展望
- **强依赖 Grid4D 骨干**：方法以 Grid4D 为形变 backbone 并沿用其超参，未在更多形变框架上验证通用性。
- **绝对增益偏小**：D-NeRF 上多数场景已接近饱和（PSNR 40+），平均 PSNR 提升约 0.77，真实贡献更多体现在定性的边界/细节与新视角抗模糊上，数值差距有限。
- **频率超参固定**：$\sigma_{fre}=0.3$、傅里叶频率采样 $\gamma_i$ 等为手工设定，对不同运动尺度场景的自适应性待验证；高斯频率 3:2 分裂比例是否随场景动态范围变化也值得深入。

## 相关工作与启发
- **动态 NeRF**：DyNeRF（Neural 3D Video）、Nerfies、D-NeRF 用形变场把 NeRF 拓展到时序，本文继承"canonical + 形变"范式但搬到高斯表示。
- **动态高斯泼溅**：迭代式（D-3DGS 逐帧传播）vs 形变式（DeformGS、4D-GS、SC-GS、Grid4D 共享 canonical + MLP 形变）；FAGS 属形变式，首次把频谱特性引入高斯表达。
- **高斯核增强**：DRK 也调斜率 $r$ 增强表达，但分化边界被 $r$ 隐式绑定；FAGS 的独立 $\beta$ 是关键改进，对后续想做"可学习高斯形状/频率"的工作有直接借鉴。
- **启发**：把"频率/频谱"作为解耦表达与运动的统一语言，可能迁移到视频生成、神经渲染中其他"细节 vs 动态"两难的任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 从频谱争抢角度重新诊断运动模糊并用可学习频率分化高斯核解决，视角新颖、解法自洽；但建立在 Grid4D + DRK 等已有部件之上，属增量式创新。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 D-NeRF/Neu3D/HyperNeRF 三个主流基准 + 完整逐组件消融 + λ 分布分析，论证扎实；不足是绝对增益偏小、未跨骨干验证通用性。
- **写作质量**: ⭐⭐⭐⭐ 问题动机（Fig.1 的两难分析）讲得清晰直观，方法公式与图示配合到位，逻辑连贯易读。
- **价值**: ⭐⭐⭐⭐ 给动态 3DGS 提供了一条即插即用、聚焦新视角抗模糊的实用路线，对追求高频细节的 4D 重建有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting](moe-gs_mixture_of_experts_for_dynamic_gaussian_splatting.md)
- [\[CVPR 2026\] TagSplat: Topology-Aware Gaussian Splatting for Dynamic Mesh Modeling and Tracking](../../CVPR2026/3d_vision/tagsplat_topology-aware_gaussian_splatting_for_dynamic_mesh_modeling_and_trackin.md)
- [\[ICLR 2026\] Fracture-GS: Dynamic Fracture Simulation with Physics-Integrated Gaussian Splatting](fracture-gs_dynamic_fracture_simulation_with_physics-integrated_gaussian_splatti.md)
- [\[ICLR 2026\] Gradient-Direction-Aware Density Control for 3D Gaussian Splatting](gradient-direction-aware_density_control_for_3d_gaussian_splatting.md)
- [\[ICLR 2026\] Uncertainty-Aware 3D Reconstruction for Dynamic Underwater Scenes](uncertainty-aware_3d_reconstruction_for_dynamic_underwater_scenes.md)

</div>

<!-- RELATED:END -->
