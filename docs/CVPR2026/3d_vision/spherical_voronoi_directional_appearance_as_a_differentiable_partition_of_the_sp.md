---
title: >-
  [论文解读] Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere
description: >-
  [CVPR 2026][3D视觉][新视角合成] 针对辐射场中"视角相关外观"长期依赖球谐函数（SH）而难以表达高频镜面反射的痛点，本文提出 **Spherical Voronoi（SV）**——用一组可学习站点把球面软划分成若干区域，作为显式球面函数表示，既比 SH/球面高斯更易优化，又能锐利建模 glint 级高光，并进一步扩展为"可学习光照探针"做空间变化反射，在 Ref-NeRF/GlossySynthetic 等反射 benchmark 上取得 SOTA（Ref-NeRF PSNR 36.09）。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "新视角合成"
  - "3D高斯泼溅"
  - "球面表示"
  - "镜面反射"
  - "可微分Voronoi"
---

# Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Di_Sario_Spherical_Voronoi_Directional_Appearance_as_a_Differentiable_Partition_of_the_CVPR_2026_paper.html)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 新视角合成, 3D高斯泼溅, 球面表示, 镜面反射, 可微分Voronoi

## 一句话总结
针对辐射场中"视角相关外观"长期依赖球谐函数（SH）而难以表达高频镜面反射的痛点，本文提出 **Spherical Voronoi（SV）**——用一组可学习站点把球面软划分成若干区域，作为显式球面函数表示，既比 SH/球面高斯更易优化，又能锐利建模 glint 级高光，并进一步扩展为"可学习光照探针"做空间变化反射，在 Ref-NeRF/GlossySynthetic 等反射 benchmark 上取得 SOTA（Ref-NeRF PSNR 36.09）。

## 研究背景与动机
**领域现状**：以 3D Gaussian Splatting（3DGS）为代表的辐射场方法是当前新视角合成的主流，每个图元用低阶球谐（通常 degree-3）存储视角相关颜色。SH 之所以流行，是因为它正交、全局支撑，loss landscape 平滑、易优化。

**现有痛点**：SH 是带限（band-limited）表示，要精确刻画尖锐高频信号（如光滑表面上的一个 glint 高光），所需系数随频率二次增长，参数量爆炸；并且在函数不连续处会出现 Gibbs 振铃伪影（论文 Figure 2 显示重建一个高光需要 ~30 个系数还带振铃）。这导致 3DGS 难以重现 glossy 表面的复杂视角相关外观——这也是当前 benchmark 在 PSNR 上"见顶"的原因之一。

**核心矛盾**：现有显式替代方案陷入"表达力 ↔ 可优化性"的两难。球面高斯（SG）、球面 Beta（SB）能表达局部支撑的尖锐信号，但因为核是紧支撑的，lobe 一旦没对准就梯度很弱、对初始化极敏感、大 concentration 时梯度病态，优化不稳定容易陷局部极小。换句话说：SH 好优化但表达力不够，SG/SB 表达力够但难优化。

**本文目标**：找到一种**既好优化、又能表达高频、还能扩展到空间变化反射**的显式球面表示，统一 diffuse 与 specular 外观建模。

**切入角度**：作者从可微分 Voronoi 图得到启发——与其用一堆相互"竞争"、彼此重叠的核去拟合球面函数，不如把球面**划分**成互不重叠的区域。划分天然给出对整个定义域的覆盖，每个方向都明确归属某个（软）区域，从而避免 SG/SB 那种 lobe 之间抢权重、留空洞的问题。

**核心 idea**：用"球面的可微分软 Voronoi 划分"代替球谐基，把方向域分成带平滑边界的可学习区域；并把同一个 SV 表示沿反射方向查询、做成空间分布的光照探针，把反射也纳入同一套显式可微框架。

## 方法详解

### 整体框架
论文先把 SV 定义为一种通用的显式球面函数表示，再把它落到辐射场的两个应用上：(i) 直接替换视角方向参数化（diffuse/一般视角相关外观），(ii) 作为光照探针做反射建模。

SV 的核心对象是球面上一组**方向站点** $s_1,\dots,s_K \in \mathbb{S}^2$ 和对应的**函数值** $c_1,\dots,c_K$。给定查询方向 $\omega$，函数值是各站点值的加权组合 $f_{SV}(\omega) = \sum_{k=1}^{K} w_k(\omega;\lambda_k)\, c_k$，权重由 softmax 给出。把它接到 3DGS 时，每个高斯额外携带 $\{\lambda_k\},\{s_k\},\{c_k\}$ 并联合优化，输出 RGB 视角相关颜色。

反射建模则走 deferred rendering（延迟着色）路线：先把 2DGS 场景光栅化成几何 buffer（位置/法线/粗糙度/diffuse 颜色），再在 lighting pass 里用"远场 cubemap + 近场 Voronoi 光照探针"合成 specular，最后与 diffuse 相加得到像素颜色。整条反射管线如下：

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["视角方向 ω"] --> B["球面软Voronoi划分<br/>softmax 加权站点"]
    B -->|直接查询 f(ω)| C["视角相关 diffuse 外观"]
    B -->|沿反射方向 f(ωr)| D["可学习光照探针<br/>kNN 探针 + 逆距离插值"]
    E["2DGS 几何pass<br/>位置/法线/粗糙度/diffuse"] --> D
    E --> F["远场 cubemap"]
    D --> G["近场镜面 Cn"]
    F --> H["远场镜面 Cf"]
    G --> I["β 混合近/远场<br/>粗糙度调温度 λ"]
    H --> I
    C --> J["最终颜色 C = D + Cspec"]
    I --> J
```

### 关键设计

**1. 球面软 Voronoi 划分：用 softmax 把球面切成互不重叠的可学习区域**

这一步直接针对 SH 表达力不足、SG/SB 又难优化的矛盾。SV 把方向 $\omega$ 处的函数值写成站点值的加权和

$$f_{SV}(\omega;\lambda,s,c) = \sum_{k=1}^{K} w_k(\omega;\lambda_k)\, c_k, \qquad w_k(\omega;\lambda) = \frac{\exp(\lambda_k\, s_k\cdot\omega)}{\sum_{k'=1}^{K}\exp(\lambda_{k'}\, s_{k'}\cdot\omega)}.$$

权重是站点方向与查询方向内积 $s_k\cdot\omega$ 的 softmax，温度 $\lambda_k>0$ 控制划分锐度：$\lambda$ 小则区域边界平滑、颜色过渡柔和；$\lambda$ 大则趋近硬 Voronoi 镶嵌、可表达尖锐不连续（论文 Figure 6 给出 $\lambda=\{1,5,25\}$ 的过渡）。当所有站点共享同一个 $\lambda$ 就是标准软球面 Voronoi；给每个站点分配各自的 $\lambda_k$ 则得到"加权"变体，具有局部自适应的角度锐度。

它之所以有效，关键在 softmax 的两个性质：**所有站点都有良好定义的梯度**（不像紧支撑核会出现某个方向梯度为零的死区），以及 softmax 归一化天然产生一个**干净、互不重叠的球面分解**——每个方向的权重和为 1，区域之间不会像 SG/SB 那样为重叠区"抢"权重、也不会留下没被任何核覆盖的空洞。论文的 2D 拟合实验（Figure 4，100 次随机初始化重复）显示 SV 始终收敛到比 SH/SG/SB 更好的重建，且方差更小。

**2. 视角方向参数化：用站点范数隐式编码温度，8 站点对齐 degree-3 SH 的自由度**

在标准辐射场（无显式反射建模）里，直接把视角方向表示换成 SV，即 $f_{SV}(\omega):\mathbb{S}^2\to\mathbb{R}^3$ 输出 RGB。为了和 degree-3 球谐做**等参数量**的公平对比，作者用 **8 个站点/高斯**：每个站点存 3 维位置 + 3 维 radiance，共 48 个可学习参数（恰好匹配 degree-3 SH 的自由度）。

一个巧妙之处是温度不单独存：站点向量 $s_k$ 的方向 $\hat s_k = s_k/\|s_k\|$ 表示单位方向，而其**范数 $\|s_k\|$ 直接当作温度** $\lambda_k$，即 $\lambda_k=\|s_k\|$。这样温度无需额外参数、随站点向量一起优化，内存随站点数线性增长。作者也指出原 Beta-Splatting backbone 每 500 iter 在测试集上挑最优 checkpoint（变相用了 test set 做 early-stopping），本文改为固定迭代数训练、不看测试集，是更干净的协议。

**3. 可学习光照探针：把 SV 沿反射方向查询，建模空间变化反射**

只沿反射方向查 $f(\omega_r)$（其中 $\omega_r = 2(\omega\cdot n)n - \omega$，沿用 Ref-NeRF 的"用反射方向而非视角方向"思路）隐含了远场光照假设——当 glossy 物体靠近其他物体/光源时，外观不仅取决于方向、还取决于位置，远场假设会失效，导致模糊重建（论文 Figure 3）。神经场可以把反射方向和位置一起 concat 进 MLP，但显式高斯表示很难这么做。

本文的解法是**空间分布的可学习光照探针**：在场景中放一组探针，每个探针 $i$ 由位置 $p_i$、混合权重 $\beta_i\in[0,1]$ 和一个 SV 函数 $(\lambda_i,s_i,c_i)$ 参数化，全部联合优化。对表面点 $P$，取其 $k$ 近邻探针 $\mathcal{N}=\text{kNN}(P)$，用归一化逆距离权重 $\tilde w_i = \|P-p_i\|^{-1}/\sum_{j\in\mathcal{N}}\|P-p_j\|^{-1}$ 插值，得到近场颜色与混合因子

$$C_n = \sum_{i\in\mathcal{N}} \tilde w_i\, f^{i}_{SV}(\omega_r), \qquad \beta = \sum_{i\in\mathcal{N}} \tilde w_i\, \beta_i.$$

每个探针把一个局部反射场显式编码成沿反射方向查询的 SV 函数，因此天生适合 3DGS 里的镜面外观建模——这是与之前探针工作（隐式/非反射方向）的关键区别。

**4. 延迟着色与粗糙度调温：用 roughness 控制 SV 锐度，统一近/远场镜面**

反射管线建在 2DGS backbone 上（法线估计更准），每个图元额外学两个材质参数：粗糙度 $r\in[0,1]$ 和 diffuse 颜色 $d\in\mathbb{R}^3$。几何 pass 把所有 2D 高斯光栅化一次，得到逐像素的位置 $P$、法线 $N$、粗糙度 $R$、diffuse 颜色 $D$。最终颜色 $C = D + C_{spec}$，其中镜面项是近/远场的空间变化混合

$$C_{spec} = \beta\, C_n + (1-\beta)\, C_f,$$

远场 $C_f$ 用可学习 cubemap 在反射方向 $\omega_r$ 处求值。一个关键耦合是**粗糙度调制温度**：

$$\lambda = (1-R)\,\lambda_{max} + R\,\lambda_{min},$$

低粗糙度对应高 $\lambda$（更锐的反射 lobe），高粗糙度则展宽 lobe。注意这里 $\lambda$ 不再像视角参数化那样直接学习，而是由表面粗糙度 $R$ 物理地推导出来——这让 SV 的角度锐度自动随材质变化，得到统一、可微、全显式的 diffuse+specular 外观模型。

## 实验关键数据

### 主实验：视角方向参数化（diffuse / 一般视角相关）
基于 Beta-Splatting backbone，每高斯 8 站点（48 参数，对齐 degree-3 SH）。SV 在所有数据集上一致超过 SH/SG/SB 等颜色参数化，甚至超过强神经基线 Zip-NeRF（PSNR，越高越好）：

| 数据集 | SH | SG | SB | SV(本文) | Zip-NeRF |
|--------|-----|-----|-----|----------|----------|
| Mip-NeRF360 | 28.09 | 28.18 | 28.12 | **28.57** | 28.55 |
| NeRF-Synthetic | 34.15 | 34.26 | 34.10 | **34.53** | 33.67 |
| DeepBlending | 29.80 | 29.67 | 29.56 | **30.48** | - |
| Tanks&Temples | 24.50 | 24.71 | 24.54 | **24.75** | 23.64 |

### 主实验：反射参数化（Voronoi 光照探针）
基于 2DGS backbone，合成场景 128 探针、真实场景 1024 探针 + 2048 站点，cubemap 分辨率 $256\times256\times6\times3$。在 Ref-NeRF / GlossySynthetic 上取得 SOTA，Ref-Real 上具竞争力（PSNR）：

| 数据集 | 3DGS | 3DGS-DR | Ref-GS | Ours |
|--------|------|---------|--------|------|
| Ref-NeRF | 30.37 | 34.13 | 35.57 | **36.09** |
| GlossySynthetic | 26.50 | 30.36 | 31.27 | **31.30** |
| Ref-Real | 23.85 | 23.83 | 23.81 | **23.91** |

值得注意的是本文略胜 Ref-GS——后者用同样的 2DGS backbone 但靠 MLP decoder 建模近场反射，而本文是**全显式**的。

### 消融：探针参数化方式（Ref-NeRF，固定参数量）
在光照探针里换不同球面表示，参数量固定，SV 大幅领先：

| 探针表示 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------|--------|--------|---------|
| SG | 34.19 | 0.967 | 0.060 |
| SB | 34.07 | 0.966 | 0.061 |
| Cubemap | 34.48 | 0.970 | 0.056 |
| **SV** | **36.09** | **0.976** | **0.050** |

### 关键发现
- **Voronoi 划分在等参数量下最具表达力**：相同 capacity 下 SV 比 SG/SB/Cubemap 都好（探针消融 +1.6 PSNR 以上），说明"空间划分"比"叠加局部核"是更紧凑高效的局部外观基。
- **优化鲁棒性是真优势而非玄学**：Figure 4 的 100 次随机初始化重复拟合显示 SV 收敛分布集中、终值更高，而 SG/SB 因紧支撑核易陷局部极小、对初始化敏感。
- **显式也能打过 MLP**：在反射 benchmark 上全显式的 SV 略胜带 MLP decoder 的 Ref-GS，且在视角参数化上超过 Zip-NeRF，说明换对表示能突破当前 benchmark 的 PSNR 瓶颈。

## 亮点与洞察
- **"划分 vs 叠加"的视角转换很巧**：把球面函数从"一堆相互竞争的核之和"重构为"互不重叠区域的软划分"，softmax 归一化既保证全域覆盖、又给所有站点稳定梯度，一举绕开 SG/SB 的初始化敏感和死梯度问题——这是本文最 "啊哈" 的地方。
- **温度参数一物多用**：在视角参数化里温度由站点范数 $\|s_k\|$ 隐式给出（零额外参数），在反射里又改由粗糙度 $R$ 物理推导（$\lambda=(1-R)\lambda_{max}+R\lambda_{min}$），同一个锐度旋钮在两个场景里分别"被学习"和"被物理约束"，设计上很统一。
- **可迁移思路**：可微分软 Voronoi 划分 + softmax 温度，本质是一种"自适应、互斥、全覆盖"的方向域分解，可迁移到任何需要在球面/方向域上拟合高频且不连续函数的任务（如 BRDF 拟合、环境光照编码、点云方向特征）。

## 局限与展望
- 作者承认远场仍用 cubemap 近似，近场探针靠 kNN 逆距离插值——探针密度（合成 128 / 真实 1024）与放置策略可能影响空间变化反射的精度，论文未深入探讨自适应探针布置。
- ⚠️ 反射管线依赖 2DGS 的法线估计质量；法线不准时反射方向 $\omega_r$ 会偏，进而影响 SV 沿反射方向的查询，这一耦合误差论文未单列消融。
- Ref-Real（真实反射）上仅与 baseline 持平/微胜，说明真实场景的近场/inter-reflection 仍是开放难题；站点数（反射用到 2048）带来的内存/速度开销随场景复杂度增长也值得关注。

## 相关工作与启发
- **vs 球谐（SH）**: SH 全局支撑、易优化但带限，高频/不连续处需大量系数并产生 Gibbs 振铃；SV 用局部自适应软划分锐利表达高频，等参数量下质量更高。
- **vs 球面高斯/Beta（SG/SB）**: SG/SB 能表达局部尖锐信号但紧支撑核导致初始化敏感、梯度病态难优化；SV 用 softmax 实现非重叠划分、全站点有稳定梯度，优化更稳。
- **vs Ref-NeRF**: 二者都用"沿反射方向查询 radiance"的思路；Ref-NeRF 用 MLP 条件化方向+位置建模空间变化反射，本文用显式可学习 SV 光照探针 + kNN 插值替代，保持全显式可微。
- **vs Ref-GS**: 同样基于 2DGS backbone，但 Ref-GS 用 MLP decoder 建模近场反射，本文全显式（Voronoi 光探针），反射 benchmark 上略胜且更易解释。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把可微分 Voronoi 划分引入球面外观表示，是对 SH/SG/SB 范式的根本性替换。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 7 个数据集、含 2D 拟合鲁棒性分析与探针消融，但缺法线误差/探针密度的系统消融。
- 写作质量: ⭐⭐⭐⭐⭐ 背景动机推导清晰，公式与图示（Figure 4/6）有力支撑论点。
- 价值: ⭐⭐⭐⭐⭐ 在反射 benchmark 取得 SOTA 且全显式，为 3DGS 高频外观建模提供了可复用的新基。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning Differentiable Hierarchies in 3D Gaussian Splatting](learning_differentiable_hierarchies_in_3d_gaussian_splatting.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2026\] MeshSplatting: Differentiable Rendering with Opaque Meshes](meshsplatting_differentiable_rendering_with_opaque_meshes.md)
- [\[CVPR 2026\] Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting](intrinsic_geometry-appearance_consistency_optimization_for_sparse-view_gaussian_.md)
- [\[CVPR 2026\] D-Prism: Differentiable Primitives for Structured Dynamic Modeling](d-prism_differentiable_primitives_for_structured_dynamic_modeling.md)

</div>

<!-- RELATED:END -->
