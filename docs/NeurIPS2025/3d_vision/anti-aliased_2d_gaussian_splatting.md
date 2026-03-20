# Anti-Aliased 2D Gaussian Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2506.11252](https://arxiv.org/abs/2506.11252)  
**作者**: Mae Younes, Adnane Boukhayma (INRIA France, University of Rennes, CNRS, IRISA)  
**代码**: [AA-2DGS](https://github.com/maeyounes/AA-2DGS)  
**领域**: 3d_vision  
**关键词**: 2D Gaussian Splatting, 抗锯齿, 新视角合成, 表面重建, Mip 滤波  

## 一句话总结

提出 AA-2DGS，通过世界空间平坦平滑核和物体空间 Mip 滤波器两个互补机制，解决 2D Gaussian Splatting 在不同采样率下渲染时的严重锯齿问题，在保持 2DGS 几何精度优势的同时显著提升多尺度渲染质量。

## 研究背景与动机

1. **2DGS 的几何优势**：2D Gaussian Splatting 使用嵌入 3D 空间的平面高斯盘代替 3DGS 的体积高斯，在 ray-splat 交点处直接评估高斯值，能获得更好的深度和法线重建精度，适用于网格恢复、物理渲染、反射建模等需要精确几何的任务。
2. **锯齿问题严重**：2DGS 在渲染分辨率与训练分辨率不同时（相机缩放、视场角变化）会产生严重锯齿伪影，极大限制了实际应用。
3. **现有 clamping 方法有害**：2DGS 原始实现采用屏幕空间 clamping（下界约束）来处理退化情况，但作者发现这种方法实际上**加剧**了锯齿伪影而非缓解。Clamping 引入梯度不连续、CUDA warp 线程分歧、跨域距离比较等问题。
4. **Mip-Splatting 无法直接迁移**：Mip-Splatting 已为 3DGS 解决了抗锯齿问题（3D 平滑滤波 + 屏幕空间 Mip 滤波），但由于 2DGS 使用完全不同的平面原语表示和渲染方式，这些方案无法直接应用。
5. **采样定理视角**：锯齿本质上是违反 Nyquist-Shannon 采样定理——高频信号在低采样率下被错误映射为低频——因此需要在采样前施加低通滤波。
6. **双重锯齿源**：Mip-Splatting 指出锯齿来自两个源头：(a) 表示本身缺乏 3D 频率约束（zoom-in 时出现高频伪影），(b) 屏幕空间滤波不足（zoom-out 时出现混叠）。AA-2DGS 需要在 2DGS 框架下同时解决这两个问题。

## 方法详解

### 整体框架

AA-2DGS 在 2DGS 基础上引入两个互补的抗锯齿机制：

- **世界空间平坦平滑核（Flat Smoothing Kernel）**：约束 2D 高斯原语的最大频率，防止 zoom-in 时出现高频伪影
- **物体空间 Mip 滤波器（Object-Space Mip Filter）**：利用 ray-splat 交点映射的仿射近似，在 splat 局部空间进行屏幕空间抗锯齿，防止 zoom-out 时出现混叠

两者共同覆盖了放大和缩小两种场景下的锯齿问题。

### 关键设计 1：世界空间平坦平滑核

**目标**：基于训练视图的 Nyquist 极限约束表示的最大频率。

**多视角频率估计**：对每个高斯原语 $k$，从所有训练视图中计算最大采样率：

$$\hat{\nu}_k = \max_{n=1\dots N} \left\{ \mathbb{1}_n(\mathbf{p}_k) \cdot \frac{f_n}{d_n} \right\}$$

其中 $f_n$ 是焦距（像素单位），$d_n$ 是深度，$\mathbb{1}_n$ 判断原语中心是否在第 $n$ 个相机视锥内。

**平坦投影**：将 Mip-Splatting 的 3D 各向同性平滑核投影到 2D 高斯所在平面，得到同方差的 2D 平滑滤波器。与原语的本征 2D 高斯卷积后，有效协方差变为：

$$\mathbf{V}_k^{\text{eff}} = \mathbf{V}_k + \sigma_{\text{smooth},k}^2 \mathbf{I}_2$$

同时调制不透明度以保持能量守恒：$\alpha_k^{\text{smooth}} = \alpha_k \frac{s_u s_v}{\sqrt{s_u^2 + \sigma^2} \cdot \sqrt{s_v^2 + \sigma^2}}$。

采样率在训练时计算，每 100 次迭代更新一次，测试时固定。

### 关键设计 2：物体空间 Mip 滤波器

**核心思想**：利用 ray-splat 交点映射 $m: \mathbf{x} \to \mathbf{u}$ 的一阶泰勒展开（仿射近似），将屏幕空间 Mip 滤波器映射到 splat 局部空间。

**推导过程**：
1. 通过仿射近似 $m(\mathbf{x}) \approx \mathbf{u}_0 + \mathbf{J}(\mathbf{x} - \mathbf{x}_0)$ 将局部坐标高斯映射到屏幕空间
2. 在屏幕空间与 Mip 滤波器（方差 $\sigma\mathbf{I}$）卷积
3. 利用高斯仿射变换性质映射回物体空间，最终得到局部空间的 Mip 滤波高斯：

$$\mathcal{G}_{\text{obj-mip},k}(\mathbf{x}) = \sqrt{\frac{1}{|\boldsymbol{\Sigma}'_{\text{local},k}|}} \exp\left(-\frac{1}{2}\mathbf{u}_k^\top (\boldsymbol{\Sigma}'_{\text{local},k})^{-1} \mathbf{u}_k\right)$$

其中 $\boldsymbol{\Sigma}'_{\text{local},k} = \mathbf{I} + \sigma \mathbf{J}\mathbf{J}^\top$。

**与传统 EWA 的区别**：传统物体空间 EWA 在原语中心做仿射近似，而 AA-2DGS 在每个像素处做近似，对大原语和极端视角更精确。

### 关键设计 3：自定义 CUDA 实现

Mip 滤波的前向和反向传播均通过自定义 CUDA 核实现，相比原始 2DGS 渲染时间增加 15-30%。Jacobian $\mathbf{J}$ 可以从 ray-splat 交点公式（Eq.6）解析计算。

### 损失函数

沿用 2DGS 的损失函数设计。新视角合成实验中禁用深度和法线正则化；表面重建实验中启用。训练 30K 迭代，高斯密度控制策略和超参数与 Mip-Splatting 一致。Mip 滤波器方差设为 0.1（近似单像素），平坦平滑滤波器方差设为 0.2。

## 实验关键数据

### Blender 数据集 — 多尺度训练 + 多尺度测试

| 方法 | Full PSNR | 1/2 PSNR | 1/4 PSNR | 1/8 PSNR | Avg PSNR |
|------|-----------|----------|----------|----------|----------|
| 2DGS | 28.58 | 30.24 | 31.42 | 27.35 | 29.40 |
| 2DGS w/o Clamping | 31.64 | 33.33 | 31.61 | 27.62 | 31.05 |
| Mip-Splatting | 32.81 | 34.49 | 35.45 | 35.50 | 34.56 |
| **AA-2DGS** | **32.68** | **34.53** | **35.65** | **35.53** | **34.60** |

### Blender 数据集 — 单尺度训练 + 多尺度测试（zoom-out）

| 方法 | Full PSNR | 1/2 PSNR | 1/4 PSNR | 1/8 PSNR | Avg PSNR |
|------|-----------|----------|----------|----------|----------|
| 2DGS | 33.05 | 27.64 | 20.61 | 16.59 | 24.47 |
| Mip-Splatting | 33.36 | 34.00 | 31.85 | 28.67 | 31.97 |
| **AA-2DGS** | **33.24** | **34.10** | **32.11** | **29.00** | **32.11** |

### Mip-NeRF 360 数据集 — 单尺度训练 + 多尺度测试（zoom-in）

| 方法 | 1× PSNR | 2× PSNR | 4× PSNR | 8× PSNR | Avg PSNR |
|------|---------|---------|---------|---------|----------|
| 2DGS | 28.82 | 24.97 | 23.79 | 23.55 | 25.28 |
| Mip-Splatting | 29.39 | 27.39 | 26.47 | 26.22 | 27.37 |
| **AA-2DGS** | **29.30** | **27.16** | **26.10** | **25.77** | **27.08** |

### DTU 表面重建（Chamfer 距离 ↓）

| 方法 | Mean Chamfer |
|------|-------------|
| 2DGS | 0.80 |
| 2DGS* (retrained) | 0.76 |
| 2DGS w/o Clamping | 0.75 |
| **AA-2DGS** | **0.74** |

### 关键发现

1. **Clamping 有害**：移除 clamping 后 2DGS 性能反而提升（Blender 多尺度 avg PSNR 从 29.40→31.05），说明原始 clamping 是误导性的设计
2. **AA-2DGS 超越 Mip-Splatting**：在 Blender 多尺度训练测试中 avg PSNR 34.60 vs 34.56；zoom-out 场景 32.11 vs 31.97
3. **几何精度保持**：DTU 上 Chamfer 距离 0.74，优于原始 2DGS 的 0.80，证明抗锯齿不牺牲几何
4. **同尺度轻微下降**：MipNeRF360 同尺度测试 PSNR 27.38 vs 2DGS 27.56，因带限约束会衰减部分高频——这是抗锯齿与锐度之间的基本 trade-off

## 亮点与洞察

1. **理论优雅**：将 Mip-Splatting 的 3D 框架巧妙适配到 2DGS 的平面原语上，平坦投影和物体空间 Mip 滤波的推导数学上简洁优美
2. **深刻洞察**：揭示 2DGS 原始 clamping 方法的根本缺陷——不仅没有抗锯齿效果，反而引入了额外伪影
3. **每像素仿射近似**：不同于传统 EWA 仅在原语中心做近似，AA-2DGS 在每个像素处计算 Jacobian 做仿射近似，对大原语更精确
4. **抗锯齿延伸到法线**：不仅 RGB 渲染抗锯齿，法线等几何属性的渲染也受益，对表面重建和反射建模有价值
5. **首个 2DGS 抗锯齿工作**：填补了 2DGS 在抗锯齿方向的研究空白

## 局限性

1. **"针状"伪影**：2D 平面原语在极端放大或极端掠射角下仍会产生针状伪影——平坦平滑核能缓解但无法根本解决零厚度问题
2. **锐度-抗锯齿 trade-off**：固定滤波参数在同训练分辨率下会轻微损失锐度（MipNeRF360 同尺度 PSNR 下降约 0.2dB）
3. **渲染开销**：自定义 CUDA 核带来 15-30% 的渲染时间增加
4. **低分辨率训练问题**：低分辨率训练时 2D 原语容易变得极薄，高分辨率渲染时出现针状伪影——这是 2D 原语的固有缺陷
5. **固定滤波参数**：平滑核和 Mip 滤波器的方差是超参数（0.2 和 0.1），可能不是所有场景的最优选择

## 相关工作

- **Mip-Splatting**：3DGS 的抗锯齿方法，AA-2DGS 的直接灵感来源，但无法直接用于 2DGS
- **3DGS**：体积高斯原语，渲染质量好但几何精度不如 2DGS
- **2DGS**：平面高斯原语，几何精度优异但缺乏有效抗锯齿
- **EWA Splatting**：经典的椭圆加权平均滤波，AA-2DGS 将其核心思想适配到 2DGS 的 ray-splat 交点框架
- **MipNeRF / Tri-MipRF**：NeRF 系列的多尺度抗锯齿方法，通过锥追踪和预滤波位置编码实现
- **Analytic Splatting / Multi-Scale GS / HDGS**：其他 3DGS 抗锯齿方案，各有计算/内存开销

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 Mip-Splatting 思想适配到 2DGS 有明确的技术挑战，物体空间 Mip 滤波器和平坦投影的推导有原创性，但整体路线延续 Mip-Splatting
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 Blender（多尺度/单尺度）、MipNeRF360（zoom-in/zoom-out/同尺度）、DTU（表面重建）三大基准，消融实验充分，clamping 分析有洞察
- **写作质量**: ⭐⭐⭐⭐ — 数学推导清晰，问题定义明确，实验组织有条理
- **价值**: ⭐⭐⭐⭐ — 填补了 2DGS 抗锯齿的空白，对需要几何精度+多尺度渲染的应用有实际意义
