# Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting

**会议**: AAAI 2026  
**arXiv**: [2511.15102](https://arxiv.org/abs/2511.15102)  
**代码**: 待公开  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, Alpha Blending, 抗锯齿, 多尺度渲染, 像素内空间分布

## 一句话总结
重新审视3DGS中的标量alpha blending，指出其忽略像素内空间变化是多尺度渲染伪影（放大erosion/缩小dilation）的根源，提出Gaussian Blending——将alpha和transmittance建模为像素内的空间分布（2D uniform window），实现实时抗锯齿且无需重训练，在多尺度Blender上PSNR从31.59→35.80。

## 研究背景与动机
1. **领域现状**：3DGS通过Gaussian splat显式表示3D场景，比NeRF渲染速度快数个量级，已成为Novel View Synthesis (NVS) 的主流方法。Mip-Splatting、Analytic-Splatting等方法通过预滤波改善了多尺度抗锯齿。
2. **现有痛点**：所有现有NVS方法在训练时未见过的采样率下仍有明显伪影——放大时出现边缘erosion（模糊），缩小时出现dilation（阶梯状伪影）。即使Analytic-Splatting做了解析积分，问题依然存在。
3. **核心矛盾**：所有方法都使用标量alpha blending——将alpha和transmittance作为标量（每像素一个值）计算。这导致前景splat会完全遮挡本不应被遮挡的背景splat，因为忽略了像素内的空间遮挡关系。当采样率变化时，这种误差被放大。
4. **本文要解决什么？** 在不牺牲实时性能的前提下，将像素内的空间变化纳入alpha blending过程，消除erosion和dilation伪影。
5. **切入角度**：观察到Gaussian splat在2D screen space上形成连续表面，其合并后的transmittance可以用简单的2D uniform distribution近似。通过动态追踪这个distribution的window范围，就能高效建模空间遮挡。
6. **核心idea一句话**：将标量alpha blending替换为空间分布alpha blending——transmittance不再是一个数，而是像素内的一个spatial window

## 方法详解

### 整体框架
Gaussian Blending在3DGS原始pipeline中替换渲染kernel：
- **输入**：与3DGS相同的Gaussian splat场景表示
- **改进点**：alpha blending阶段，将标量transmittance $T_i$ 替换为2D uniform distribution表示（中心 $x_i$、尺寸 $l_i$、值 $t_i$）
- **输出**：更准确的像素颜色，特别是在训练时未见过的采样率下

### 关键设计

1. **空间Transmittance Distribution**:
   - 做什么：用2D uniform window追踪像素内transmittance的空间分布
   - 核心思路：传统方法中 $T_i = \prod_{j=1}^{i-1}(1-\alpha_j(p))$ 是标量；Gaussian Blending将其表示为window $(x_i, l_i, t_i)$，初始为整个像素区域（$x_1=p, l_1=[1,1]^\top, t_1=1$）。每渲染一个splat后，window会根据该splat的空间覆盖而收缩——被遮挡区域的transmittance降低，未被遮挡区域保持高transmittance
   - 设计动机：物理上正确的渲染需要对像素区域积分 $C_p^p = \int_p \sum_i T_i^p(x)\alpha_i(x)c_i dx$，但直接计算是指数复杂度。观察到Gaussian splat聚集形成连续表面，合并后的transmittance近似uniform分布，因此用window近似即可

2. **Weight计算（Splat响应积分）**:
   - 做什么：计算当前splat在transmittance window内的积分响应
   - 核心思路：对2D Gaussian做特征值分解找到主轴，将window旋转对齐主轴，然后分解为两个独立的1D Gaussian积分：$\int w_i(x)dx = t_i \cdot o_i \cdot I^0_{\sigma_1}(u_1,u_2) \cdot I^0_{\sigma_2}(v_1,v_2)$，其中 $I^k_\sigma(a,b)$ 是1D Gaussian的 $k$ 阶矩
   - 设计动机：直接2D积分无closed-form，利用特征值分解+旋转对齐可分解为可解析计算的1D积分

3. **Window更新（Transmittance分布演化）**:
   - 做什么：渲染每个splat后更新spatial window
   - 核心思路：利用1阶和2阶矩来匹配更新后的transmittance分布。新window的中心和大小通过矩匹配计算，确保remaining transmittance的空间分布被准确追踪。window会逐渐收缩到尚未被遮挡的区域
   - 设计动机：高transmittance区域应保持对背景splat的可见性，低transmittance区域应抑制重复渲染

### 损失函数 / 训练策略
- **Training-free**：Gaussian Blending是纯渲染方法，不需要额外训练
- **Drop-in replacement**：可直接替换现有3DGS方法的渲染kernel
- 通过优化CUDA实现保持实时渲染速度，无额外内存开销
- 也提供 $\text{GB}_\text{test}$ 变体：仅在test时应用Gaussian Blending

## 实验关键数据

### 主实验
Multi-scale Blender数据集（×1训练，×1/2~×1/8测试）的PSNR结果：

| 方法 | ×1 | ×1/2 | ×1/4 | ×1/8 | Avg. |
|------|-----|------|------|------|------|
| 3DGS | 33.57 | 27.04 | 21.43 | 17.74 | 24.95 |
| Mip-Splatting | 33.54 | 34.09 | 31.50 | 27.80 | 31.73 |
| Analytic-Splatting | 33.78 | 34.20 | 31.16 | 27.22 | 31.59 |
| **Gaussian Blending** | **33.92** | **35.80** | **36.82** | **35.79** | **35.58** |
| **Analytic+GB_test** | 33.62 | 35.72 | 37.36 | 36.51 | **35.80** |

×1/8缩小时：3DGS的17.74 → Gaussian Blending的35.79，提升**18dB**！

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 标量alpha blending | 基线 | erosion+dilation伪影 |
| Structured pruning (不用window) | 稍有改善 | 不追踪空间分布 |
| Gaussian Blending (full) | 最优 | 动态window追踪 |
| 仅test时应用GB | 接近full | 无需重训练也有效 |

### 关键发现
- **标量alpha blending是多尺度伪影的根源**——不是prefiltering不够，而是blending本身有问题。即使Analytic-Splatting做了完美的像素积分，标量transmittance仍然导致边缘伪影
- **在未见尺度上提升巨大（×1/4: +5.7dB, ×1/8: +8.6dB vs Analytic-Splatting）**，同时在训练尺度（×1）上也略有提升
- **与现有方法正交互补**：可叠加在Mip-Splatting或Analytic-Splatting上进一步提升
- **保持实时渲染速度**，无额外内存开销

## 亮点与洞察
- **问题诊断精准**：不是提出新的抗锯齿滤波器，而是从"blending本身是错的"这个角度切入，找到了所有NVS方法共享的根本限制
- **Uniform distribution近似transmittance**的假设虽然粗糙但有效——因为Gaussian splat确实倾向于聚集形成连续表面，合并后的alpha分布趋近uniform。这个观察很有insight
- **Drop-in replacement设计**非常实用：无需重新训练模型，只替换渲染kernel就能在任意3DGS方法上获得多尺度抗锯齿

## 局限性 / 可改进方向
- **Uniform distribution近似在半透明物体上可能不准确**：烟雾、玻璃等场景的alpha分布远非uniform
- **Window作为方形（axis-aligned rotation ≤45°）的简化**可能在某些极端splat分布下引入误差
- **只在Blender和Mip-NeRF 360上评估**：缺少更大规模真实场景（如城市级重建）的验证
- **没有与supersampling做直接对比**来验证近似的精度

## 相关工作与启发
- **vs Mip-Splatting**: 做3D+2D prefiltering处理频率混叠，但alpha blending仍是标量。GB从blending本身入手，两者互补（叠加后+3.8dB avg）
- **vs Analytic-Splatting**: 做了像素面积上的解析积分替代点采样，但transmittance仍是标量。GB的空间transmittance可以叠加其上
- **vs Supersampling**: 物理上正确但计算昂贵。GB用uniform近似以极低成本逼近supersampling效果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从"alpha blending本身是错的"这个视角切入，不是增量改进而是重新思考基础渲染机制
- 实验充分度: ⭐⭐⭐⭐ 多尺度Blender和Mip-NeRF 360，多种baseline对比，与多种方法叠加测试
- 写作质量: ⭐⭐⭐⭐⭐ Figure 2的对比图极其直观地展示了标量blending vs Gaussian Blending的本质区别
- 价值: ⭐⭐⭐⭐⭐ Drop-in replacement + 实时 + 无需重训 = 极高实用价值，可能成为3DGS渲染的新标准
