---
title: >-
  [论文解读] Jigsaw++: Imagining Complete Shape Priors for Object Reassembly
description: >-
  [ICCV 2025][物体重组] Jigsaw++ 提出了一种基于生成模型的完整形状先验学习方法，通过"retargeting"策略将部分组装的碎片点云映射到完整物体的形状空间，与现有组装算法正交地提升重组质量。 物体重组（Object Reassembly）广泛应用于数字考古、机器人家具组装和骨骼修复等场景…
tags:
  - "ICCV 2025"
  - "物体重组"
  - "3D形状补全"
  - "点云生成"
  - "Rectified Flow"
  - "形状先验"
---

# Jigsaw++: Imagining Complete Shape Priors for Object Reassembly

**会议**: ICCV 2025  
**arXiv**: [2410.11816](https://arxiv.org/abs/2410.11816)  
**代码**: [GitHub](https://github.com/Jiaxin-Lu/Jigsaw-PlusPlus)  
**领域**: 其他  
**关键词**: 物体重组, 3D形状补全, 点云生成, Rectified Flow, 形状先验

## 一句话总结

Jigsaw++ 提出了一种基于生成模型的完整形状先验学习方法，通过"retargeting"策略将部分组装的碎片点云映射到完整物体的形状空间，与现有组装算法正交地提升重组质量。

## 研究背景与动机

物体重组（Object Reassembly）广泛应用于数字考古、机器人家具组装和骨骼修复等场景，分为零件组装（part assembly）和碎片组装（fracture assembly）两大类。现有方法的核心缺陷在于：**缺乏对完整物体的全局理解**。当输入仅有部分碎片时，现有方法高度依赖类别特定模板，无法泛化到多样化的物体类型。

具体而言，已有方法存在三方面不足：

**逐片处理**：只关注单个碎片或其断裂面的信息，忽略了完整物体的全局形状约束

**模板依赖**：需要预先知道物体类别或已有完整模板，限制了适用范围

**缺失碎片问题**：真实场景中常有碎片丢失，但现有方法假设输入碎片完整

Jigsaw++ 的核心动机是：**学习一个类别无关的完整形状先验**，从部分组装结果中"想象"出完整物体，作为额外信息层来提升下游组装算法的表现。

## 方法详解

### 整体框架

Jigsaw++ 采用两阶段设计：
1. **第一阶段（形状先验学习）**：训练一个点云生成模型，学习完整物体的形状分布
2. **第二阶段（Retargeting 重建）**：从部分组装的输入出发，通过微调的生成模型重建完整形状

### 关键设计

1. **点云-图像双向映射**：这是整个框架的基石。将点云坐标 $\mathbf{o}_i \in [0,1]^3$ 通过映射函数 $f(\mathbf{o}_i) = \lfloor 255 \mathbf{o}_i \rfloor$ 转换为 RGB 颜色空间，再通过指定相机角度进行光栅化渲染。逆映射 $f'(\mathbf{c}_i) = \frac{1}{255}\mathbf{c}_i$ 可从彩色图像恢复 3D 坐标。这种设计的巧妙之处在于：（1）利用预训练的 LEAP（image-to-3D）模型，间接获取海量 2D 数据的知识；（2）突破了点云数量的限制，支持任意数量的输入输出点。

2. **基于 Rectified Flow 的联合生成模型**：采用 Rectified Flow 作为生成框架，联合生成全局嵌入 $\mathbf{g}$ 和重建潜变量 $\mathbf{r}$。Rectified Flow 通过 ODE 实现两分布间的传输映射：

$$X_t = (1-t)X_0 + tX_1, \quad \frac{d}{dt}X_t = X_1 - X_0$$

其优势是学到近似线性的轨迹，使前向和逆向采样都很高效。模型使用 U-ViT 作为骨干网络，在编码阶段利用 DINOv2 特征提取器提供跨类别的泛化能力。

3. **Retargeting 策略**：这是 Jigsaw++ 最核心的贡献。给定部分组装物体 $\hat{O}$ 及其潜变量 $\hat{\mathbf{x}}_1$，首先通过逆 ODE 求解得到 $\hat{\mathbf{x}}_0$。由于输入不是完整物体，$\hat{\mathbf{x}}_0$ 在 $\pi_0 = \mathcal{N}(0,I)$ 下的似然很低。因此应用 Langevin 动力学进行调整：

$$\mathbf{x}_0 = \alpha \hat{\mathbf{x}}_0 + \sqrt{1-\alpha^2}\xi, \quad \xi \sim \mathcal{N}(0,I)$$

然后用 $(\mathbf{x}_0, \mathbf{x}_1)$ 对进行微调，使模型学会从不完整输入到完整形状的映射。Rectified Flow 的线性轨迹特性将逆采样步数压缩到正常的 $1/25$，大幅降低微调成本。

### 损失函数 / 训练策略

- **生成阶段**：标准 Rectified Flow 训练目标 $\mathbb{E}\|\frac{d}{dt}X_t - v(X_t, t)\|$
- **Retargeting 阶段**：微调目标 $\mathbb{E}_{\mathbf{x}_0,\mathbf{x}_1}\|(\mathbf{x}_0 - \mathbf{x}_1) - v(\mathbf{x}_t, t)\|^2$
- 使用 Breaking Bad 训练集的 407 个物体（34075 个碎片模式）训练，不提供类别信息

## 实验关键数据

### 主实验

**Breaking Bad 数据集（碎片组装）**：

| 方法 | CD (×10⁻³) ↓ | Precision (%) ↑ | Recall (%) ↑ |
|------|-------------|----------------|-------------|
| SE(3) | 22.4 | 20.2 | 22.5 |
| SE(3) + Jigsaw++ | 14.3 | 37.8 | 36.6 |
| Jigsaw | 10.5 | 45.6 | 42.7 |
| Jigsaw + Jigsaw++ | **4.5** | **48.7** | **49.5** |

**PartNet 数据集（零件组装，DGL 基线）**：

| 类别 | CD ↓ | Precision ↑ | Recall ↑ |
|------|------|------------|---------|
| Chair (DGL) | 47.8 | 21.5 | 20.0 |
| Chair + Jigsaw++ | 41.0 | 52.0 | 33.6 |
| Table (DGL) | 53.6 | 16.6 | 15.4 |
| Table + Jigsaw++ | 42.6 | 53.6 | 31.0 |

### 消融实验

| 配置 | 说明 |
|------|------|
| 逆采样比 k=1/10 | 最优设置，完整逆采样会过度模仿输入 |
| k=1/25 | 性能仍可接受，验证了 Rectified Flow 的步数压缩能力 |
| α=1 | 输出几乎复制输入 |
| α 逐渐减小 | 输出逐步偏向完整物体，但具体形状可能偏离 |
| 20% 碎片缺失 | CD 仅从 1.8 升至 2.0，precision/recall 几乎不变 |

### 关键发现

- Jigsaw++ 与现有组装算法正交，即使基线算法较弱（如 SE(3)），加入形状先验也能大幅提升
- 使用 GT 形状先验做碎片匹配可将 Jigsaw 的误差降低 50%，即使加入 20% 噪声仍显著优于原始方法
- Breaking Bad 上改进更大（物体较小适合颜色映射），PartNet 上 precision 提升 30+ 个百分点

## 亮点与洞察

- **正交性设计**：不是替换现有方法而是作为补充层，任何组装算法都可以受益
- **2D-3D 桥接**：巧妙利用坐标-颜色映射+LEAP 避免了 3D 数据稀缺问题
- **Rectified Flow 的工程优势**：线性轨迹使得 retargeting 微调成本极低，逆采样只需 1/25 步数

## 局限与展望

- **尺寸限制**：颜色映射对大尺寸物体（如路灯）失效，图像分辨率不足以表达
- **未见类别泛化**：模型对训练集中未见的物体类型重建能力有限
- **拓扑约束**：对复杂几何（如杯子把手）的拓扑关系难以准确重构
- **下游利用**：目前尚无现成的组装算法能充分利用生成的形状先验

## 相关工作与启发

- LEAP（image-to-3D）的复用思路值得借鉴，将预训练大模型迁移到数据稀缺的 3D 任务
- Rectified Flow 在需要"条件重定向"的生成任务中有独特优势
- 形状先验可用于更多下游任务，如物体识别、机器人抓取中的形状推理

## 评分

- **新颖性**: ⭐⭐⭐⭐ retargeting + 颜色映射方案原创性强
- **实验充分度**: ⭐⭐⭐⭐ 两个数据集 + 多项消融实验 + 缺失碎片鲁棒性测试
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，但数学符号较多
- **价值**: ⭐⭐⭐⭐ 为物体重组提供了全新范式方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Tokenisation is NP-Complete](../../ACL2025/others/tokenisation_is_np-complete.md)
- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)
- [\[ACL 2025\] Meta-Learning Neural Mechanisms rather than Bayesian Priors](../../ACL2025/others/meta-learning_neural_mechanisms_rather_than_bayesian_priors.md)
- [\[ECCV 2024\] Object-Aware NIR-to-Visible Translation](../../ECCV2024/others/object-aware_nir-to-visible_translation.md)
- [\[CVPR 2025\] Radio Frequency Ray Tracing with Neural Object Representation for Enhanced RF Modeling](../../CVPR2025/others/radio_frequency_ray_tracing_with_neural_object_representation_for_enhanced_rf_mo.md)

</div>

<!-- RELATED:END -->
