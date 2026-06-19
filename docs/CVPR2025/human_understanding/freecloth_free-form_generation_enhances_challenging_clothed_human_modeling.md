---
title: >-
  [论文解读] FreeCloth: Free-Form Generation Enhances Challenging Clothed Human Modeling
description: >-
  [CVPR 2025][人体理解][着装人体建模] 提出 FreeCloth 混合框架，将人体表面分为"裸露/变形/生成"三类区域，对贴身衣物用 LBS 变形、对宽松服装（裙子、长裙）用无 LBS 约束的自由形态生成器建模，在 ReSynth 数据集上取得 SOTA，尤其在宽松服装场景下大幅超越现有方法。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "着装人体建模"
  - "自由形态生成"
  - "线性混合蒙皮"
  - "点云"
  - "宽松服装"
---

# FreeCloth: Free-Form Generation Enhances Challenging Clothed Human Modeling

**会议**: CVPR 2025  
**arXiv**: [2411.19942](https://arxiv.org/abs/2411.19942)  
**代码**: [https://alvinyh.github.io/FreeCloth](https://alvinyh.github.io/FreeCloth)  
**领域**: 人体理解  
**关键词**: 着装人体建模, 自由形态生成, 线性混合蒙皮, 点云, 宽松服装

## 一句话总结

提出 FreeCloth 混合框架，将人体表面分为"裸露/变形/生成"三类区域，对贴身衣物用 LBS 变形、对宽松服装（裙子、长裙）用无 LBS 约束的自由形态生成器建模，在 ReSynth 数据集上取得 SOTA，尤其在宽松服装场景下大幅超越现有方法。

## 研究背景与动机

**领域现状**：数据驱动的着装人体建模方法通常预测 canonical 空间中的局部变形，再通过 LBS（线性混合蒙皮）将变形转换到姿态空间。这种范式在紧身衣物上效果良好，因为衣物紧贴人体、服从骨骼运动。

**现有痛点**：对于宽松服装（裙子、长裙），LBS 范式根本性地失效。原因在于当衣物远离身体时（如两腿之间的裙摆区域），从姿态空间到 canonical 空间的映射（canonicalization）变得病态——没有明确的骨骼对应关系，导致点云被"撕裂"成裤子形状的伪影。POP、SkiRT 等工作尝试通过模板细化缓解问题，但本质上仍受限于 LBS 变换框架。

**核心矛盾**：LBS 提供了宝贵的结构先验（骨骼运动引导变形），但对于远离身体的宽松部分，这个先验反而成为约束。完全抛弃 LBS 又会导致关节区域建模不准确。需要在"利用结构先验"和"保持表达灵活性"之间找到平衡。

**本文目标** 如何对宽松服装区域绕开 LBS 限制进行建模，同时保留 LBS 在贴身区域的优势？

**切入角度**：作者认为应该根据衣物区域与身体的距离关系采用不同的建模策略。贴身区域受骨骼运动影响大，适合 LBS；宽松区域受运动影响小，更适合直接生成。这种混合思路直觉上很自然，但此前没有被系统化实现。

**核心 idea**：将人体分割为裸露/贴身变形/宽松生成三类区域，用 LBS 处理贴身衣物变形、用自由形态点云生成器处理宽松服装，实现"该变形就变形、该生成就生成"的混合建模。

## 方法详解

### 整体框架

输入为 posed SMPL-X 人体和服装类型，输出为完整着装人体点云。Pipeline：(1) 基于 Clothing-cut Map 将人体分为三区域；(2) 裸露区域直接复制；(3) 蓝色贴身区域用 LBS 变形网络生成 $X^d$；(4) 绿色宽松区域用自由形态生成器生成 $X^g$；(5) 三部分合并得到完整点云 $X$。

### 关键设计

1. **人体部件分割（Clothing-cut Map）**:

    - 功能：自动确定哪些区域该用 LBS 变形、哪些该用自由形态生成
    - 核心思路：先定位不受服装覆盖的裸露区域（头、手、脚）；然后利用 SAM 基础模型在渲染的法线图上分割宽松服装区域，反投影到 3D 空间标记出需要生成的区域；剩余为贴身变形区域
    - 设计动机：分割精度直接决定两个分支的协同效果。消融实验证明，没有分割引导时衣裙会在腿部区域撕裂，因为 LBS 和生成器会在同一区域产生冲突

2. **LBS 变形网络（贴身区域）**:

    - 功能：对靠近身体的衣物预测姿态依赖的变形
    - 核心思路：用 PointNet++ 从 posed body 提取多尺度局部姿态特征 $\phi_k^p$，通过重心坐标插值得到每个点的连续局部姿态编码 $z_i^p$。结合局部 + 全局服装编码，通过 Pose Decoder 预测 canonical 空间中的位移 $r_i^c$ 和法线 $n_i^c$，最终通过 LBS 变换 $T_i$ 转到 posed 空间：$x_i^d = T_i \cdot (p_i^c + r_i^c)$
    - 设计动机：局部姿态编码比全局编码能更好捕捉细粒度的衣物褶皱变化，这已被 CloSET 等前人工作验证

3. **自由形态生成器（宽松区域）**:

    - 功能：完全绕开 LBS，直接在 posed 空间生成宽松衣物点云
    - 核心思路：设计结构感知姿态编码——将人体分成 $K_b$ 个语义部件，用 PointNet++ 提取各部件特征，Max-Pooling 融合为全局姿态编码 $h^p$。生成器（基于修改版 SpareNet）以 $h^p$ 和全局服装编码 $h^g$ 为条件，直接生成 posed 空间的点集：$X^g = \mathcal{G}(h^p, h^g)$。全局服装编码 $h^g$ 与 LBS 分支共享，确保两个分支生成的衣物类型一致
    - 设计动机：宽松裙摆受骨骼运动影响小，更像是"根据当前姿态补全点云"的任务。部件级姿态编码比直接用全局姿态向量更好地捕捉了骨骼与裙摆形态的相关性（消融实验证明无此设计时裙摆朝向与姿态不匹配）

### 损失函数 / 训练策略

总损失为五项加权和：$\mathcal{L} = \lambda_{cd}\mathcal{L}_{cd} + \lambda_n\mathcal{L}_n + \lambda_{rd}\mathcal{L}_{rd} + \lambda_{rg}\mathcal{L}_{rg} + \lambda_{col}\mathcal{L}_{col}$。

- **Chamfer Distance $\mathcal{L}_{cd}$**：预测点云与 GT 的双向最近点距离
- **法线损失 $\mathcal{L}_n$**：L1 法线误差
- **位移正则 $\mathcal{L}_{rd}$**：约束贴身区域变形不要过大
- **服装编码正则 $\mathcal{L}_{rg}$**：防止编码过大
- **碰撞损失 $\mathcal{L}_{col}$**：利用身体 SDF 场惩罚生成点穿透身体表面，$\max\{\epsilon - d(x_j^g), 0\}$

端到端训练，两个分支的网络和服装编码联合优化。

## 实验关键数据

### 主实验

| Subject | FID↓ (Ours/POP) | MSE↓ (Ours/POP) | 说明 |
|---------|------|------|------|
| All | **37.75** / 57.87 | **2.61** / 2.88 | 整体 SOTA |
| felice-004 (最宽松) | **42.41** / 66.43 | 5.24 / 5.80 | 最难场景大幅领先 |
| janett-025 | **27.95** / 52.55 | **1.92** / 2.02 | 长裙优势明显 |

感知实验中 63.4% 人类评估者偏好本方法，GPT-4o 也给出 56% 偏好率。最宽松的两条裙子上超过 85% 评估者选择本方法。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅 LBS 变形 (a) | 裤子形撕裂 | 验证 LBS 的根本局限 |
| 仅下半身生成 (b) | 腿部噪声、不连续 | 完全不用 LBS 不行 |
| 无碰撞损失 (c) | 衣物穿透身体 | 碰撞约束必要 |
| 无部件级编码 (d) | 裙摆朝向错误 | 结构感知设计关键 |
| 无 Clothing-cut Map | 衣裙撕裂穿透 | 分割引导不可或缺 |
| Full model (e) | 最优质量 | 混合方案上限最高 |

### 关键发现

- 碰撞损失 + 姿态增强组合使用效果最好，说明生成器需要物理约束防止不合理形态
- Clothing-cut Map 的消融最为显著：没有它时 LBS 和生成器会在重叠区域产生严重冲突
- FITE 虽然定量接近，但存在"封闭表面"伪影（将开放裙摆错误封闭），本方法天然避免了此问题

## 亮点与洞察

- **混合建模范式的简洁有效**：不追求单一方法解决所有问题，而是根据区域特性分而治之。这个思路简单直觉但效果显著，核心在于 Clothing-cut Map 提供了合理的分界
- **自由形态生成绕开 LBS 的设计很启发性**：将宽松服装建模等价为"条件点云补全"任务，拓展了着装人体建模的方法论边界
- **结构感知姿态编码可迁移**：部件级特征提取 + Max-Pooling 的设计思路可应用于其他需要理解人体部件与附属物关系的任务，如手持物体生成、配饰建模

## 局限与展望

- 基于点云表示，视觉质量受点密度限制，难以表达极其精细的褶皱纹理
- Clothing-cut Map 依赖 SAM 预分割，对于新服装类型可能需要调整
- 只在合成数据集 ReSynth 上评估，缺少真实扫描数据验证
- 生成器是特定于每件服装的（通过服装编码），换装泛化能力未验证
- 未来可结合 3D Gaussian Splatting 实现有纹理的实时渲染，作者也提及了这一方向

## 相关工作与启发

- **vs POP**：POP 对所有区域使用 LBS 变形，宽松裙试图"拉伸"两腿间的点导致撕裂。FreeCloth 从根本上避免了这个问题
- **vs FITE**：FITE 学习隐式衣物模板并在其上做 LBS 粗到精细化，但隐式场难以处理开放表面，导致裙摆被"封闭"。FreeCloth 的点云生成天然支持开放拓扑
- **vs DPF**：DPF 完全摆脱 LBS 优化平滑变形场，但需要逐帧优化不实用。FreeCloth 的混合方案既保留了前馈推理效率又获得了灵活性

## 评分

- 新颖性: ⭐⭐⭐⭐ 混合建模思路直觉自然，但"分区域策略"本身不算全新，核心贡献在于系统化实现和自由形态生成器设计
- 实验充分度: ⭐⭐⭐⭐ 消融全面、感知实验有说服力，但只在合成数据集上评估是遗憾
- 写作质量: ⭐⭐⭐⭐ 图表制作精美，动机阐述清晰，方法描述有条理
- 价值: ⭐⭐⭐⭐ 对宽松服装建模的实际问题提供了有效解决方案，开拓了混合建模新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] TELA: Text to Layer-wise 3D Clothed Human Generation](../../ECCV2024/human_understanding/tela_text_to_layer-wise_3d_clothed_human_generation.md)
- [\[CVPR 2025\] RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars](rgbavatar_reduced_gaussian_blendshapes_for_online_modeling_of_head_avatars.md)
- [\[CVPR 2025\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[CVPR 2025\] Any6D: Model-free 6D Pose Estimation of Novel Objects](any6d_model-free_6d_pose_estimation_of_novel_objects.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)

</div>

<!-- RELATED:END -->
