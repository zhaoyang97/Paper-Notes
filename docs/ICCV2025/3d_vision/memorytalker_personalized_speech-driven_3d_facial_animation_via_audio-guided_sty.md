---
title: >-
  [论文解读] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization
description: >-
  [3D视觉] 提出 MemoryTalker，通过两阶段训练策略（Memorizing + Animating）利用键值记忆网络存储通用面部运动，并通过音频驱动的风格化记忆实现仅凭音频即可生成个性化 3D 面部动画，无需任何额外先验信息。 语音驱动的 3D 面部动画旨在从语音信号合成与说话人风格匹配的面部运动序列…
tags:
  - "3D视觉"
---

# MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization

- **会议**: ICCV 2025
- **arXiv**: [2507.20562](https://arxiv.org/abs/2507.20562)
- **领域**: 3D视觉
- **关键词**: Speech-Driven 3D Facial Animation, Memory Network, Speaking Style, Personalization, Key-Value Memory

## 一句话总结

提出 MemoryTalker，通过两阶段训练策略（Memorizing + Animating）利用键值记忆网络存储通用面部运动，并通过音频驱动的风格化记忆实现仅凭音频即可生成个性化 3D 面部动画，无需任何额外先验信息。

## 研究背景与动机

语音驱动的 3D 面部动画旨在从语音信号合成与说话人风格匹配的面部运动序列，是 VR 远程呈现、角色动画等沉浸式应用的关键技术。核心挑战在于：不仅需要精确的语音-运动同步，还要捕捉不同说话人的个人风格（嘴巴张合幅度、嘟嘴程度等）。

现有方法存在两大限制：

**One-hot 编码方法**（FaceFormer、CodeTalker 等）：用训练集中说话人的 ID 向量表示风格，推理时无法处理未见过的说话人，且同一音频使用不同 ID 会产生不同结果，不具泛化性。

**参考 3D 网格方法**（Imitator、Mimic、Yang et al.）：推理时需要额外提供一段目标说话人的 3D 面部运动序列来编码风格，但在实际应用中获取这些数据极不实用。

本文的核心目标是：**仅用音频输入，无需任何额外先验（ID 标签、参考 3D 网格），即可生成反映说话人个人风格的 3D 面部动画**。这是首个在推理时不需要额外先验信息的个性化语音驱动 3D 面部动画方法。

## 方法详解

### 整体框架

MemoryTalker 采用两阶段训练策略：

- **第一阶段（Memorizing）**：构建面部运动记忆网络，存储和召回与语音对应的通用面部运动
- **第二阶段（Animating）**：学习从音频中提取说话风格特征，将通用运动记忆风格化为个性化记忆，生成个性化面部动画

### 关键设计

**1. 面部运动记忆网络（Motion Memory）**

设计键值记忆 $\mathbf{M}_m \in \mathbb{R}^{n \times c}$，含 $n$ 个槽位、$c$ 维通道：

- **写入**：运动编码器 $E_m$ 将面部运动 $v^t$ 编码为特征 $f_m^t$，通过注意力机制计算与各槽位的相似度作为值地址向量 $\mathbf{V}_m^t$，加权求和得到召回特征 $\hat{f}_{m,val}^t$
- **读取**：利用预训练 ASR 模型（HuBERT）提取文本表征 $f_{txt}^t$ 作为查询键，经 softmax 得到键地址向量 $\mathbf{K}_{txt}^t$，从记忆中召回通用运动特征 $\hat{f}_{m,key}^t$

使用文本表征而非直接用音频特征作为查询的关键原因：同一音素不同说话人的音频风格差异大，但文本表征可以将这种差异抹平，映射到一致的面部运动（例如所有说"who"的人嘴唇都先合拢再圆唇）。

**2. 风格化记忆网络（Stylized Motion Memory）**

从音频的梅尔频谱中通过风格编码器 $E_s$ 提取说话风格特征 $f_s$。设计风格权重 $\tilde{w}_s$ 对每个记忆槽位进行加权：

$$\tilde{w}_s = \text{sigmoid}(\psi'_{\rightarrow n}(f_s)) \cdot \psi_{\rightarrow 1}(f_s)$$

其中 sigmoid 部分对每个槽位评分，标量缩放因子控制整体强度。通过风格权重将通用记忆 $\mathbf{M}_m$ 转化为风格化记忆 $\tilde{\mathbf{M}}_m$，使不同说话人的同一音素产生不同幅度的面部运动。

**3. 运动解码器**

基于 Transformer 解码器结构，以文本表征和召回的运动特征拼接为输入，生成最终的 3D 面部运动：

$$\hat{v}^t = D_m([f_{txt}^t; \hat{f}_{m,key}^t], f_{txt}^t)$$

### 损失函数

**第一阶段**：
$$\mathcal{L}_{1\text{-stage}} = \mathcal{L}_{mse} + \mathcal{L}_{vel} + \lambda_1(\mathcal{L}_{mem} + \mathcal{L}_{align})$$

- $\mathcal{L}_{mse}$：运动重建损失（预测与真值的 L2 距离）
- $\mathcal{L}_{vel}$：速度损失（解决帧间抖动）
- $\mathcal{L}_{mem}$：记忆重建损失（确保运动信息写入记忆）
- $\mathcal{L}_{align}$：KL 散度对齐损失（对齐文本键地址与运动值地址）
- $\lambda_1 = 0.01$

**第二阶段**（冻结第一阶段所有参数，仅训练风格编码器）：
$$\mathcal{L}_{2\text{-stage}} = \mathcal{L}_{mse} + \mathcal{L}_{vel} + \lambda_2(\mathcal{L}_{lip} + \mathcal{L}_{style})$$

- $\mathcal{L}_{lip}$：唇部顶点损失（聚焦下半脸区域的精细运动）
- $\mathcal{L}_{style}$：三元组损失（拉近同一说话人、推远不同说话人的风格特征）
- $\lambda_2 = 0.01$

## 实验关键数据

### 主实验表格（VOCASET 数据集）

| 方法 | FVE↓(×10⁻⁶) | LVE↓(×10⁻⁵) | FID↓(×10⁻¹) | LDTW↓(×10⁻⁵) | Lip-max↓(×10⁻⁴) |
|------|-------------|-------------|-------------|--------------|-----------------|
| FaceFormer | 0.639 | 0.413 | 3.583 | 0.507 | 0.452 |
| CodeTalker | 0.721 | 0.498 | 3.713 | 0.554 | 0.484 |
| SelfTalk | 0.593 | 0.382 | 3.279 | 0.475 | 0.416 |
| UniTalker | 0.570 | 0.382 | 3.256 | 0.507 | 0.407 |
| **MemoryTalker** | **0.506** | **0.293** | **3.045** | **0.418** | **0.331** |

在 VOCASET 上所有指标全面 SOTA，FVE 降低 11.2%，LVE 降低 23.3%。

### BIWI 数据集

| 方法 | FVE↓(×10⁻⁴) | LVE↓(×10⁻⁴) | FID↓(×10⁻¹) |
|------|-------------|-------------|-------------|
| UniTalker | 0.919 | 0.196 | 7.234 |
| **MemoryTalker** | **0.901** | **0.187** | **7.202** |

跨数据集同样最优。

### 效率对比

| 方法 | 推理时间 | 参数量 |
|------|----------|--------|
| CodeTalker | 297.6 ms | 315M |
| SelfTalk | 10.1 ms | 450M |
| UniTalker | 9.7 ms | 313M |
| **MemoryTalker** | **7.8 ms** | **94M** |

推理速度约 120 fps，参数量仅 94M，兼具效率和性能。

### 消融实验

| 配置 | FVE↓ | LVE↓ |
|------|------|------|
| 无记忆网络（baseline） | 0.638 | 0.460 |
| + 第一阶段（记忆） | 0.531 | 0.313 |
| + 第二阶段（风格化） | **0.506** | **0.293** |

两阶段均有明显贡献。去掉 $\mathcal{L}_{style}$（三元组损失）训练不稳定、性能显著下降，说明风格区分至关重要。

### 用户研究

与 5 种 SOTA 方法比较（33 名参与者），在唇同步、真实感和说话风格三项上均获 >79% 的偏好率。

### 关键发现

- **t-SNE 可视化**：第一阶段召回的运动特征各说话人混在一起（通用运动），第二阶段各说话人清晰聚类（个性化风格被成功捕捉）
- **One-hot 方法的内在缺陷**：同一音频用不同训练集 ID 推理，FaceFormer 的 FVE 标准差为 0.036、CodeTalker 为 0.056，说明 one-hot 编码无法稳定表示说话风格

## 亮点与洞察

1. **首个仅用音频实现个性化的方法**：不需要 ID 标签或参考 3D 网格，极大提升实用性
2. **键值记忆桥接模态鸿沟**：巧妙利用文本表征作为键来消除音频中的风格差异，再通过风格化权重注入个性化
3. **两阶段解耦设计**：先学通用运动再学个性化，避免了两个目标的冲突
4. **极致效率**：94M 参数 + 7.8ms 推理，适合实时 VR/元宇宙部署
5. **风格权重的设计**：sigmoid 评分 × 标量缩放，简洁有效地控制每个记忆槽位的个性化程度

## 局限性

1. 仅建模嘴部和面部运动，未涵盖眼部运动、眉毛表情等上半脸区域
2. 依赖预训练 ASR 模型（HuBERT）的质量，对非英语语音的泛化性未验证
3. 实验数据集规模较小（VOCASET 仅 12 人 480 序列），大规模泛化性存疑
4. 未处理情感语音的情况，仅针对中性说话风格
5. 记忆网络槽位数 $n$ 的选择缺乏系统研究

## 相关工作

- **One-hot 方法**：FaceFormer、CodeTalker 用训练集 ID 编码风格，无法泛化到新说话人
- **参考运动方法**：Imitator（参考 2D 视频）、Mimic（参考 3D 运动做风格-内容解耦）、Yang et al.（渐进式风格注入），但推理时均需额外输入
- **记忆网络**：在目标跟踪、少样本学习、异常检测中广泛应用，本文首次将键值记忆网络用于跨模态的语音-3D 运动对齐

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次实现仅音频驱动的个性化 3D 面部动画，记忆网络的跨模态桥接设计新颖
- **实用性**: ⭐⭐⭐⭐⭐ — 无需额外先验、极低延迟、小参数量，直接可部署
- **实验质量**: ⭐⭐⭐⭐ — 定量+定性+用户研究全面，消融充分，但数据集规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图示直观，动机论证有力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](../../CVPR2025/3d_vision/multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)

</div>

<!-- RELATED:END -->
