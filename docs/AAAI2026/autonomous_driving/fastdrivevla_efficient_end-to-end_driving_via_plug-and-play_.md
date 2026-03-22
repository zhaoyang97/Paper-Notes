# FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning

**会议**: AAAI 2026  
**arXiv**: [2507.23318](https://arxiv.org/abs/2507.23318)  
**代码**: 未公开  
**领域**: 多模态 VLM / 自动驾驶  
**关键词**: VLA 模型加速, 视觉 token 剪枝, 前景重建, 自动驾驶, plug-and-play

## 一句话总结

提出 FastDriveVLA，通过 MAE 风格的前景像素重建训练轻量级 plug-and-play 的 ReconPruner 模块（仅 0.07B），利用对抗前景-背景重建策略优先保留驾驶决策所需的前景 token，在 nuScenes 开环规划基准上各剪枝率均达 SOTA，一次训练可迁移至同一视觉编码器的不同 VLA 模型。

## 研究背景与动机

Vision-Language-Action（VLA）模型在端到端自动驾驶中展现了强大的场景理解和动作推理能力，但视觉编码器产生的大量 token（如 3249 个）带来巨大计算开销和推理延迟，严重制约实车部署。现有 VLM token 剪枝方法存在两个根本问题：

1. **基于注意力的方法**（FastV, SparseVLM）：依赖文本-视觉注意力分数评估 token 重要性，但驾驶场景中指令固定且简洁，无法提供有效的 token 选择引导
2. **基于相似度的方法**（VisPruner, DivPrune）：通过 token 多样性选择子集，但驾驶场景中前景区域（车道、行人、车辆）才是决策关键，基于相似度可能错误保留大量无用背景 token

核心 insight：**人类驾驶员专注于相关前景区域**——保留编码前景信息的视觉 token 是有效决策的关键，背景 token 可以安全丢弃。

## 方法详解

### 整体框架

训练一个轻量 ReconPruner 模块（0.07B 参数），插在 VLA 的视觉编码器之后。通过 MAE 风格重建估计每个 token 的前景显著性分数，按 Top-K 策略保留高分 token、丢弃低分 token。训练完成后可即插即用到共享同一视觉编码器的不同 VLA 模型，无需重训。

### 关键设计

1. **ReconPruner 架构**：由一个 PrunerLayer（Qwen2.5-VL-3B 的单个解码器层）和一个 Scorer（$\mathbb{R}^{D \times 1}$ 的线性层）组成。引入可学习查询 token $Q \in \mathbb{R}^{1 \times D}$，与视觉 token 联合输入 PrunerLayer，通过 Hadamard 积融合后由 Scorer 输出显著性分数 $S \in \mathbb{R}^{N \times 1}$
2. **对抗前景-背景重建策略**：仅靠前景重建会导致退化解——ReconPruner 给所有 token 都打高分来提升重建性能。受 GAN 启发，额外要求用低分 token 重建背景区域，形成对抗约束：前景 token 重建前景→质量高，背景 token 重建背景→质量也高，迫使模型学会区分。使用 STE（Straight-Through Estimator）解决二值 mask 不可微的问题
3. **nuScenes-FG 数据集**：定义驾驶场景前景（人、道路、车辆、交通标志、交通护栏），用 Grounded-SAM 对 nuScenes 进行分割标注，构建 241K 图像-mask 对，覆盖六个相机视角
4. **Plug-and-Play 泛化**：一次针对特定视觉编码器（如 CLIP-ViT）训练 ReconPruner，可迁移到使用相同编码器的 Impromptu-VLA 等不同 VLA 模型

### 损失函数

$$\mathcal{L}_{all} = \alpha \mathcal{L}_{fore} + (1-\alpha) \mathcal{L}_{back}, \quad \alpha=0.5$$

前景/背景损失均为 MSE + SSIM 加权组合（$\lambda=0.2$）。重建解码器由 6 层 Qwen2.5-VL-3B 解码器 + 前馈重建头组成。

## 实验关键数据

### 主实验：nuScenes 开环规划（基于 Impromptu-VLA）

| 方法 | 保留 token | L2 Avg (cm)↓ | Collision Avg (%)↓ | Intersection Avg (%)↓ | 相对性能 |
|------|-----------|-------------|-------------------|----------------------|---------|
| 原始（100%=3249） | 3249 | 31.83 | 0.24 | 2.80 | 100% |
| FastV (↓25%) | 2436 | 32.29 | 0.31 | 2.87 | 98.6% |
| SparseVLM (↓25%) | 2436 | 32.18 | 0.28 | 2.81 | 98.9% |
| DivPrune (↓25%) | 2436 | 32.24 | 0.30 | 2.86 | 98.7% |
| **FastDriveVLA (↓25%)** | **2436** | **31.80** | **0.26** | **2.77** | **100.1%** |
| FastV (↓50%) | 1624 | 32.59 | 0.33 | 2.99 | 97.7% |
| VisPruner (↓50%) | 1624 | 32.25 | 0.27 | 2.95 | 98.7% |
| **FastDriveVLA (↓50%)** | **1624** | **32.10** | **0.25** | **2.94** | **99.1%** |

### 消融实验：关键设计贡献

| 配置 | Collision Avg (%) | 说明 |
|------|-------------------|------|
| 仅前景重建 | 较高 | 退化解：所有 token 都获高分 |
| + 对抗背景重建 | 显著下降 | 有效区分前景/背景 |
| + nuScenes-FG 前景 mask | 最优 | 高质量标注进一步提升 |
| Plug-and-play 迁移 | 持平原始训练目标 | 验证跨 VLA 迁移能力 |

### 关键发现

- **25% 剪枝时几乎无损**：FastDriveVLA 在剪掉 25% token 后 L2 误差甚至低于未剪枝模型（31.80 vs 31.83），Collision 率从 0.24% 仅升至 0.26%
- **50% 剪枝时优势显著**：对比其他方法在 50% 剪枝时 Collision 率 0.27-0.33%，FastDriveVLA 仅 0.25%，几乎零退化
- **前景感知 > 通用剪枝**：所有通用 VLM 剪枝方法在驾驶场景中效果均不如前景感知策略

## 亮点与洞察

- **驾驶场景特定设计**：从人类驾驶直觉出发（关注前景忽略背景），将 domain knowledge 注入 token 剪枝策略，思路清晰且有效
- **MAE 重建作为前景检测代理**：巧妙避免了额外的检测模型——前景 token 能重建出有意义的像素，背景 token 重建结果平坦，利用重建能力差异进行区分
- **对抗训练解决退化解**：GAN 思想的精妙应用——不是做生成对抗，而是做前景/背景的重建对抗
- **极轻量设计**：ReconPruner 仅 0.07B 参数（PrunerLayer + Scorer），几乎不增加推理开销

## 局限性

- 前景定义是静态的（预定义类别），未考虑动态重要性变化（如突然冲出的行人应获更高权重）
- 仅在 nuScenes 上验证，未涉及 Waymo、KITTI 等其他驾驶数据集
- nuScenes-FG 依赖 Grounded-SAM 自动标注，标注质量可能存在噪声
- 未分析具体的推理加速比和 FPS 提升
- 重建解码器（6 层 Qwen2.5-VL-3B）在训练阶段引入额外开销，虽然推理时不需要

## 相关工作

| 方法类别 | 代表方法 | 剪枝准则 | 驾驶场景表现 |
|----------|---------|---------|-------------|
| 基于注意力 | FastV, SparseVLM | 文本-视觉注意力分数 | 差：驾驶指令固定简洁 |
| 基于相似度 | VisPruner, DivPrune | token 多样性 | 差：保留无关背景 |
| 投影器压缩 | TokenPacker, Matryoshka | 重训整个模型 | 成本高，不 plug-and-play |
| **前景重建（本文）** | **FastDriveVLA** | **前景显著性分数** | **最优：保留决策关键 token** |

## 评分

- 新颖性: ⭐⭐⭐⭐ 前景重建剪枝思路新颖，对抗策略巧妙
- 实验充分度: ⭐⭐⭐⭐ 多剪枝率对比，消融完整
- 写作质量: ⭐⭐⭐⭐ motivation 从人类直觉出发清晰
- 价值: ⭐⭐⭐⭐ 对 VLA 模型实车部署有直接实用价值
