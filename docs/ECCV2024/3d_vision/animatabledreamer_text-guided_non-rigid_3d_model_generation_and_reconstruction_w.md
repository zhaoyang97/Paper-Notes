# AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation

**会议**: ECCV 2024  
**arXiv**: [2312.03795](https://arxiv.org/abs/2312.03795)  
**代码**: https://zz7379.github.io/AnimatableDreamer/ (有)  
**领域**: 3D视觉 / 生成模型  
**关键词**: Text-to-4D, Non-rigid 3D, Score Distillation, Skeleton Animation, Canonical Space

## 一句话总结
提出 AnimatableDreamer，通过 Canonical Score Distillation (CSD) 技术，从单目视频提取骨骼和运动后生成文本引导的可动画化 3D 非刚体模型，在生成质量和时序一致性上全面超越现有方法。

## 研究背景与动机
1. **领域现状**：文本到 3D 生成（SDS/DreamFusion）已能生成高质量静态 3D 物体，但可变形/非刚体物体的生成仍然困难。现有方法要么限于特定类别（人体），要么无法保证不同姿态下的形态一致性。

2. **现有痛点**：
   - 直接对可动画物体施加 vanilla SDS 会破坏运动一致性——不同姿态下生成的表面不连贯
   - 从单目视频重建可变形物体时，不可见区域（如动物的另一侧）几何质量差
   - 现有方法（BANMo）需要多段视频才能获取好的 3D 重建

3. **核心矛盾**：想生成可动画化的非刚体 3D 模型，但 SDS 监督在 canonical space 和观测空间之间缺乏一致性桥梁

4. **本文要解决什么？** (a) 如何从单目视频提取可复用的骨骼/蒙皮；(b) 如何在骨骼约束下用文本引导生成新的可动画 3D 模型

5. **切入角度**：将变形场（warping field）作为 canonical space 和观测空间的桥梁，让扩散模型的梯度通过 warp 传回 canonical model

6. **核心 idea 一句话**：Canonical Score Distillation——在被观测的变形姿态上计算扩散先验梯度，通过可微 warp 反传到 canonical 模型，确保所有姿态下的一致性

## 方法详解

### 整体框架
两阶段：(1) 提取阶段——从单目视频学习隐式关节模型（NeuS + 骨骼变形场），用 CSD 增强不可见区域；(2) 生成阶段——在提取的骨骼上用 CSD + MVDream 生成文本引导的新 3D 模型

### 关键设计

1. **Implicit Articulate Model**:
   - 做什么：用 NeuS（SDF + 颜色 + 特征描述子）表示 canonical 模型，blend skinning 做变形
   - 核心思路：B 根骨骼用高斯分布建模蒙皮权重，通过 Mahalanobis 距离计算。骨骼变换用 MLP + Fourier 时间嵌入学习
   - 设计动机：canonical space 保证时序一致性，骨骼蒙皮提供可控的动画能力

2. **Skeleton Construction**:
   - 做什么：从学到的骨骼模型构建结构化骨架，约束生成过程
   - 核心思路：用 DINOv2 特征计算骨骼间的语义关联 + 蒙皮权重计算形态关联，综合得到骨骼连接强度 $\mathcal{T}_{j,k}$。施加平移/角度约束防止不合理变形
   - 设计动机：骨架约束确保生成的新物体保持合理的运动模式

3. **Canonical Score Distillation (CSD)** ← 核心贡献:
   - 做什么：让扩散模型的监督信号通过 warp 机制传回 canonical 模型
   - 核心思路：$\nabla_\phi \mathcal{L}_{CSD} = \mathbb{E}[\underbrace{(\epsilon_\theta - \epsilon)}_{\text{扩散先验}} \cdot \underbrace{\frac{\partial \mathcal{R}(\mathbf{X}_*)}{\partial \mathbf{X}_*}}_{\text{Canonical 渲染}} \cdot \underbrace{\frac{\partial W(\mathbf{X}^t)}{\partial \phi_w}}_{\text{Warp 精化}}]$
   - 三个梯度项：扩散先验提供外观指导 → canonical 渲染保证一致性 → warp 精化优化变形参数
   - 使用 MVDream（多视角扩散模型）同时渲染 4 个正交视角，保证 3D 一致性
   - 设计动机：普通 SDS 只在观测空间计算梯度，不保证 canonical space 一致。CSD 通过 warp 链式法则确保梯度传到 canonical model

### 损失函数 / 训练策略
- 提取阶段：$\mathcal{L}_{Ext} = \mathcal{L}_{recon}(\text{RGB+轮廓+光流}) + \mathcal{L}_{CSD} + \mathcal{L}_{reg}(\text{特征匹配+循环一致性})$
- 生成阶段：$\mathcal{L}_{Gen} = \mathcal{L}_{skel} + \mathcal{L}_{bone} + \mathcal{L}_{CSD} + \mathcal{L}_{reg}$
- 渲染 200×200 分辨率，4 个正交视图
- 单 A800 GPU 训练约 5 小时，12000 迭代

## 实验关键数据

### 生成任务

| 方法 | CLIP↑ | CLIP-T↑ | R-Precision@10↑ | GPT Eval3D↑ |
|------|-------|---------|-----------------|-------------|
| ProlificDreamer | 33.1 | 95.9 | 56.3 | 959 |
| MVDream | 34.8 | 94.4 | 31.2 | 979 |
| **AnimatableDreamer** | **38.2** | **96.6** | **87.5** | **1098** |

### 单目重建任务（Chamfer Distance ↓ / F-score@2% ↑）

| 方法 | 视频数 | Cat-Coco | Cat-Pikachu | Penguin | Shiba |
|------|--------|----------|-------------|---------|-------|
| BANMo | 1 | 10.7/15.3 | 3.71/57.3 | 6.47/43.9 | 6.81/36.6 |
| RAC | 1 | 6.25/42.2 | 3.60/60.2 | 4.68/53.7 | 7.94/30.1 |
| **Ours** | **1** | **3.65/63.3** | **2.0/88.9** | **3.7/64.0** | **4.54/53.9** |

### 消融实验

| 配置 | CLIP↑ | R-Precision↑ | 说明 |
|------|-------|-------------|------|
| w/o bone+skel | 27.1 | 35.6 | 无骨骼约束，完全崩溃 |
| w/o skel | 28.4 | 40.1 | 无骨架约束，运动塌缩 |
| w/o bone | 37.8 | 81.7 | 无骨骼表面约束，质量略降 |
| **Full model** | **38.2** | **87.5** | 完整模型最优 |

### 关键发现
- **CSD 对重建的贡献巨大**：Cat-Coco 上无 CSD 时 CD 从 3.65→8.34（+128%），F-score 从 63.3→32.6（-48%）
- **R-Precision 提升最显著**：87.5% vs MVDream 31.2%，说明骨骼约束极大提升了文本-模型一致性
- **单目重建超越多视频方法**：在 Cat-Coco 上比 4 视频 BANMo（4.66）还好（3.65），说明 CSD 有效弥补了单视角信息不足

## 亮点与洞察
- **CSD 的关键 insight**：将 warp 作为 canonical space 和观测空间的可微桥梁——这个思路可以推广到任何需要在变换空间上做 SDS 的场景（如布料模拟、流体等）
- **骨骼约束的两层设计**：语义关联（DINOv2 特征相似度）+ 形态关联（蒙皮权重重叠度）共同决定骨骼连接强度——比纯几何或纯语义的方式更鲁棒
- **生成+重建统一框架**：同一套 CSD 技术既能用于从视频重建（增强不可见区域），又能用于从文本生成（保证动画一致性）

## 局限性 / 可改进方向
- **显存消耗大**：CSD 需要从相机空间到 canonical 空间的长梯度链 + MVDream 同时处理 4 张图，显存需求高
- **分辨率受限**：只能渲染 200×200，限制了细节质量
- **仅限骨骼驱动的变形**：无法处理拓扑变化（如物体分裂）或布料等非骨骼驱动的变形
- **依赖视频质量**：骨骼提取依赖 HOI 检测和光流质量，视频遮挡严重时可能失败

## 相关工作与启发
- **vs DreamFusion/ProlificDreamer**: 这些生成静态 3D，本文生成可动画化 3D。CSD 是 SDS 的自然扩展
- **vs BANMo/RAC**: 这些纯重建方法依赖多视频。本文用 CSD 的扩散先验弥补单视频的信息缺失
- **vs 4D generation (MAV3D等)**: MAV3D 直接在 4D NeRF 上做 SDS，没有 canonical space 保证一致性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ CSD 概念新颖，将 warp 作为 SDS 的可微桥梁是重要贡献
- 实验充分度: ⭐⭐⭐⭐ 生成+重建双任务验证，消融详细，但数据集规模较小
- 写作质量: ⭐⭐⭐⭐ 方法图清晰，CSD 公式推导清楚
- 价值: ⭐⭐⭐⭐⭐ 开辟了可动画化 3D 生成的新范式，CSD 思路可广泛复用
