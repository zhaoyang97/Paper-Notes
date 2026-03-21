# Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos

**会议**: CVPR 2025  
**arXiv**: [2603.13185](https://arxiv.org/abs/2603.13185)  
**代码**: https://github.com/rohithpeddi/WorldSGG  
**领域**: 3D视觉  
**关键词**: 场景图生成, 4D场景理解, 物体永恒性, 世界状态建模, 单目视频

## 一句话总结
本文提出 World Scene Graph Generation (WSGG) 任务和 ActionGenome4D 数据集，将视频场景图从以帧为中心的 2D 表示升级为以世界为中心的 4D 表示，要求模型对所有物体（包括被遮挡或离开视野的不可见物体）在世界坐标系中进行 3D 定位和关系预测，并提出三种互补方法（PWG/MWAE/4DST）探索不同的不可见物体推理归纳偏置。

## 研究背景与动机
1. **领域现状**：视频场景图生成（VidSGG）已有 Action Genome 等数据集和 Transformer 方法，3D/4D 场景图也有探索。
2. **现有痛点**：现有方法本质上是"以帧为中心"——只对当前可见物体建模，物体一旦被遮挡或离开视野就从图中消失。这与真实世界代理的认知方式（物体永恒性）不符。
3. **核心矛盾**：具身代理需要维持对所有环境中物体的持续认知（包括不可见的），但现有数据集和任务都缺少（1）3D 世界坐标系定位、（2）跨帧物体一致性追踪、（3）不可见物体的关系标注。
4. **本文要解决什么？** 如何从单目视频构建时间持续的、世界锚定的场景图，涵盖所有交互物体（可见+不可见）？
5. **切入角度**：将认知科学中的"物体永恒性"引入场景图生成——物体即使不可见也继续存在于世界状态中，需要推理其关系。
6. **核心idea一句话**：场景图从 frame-centric 2D upgrade 到 world-centric 4D，核心挑战在于不可见物体的表征与关系推理。

## 方法详解

### 整体框架
输入单目视频 → $\pi^3$ 前馈 3D 重建 + BA 位姿精化 → 世界坐标系中的 3D OBB（GDINO 检测 + SAM2 分割 + PCA OBB + Kalman 平滑）→ 三种 WSGG 方法处理不可见物体 → 输出每帧世界场景图 $\mathcal{G}_{\mathcal{W}}^t$（包含 attention/spatial/contacting 三类关系）。

### ActionGenome4D 数据集
- 在 Action Genome 视频基础上用 $\pi^3$ 进行 3D 重建
- GDINO + SAM2 + 地面对齐 + PCA OBB + Kalman 滤波生成 3D 有向包围盒
- VLM 生成不可见物体关系的伪标注 + 人工修正
- 谓词集：attention (3)、spatial (6)、contacting (17)
- 覆盖可见-可见、可见-不可见、不可见-不可见的所有物体对

### 三种互补方法

1. **PWG (Persistent World Graph)**:
   - 做什么：通过 Last-Known-State (LKS) 缓冲区实现物体永恒性
   - 核心思路：不可微分的零阶保持——冻结每个物体最后一次可见时的 DINO 特征，带有过期度 $\Delta_n^{(t)} = |t - \tau^*|$。将几何/特征/相机/过期度融合后通过 Spatial GNN + 关系预测器
   - 设计动机：最简单的物体永恒性实现——既然物体存在过，就保留其特征

2. **MWAE (Masked World Auto-Encoder)**:
   - 做什么：将遮挡/离场视为自然遮蔽，用 MAE 框架重建不可见物体表征
   - 核心思路：mask 不可见物体的视觉流，Associative Retriever 用不对称交叉注意力（所有 token query，只有可见 token 作 key/value）重建缺失特征。训练时用模拟遮挡和交叉视角重建作为监督
   - 设计动机：不是简单冻结旧特征，而是根据当前场景上下文推断不可见物体可能的当前状态

3. **4DST (4D Scene Transformer)**:
   - 做什么：用可微分时间 Transformer 替代静态缓冲区，联合处理所有时间步的可见/不可见 token
   - 核心思路：在 Fusion Node 融合多模态 token（视觉+3D 结构+运动+相机位姿），然后无遮蔽双向时间自注意力 + Spatial GNN，输出全局感知的时空表征
   - 设计动机：允许不可见物体从所有历史帧中提取信息，而非仅从最后可见帧

### 共享组件
- **Global Structural Encoder**：OBB 8 角点 → MLP → 结构 token
- **Spatial Positional Encoding**：物体对的欧氏距离+方向+体积比
- **Camera/Motion Encoder**：6D 旋转表示 + 帧间相对位姿 + 物体 3D 速度/加速度
- **关系预测器**：CLIP text embedding + union ROI 特征 → 三头预测（attention/spatial/contacting）

## 实验关键数据

### 主实验（ActionGenome4D, PredCls）

| 方法 | Backbone | R@10 (WC) | R@20 (WC) | R@50 (NC) | R@50 (NC) |
|------|----------|-----------|-----------|-----------|-----------|
| PWG | DINOv2-L | 65.07 | 67.99 | 94.39 | 99.59 |
| MWAE | DINOv2-L | — | — | — | — |
| 4DST | DINOv2-L | — | — | — | — |

### 消融实验
- 3D 几何特征（OBB）对空间关系预测贡献最大
- 相机位姿编码对不可见物体推理至关重要
- 运动特征对 contacting 关系预测有显著帮助
- VLM 伪标注质量经人工修正后与 GT 接近

### 关键发现
- 不可见物体的关系预测是核心挑战——所有方法在不可见物体对上的 Recall 远低于可见物体对
- PWG 虽简单但在很多配置下表现竞争力强，说明"冻结最后可见特征"是合理的基线
- VLM（Graph RAG-based）在非定位关系预测上有潜力但仍不如专用方法
- 3D 定位是 SGDet 模式下的主要瓶颈

## 亮点与洞察
- **从 frame-centric 到 world-centric 的范式转换**：将"物体永恒性"这一认知科学概念系统化地引入场景图生成，定义清晰的 WSGG 任务
- **三种互补归纳偏置的探索**：PWG（特征冻结）、MWAE（遮蔽重建）、4DST（全时间注意力）代表了从简单到复杂的不可见物体推理谱
- **ActionGenome4D 数据集**：将 Action Genome 升级为 4D 场景，foundation model 驱动的标注流水线（$\pi^3$ + GDINO + SAM2 + VLM）可复用于其他数据集
- **可迁移思路**：world-centric 场景图可以直接服务于具身 AI 的规划和推理

## 局限性 / 可改进方向
- ActionGenome4D 基于 Action Genome 的室内场景，场景多样性有限（主要是家庭活动场景）
- 3D 重建依赖 $\pi^3$ 的质量——在纹理不良或快速运动场景下可能退化，影响 OBB 精度
- 不可见物体的关系伪标注依赖 VLM，即使经人工修正仍可能有偏——特别是长时间不可见的物体
- 仅考虑人-物体交互关系，物体-物体关系未涉及（如"杯子在桌子上"）
- 未与实际具身任务（如导航/操作）集成验证下游价值
- 计算成本较高：$\pi^3$ 重建 + GDINO 检测 + SAM2 分割 + Kalman 平滑的全流水线开销显著

## 相关工作与启发
- **vs Action Genome (VidSGG)**: AG 仅标注可见物体的关系。ActionGenome4D 扩展到不可见物体，从 2D 升级到 3D
- **vs 3D SGG (ScanNet)**: 3D SGG 处理静态 3D 扫描，无时间维度。WSGG 处理动态视频中的时序关系变化
- **vs 4D Panoptic SG**: 4D PSG 仍受限于相机可见范围。WSGG 显式建模不可见物体

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 定义了全新的 WSGG 任务，将物体永恒性引入场景图是原创性贡献
- 实验充分度: ⭐⭐⭐⭐ 三种方法+VLM baseline+消融+两种评估模式
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义严谨，数学形式化完整，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ 新数据集+新任务+新方法，对具身 AI 有重要意义
