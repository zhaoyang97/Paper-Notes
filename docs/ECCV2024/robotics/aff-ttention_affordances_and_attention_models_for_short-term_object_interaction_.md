# AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation

**会议**: ECCV 2024  
**arXiv**: [2406.01194](https://arxiv.org/abs/2406.01194)  
**代码**: https://github.com/lmur98/AFFttention (有)  
**领域**: 视频理解 / Agent  
**关键词**: Short-Term Anticipation, Affordances, Egocentric Video, Attention, Object Interaction

## 一句话总结
提出 STAformer 架构和两个基于 affordance 的模块（环境 affordance 数据库 + 交互热点），将第一人称视频中的短期物体交互预测（STA）在 Ego4D 和 EPIC-Kitchens 上提升了 30-45% 的相对性能。

## 研究背景与动机
1. **领域现状**：短期物体交互预测（STA）要求从第一人称视频中同时预测下一个交互物体的位置（bbox）、类别（noun）、动作（verb）和接触时间（time-to-contact）。这是可穿戴助手和人机交互的关键能力。

2. **现有痛点**：
   - 之前方法（StillFast、TransFusion、GANO）大多基于卷积或简单特征融合，视频动态建模不充分
   - 图像分支和视频分支的融合方式粗糙（求和或拼接），没有充分利用两者的互补信息
   - 预测缺乏对人类行为模式的先验约束，容易产生与场景功能不符的误检

3. **核心矛盾**：STA 需要同时理解高分辨率空间细节（检测物体位置）和时序动态（预测未来动作），但现有方法在这两个维度上的融合是浅层的

4. **本文要解决什么？** (a) 如何更好地融合图像和视频特征用于 STA？(b) 如何利用环境中的交互先验来约束预测？

5. **切入角度**：从两个互补方向出发——结构化的 attention 融合（STAformer）+ 基于人类行为规律的 affordance 先验（Environment Affordances + Interaction Hotspots）

6. **核心 idea 一句话**：用 frame-guided temporal pooling + dual cross-attention 实现精细的图像-视频融合，再用 affordance 数据库和交互热点从语义和空间两个层面约束预测

## 方法详解

### 整体框架
输入为高分辨率图像 $I_T$（最后一帧）和低分辨率视频片段 $\mathcal{V}_{T-t:T}$。通过 DINOv2 提取 2D 图像 token，TimeSformer 提取 3D 视频 token。经过三阶段处理：(1) Frame-guided Temporal Pooling 将 3D 视频特征对齐到最后一帧的 2D 空间；(2) Dual Image-Video Attention 双向交叉注意力融合图像和视频特征；(3) 特征金字塔 + 预测头输出检测结果。最后由两个 affordance 模块对预测进行后处理精化。

### 关键设计

1. **Frame-guided Temporal Pooling Attention**:
   - 做什么：将 TimeSformer 输出的 3D 时空 token 压缩到 2D 空间，与最后一帧对齐
   - 核心思路：用最后一帧的视频 token 作为 query，所有帧的 token 作为 key/value，通过残差交叉注意力实现自适应时间池化：$\Phi_{3D}^{2D} = \Phi_{3D}(\mathcal{V}_T) + A(Q_{last}, K_{all}, V_{all})$
   - 设计动机：比 mean pooling 或 conv pooling 更灵活，能根据最后一帧的空间位置自适应地聚合时序信息，消融实验证明比 mean pooling 提升 0.76 mAP All

2. **Dual Image-Video Attention**:
   - 做什么：双向交叉注意力，让图像 token 吸收视频动态信息，同时让视频 token 获取高分辨率空间细节
   - 核心思路：两个并行的残差 cross-attention 层——image-guided（图像做 Q，视频做 KV）和 video-guided（视频做 Q，图像做 KV），各自输出精化后的 token
   - 设计动机：单向注意力（只从图像→视频或视频→图像）效果不如双向，因为两种模态各有互补信息。消融显示双向比 sum fusion 提升约 0.1 mAP All

3. **Environment Affordance Module**:
   - 做什么：从训练集构建一个"环境-交互"数据库，推理时匹配当前场景到数据库中功能相似的区域，获取可能的 noun/verb 分布
   - 核心思路：(a) 从训练视频中用 Siamese 网络聚类出 activity-centric zones；(b) 每个 zone 记录出现过的 noun/verb 集合；(c) 推理时用 EgoVLP-v2 编码当前视频，KNN 找最近的 2K 个 zone（K 个视觉匹配 + K 个文本匹配）；(d) 按相似度加权得到 affordance 概率分布 $p_{aff}(n|\mathcal{V}') \propto \exp(\sum S_i \cdot \mathbb{1}_{n \in \mathcal{N}^{Z_i}})$；(e) 与 STA 模型预测做贝叶斯融合
   - 设计动机：相似环境中的人类行为具有一致性（如厨房→拿刀切菜），这种先验可以约束预测的 noun/verb 分布，减少不合理预测

4. **Interaction Hotspot Module**:
   - 做什么：通过观察手和物体的轨迹，预测未来交互可能发生的空间位置分布
   - 核心思路：基于 Liu et al. 的手运动预测方法改进——用在 Ego4D 上微调的手-物体检测器 + EgoVLP 特征，输出每个像素的交互概率图 $p_{ih}(x,y)$，然后用检测框中心处的概率值来重新加权 STA 检测的置信度
   - 设计动机：affordance 数据库解决了"什么交互"的问题，hotspot 解决了"在哪交互"的问题，两者互补

### 损失函数 / 训练策略
- 采用 StillFast 的 Faster-RCNN 预测头，包含 bbox 回归 + 分类 + 置信度损失
- DINOv2 的最后 3 个 block 微调，TimeSformer 的最后 3 个 block 微调
- Affordance 模块和 Hotspot 模块是后处理，不参与端到端训练

## 实验关键数据

### 主实验

| 数据集 | 指标 (mAP All) | STAformer+AFF | 之前 SOTA | 相对提升 |
|--------|---------------|---------------|-----------|---------|
| Ego4D v1 val | All Top-5 mAP | 3.77 | 2.60 (TransFusion) | +45.0% |
| Ego4D v2 val | All Top-5 mAP | 5.67 | 3.99 (GANO v2) | +42.1% |
| Ego4D v2 test | All Top-5 mAP | 6.75 | 5.18 (Language NAO) | +30.3% |
| EPIC-Kitchens val | All Top-5 mAP | 4.69 | 3.28 (StillFast) | +42.9% |

### 消融实验

| 配置 | N mAP | N+V mAP | All mAP | 说明 |
|------|-------|---------|---------|------|
| StillFast baseline | 16.21 | 7.47 | 2.48 | CNN baseline |
| DINOv2 image only (A1) | 17.48 | 8.64 | 2.52 | 仅图像已超过 baseline |
| + TimeSformer mean pool (B1) | 16.67 | 8.38 | 2.63 | Mean pooling 帮助不大 |
| + Frame-guided pooling (B3) | 19.78 | 10.04 | 3.39 | 关键设计，大幅提升 |
| + Dual attention (C1) | 20.08 | 10.21 | 3.47 | 双向注意力小幅提升 |
| + Finetune DINOv2 (C2=STAformer) | 21.71 | 10.75 | 3.53 | 微调图像编码器重要 |
| + Multi-head variant (C5) | 23.02 | 11.57 | 3.85 | 多头注意力进一步提升 |

### 关键发现
- **Frame-guided temporal pooling 贡献最大**：从 B1→B3 提升了 0.76 mAP All，是 STAformer 中最关键的组件
- **Environment affordance 对 StillFast 也有效**：从 2.48→2.85 (StillFast+AFF)，说明 affordance 模块是模型无关的
- **双向 attention 优于单向**：C2 (双向) 优于 C3 (I→V) 和 C4 (V→I)，两种信息流都有价值
- **N+V 指标获益最大**：affordance 主要通过约束 noun/verb 分布来提升语义预测，N+V 的相对提升高达 58.9%

## 亮点与洞察
- **Affordance 数据库作为外部记忆**：这是一种优雅的非参数化方式来注入领域知识——不需要重训模型，只需匹配数据库。这种 retrieval-augmented 的思路可以迁移到任何需要场景先验的任务
- **后处理模块的模型无关性**：Environment affordance 和 interaction hotspot 都是后处理，可以即插即用到任何 STA 模型上（论文在 StillFast 和 STAformer 上都验证了），这大幅提升了实用价值
- **视觉+文本双路检索 affordance**：用 visual similarity 找外观相似的场景，用 cross-modal similarity 找功能相似但外观不同的场景（如不同国家的厨房），这个双路策略很巧妙

## 局限性 / 可改进方向
- **Affordance 数据库需要大量训练数据标注**：构建 zone 需要训练 Siamese 网络和聚类，对新域的泛化可能受限
- **后处理方式非端到端**：affordance 和 hotspot 模块作为后处理无法反向传播优化，如果集成到训练循环中可能效果更好
- **时间复杂度**：KNN 检索 affordance 数据库在推理时增加了额外计算，论文没有讨论实时性
- **EPIC-Kitchens STA 标注是作者自己生成的**：这个新 benchmark 的标注质量和评估公平性需要社区验证

## 相关工作与启发
- **vs StillFast**: StillFast 用 R50+X3D 的 CNN 架构 + sum fusion，本文用 DINOv2+TimeSformer 的 Transformer 架构 + dual attention，在架构层面全面升级
- **vs EGO-TOPO**: EGO-TOPO 用神经网络直接预测 affordance，本文改用 KNN 检索数据库——实验证明检索方式更好（EGO-TOPO 的 affordance 应用到 StillFast 后性能反而下降）
- **vs TransFusion**: TransFusion 引入语言编码器做多模态融合，但本文的 affordance 方法在不需要额外语言输入的情况下实现了更好的语义约束

## 评分
- 新颖性: ⭐⭐⭐⭐ affordance 数据库和 STAformer 设计有新意，但各个组件单独看并不算突破性
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、详细消融、模块可迁移性验证、充分对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，但公式符号较多需要耐心阅读
- 价值: ⭐⭐⭐⭐ 在 STA 任务上取得大幅领先，affordance 检索思路可复用
