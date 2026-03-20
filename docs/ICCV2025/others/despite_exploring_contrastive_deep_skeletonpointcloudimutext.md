 bu# DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding

**会议**: ICCV 2025  
**arXiv**: [2506.13897](https://arxiv.org/abs/2506.13897)  
**代码**: 无（论文提到将公开预训练编码器、代码和数据）  
**领域**: 3D视觉 / 动作识别  
**关键词**: 多模态对比学习, LiDAR点云, 人体动作识别, 跨模态检索, 隐私保护  

## 一句话总结

提出 DeSPITE，一个将 LiDAR 点云、骨架姿态、IMU 信号和文本四种模态对齐到联合嵌入空间的对比学习框架，首次以 LiDAR（而非 RGB）作为核心视觉模态，实现了跨模态匹配/检索等此前不可能的任务，同时作为有效的 HAR 预训练策略在 MSR-Action3D 和 HMPEAR 上取得 SOTA。

## 背景与动机

RGB 摄像头在医疗、监控等隐私敏感场景中受限，而 LiDAR 天然保护隐私。此前的多模态对比学习方法（ImageBind、IMU2CLIP、MotionCLIP、LAVIMO 等）全部以 RGB 作为主要视觉模态来"绑定"其他模态。然而，将 LiDAR 点云与 IMU、骨架等模态进行对比对齐这一方向几乎没有被探索过。作者提出了一个关键研究问题：**如果我们只依赖 LiDAR 作为多模态对比学习中的主要视觉模态，会发生什么？**

## 核心问题

1. LiDAR 点云、骨架姿态、IMU 信号三种隐私保护模态之间的跨模态匹配和检索在此前完全未被研究。
2. 点云 HAR 领域缺乏通用预训练模型，现有自监督方法仅在小数据集上进行预训练。
3. 缺少同时包含点云-骨架-IMU-文本注释的大规模数据集。

## 方法详解

### 整体框架

DeSPITE 的核心思想很直白：人体运动在不同传感器模态下具有天然对应关系——同一个人做同一个动作，LiDAR 看到的点云序列、穿戴的 IMU 信号、提取的骨架姿态，描述的是同一件事。DeSPITE 利用这种对应关系，通过 InfoNCE 对比损失将四种模态映射到同一个 512 维嵌入空间。

每种模态使用独立编码器：
- **点云**: PST-Transformer + SimCLR 投影头
- **IMU**: 2 层 LSTM
- **骨架**: ACTOR 编码器（Transformer VAE）
- **文本**: 冻结的 CLIP 文本编码器（作为"锚"模态）

输入统一为 24 帧窗口，每帧点云通过最远点采样降至 256 点。

### 关键设计

1. **灵活的模态组合**：作者训练了所有可能的模态子集组合（DeSPIE=骨架+点云+IMU+文本中去掉T，DeSPE=骨架+点云，DePIE=点云+IMU 等），系统考察每种模态的贡献。

2. **LIPD-Babel 数据集构建**：这是本文一个重要的工程贡献。作者将 LIPD 数据集（包含点云-IMU-骨架但无活动标注）与 Babel 数据集（包含 AMASS 运动数据的文本标注）通过序列 ID 映射合并。由于两个数据集帧率不同（Babel 30 FPS vs LIPD 10 FPS），需要下采样对齐。构建了两个版本：
   - LIPD-Babel-v1：用于匹配和检索评估（502K/85K 训练/测试窗口）
   - LIPD-Babel-v2：用于 HAR 分类评估（403K/58K 训练/测试窗口，含文本标注）

3. **文本作为可选绑定模态**：并非所有训练样本都有文本标注，通过布尔掩码 $tm$ 处理缺失文本的样本——仅在有文本配对的子集上计算文本对比损失。

4. **一个有趣的发现**：文本模态对匹配和检索任务有害，但对 HAR 微调有益。这说明 CLIP 文本嵌入空间的语义结构有助于学习更具判别性的活动特征，但其粗粒度特性损害了精细的时空对齐。

### 损失函数 / 训练策略

总损失由两部分组成：

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{text} + \beta \mathcal{L}_{M}$$

其中 $\alpha = \beta = 0.5$。

- $\mathcal{L}_{text}$：每种传感模态与文本之间的 InfoNCE 对比损失（仅在有文本标注的样本上计算）
- $\mathcal{L}_{M}$：所有传感模态两两之间的 InfoNCE 对比损失（点云↔IMU、点云↔骨架、IMU↔骨架）

每对模态的损失是双向对称的：$\mathcal{L}_{a,b} = \frac{1}{2}(\mathcal{L}_{a \to b} + \mathcal{L}_{b \to a})$

训练 145 epochs，Adam 优化器，lr=1e-4，batch size 1024，可学习温度参数 $\tau$。使用随机平移、缩放和高斯噪声增强防止过拟合。HAR 微调使用 SGD，warmup 到 lr=0.01，35 epochs。

## 实验关键数据

### MSR-Action3D（点云 HAR，clip-level accuracy）

| 方法 | 准确率 |
|------|--------|
| PST-Transformer (baseline) | 93.73 |
| PST-Transformer + MaST-Pre | 94.08 |
| PST-Transformer + M2PSC | 94.84 |
| PvNext | 94.77 |
| KAN-HyperpointNet | 95.59 |
| **PST-Transformer + DeSPIE (Ours)** | **95.47** |
| **PST-Transformer + DeSPITE (Ours)** | **95.47** |

### HMPEAR（点云 HAR，segment-level accuracy）

| 方法 | 模态 | Acc(Seg) |
|------|------|----------|
| PST-Transformer | PC | 65.94 |
| PEAR-Proj (BestAR) | RGB+PC | 66.0 |
| **PST-Transformer + DePITE (Ours)** | **PC** | **70.65 (+4.71)** |

### LIPD-Babel-v2（多模态 HAR）

| 方法 | 模态 | Acc(Seg) |
|------|------|----------|
| PST-Transformer (scratch) | PC | 67.38 |
| LSTM (scratch) | IMU | 65.62 |
| ACTOR (scratch) | Skeleton | 68.23 |
| PST-Transformer + DeSPITE | PC | 69.00 (+1.62) |
| LSTM + DeSPIE | IMU | 69.21 (+3.59) |
| ACTOR + DeSPITE | Skeleton | 70.64 (+2.41) |

### 消融实验要点

1. **文本模态的双刃剑效应**：带文本训练的模型（DeSPITE、DePITE 等）在匹配和时序检索任务中几乎全面差于不带文本的模型（DeSPIE、DePIE 等），但在 HAR 微调中表现更好。这暗示 CLIP 文本空间的语义信息有助于分类但损害精细对齐。

2. **模态组合的影响**：更多模态参与预训练通常对下游 HAR 有利——DeSPITE/DeSPIE/DePITE 一致优于两模态变体。

3. **跨模态难度**：IMU↔骨架 的匹配和检索最容易，点云↔骨架 次之，IMU↔点云 最难，这与直觉一致（IMU 和骨架都直接描述关节运动，点云更抽象）。

4. **冻结 vs 微调**：详细消融表明微调通常优于线性/非线性探测，但探测结果也不错，说明预训练确实学到了有意义的表征。

## 亮点 / 我学到了什么

1. **研究问题提得好**：将 LiDAR 而非 RGB 作为多模态对比学习的核心视觉模态，这是一个简洁但未被探索的方向。隐私保护的动机也很实际。

2. **系统的实验设计**：训练了所有模态组合（DeSPE、DeSIE、DePIE…），实验工作量巨大，但给出了清晰的模态贡献全景图。

3. **数据集构建的工程价值**：LIPD-Babel 的构建虽然不是技术上的创新，但通过巧妙的序列 ID 映射和帧率对齐，为社区提供了一个此前不存在的四模态训练资源。

4. **"文本有害但有用"的发现**：文本模态在匹配/检索中有害但在分类中有用，这个看似矛盾的发现值得深入思考——反映了语义对齐粒度和下游任务粒度之间的 mismatch。

## 局限性 / 可改进方向

1. **方法本身较为直接**：四个独立编码器 + InfoNCE 对比损失，没有太多架构创新。每种模态使用现有编码器（PST-Transformer、LSTM、ACTOR），贡献更多在于"探索"而非"方法"。

2. **数据质量依赖合成**：LIPD 数据集大量依赖合成的 LiDAR 点云和 IMU 数据，真实场景中的噪声和遮挡问题未被充分讨论。

3. **文本对齐效果不理想**：与 TMR++ 相比，文本检索性能明显落后（R-Top-1: 42.55 vs 55.54），说明 CLIP 文本空间与运动模态的对齐还有较大提升空间。

4. **单人场景限制**：虽然通过人工构造模拟多人场景，但数据本身是单人运动捕捉，真实多人交互场景中的遮挡、干扰等问题未被考虑。

5. **编码器选择未经优化**：IMU 用 LSTM 在 2026 年略显过时，更现代的架构（如 Transformer 或 Mamba）可能带来更好的 IMU 编码。

## 与相关工作的对比

| 方法 | 核心视觉模态 | 对齐模态 | 主要应用 |
|------|-------------|----------|----------|
| CLIP | RGB | Text | 通用视觉-语言 |
| ImageBind | RGB | 6种模态 | 通用多模态 |
| IMU2CLIP | RGB (CLIP) | IMU | IMU 检索 |
| MotionCLIP | RGB (CLIP) | Skeleton | 运动生成 |
| LAVIMO | RGB | Skeleton+Text | 骨架检索 |
| **DeSPITE** | **LiDAR PC** | **Skeleton+IMU+Text** | **隐私保护HAR、跨模态检索** |

DeSPITE 的核心差异化在于：完全放弃 RGB，以 LiDAR 点云为中心构建多模态嵌入空间。

## 与我的研究方向的关联

- ideas/ 中未找到直接相关的研究想法。
- 对比学习对齐多模态的框架可以迁移到其他涉及隐私保护的领域（如医疗影像中不同传感器模态的对齐）。
- "文本帮助分类但损害检索"这个发现在设计多模态预训练策略时值得注意。

## 评分

- **新颖性**: ⭐⭐⭐ — 研究问题新颖（LiDAR 替代 RGB），但方法本身是标准的对比学习套路
- **实验充分度**: ⭐⭐⭐⭐⭐ — 极其全面，所有模态组合、多任务、多数据集、大量消融
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述得当，实验展示系统
- **对我的价值**: ⭐⭐ — 方向距离较远，但多模态对比对齐的实验方法论值得参考
