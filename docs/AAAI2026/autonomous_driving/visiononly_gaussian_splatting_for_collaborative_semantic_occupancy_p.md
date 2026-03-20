# Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction (Oral)

**会议**: AAAI 2026  
**arXiv**: [2508.10936v2](https://arxiv.org/abs/2508.10936v2)  
**代码**: [https://github.com/ChengChen2020/VOGS-CP](https://github.com/ChengChen2020/VOGS-CP)  
**领域**: Collaborative Perception / 3D Occupancy Prediction  
**关键词**: Collaborative Perception, 3D Gaussian Splatting, Semantic Occupancy, V2X, Multi-Agent  

## 一句话总结
首次将 3D 高斯 Splatting 作为多智能体协同感知的通信媒介和中间表征，利用高斯基元的刚体变换可解析性和稀疏性，通过高斯打包（ROI 裁剪+刚体变换）和跨智能体邻域融合模块，实现了高效且可解释的视觉协同语义占用预测。

## 核心问题
协同感知（Collaborative Perception）通过 V2X 通信让多辆联网车辆共享信息，克服遮挡和扩展感知范围。但将其扩展到 3D 语义占用预测（SOP）面临挑战：
1. **通信效率**：密集 3D 体素特征（$X \times Y \times Z \times D$）的通信量过大；平面特征（BEV/tri-plane）丢失了高度和 3D 几何信息
2. **跨智能体对齐**：隐式平面特征的跨车辆空间对齐复杂，需要额外的深度监督
3. **融合策略**：多智能体的预测可能存在噪声、冗余和不一致性

现有方法 CoHFF（首个协同 SOP 框架）依赖 tri-plane 特征、需要额外深度估计网络、多阶段多任务训练，系统复杂度高。

## 关键方法
- **3D 高斯基元作为通信接口**：每个基元包含均值 $\mathbf{m} \in \mathbb{R}^3$、尺度 $\mathbf{s} \in \mathbb{R}^3$、旋转四元数 $\mathbf{r} \in \mathbb{R}^4$、不透明度 $a \in \mathbb{R}$、语义 $\mathbf{c} \in \mathbb{R}^{|C|}$。相比密集体素特征大幅减少通信量，且天然支持刚体变换
- **Gaussian Packaging**：
  - **刚体变换**：利用高斯分布在刚体变换下的封闭形式解（均值做旋转+平移，协方差做旋转，尺度/不透明度/语义不变），实现简洁的跨智能体对齐
  - **ROI 裁剪**：只传输变换后落在 ego 车辆感兴趣区域（ROI）内的高斯基元，大幅降低通信负载
- **Cross-Agent Gaussian Fusion**：
  - 对每个 ego 高斯建立半径 $\rho$ 的邻域，从邻居车辆的高斯中收集配对特征（包含相对位移、尺度差、四元数余弦、不透明度、语义）
  - MLP 将配对特征映射为修正提议（位置残差、新尺度/旋转/不透明度/语义提议）
  - 通过 mean pooling 或 attention pooling 聚合邻域提议
  - 语义更新使用基于置信度的混合（ego 与邻居的语义按各自最大 logit 加权融合）
- **端到端单阶段训练**：使用交叉熵 + Lovász-Softmax 损失，不需要额外深度监督

## 亮点 / 我学到了什么
- **显式 3D 表征作为通信媒介的优势**：与 BEV/tri-plane 等隐式特征相比，3D 高斯基元具有可解释性（每个基元有明确的位置、形状、语义含义）、刚体变换的解析解、天然的稀疏性和 ROI 裁剪能力。这个选择本身就是一个有价值的研究洞察
- **零样本即有效**：即使不做联合训练，直接拼接多智能体的高斯基元（zero-shot）就能带来感知提升（IoU 67.88 vs 67.76 单智能体），这验证了显式 3D 表征的根本优势
- **通信效率的极端表现**：仅用 CoHFF 34.6% 的通信量（0.27MB vs 0.78MB），仍能在 mIoU 上超越 CoHFF (+1.86)
- **高斯基元的"一个大高斯表示空闲空间"策略**：单智能体 IoU 从 38.52（CoHFF baseline）跳到 67.76，其中很大一部分来自用一个大型高斯显式建模空闲空间，让其他高斯专注于被占用区域
- **邻域融合的语义置信度混合**：$\alpha_k = \text{conf}(\mathbf{c}_k) / (\text{conf}(\mathbf{c}_k) + \text{conf}(\bar{\mathbf{c}}_k^\star))$，用最大 logit 作为置信度的简单策略有效处理了语义冲突

## 局限性 / 可改进方向
- 仅在仿真数据 Semantic-OPV2V 上验证，缺少真实世界数据集（如 V2V4Real、DAIR-V2X）的测试
- 以 GaussianFormer（v1）为基础单智能体模型，未使用更新的 GaussianFormer-2 或 SplatSSC
- 高斯基元数量（25600）是固定的，未探索自适应数量策略
- 跨智能体融合模块相对简单（MLP + pooling），可用更复杂的图注意力或 cross-attention 替代
- 对通信延迟、丢包等真实 V2X 通信挑战未做分析
- 没有处理动态物体的显式策略

## 实验关键数据

**Semantic-OPV2V 语义占用预测：**

| 方法 | 设置 | IoU | mIoU |
|------|------|-----|------|
| CoHFF (单智能体) | Single | 38.52 | 24.81 |
| GaussianFormer (单智能体) | Single | 67.76 | 29.20 |
| CoHFF (协同) | Collab | 50.46 | 34.16 |
| Ours Zero-shot | Collab | 67.88 | 30.54 |
| Ours Naive Fusion | Collab | 70.10 | 36.02 |
| **Ours Learned Fusion** | **Collab** | **72.87** | **37.44** |

**BEV 分割（IoU %）：** Ours vs CoHFF = 70.25/82.69/79.37 vs 47.40/63.36/40.27 (Vehicle/Road/Others)

**通信量：** Ours 0.27MB vs CoHFF 0.78MB（仅 34.6%），mIoU 仍超 CoHFF (+1.86)

## 与我的研究方向的关联
- **3DGS 在感知中的新应用**：将 3D 高斯基元不仅作为场景表征，更作为多智能体通信接口，是 3DGS 应用范围的有意义扩展
- **与 SplatSSC 的联系**：都基于 GaussianFormer 的 object-centric 范式，但本文处理多智能体融合，SplatSSC 处理单智能体深度引导初始化。两者技术可互补
- **协同感知 + 高效表征**：高斯基元天然的稀疏性和可解释性使其成为带宽受限场景下的理想通信载体，对 V2X 研究有启发
- **端到端可训练的多智能体系统**：不需要多阶段训练，proof of concept 简洁有力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
