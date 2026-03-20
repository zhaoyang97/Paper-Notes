# Amodal Depth Anything: Amodal Depth Estimation in the Wild

**会议**: ICCV 2025  
**arXiv**: [2412.02336](https://arxiv.org/abs/2412.02336)  
**代码**: [https://github.com/zhyever/Amodal-Depth-Anything](https://github.com/zhyever/Amodal-Depth-Anything) (有)  
**领域**: 3D Vision / Depth Estimation / Amodal Perception  
**关键词**: 非模态深度估计, 相对深度, 遮挡感知, Depth Anything V2, 条件流匹配  

## 一句话总结
提出非模态相对深度估计新范式，构建大规模真实数据集ADIW（564K），基于Depth Anything V2和DepthFM设计两个互补框架（Amodal-DAV2和Amodal-DepthFM），通过最小化修改预训练模型实现遮挡区域深度预测，在ADIW上RMSE比之前SOTA提升27.4%。

## 背景与动机
- **非模态深度估计**旨在预测场景中被遮挡物体不可见部分的深度，是一个新兴且具有挑战性的任务
- 先前方法（AmodalSynthDrive、Amodal-3D-FRONT）依赖**合成数据集**且聚焦于**度量深度**，存在严重的领域偏移问题，零样本泛化能力差
- 合成数据集制作成本高（需手动逐一放置遮挡物），难以规模化
- 现实场景中**不存在能采集遮挡区域真实深度的设备**，数据获取是核心瓶颈
- 近年来Depth Anything等模型在**相对深度**估计上展现了强大的泛化能力，为非模态深度提供了新的解决思路

## 核心问题
1. **如何在无真实遮挡深度标注的情况下构建大规模训练数据？**
2. **如何利用预训练深度模型的强大先验来预测遮挡区域的深度？**
3. **度量深度 vs 相对深度：哪种范式更适合非模态深度估计的泛化？**

## 方法详解
### 整体框架
输入：观测图像 $I_o$、观测深度图 $D_o$、目标非模态掩码 $M_a$ → 输出：包含遮挡部分深度的非模态深度图。提出两个互补框架：确定性的Amodal-DAV2和生成式的Amodal-DepthFM。

### 关键设计
1. **ADIW数据集构建Pipeline**:
   - 利用SAM在SA-1B上自动生成分割掩码，通过启发式算法[pix2gestalt]筛选完整物体
   - 采用合成策略：将前景物体叠加到背景图上形成观测图像 $I_o$
   - 分别用Depth Anything V2 (ViT-G) 对 $I_o$ 和背景图 $I_b$ 生成深度图
   - **Scale-and-Shift对齐**：由于前景物体改变了背景的相对深度，需通过最小二乘法计算缩放因子 $s$ 和偏移因子 $t$ 对齐两个深度图，确保训练标签一致性
   - 最终生成**564K**训练样本

2. **Amodal-DAV2（确定性模型）**:
   - 在DAV2的ViT编码器RGB Conv层旁并行添加**Guidance Conv层**，接受 $D_o$ 和 $M_a$ 作为额外引导通道
   - Guidance Conv权重**零初始化**，保证初始时模型忽略额外输入，逐步学习
   - 在DPT Head输入特征上添加**LayerNorm**稳定训练
   - 全模型端到端微调，修改极少

3. **Amodal-DepthFM（生成式模型）**:
   - 基于DepthFM的条件流匹配框架
   - 修改UNet第一层Conv以接受额外的引导通道（$D_o$ 和 $M_a$）
   - 前8通道使用预训练权重，额外2通道零初始化
   - 推理时应用**Scale-and-Shift对齐**将预测深度与观测深度在可见共享区域对齐，提升一致性
   - 生成式特性允许输出**多个合理的遮挡深度结构**

### 损失函数 / 训练策略
- **Amodal-DAV2**: Scale-Invariant Log (SILog) Loss，$\lambda=0.85$，在**整个物体**（包括可见+不可见）上监督（而非仅不可见部分），有助于理解整体场景结构
- **Amodal-DepthFM**: 条件流匹配目标 $\min_\theta \mathbb{E}_{t,z,p(x_0)} \|v_\theta(t, \phi_t(x_0)) - (x_1 - x_0)\|$，训练时添加高斯噪声增强
- 训练超参：
  - Amodal-DAV2: batch 32, lr 1e-5, 50K iters
  - Amodal-DepthFM: batch 128, lr 3e-5, 15K iters
  - Adam优化器，指数学习率衰减，梯度裁剪0.01
  - 4×A100 GPU

## 实验关键数据
| 数据集 | 指标 | 本文最优 | 之前SOTA | 提升 |
|--------|------|----------|----------|------|
| ADIW (Overall) | RMSE↓ | **3.682** (Amodal-DAV2-L) | 5.114 (pix2gestalt‡) | 27.4% |
| ADIW (Overall) | δ(%)↑ | **93.251** | 88.717 (pix2gestalt‡) | +4.5pp |
| ADIW (Easy) | RMSE↓ | 对应行表现优 | 5.067 | - |
| ADIW (Hard) | RMSE↓ | 对应行表现优 | 5.641 | - |

**不同方法在ADIW上的Overall对比：**
| 方法 | RMSE↓ | δ(%)↑ |
|------|-------|-------|
| Jo et al.† | 10.260 | 56.118 |
| Sekkat et al.† | 11.264 | 49.222 |
| Sekkat et al.†‡ | 5.194 | 88.367 |
| pix2gestalt‡ | 5.114 | 88.717 |
| Amodal-DepthFM (Full) | 4.645 | 92.295 |
| **Amodal-DAV2-L (Full)** | **3.682** | **93.251** |

### 消融实验要点
**Amodal-DAV2消融：**
- 无引导信号：RMSE 7.549→3.682（引导至关重要）
- 无掩码引导：RMSE 4.369→3.682（掩码提升显著）
- 仅监督不可见部分：RMSE 3.845 vs 全对象监督3.682（全对象监督更优）
- 推理时加对齐反而降低性能：RMSE 4.015（确定性模型已学到一致的深度）

**Amodal-DepthFM消融：**
- Scale-and-Shift对齐对DepthFM**至关重要**：RMSE 5.410→4.645（生成模型输出不够一致，需要对齐）
- 场景级监督（⑥）RMSE 4.608与物体级（④）4.645接近但log10指标更差

## 亮点
1. **问题范式创新**：将非模态深度估计从度量深度转向相对深度，大幅提升泛化能力
2. **数据构建可扩展**：基于SA-1B和合成策略的数据pipeline无需人工标注，可自动生成564K样本
3. **最小化修改预训练模型**：零初始化的Guidance Conv设计巧妙，既利用预训练知识又引入额外引导
4. **两种互补范式**：Amodal-DAV2精度高，Amodal-DepthFM细节好且支持多样化预测
5. **不依赖RGB先验**：直接预测遮挡深度，避免了Inpainting方法的级联误差

## 局限性 / 可改进方向
1. **依赖非模态掩码质量**：不准确或模糊的掩码会导致级联误差
2. **微调后细节捕获能力略有下降**：可能因SA-1B数据集本身的多样性有限
3. **仅支持单帧**：可扩展到视频场景的时序非模态深度估计
4. **单一任务**：未来可构建统一框架同时预测非模态分割、RGB、深度、法线等
5. **数据来源受限于SAM数据集**：引入更多高质量复杂目标数据集可能进一步提升性能

## 与相关工作的对比
- **vs Sekkat et al. / Jo et al.**：先前方法基于合成数据+度量深度，泛化能力差；本文用真实图像+相对深度，RMSE降低>50%
- **vs Invisible Stitch / pix2gestalt**：Inpainting-based方法依赖RGB重建质量，存在级联误差（先Inpaint再估深度）；本文直接从观测深度+掩码回归，更鲁棒
- **vs Depth Anything V2**：DAV2只估计可见像素深度；本文扩展为遮挡区域深度，是任务层面的补充
- **vs DepthFM**：本文保留DepthFM的生成特性（多样化预测），但细节质量更好的同时精度不如确定性方案

## 启发与关联
- **Depth Anything作为教师模型的潜力**：与 `ideas/object_detection/20260316_multi_teacher_dense_distillation.md` 中将Depth Anything作为几何教师的思路一致，证明了大规模深度基础模型在下游任务中的通用价值
- **基础模型伪标签的范式**：与 `ideas/3d_vision/20260317_foundation_aux_hand_pose.md` 类似，用DepthAnything生成伪深度标签弥补真实标注的缺失
- **零初始化引导层设计**：这种"在预训练模型旁路添加零初始化通道"的做法（类似ControlNet思想）是一种通用范式，可迁移到其他需要额外条件输入的基础模型适配场景
- **轻量化方向**：可参考 `ideas/3d_vision/20260317_lightweight_illusion_depth_fusion.md` 的知识蒸馏思路，将Amodal-DAV2-L蒸馏为更轻量的版本

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题范式（相对深度替代度量深度）和数据构建流程有创新，但模型设计本身修改较为简单
- 实验充分度: ⭐⭐⭐⭐ 消融实验全面覆盖引导信号、监督策略、对齐策略，但缺少与更多深度基础模型的对比以及不同数据规模的消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰、图表丰富，方法描述详尽，但部分表格数据排版稍乱
- 价值: ⭐⭐⭐⭐ 开辟了非模态深度估计的新方向，数据pipeline可复用，对3D重建和场景理解有实际应用价值
