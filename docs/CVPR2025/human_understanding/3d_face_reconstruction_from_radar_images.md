# 3D Face Reconstruction From Radar Images

**会议**: CVPR 2025  
**arXiv**: [2412.02403](https://arxiv.org/abs/2412.02403)  
**代码**: 无  
**领域**: 人体理解 / 3D人脸重建 / 雷达成像  
**关键词**: mmWave雷达, 3DMM, Analysis-by-Synthesis, 可学习渲染器, 人脸重建

## 一句话总结
首次从毫米波雷达图像进行3D人脸重建：用物理雷达渲染器生成合成数据集训练CNN编码器估计BFM参数，再通过学习一个可微分雷达渲染器构建model-based autoencoder，在合成数据上实现2.56mm平均点距精度，并可在推理时无监督优化参数。

## 研究背景与动机

1. **领域现状**：3D人脸重建在RGB图像领域已经非常成熟，基于3DMM（如BFM、FLAME）的方法可以从单张RGB图像高精度重建人脸。同时，毫米波雷达作为一种非光学传感器，因其能穿透织物、不依赖光照的特性，已在人体姿态估计、安检扫描等场景中得到应用。

2. **现有痛点**：光学传感器在睡眠实验室等场景中存在本质局限——需要光照、无法穿透被子/枕头，患者需要暴露在摄像头下。而雷达虽然能解决这些问题，但雷达图像的分辨率低（空间分辨率约4mm×4mm×11mm）、成像特性与RGB完全不同（依赖表面法线反射，非所有面部区域都可见），导致直接将RGB重建方法迁移到雷达域行不通。

3. **核心矛盾**：雷达域缺乏可微分渲染器。RGB领域的model-based autoencoder方法（如Tewari et al.的Analysis-by-Synthesis框架）依赖可微分渲染，但物理雷达渲染器（基于光线追踪+后向投影）既不可微分也速度太慢（每张2分钟），无法直接用于训练。

4. **本文要解决什么？**
   - 如何从雷达图像（amplitude image / depth image）估计3DMM参数进行人脸重建？
   - 如何在没有可微分雷达渲染器的情况下实现Analysis-by-Synthesis训练范式？
   - 如何弥合合成雷达图像与真实雷达图像之间的domain gap？

5. **切入角度**：用神经网络学习一个"可微分雷达渲染器"来近似物理渲染器，从而把RGB领域成熟的model-based autoencoder框架迁移到雷达域。

6. **核心idea一句话**：用学习到的可微分雷达渲染器替代物理渲染器，构建model-based autoencoder，使3D人脸重建可以从雷达图像通过Analysis-by-Synthesis方式无监督优化。

## 方法详解

### 整体框架
输入是雷达amplitude image（或depth image，或两者组合），输出是BFM 2019的形状参数$\alpha \in \mathbb{R}^{10}$、表情参数$\gamma \in \mathbb{R}^{7}$和姿态参数（平移$t \in \mathbb{R}^3$、旋转$T \in \mathbb{R}^3$）。整个pipeline分为两个阶段：

1. **编码器（全监督训练）**：三个CNN分别预测形状、表情、姿态参数，用L2参数损失训练
2. **自编码器（Analysis-by-Synthesis）**：将预训练编码器与学习到的可微分渲染器（解码器）组合，联合优化参数损失+图像重建损失

### 关键设计

1. **物理雷达渲染器与合成数据集**
   - 做什么：生成10,000张合成雷达图像用于训练
   - 核心思路：基于BFM 2019的face12 mask，高斯采样形状向量$\alpha \sim \mathcal{N}(0,1)$和表情向量$\gamma \sim \mathcal{N}(0,1)$生成人脸mesh，然后用Schüssler et al.的物理雷达渲染器（光线追踪+后向投影算法）生成对应的雷达amplitude image。姿态包括$\pm 5$cm平移、$\pm 5°$偏航、$\pm 10°$俯仰/翻滚。渲染参数（材质因子、天线尺寸）也随机采样以增加多样性
   - 设计动机：真实雷达数据极其稀缺（论文只有4个人的数据），必须依赖合成数据训练。物理渲染器虽然慢（每张2分钟）但可以离线批量生成数据集

2. **编码器架构（三网络并行）**
   - 做什么：从雷达图像预测3DMM参数
   - 核心思路：借鉴Chang et al.的设计，用2个ResNet-50分别预测形状$\alpha$和表情$\gamma$，1个AlexNet预测姿态$(t, T)$。ResNet输出经tanh层缩放到$[-3, 3]$范围（覆盖99.8%的采样值）。训练时随机采样dynamic range $[-15, -30]$ dB做数据增强，评估时固定$-20$ dB
   - 设计动机：形状和表情是高维语义特征需要更大网络（ResNet-50），姿态是低维几何量用轻量AlexNet即可。tanh缩放限制输出范围，避免预测出不合理的3DMM参数

3. **可学习雷达渲染器（反向ResNet-50）**
   - 做什么：从3DMM参数生成雷达图像，替代不可微分的物理渲染器
   - 核心思路：将ResNet-50的层**反向排列**作为解码器，前端加一个全连接层将参数映射到第一个卷积层。输入是形状、表情、姿态和渲染参数，输出是amplitude/depth图像。在合成数据上独立预训练，冻结后作为自编码器的解码器
   - 设计动机：物理渲染器不可微分且慢，而训练一个神经网络渲染器可以实现：(a) 可微分——使Analysis-by-Synthesis成为可能；(b) 快2000倍（58ms vs 2分钟）；(c) 作为解码器约束编码器的latent space结构

4. **Model-based Autoencoder**
   - 做什么：联合编码器和解码器，通过图像重建损失提供额外监督
   - 核心思路：将预训练编码器和冻结的解码器串联。训练时解码器权重固定（保持latent space结构），只更新编码器权重。总损失为：
     $$L_{train} = L_{image} + \lambda \cdot L_{params}$$
     其中$L_{image}$是输出与输入图像的L2损失，$L_{params}$是预测参数与GT参数的L2损失，$\lambda = 1$
   - 设计动机：仅靠参数损失训练编码器是point-wise监督，加上图像重建损失相当于在像素空间施加全局一致性约束，起到正则化效果。更关键的是：推理时可以冻结编码器和解码器，仅通过反向传播图像损失来优化latent变量（参数），实现无监督的test-time optimization

### 损失函数 / 训练策略
- **编码器训练**：L2参数损失，Adam优化器，学习率线性从0.01衰减到0.001（前150 epoch），共200 epoch，batch 50
- **解码器训练**：L2图像重建损失，学习率固定0.001，300 epoch，batch 150
- **自编码器训练**：$L_{train} = L_{image} + L_{params}$，解码器权重冻结，学习率0.001，300 epoch，batch 50
- **推理优化**：冻结编码器+解码器，仅优化latent variables通过$L_{image}$
- 硬件：Nvidia A100训练，自编码器约4小时

## 实验关键数据

### 主实验

| 方法 | 输入类型 | L2 Shape ↓ | L2 Expr ↓ | L2 Total ↓ | 点距(mm) ↓ |
|------|----------|-----------|----------|-----------|-----------|
| Baseline (Mean) | - | 0.991 | 1.000 | 0.808 | 4.42 |
| Baseline (Random) | - | 2.012 | 2.059 | 1.648 | 6.28 |
| Encoder | Amplitude | 0.840 | 0.921 | 0.655 | 3.47 |
| Encoder | Depth | 0.740 | 0.910 | 0.608 | 2.77 |
| Encoder | Amp.+Depth | 0.735 | 0.885 | 0.598 | 2.82 |
| **Autoencoder** | Amplitude | 0.770 | 0.815 | 0.594 | 3.29 |
| **Autoencoder** | Depth | 0.634 | 0.850 | 0.544 | **2.56** |
| **Autoencoder** | Amp.+Depth | 0.640 | 0.820 | 0.539 | 2.61 |

### 消融实验（编码器 vs 自编码器 / 不同输入模态）

| 配置 | 点距(mm) | 说明 |
|------|---------|------|
| Autoencoder + Depth | **2.56** | 最佳配置 |
| Autoencoder + Amp.+Depth | 2.61 | 双模态略差于纯深度，可能因amplitude噪声 |
| Encoder + Depth | 2.77 | 无autoencoder正则化，差0.21mm |
| Encoder + Amp.+Depth | 2.82 | 编码器双模态 |
| Autoencoder + Amplitude | 3.29 | 纯amplitude信息不足 |
| Encoder + Amplitude | 3.47 | 最差的有效配置 |

### 关键发现
- **Autoencoder始终优于Encoder**：在shape和expression参数预测上，autoencoder版本的L2 error均低于纯encoder版本。额外的图像重建损失起到了正则化作用
- **Depth图像显著优于Amplitude图像**：depth输入使点距从3.29mm降到2.56mm（改善22%）。原因是depth图像的面部像素值分布更均匀，不像amplitude图像有小区域高值峰
- **姿态影响shape/expression预测**：中性姿态下的cosine similarity对角线更明显，说明姿态变化增加了shape/expression估计的难度
- **Shape识别在真实数据上可行**：cosine similarity分析表明，相同人的shape参数在真实图像上仍有较高相似度（可用于人脸识别）
- **Expression识别在真实数据上困难**：表情参数在真实数据上没有明显的聚类结构，仅identity-specific模型可以识别表情
- 可学习渲染器速度是物理渲染器的**2000倍+**（58ms vs 2min）

## 亮点与洞察
- **首次将雷达图像用于3D人脸重建**：开辟了全新的传感模态应用场景（睡眠监测、黑暗环境、穿透遮挡物），不是对现有RGB方法的简单改进，而是解决了一个全新的问题。这个"第一次"本身就有重要的开创性价值
- **用学习渲染器绕过不可微分渲染器**：物理雷达渲染器基于光线追踪，天然不可微分。作者没有尝试推导微分形式（极其困难），而是训练一个神经网络来近似它。这个思路很实用——任何不可微分的渲染/模拟过程都可以用这种方式"封装"成可微分模块。可迁移到其他physics-based但non-differentiable的仿真场景
- **Test-time optimization via image loss**：推理时固定所有权重，仅通过图像重建损失反传优化latent参数，是一种elegant的无标注自适应方式。这本质上是利用了重建一致性作为自监督信号
- **渲染参数也作为预测目标**：编码器不仅预测3DMM参数，还预测dynamic range、材质因子、天线尺寸等渲染参数，让网络学会理解成像过程本身

## 局限性 / 可改进方向
- **真实数据极度匮乏**：仅4个男性欧洲人、每人5个表情，diversity严重不足。这导致无法充分验证合成→真实的泛化能力。未来需要大规模真实雷达人脸数据集
- **Synthetic-to-real domain gap**：合成图像和真实图像存在明显的pattern差异和scale差异，论文中真实数据的shape/expression重建质量显著下降
- **表情识别失败**：在真实数据上无法有效识别表情变化，只有identity-specific模型可以做到，严重限制了睡眠监测等应用的实用性
- **3DMM参数空间有限**：仅用10维shape和7维expression，覆盖约85%形状方差和76%表情方差，丢失了大量细节信息
- **改进思路**：(1) 用domain adaptation/domain randomization减小synthetic-real gap；(2) 扩展到FLAME模型获取更丰富的表情空间；(3) 引入self-supervised contrastive learning利用无标注真实数据；(4) 多视角雷达阵列融合提升重建完整性

## 相关工作与启发
- **vs Tewari et al. (MoFA)**：都用model-based autoencoder做Analysis-by-Synthesis人脸重建，但MoFA用可微分的OpenGL渲染器，本文因为雷达域没有可微分渲染器所以学了一个。核心贡献不在autoencoder框架本身，而在将该框架迁移到雷达域
- **vs Xie et al.**：首个雷达3D人脸重建工作，但基于landmark检测+FLAME拟合（两步pipeline），本文直接回归3DMM参数（端到端），且autoencoder框架支持无监督优化
- **vs Chen et al. (ImmFusion)**：做雷达+RGB融合的全身重建，本文纯雷达无需RGB辅助，聚焦面部而非全身

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将3D人脸重建从RGB扩展到雷达图像，学习渲染器绕过不可微分物理渲染器的思路有新意
- 实验充分度: ⭐⭐⭐ 合成数据实验充分（多种输入模态、encoder vs autoencoder对比），但真实数据仅4人且diversity不足
- 写作质量: ⭐⭐⭐⭐ 结构清晰，motivation到method的推导逻辑通顺，图表质量高
- 价值: ⭐⭐⭐ 开创性工作有启发意义，但受限于数据规模和真实效果，离实用还有距离
