# NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration

**会议**: ECCV 2024  
**arXiv**: [2309.07322](https://arxiv.org/abs/2309.07322)  
**代码**: [https://github.com/uncbiag/NePhi](https://github.com/uncbiag/NePhi) (有)  
**领域**: 医学图像 / 3D视觉  
**关键词**: 医学图像配准, 微分同胚变换, 隐式神经表示, 形变场, 多分辨率配准  

## 一句话总结
NePhi用隐式神经网络（SIREN）替代传统的体素化形变场来表示配准变换，通过编码器预测latent code + 可选的测试时优化实现快速且近似微分同胚的医学图像配准，在多分辨率设置下与SOTA精度相当但内存降低5倍。

## 背景与动机
医学图像配准是将两张图像对齐到相同坐标系的基础任务。现有的学习方法大多用体素化形变场（voxel-based deformation fields）表示变换，这在高分辨率3D图像上存在显著问题：内存消耗巨大（立方级增长），且难以保证变换的拓扑正确性（微分同胚性）。已有的基于神经网络的配准方法虽然用INR表示形变，但依赖纯优化推理，速度极慢。

## 核心问题
如何设计一种既能像体素方法一样快速推理，又能保证良好的变换规则性（近似微分同胚），同时大幅降低内存消耗的配准方法？

## 方法详解

### 整体框架
NePhi的pipeline分两阶段：
1. **训练阶段**：编码器接收图像对，预测一组latent codes，这些codes作为条件输入SIREN网络来生成形变场
2. **推理阶段**：可以直接用编码器预测结果（快速），也可以进一步做instance optimization（更精确但稍慢）

输入为一对3D医学图像（moving + fixed），输出为一个连续的形变场函数，通过SIREN网络参数化。

### 关键设计
1. **SIREN-based形变表示**: 不用离散体素格点存储形变向量，而是用SIREN（Sinusoidal Representation Network）作为连续函数来表示形变场。任意坐标点的形变可以通过网络前向传播得到，分辨率不受固定网格限制。这是内存效率的核心来源——网络参数量远小于高分辨率体素网格。

2. **Latent Code条件化**: 不同图像对的形变通过不同的latent code来区分。编码器（类CNN架构）从图像对中预测latent code，然后将其注入SIREN的各层作为conditioning信号，使同一个SIREN主干能表示不同的形变。

3. **多分辨率配准策略**: 采用coarse-to-fine的多分辨率方案。先在低分辨率下估计粗略形变，再在高分辨率下精化。因为SIREN是连续函数表示，分辨率切换不需要改变网络结构，只需在不同密度的采样点上评估即可。

4. **近似微分同胚性保证**: 将SIREN的输出解释为速度场（velocity field），通过积分（scaling and squaring）得到最终形变场。速度场的连续性和光滑性自然地促进微分同胚性，不需要额外正则化约束。

### 损失函数 / 训练策略
- 图像相似性损失（如NCC/MSE）驱动配准精度
- 支持instance optimization：测试时可以冻结SIREN权重，仅优化latent code来细化配准结果
- 训练时的内存优势来自于：SIREN是在随机采样的坐标子集上评估的，不需要一次性计算整个体积的形变

## 实验关键数据
| 数据集 | 设置 | 关键发现 |
|--------|------|----------|
| 2D合成数据 | 单分辨率 | 验证基本原理可行，形变规则性优于体素方法 |
| DirLab COPDGene 肺 | 多分辨率 | 配准精度与VoxelMorph等SOTA相当 |
| OASIS 脑 | 多分辨率+IO | 匹配SOTA (如TransMorph, VoxelMorph) 精度，内存减少5倍 |

### 消融实验要点
- SIREN vs 体素表示：单分辨率下精度相当，但内存显著降低
- 有无instance optimization：IO带来明显的精度提升，是缩小与体素方法差距的关键
- 内存对比：3D高分辨率下，NePhi的训练和推理内存消耗约为体素方法的1/5

## 亮点 / 我学到了什么
- **用连续函数替代离散网格**是解决3D高分辨率配准内存瓶颈的优雅方案
- **latent code + universal decoder**的架构思路具有通用性，可以迁移到其他需要个性化连续函数的场景
- 随机坐标子采样训练策略巧妙回避了全分辨率前向传播的内存问题
- 实验展示了accuracy-memory-regularity三角权衡的全面分析，方法论值得学习

## 局限性 / 可改进方向
- 单分辨率下精度与体素方法持平但未超越，需要多分辨率+IO才能匹配SOTA
- Instance optimization增加了推理时间，对于需要实时配准的场景仍有挑战
- SIREN的频率参数选择可能影响能捕获的形变频率范围
- 尚未在超大规模数据集上验证泛化性
- → 相关idea: [Hash-Accelerated Neural Deformation Fields](../../../ideas/medical_imaging/20260317_hash_accelerated_neural_deformation.md)

## 与相关工作的对比
- **VoxelMorph/TransMorph**: 体素化方法精度高但内存大；NePhi内存小但需要IO补精度
- **IDIR/NIR (纯优化INR配准)**: NePhi通过编码器预测latent code大幅加速推理，无需从头优化
- **GradICON**: 重点在梯度逆一致性保证微分同胚；NePhi通过连续速度场积分方案实现

## 与我的研究方向的关联
- 已有直接相关idea: [Hash-Accelerated Neural Deformation Fields for Real-Time Diffeomorphic Registration](../../../ideas/medical_imaging/20260317_hash_accelerated_neural_deformation.md)，思路是用hash encoding替代SIREN加速
- NePhi的latent code条件化方案可以启发多模态对齐任务中的个性化变换建模

## 评分
- 新颖性: ⭐⭐⭐⭐ 将INR引入配准的功能性表示是有新意的，但类似思路在NeRF领域已广泛使用
- 实验充分度: ⭐⭐⭐⭐ 2D+3D多数据集，memory/accuracy/regularity三维对比全面
- 写作质量: ⭐⭐⭐⭐ 清晰的框架图和系统性的对比分析
- 对我的价值: ⭐⭐⭐⭐ 已经基于此论文生成了具体的idea，具有直接参考价值
