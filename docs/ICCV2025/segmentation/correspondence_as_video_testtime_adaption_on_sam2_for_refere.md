# Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild

**会议**: ICCV 2025  
**arXiv**: [2508.07759](https://arxiv.org/abs/2508.07759)  
**代码**: [https://github.com/wanghr64/cav-sam](https://github.com/wanghr64/cav-sam)  
**领域**: 分割 / 少样本分割 / 跨域分割  
**关键词**: SAM2, reference segmentation, 伪视频, 扩散模型过渡, test-time adaptation, few-shot segmentation  

## 一句话总结
将reference-target图像对之间的对应关系表示为用扩散模型生成的伪视频序列，利用SAM2的iVOS能力进行分割，结合test-time轻量微调对齐几何变化，在跨域few-shot分割上比SOTA方法提升约5% mIoU，且无需meta-training。

## 背景与动机
SAM/SAM2虽然分割能力强大，但在下游领域（医学影像、遥感等）存在域差异和类别新颖性问题。Reference segmentation（给定参考图像+mask来分割目标图像）是一种有前途的适应方案，但现有方法（FSS/CD-FSS）主要依赖meta-learning，需要大量episodic训练数据和计算资源。一个关键洞察是：SAM2已经具备了iVOS（交互式视频目标分割）能力——只要标注视频中某几帧，就能跟踪分割整个视频。如果能把reference-target图像对"伪装"成视频帧，就能直接利用SAM2的iVOS能力，避免meta-training。

## 核心问题
如何将离散的reference-target图像对转化为SAM2可处理的连续伪视频序列？核心挑战有二：(1) 语义差异——iVOS跟踪的是同一实例，但reference segmentation需要处理同类不同实例的类内变化；(2) 几何变化——reference和target中物体的姿态、大小、遮挡可能差异巨大，超出iVOS的平滑假设。

## 方法详解

### 整体框架
CAV-SAM（Correspondence As Video for SAM）包含两个核心模块：(1) DBST用扩散模型在reference和target之间生成中间过渡帧，构建语义平滑的伪视频序列；(2) TTGA通过test-time微调获取prototype向量来激活伪视频帧中的目标物体，作为SAM2的额外prompt，对齐几何变化。全程不需要meta-training。

### 关键设计
1. **DBST（Diffusion-Based Semantic Transition）**：基于DiffMorpher改造，用扩散模型在reference和target图像之间生成语义过渡序列。具体做法：分别对两张图像LoRA微调扩散模型，得到$\Delta\theta_r$和$\Delta\theta_t$，然后对LoRA参数做线性插值$\Delta\theta_\alpha = (1-\alpha)\Delta\theta_r + \alpha\Delta\theta_t$，同时对DDIM反转的latent noise做球面线性插值。通过不同$\alpha$值（0.2-0.8共9帧）生成中间帧，得到语义平滑过渡的伪视频。关键优化：去掉DiffMorpher中不必要的视觉精炼模块（基础语义过渡对SAM2已足够），显著降低计算成本。

2. **TTGA（Test-Time Geometric Alignment）**：仅用reference图像和mask做轻量test-time微调（只微调SAM2 image encoder的FPN neck层，100步）。提出增强循环一致性（ACC）损失：用prototype向量在增强图像上预测pseudo-label，再用结果反过来在原图上验证。微调后的encoder提取更好的特征，prototype向量用来激活伪视频帧中的目标区域（用余弦相似度+Otsu阈值），生成pseudo-label作为SAM2的额外prompt，帮助其对齐几何变化。

3. **核心洞察——简单拼接就接近SOTA**：即使只是简单拼接reference-target图像，SAM2就能达到60.68 mIoU（接近CD-FSS SOTA的~61）。这证明了iVOS模型用于reference segmentation的巨大潜力。DBST+TTGA进一步提升到64.06。

### 损失函数 / 训练策略
- DBST：LoRA rank=16, lr=2e-4, 200步，DDIM inversion 20步
- TTGA：lr=1e-3, cosine annealing, 100步，仅微调FPN neck
- 损失：$\mathcal{L} = \mathcal{L}_{aug} + \mathcal{L}_{cyc}$（BCE loss）
- 使用SAM2-tiny模型

## 实验关键数据
| 方法 | 训练需求 | Deepglobe | ISIC | Chest X-Ray | FSS-1000 | Avg (1-shot) |
|------|----------|-----------|------|-------------|----------|-------------|
| SSP (ECCV22) | 大规模meta-train | 40.48 | 35.09 | 74.23 | 79.03 | 57.20 |
| APSeg (CVPR24) | meta-train+SAM | 35.94 | 45.43 | 84.10 | 79.71 | 61.30 |
| DR-Adaptor (CVPR24) | meta-train | 41.29 | 40.77 | 82.35 | 79.05 | 60.86 |
| **CAV-SAM** | **TTA only** | **39.11** | **50.36** | **86.97** | **79.78** | **64.06** |

- 平均mIoU比SOTA方法提升约**5%**（61.30→64.06 for 1-shot）
- 5-shot场景更强：69.14 avg，比APSeg的65.09提升4.05
- 在最挑战的Chest X-Ray上提升最大：84.10→86.97（+2.87）
- 简单拼接baseline已达60.68，DBST+TTGA逐步提升到64.06
- 也可以用DEVA替代SAM2：49.57→59.93

### 消融实验要点
- DBST贡献+2.0 mIoU，TTGA贡献+1.38 mIoU
- 启发式方法（mixup/affine增强）反而比拼接baseline更差（52.21/56.84 vs 60.68），证明自然的语义过渡很重要
- ACC优于ABC（64.06 vs 61.54），循环一致性比双向一致性更鲁棒
- DBST对视觉质量不敏感——轻量版输出虽不如完整DiffMorpher好看，但分割效果一致

## 亮点
- **视角转换极具创造性**：把reference segmentation重新定义为"伪视频的iVOS问题"，绕过了meta-learning框架的全部复杂性
- **SAM2拼接就接近SOTA**的发现很有启发性——说明iVOS模型已经隐式学到了强大的图像间对应能力
- **扩散模型用于语义过渡**：不是用来生成"好看"的图像，而是用来构建自然的语义桥梁，应用角度新颖
- **完全无meta-training**：只需test-time微调100步，极大降低了数据和计算需求
- **语义一致性处理**：当reference和target类别不同时，DBST自然生成"无意义"过渡，TTGA不激活任何区域，SAM2不产生分割——系统会自动"失败"

## 局限性 / 可改进方向
- 扩散模型生成伪视频有一定时间开销（LoRA训练+DDIM反转）
- 在Deepglobe（区域分割而非物体分割）上效果不如meta-learning方法
- 伪视频帧数（N_v=9）是固定的，更长的过渡序列可能进一步改善
- SAM2-tiny的容量可能限制了分割精度，更大的SAM2可能带来更好效果
- test-time微调100步对实时应用仍然较慢

## 与相关工作的对比
- **vs. APSeg/VRP-SAM**：这些方法给SAM引入meta-learning prompt encoder，需要大量episodic训练；CAV-SAM完全不需要meta-training
- **vs. CD-FSS方法（PATNet/IFA/DR-Adaptor）**：这些需要在大量数据上meta-train；CAV-SAM用diffusion+TTA替代
- **vs. SAM2Long**：SAM2Long解决的是长视频中的错误累积；CAV-SAM解决的是将离散图像对转化为伪视频的问题——不同场景但都是对SAM2的创新应用

## 启发与关联
- **idea潜力**：伪视频思路可以扩展到其他需要pair-wise对应的视觉任务（如图像检索、配准、变化检测）
- 扩散模型作为"语义桥梁"的应用角度很新，值得在其他跨域/跨模态任务中探索
- 与SAM2Long的组合使用：DBST生成伪视频 + SAM2Long的树搜索 = 更鲁棒的inference pipeline

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "对应关系即视频"的视角转换是真正的创意，扩散模型生成语义过渡序列的应用角度全新
- 实验充分度: ⭐⭐⭐⭐ 4个CD-FSS数据集、多种baseline对比、启发式方法vs扩散方法、ACC vs ABC消融
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰（Figure 2的语义差异/几何变化分析），方法描述系统
- 价值: ⭐⭐⭐⭐ 无meta-training的reference segmentation是重要方向，但扩散模型开销限制了实时性
