# Active Generation for Image Classification

**会议**: ECCV 2024  
**arXiv**: [2403.06517](https://arxiv.org/abs/2403.06517)  
**代码**: [https://github.com/hunto/ActGen](https://github.com/hunto/ActGen) (有)  
**领域**: 图像生成 / 数据增强 / 图像分类  
**关键词**: 主动学习, 扩散模型数据增强, 注意力引导生成, 对抗样本生成, 图像分类  

## 一句话总结
ActGen将主动学习思想引入扩散模型数据增强，通过识别分类器的错分样本并以注意力掩码引导+梯度对抗引导生成"难样本"，仅用10%的合成数据量即超越了此前需要近等量合成数据的方法，在ImageNet上ResNet-50获得+2.26%的精度提升。

## 背景与动机
利用扩散模型生成合成数据来改善图像分类是近年的热点方向。Azizi等人(2023)用Imagen生成与ImageNet等量的120万张图才获得+1.78%的精度提升，计算代价极高、性价比低。其他方法如Da-Fusion、Real Guidance等也采用"先大量生成、再用于训练"的两阶段策略，无法感知模型实际需要什么样的数据，导致大量冗余生成。

核心问题在于：**生成的图像是否真的对模型有用？** 大部分随机生成的图像对已经学好的模式是冗余的，真正有价值的是那些模型还没学好的困难样本。

## 核心问题
如何让扩散模型**有目标地**生成对分类器训练最有帮助的图像，而不是无差别地大量生成？即：在大幅减少生成数量的前提下，如何最大化合成数据对分类精度的贡献？

## 方法详解

### 整体框架
ActGen是一个"训练感知"的在线生成增强框架。它将数据生成和模型训练统一到同一个循环中：
1. 从训练集中划分出一个验证集（如10K张）
2. 每个epoch结束后，用当前模型在验证集上推理，找出**被错分的样本**作为"难样本原型"
3. 以这些难样本作为引导图，通过Stable Diffusion生成变体
4. 将生成的图像加入训练集，继续训练
5. 仅在训练前半程进行生成（后半程学习率小，新数据作用有限）

这种设计实现了"按需生成"——模型哪里弱就补哪里。

### 关键设计

1. **注意力图像引导（Attentive Image Guidance）**: 以错分的真实图像作为引导，在DDPM每个去噪步中，将生成的latent与引导图像的latent做插值。为了保持前景物体不变但背景多样化，利用cross-attention层中文本embedding与图像特征之间的注意力来提取前景掩码$\mathbf{m}_t$，只在前景区域做引导插值：$\tilde{\mathbf{x}}_{t-1} = \mathbf{x}_{t-1} + \mathbf{m}_t \odot \gamma_t (\mathbf{x}_{t-1}^{(g)} - \mathbf{x}_{t-1})$，其中$\gamma_t$由sigmoid函数控制引导强度。

2. **对比损失引导（Contrastive Loss）**: 维护一个memory bank存储所有已生成图像的latent。用欧氏距离度量当前生成latent与同类已生成图像的距离，当距离小于margin $\rho=200$时施加惩罚，避免重复生成相似的图像，增加多样性。

3. **对抗分类损失（Adversarial Classification Loss）**: 将去噪后的图像送入当前分类器，计算**负交叉熵损失** $\mathcal{L}_{adv} = -\text{CE}(\Omega(\mathbf{o}_t), y)$，即最大化分类损失。这促使生成的图像在模糊、遮挡、对比度变化等方面更"难"，而不是简单加噪。

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{contra} + \lambda \mathcal{L}_{adv}$
- 梯度更新的是**文本embedding** $\mathbf{c}$（而非图像latent），通过归一化梯度下降：$\mathbf{c}_{t-1} = \mathbf{c}_t - \nu \frac{\nabla_{\mathbf{c}_t}\mathcal{L}}{\|\nabla_{\mathbf{c}_t}\mathcal{L}\|_2}$
- 对抗样本生成概率随训练进行逐步增大（借鉴课程学习，概率为$0.5 \times \frac{\text{epoch}}{\text{total\_epoch}}$）
- 梯度引导仅在DDPM前10步（共40步）中做，平衡计算开销与生成质量
- 使用Stable Diffusion V2.1 base，40步DDPM采样，classifier-free guidance scale=15

## 实验关键数据

| 数据集 | 模型 | 生成量(M) | 生成/真实 | 本文ACC | 之前SOTA | 提升 |
|--------|------|----------|-----------|---------|----------|------|
| ImageNet | ResNet-50 | 0.13 | 10% | 78.65 | 78.17 (Azizi, 94%) | +2.26 vs baseline |
| ImageNet | ResNet-152 | 0.13 | 10% | 80.87 | 80.15 (Azizi, 94%) | +2.28 vs baseline |
| ImageNet | ViT-S/16 | 0.08 | 6% | 81.12 | 81.00 (Azizi, 94%) | +1.23 vs baseline |
| ImageNet | DeiT-B/16 | 0.08 | 6% | 83.29 | 82.84 (Azizi, 94%) | +1.50 vs baseline |
| CIFAR-10 | ResNet-50 | 9.6K | 19% | 95.53 | 95.26 (SD random) | +0.51 vs baseline |
| CIFAR-100 | ResNet-50 | 9.6K | 19% | 77.33 | 77.17 (SD random) | +0.27 vs baseline |
| EuroSAT few-shot | CLIP-RN50 | 2K | - | 87.25(16-shot) | 86.27 (Real guidance) | +0.98 |

### 消融实验要点
- **各模块贡献递增**: Random→Image Guidance(+1.29)→Attentive IG(+0.22)→+$\mathcal{L}_{contra}$(+0.21)→+$\mathcal{L}_{adv}$(+0.29)，所有模块都有正向贡献
- **验证集大小**: <1K时类别覆盖不足性能差，≥5K后趋于稳定，默认用10K
- **生成数量**: 1K即有提升，130K为sweet spot(78.65)，过多(1300K)反而下降到78.03，但调整选择阈值后可到79.18
- **训练开销**: 总计15.2 GPU days（生成4.5+训练10.7），比Azizi等人的~40+ GPU days显著更少
- **对比Focal Loss**: Focal Loss在ImageNet上反而降0.36，而ActGen提升2.26，说明"生成难样本"比"加权难样本"更有效

## 亮点
- **主动学习+生成模型的巧妙结合**: 不是被动生成再筛选，而是根据模型弱点主动生成需要的数据，这个范式转换很优雅
- **注意力掩码实现前景保留+背景多样化**: 利用cross-attention天然定位前景的特性做选择性引导，无需额外分割模型
- **通过更新text embedding实现梯度引导**: 不直接操控图像latent，而是通过text embedding间接控制，保证了扩散模型生成质量
- **极高的数据效率**: 仅用10%的合成数据量就超越了100%数据量的方法，具有很强的实用价值
- **课程学习式的对抗策略**: 对抗强度随训练渐增，避免早期生成过难的噪声样本干扰训练

## 局限性 / 可改进方向
- **依赖预训练扩散模型质量**: 对于Stable Diffusion覆盖不好的领域（如医学图像、遥感），效果可能受限
- **只考虑错分样本作为难样本**: 低confidence正确预测的边界样本也可能有价值，选择策略可以更丰富
- **验证集划分减少了训练数据**: 从训练集分出10K作为验证集，对小数据集影响更大
- **生成开销仍然不小**: 虽然大幅减少了生成量，但4.5 GPU days的生成开销对于资源受限的场景仍有门槛
- **没有探索多轮迭代生成**: 当前每epoch生成一次，是否可以更有策略地控制生成频率？
- **可扩展到其他任务**: 目标检测、语义分割等密集预测任务的"难样本"定义更复杂，需要新的适配 → 参见 [ideas/object_detection/20260317_active_freq_detection.md](../../../ideas/object_detection/20260317_active_freq_detection.md)

## 与相关工作的对比
- **vs. Azizi et al. (2023)**: Azizi用Imagen生成等量数据(120万张)获+1.78%，ActGen用10%(13万张)获+2.26%。且Azizi用的是非开源的更强Imagen，ActGen用开源Stable Diffusion。关键区别：Azizi是"两阶段+无目标生成"，ActGen是"在线+主动生成"。
- **vs. Real Guidance (He et al., 2023)**: Real Guidance用真实图像引导扩散模型生成，但不感知模型状态，也不控制生成难度。ActGen在其基础上加入了模型感知(错分样本选择)和难度控制(对抗损失)。
- **vs. Da-Fusion (Trabucco et al., 2023)**: Da-Fusion通过学习文本embedding的扰动来增加多样性，但同样是模型无关的生成策略。ActGen的对比损失和对抗损失提供了更细粒度的控制。

## 启发与关联
- **与主动学习检测idea的关联**: [ActiveFreq检测](../../../ideas/object_detection/20260317_active_freq_detection.md) 中"频域难度评估"与ActGen的"错分样本识别"思路相通，可以考虑将频域特征作为难度补充信号来指导生成
- **可扩展思路**: ActGen的框架可推广到目标检测（生成包含难检测物体的场景）、语义分割（生成边界模糊的区域）等任务
- **text embedding梯度更新**是一个通用的扩散模型控制技术，可以用于其他需要对生成内容做细粒度控制的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 主动学习+扩散生成的结合思路新颖，但各个子模块技术并非全新
- 实验充分度: ⭐⭐⭐⭐ 覆盖ImageNet/CIFAR/few-shot多场景，消融丰富，但缺少更多实际应用场景
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机和方法阐述流畅
- 价值: ⭐⭐⭐⭐⭐ 数据效率提升显著，具有很强的实用意义和范式引领价值
