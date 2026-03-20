# SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

**会议**: ICCV 2025  
**arXiv**: [2503.09641](https://arxiv.org/abs/2503.09641)  
**代码**: [https://github.com/NVlabs/Sana](https://github.com/NVlabs/Sana)  
**领域**: 图像生成 / 扩散模型加速  
**关键词**: one-step diffusion, consistency distillation, flow matching, adversarial distillation, 实时生成  

## 一句话总结
将预训练的SANA flow matching模型通过无损数学变换转化为TrigFlow，结合连续时间一致性蒸馏（sCM）和潜空间对抗蒸馏（LADD）的混合策略，实现统一的1-4步自适应高质量图像生成，1步生成1024×1024图像仅需0.1s（H100），以7.59 FID和0.74 GenEval超越FLUX-schnell且速度快10倍。

## 背景与动机
扩散模型通常需要20-100步采样，严重限制了实时应用。现有加速方法分为轨迹蒸馏（如一致性模型CM）和分布蒸馏（如GAN/VSD），但各有局限：GAN训练不稳定易模式崩塌，VSD需要额外扩散模型增加显存开销，离散时间CM在极少步（<4步）时质量退化。更关键的是，连续时间一致性模型（sCM）需要TrigFlow格式的模型，而主流模型（SANA/FLUX/SD3）都是flow matching，如果从头训练TrigFlow模型开销巨大。

## 核心问题
如何在不从头训练的前提下，将现有flow matching模型高效转化为支持1步生成的一致性模型，同时保持生成质量和多步灵活性？

## 方法详解

### 整体框架
SANA-Sprint基于预训练的SANA模型，三步走：(1) 无损数学变换将flow matching转为TrigFlow格式；(2) 用sCM进行连续时间一致性蒸馏保持与teacher对齐 + LADD对抗蒸馏增强单步保真度；(3) 统一训练出step-adaptive模型，1-4步共用同一模型。

### 关键设计
1. **Training-Free Flow→TrigFlow变换**：核心创新——通过严格的数学推导（Proposition 3.1），将flow matching模型的输入/输出通过可微变换转化为TrigFlow格式。变换公式涉及时间重映射（$t_{FM} = \sin(t_{Trig})/(\sin(t_{Trig})+\cos(t_{Trig}))$）、输入尺度调整和输出线性组合。理论和实验验证完全无损（FID 5.81→5.73，50步）。这消除了从头预训练TrigFlow的需求，使sCM框架可以直接应用于任何flow matching模型。

2. **sCM + LADD混合蒸馏**：sCM通过连续时间一致性loss学习ODE轨迹上的自一致映射，保持与teacher的分布对齐和多样性；LADD用teacher模型特征作为判别器在潜空间进行对抗学习，增强单步生成的保真度。两者互补：sCM alone FID=8.93, LADD alone FID=12.20, 结合 FID=8.11。

3. **训练稳定化技术**：(a) Dense Time Embedding——将噪声系数从1000t改为t，使时间导数稳定、收敛加速；(b) QK-Normalization——在self/cross attention中加入RMS norm稳定大模型（1.6B）的梯度，防止训练崩溃；(c) Max-Time Weighting——在LADD中以50%概率将训练时间步设为π/2（即纯噪声），显著增强1步生成质量。

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{sCM} + 0.5 \cdot \mathcal{L}_{adv}$
- Teacher从SANA-1.5 4.8B剪枝微调；先微调teacher 5K步适配dense time embedding+QK norm，再蒸馏student 20K步
- 32卡A100，全局batch size 512

## 实验关键数据
| 方法 | Steps | Params | FID↓ | GenEval↑ | Latency(A100) |
|------|-------|--------|------|----------|---------------|
| FLUX-dev | 50 | 12B | 10.15 | 0.67 | 23.0s |
| SANA-1.6B | 20 | 1.6B | 5.76 | 0.66 | 1.2s |
| FLUX-schnell | 4 | 12B | 7.94 | 0.71 | 2.10s |
| SD3.5-Turbo | 4 | 8B | 11.97 | 0.72 | 1.15s |
| **SANA-Sprint 0.6B** | **4** | **0.6B** | **6.48** | **0.76** | **0.32s** |
| **SANA-Sprint 1.6B** | **4** | **1.6B** | **6.54** | **0.77** | **0.31s** |
| FLUX-schnell | 1 | 12B | 7.26 | 0.69 | 0.68s |
| SDXL-DMD2 | 1 | 0.9B | 7.10 | 0.59 | 0.32s |
| **SANA-Sprint 0.6B** | **1** | **0.6B** | **7.04** | **0.72** | **0.21s** |
| **SANA-Sprint 1.6B** | **1** | **1.6B** | **7.69** | **0.76** | **0.21s** |

- 1步生成：7.04 FID + 0.72 GenEval，超越FLUX-schnell (7.26/0.69)且速度快3.2x
- RTX 4090上0.31s生成1024×1024，H100上0.1s——真正的AIPC级实时
- ControlNet集成：1024×1024仅0.25s（H100），实现手绘草图→图像的实时交互
- 统一step-adaptive：同一模型在1-4步均表现优异，无需step-specific训练
- 比teacher SANA快8.4x，FLUX-schnell快64.7x（Transformer部分）

### 消融实验要点
- sCM+LADD >> sCM alone >> LADD alone
- Dense time embedding（t vs 1000t）：消除1000x梯度放大，显著改善稳定性
- QK-Norm对1.6B模型至关重要（无则训练崩溃）
- Max-time weighting 50%最优（0%→9.44 FID，50%→8.32 FID）
- CFG embedding提升CLIP score +0.94
- Flow→TrigFlow变换完全无损（FID差异<0.1）

## 亮点
- **Flow→TrigFlow无损变换**是突破性贡献：让任何flow matching模型（FLUX/SD3等）都能直接使用sCM蒸馏框架，无需重新预训练
- **混合蒸馏（sCM+LADD）**互补性强：sCM保对齐和多样性，LADD保单步保真度
- **真正的AIPC级性能**：RTX 4090上0.31s，H100上0.1s——这是消费级GPU实时T2I的里程碑
- **统一step-adaptive**：同一模型1-4步均高质量，大幅简化部署
- **稳定化技术通用性强**：Dense time embedding和QK-Norm可以迁移到其他蒸馏方法

## 局限性 / 可改进方向
- SANA-Sprint 1.6B在1步时FID略逊于0.6B（7.69 vs 7.04）——大模型单步可能需要更多蒸馏迭代
- sCM的JVP计算目前不支持Flash Attention，需使用Linear Attention替代
- 仅在SANA架构上验证，虽声称适用于FLUX/SD3但未实际测试
- ControlNet仅验证了HED边缘条件，其他condition类型未测试

## 与相关工作的对比
- **vs. FLUX-schnell**：同为蒸馏模型，SANA-Sprint用0.6B参数在FID/GenEval上超越12B的FLUX-schnell，速度快10x+
- **vs. LCM/PCM**：离散时间CM在<4步时质量退化严重（SDXL-LCM 1步FID=50.51），SANA-Sprint基于sCM消除离散化误差
- **vs. DMD2**：DMD2是纯分布蒸馏（VSD），需要step-specific模型；SANA-Sprint是混合蒸馏+step-adaptive
- **vs. Dense2MoE**：Dense2MoE用MoE加速DiT推理；SANA-Sprint用蒸馏减少步数——两种加速路径正交

## 启发与关联
- Flow→TrigFlow变换使sCM从理论工具变为实用工具，可能引发一波将现有flow matching模型加速到1步的工作
- sCM+GAN混合策略的思路可以迁移到视频扩散模型的加速——视频生成更需要实时性
- 这项工作与Dynamic-DINO的MoE方法正交——MoE减参数×sCM减步数可能实现更极致的加速

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Flow→TrigFlow无损变换是理论创新，sCM+LADD混合框架设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 6种方法在1/2/4步全面对比，稳定化技术逐项消融，时间步搜索分析详尽
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨（含完整证明），Figure 1/2极具说服力
- 价值: ⭐⭐⭐⭐⭐ 消费级GPU实时T2I的里程碑，开源代码+模型，对社区影响巨大
