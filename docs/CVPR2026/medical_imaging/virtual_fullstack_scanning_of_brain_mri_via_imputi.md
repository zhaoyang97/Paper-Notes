# CodeBrain: Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code

**会议**: CVPR 2026  
**arXiv**: [2501.18328](https://arxiv.org/abs/2501.18328)  
**代码**: [GitHub](https://github.com/ycwu1997/CodeBrain)  
**领域**: 医学影像 / MRI合成  
**关键词**: brain MRI, modality imputation, scalar quantisation, any-to-any synthesis, VQ-VAE  

## 一句话总结
CodeBrain将脑MRI多模态补全(any-to-any imputation)重新定义为区域级全栈量化码预测问题：Stage I用有限标量量化(FSQ)将完整MRI集编码为紧凑code map + 模态无关公共特征，Stage II从不完整模态预测code map(用grading loss保持量化空间平滑性)，在IXI和BraTS 2023上超越5种SOTA方法，生成的模态可接近真实数据的脑肿瘤分割性能。

## 背景与动机
脑MRI包含多种模态(T1/T2/PD/FLAIR/T1Gd)，各自强调不同解剖/病理特征，但临床中因扫描时间、成本、造影剂风险等限制无法采全。现有统一补全方法的局限：(1) 全局条件(binary vector/learnable query)无法捕获区域级跨模态变化；(2) 模态特定模块(decoder per modality)参数量随模态数线性增长，泛化差；(3) 像素级翻译缺乏对跨模态关系的显式建模。

## 核心问题
如何构建统一模型，能从任意不完整MRI模态组合补全缺失模态(any-to-any)，且不需要模态特定模块？

## 方法详解

### 整体框架
两阶段pipeline：Stage I(学紧凑表示)：完整MRI集 → 后验编码器E_posterior → FSQ量化得full-stack code map Z^_full (d=6, h×w空间)。同时，不完整输入 → 公共编码器E_c → 模态无关特征F_c。解码器D从[Z^_full, F_c]重建完整MRI。Stage II(学code预测)：不完整输入 → 先验编码器E_prior → 预测code map Z~_full，用grading loss监督(target来自Stage I)。推理时：不完整输入 → E_prior预测code → E_c提取公共特征 → D解码输出补全结果。

### 关键设计
1. **有限标量量化(FSQ)区域级表示**: 将complete MRI编码到d=6个通道的code map，每个位置的6-维码代表一个图像patch的全栈特征。量化为L=[8,8,8,5,5,5]级离散值(共64K码本大小)，用round操作+straight-through estimator做端到端训练。vs VQ-VAE的codebook学习：FSQ无需学码本，直接round到有限整数值，训练更稳定。关键优势：所有完整MRI样本投射到同一个有限量化空间，使跨模态翻译简化为离散code预测。

2. **模态无关公共特征F_c**: 从任意不完整输入提取，训练时随机mask模态(0<K<N)，确保F_c不依赖特定模态。F_c补充量化表示丢失的高频细节——无F_c时PSNR降4.17dB。解码时[Z^_full, F_c]拼接输入D。

3. **Grading Loss做code预测**: 将离散code预测建模为序位回归(ordinal regression)而非分类——每个标量值转换为有序二值决策序列o=[1,1,...,1,0,...,0]，用BCE loss训练。vs cross-entropy：保持量化空间中相邻code的语义相似关系(CE假设所有类等距)。实验验证grading优于classification(PSNR 29.50 vs 29.24)。

### 损失函数 / 训练策略
Stage I: L_rec = λ_[m,a] × L_psnr(重建) + L_gan(LSGAN)。λ_m=20(缺失模态权重高), λ_a=5。Stage II: L_grad = BCE(O~, O^)。NAFNet骨干，AdamW lr=1e-4，batch 48，每stage 300 epochs，8×4090 GPU共2.38天。

## 实验关键数据

### IXI数据集(T1/T2/PD三模态, 平均9种场景)
| 方法 | PSNR↑ | SSIM↑ | MAE↓ |
|------|-------|-------|------|
| MMGAN | 27.94 | 91.22 | 20.75 |
| MMT | 28.38 | 91.79 | 19.55 |
| M2DN | 28.40 | 92.08 | 19.18 |
| Zhang et al. | 29.31 | 92.96 | 17.83 |
| **CodeBrain** | **29.84** | **93.35** | **16.97** |

### BraTS 2023(T1/T2/FLAIR/T1Gd四模态)
| 方法 | PSNR↑ | SSIM↑ | MAE↓ |
|------|-------|-------|------|
| Zhang et al. | 24.95 | 89.90 | 21.68 |
| **CodeBrain** | **25.26** | **90.42** | **21.14** |

### 下游脑肿瘤分割(3D Dice %)
CodeBrain补全的T1Gd → 分割Dice 61.44%，vs Full Modalities 87.40%，vs零填充仅6.2%。T1Gd是最难补全的增强模态。

### 消融实验要点
- **去掉F_c**: 重建PSNR从34.32降至30.15 (-4.17dB)，说明公共特征对细节恢复关键
- **Classification vs Grading**: Grading优于Classification (29.50 vs 29.24 PSNR)
- **PSNR loss vs L1 loss**: PSNR loss略优(34.32 vs 34.08)但SSIM略降
- **量化条件 vs 连续条件**: 离散码比连续变量做条件效果更好(infinite连续→补全性能大幅下降)，量化提供合适的表达力-可预测性平衡

## 亮点 / 我学到了什么
- **将图像翻译问题转化为离散码预测**: 这是核心insight——比直接pixel-to-pixel翻译更简单、更统一。量化后所有模态共享同一有限空间，消除了模态特定模块的需求
- **FSQ比VQ-VAE更稳健**: 不需要学codebook(避免codebook collapse)，直接round到整数值+STE，工程简洁
- **Grading loss保持量化空间结构**: ordinal regression保留"3比2大但比4小"的关系，是处理有序离散预测的好方案
- **模态无关公共特征的设计**: 通过训练时随机mask实现"任何子集都能提取F_c"，简洁优雅

## 局限性 / 可改进方向
- 仅做2D slice处理，未利用3D空间信息(作者在limitation中提到)
- T1Gd的补全质量仍显著低于真实数据(分割Dice 61.44% vs 87.40%)，增强造影模态难以从非造影模态完全推断
- 可能仍存在hallucination问题
- 训练时间2.38天(8×4090)，资源需求较高
- 仅在脑MRI上验证，其他部位(心脏/腹部)的泛化性未知

## 与相关工作的对比
- **vs MMT/M2DN**: 用全局条件(binary vector/modality query)，无法捕获区域级变化。CodeBrain的量化码在每个patch位置都有不同值
- **vs Zhang et al.[49]**: 用模态共享+模态特定双编码器，参数随模态数增长。CodeBrain完全模态无关
- **vs VQGAN/VQVAE**: 这些做单模态生成，CodeBrain将VQ扩展到跨模态翻译，且用FSQ替代传统VQ避免codebook问题

## 与我的研究方向的关联
- "将跨模态翻译转化为离散码预测"的范式可能迁移到其他多模态生成任务
- Grading loss适用于任何需要预测有序离散值的场景
- 与VLM的token化(tokenization)思路有概念联系

## 评分
- 新颖性: ⭐⭐⭐⭐ 将MRI补全重定义为code prediction是新视角，FSQ+grading loss组合有效
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、多场景(O→O/M→O)、详细消融(F_c/grading/loss/条件类型/权重)、下游分割验证
- 写作质量: ⭐⭐⭐⭐ 清晰完整，图表信息量大，方法动机论证充分
- 对我的价值: ⭐⭐⭐ 医学影像非核心方向，但量化码做跨模态翻译的思路有启发
