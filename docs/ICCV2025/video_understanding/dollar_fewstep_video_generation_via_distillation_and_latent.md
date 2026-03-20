# DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization

**会议**: ICCV 2025  
**arXiv**: [2412.15689](https://arxiv.org/abs/2412.15689)  
**代码**: 无（未提及）  
**领域**: 视频生成 / 扩散模型加速  
**关键词**: video generation, distillation, consistency distillation, latent reward, few-step, VBench  

## 一句话总结
结合变分分数蒸馏（VSD）和一致性蒸馏实现few-step视频生成，同时提出潜空间奖励模型微调方法进一步优化生成质量，4步生成的10秒视频（128帧@12FPS）在VBench上达82.57分超越teacher模型和Gen-3/Kling等基线，1步蒸馏实现278.6倍加速。

## 背景与动机
视频扩散模型需要大量采样步（通常50+）才能生成高质量视频，计算成本极高。SANA-Sprint等工作已经在图像领域实现了1-4步生成，但视频领域的步骤蒸馏更具挑战——视频需要在时间维度上保持一致性，简单减少步数容易导致时间闪烁和质量退化。同时，现有视频蒸馏方法通常只能用于特定步数（如只能4步），缺乏灵活性。

## 核心问题
如何在大幅减少采样步数的同时保持视频的质量和多样性？如何通过奖励信号进一步优化蒸馏后模型的特定质量指标？

## 方法详解

### 整体框架
DOLLAR分两阶段：(1) 混合蒸馏——结合变分分数蒸馏(VSD)和一致性蒸馏将teacher的50步能力压缩到1-4步；(2) 潜空间奖励优化——用latent reward model对蒸馏后的student进一步微调，提升特定质量维度。

### 关键设计
1. **VSD + 一致性蒸馏的混合方案**：VSD通过分布匹配将teacher的多步分布对齐到student的少步分布，保证生成多样性；一致性蒸馏确保student在不同步数下的输出一致，避免step-specific训练。两者互补——VSD保多样性，一致性保质量。

2. **潜空间奖励模型微调**：不需要reward模型可微，而是在潜空间中操作——大幅减少GPU显存需求。可以针对任意指定的奖励指标（如美学质量、文本对齐、时间一致性等）进行优化。这使得蒸馏后的模型不仅速度快，还能在特定维度上超越teacher。

3. **10秒长视频的few-step生成**：验证了128帧@12FPS（约10秒）视频的few-step生成——这比之前大多数方法验证的短视频（2-4秒）更具挑战性。

### 损失函数 / 训练策略
VSD loss + consistency loss混合，后接latent reward fine-tuning。

## 实验关键数据
| 方法 | Steps | VBench Score |
|------|-------|-------------|
| Teacher (50步 DDIM) | 50 | < 82.57 |
| Gen-3 | - | < 82.57 |
| T2V-Turbo | - | < 82.57 |
| Kling | - | < 82.57 |
| **DOLLAR (4步)** | **4** | **82.57** |

- 4步student在VBench上**超越teacher模型**及Gen-3、Kling等商业模型
- 1步蒸馏实现**278.6倍加速**，接近实时
- 人类评估进一步验证4步模型优于50步teacher
- 10秒长视频（128帧）验证——更长更具挑战

### 消融实验要点
- 混合蒸馏 > 单独VSD > 单独一致性蒸馏
- Latent reward微调在VBench上进一步提升分数
- Latent reward的优势：内存高效，不要求reward可微

## 亮点
- **蒸馏后student超越teacher**：4步生成质量超过50步采样——这是SANA-Sprint在图像领域也实现的目标，现在扩展到了视频
- **278.6倍加速**的1步生成使实时视频生成成为可能
- **Latent reward微调**是practical的创新——不需要reward可微，内存友好，且可以针对任意质量维度优化
- **10秒视频验证**比大多数同类工作更长，更接近实际应用需求
- 来自Adobe Research，实用导向

## 局限性 / 可改进方向
- 未开源代码/模型
- VBench作为评估指标的局限性——可能不完全反映人类偏好
- 蒸馏过程的训练成本未详细报告
- 未与最新的CogVideoX/Wan等开源模型比较

## 与相关工作的对比
- **vs. SANA-Sprint**：SANA-Sprint用sCM+LADD在图像领域实现1步生成；DOLLAR用VSD+一致性蒸馏在视频领域实现——都是混合蒸馏思路，但domain不同
- **vs. T2V-Turbo**：T2V-Turbo也做视频步骤蒸馏，DOLLAR在VBench上超越
- **vs. AnimateLCM**：AnimateLCM基于LCM做视频加速，DOLLAR的混合方案更先进

## 启发与关联
- Latent reward微调的思路可以与VACE结合——在统一视频编辑框架中用reward微调特定编辑质量
- 将DOLLAR的蒸馏方法应用到Dita的动作扩散去噪上——加速机器人控制的响应时间

## 评分
- 新颖性: ⭐⭐⭐⭐ VSD+一致性混合蒸馏在视频领域是新应用，latent reward微调实用
- 实验充分度: ⭐⭐⭐⭐ VBench全面评估，人类评估验证，10秒长视频测试
- 写作质量: ⭐⭐⭐⭐ 方法清晰
- 价值: ⭐⭐⭐⭐ 视频扩散模型加速的重要进展，接近实时视频生成
