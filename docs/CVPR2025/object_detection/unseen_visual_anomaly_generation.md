---
title: >-
  [论文解读] Unseen Visual Anomaly Generation
description: >-
  [CVPR 2025][目标检测][视觉异常生成] 提出 AnomalyAny 框架，利用预训练 Stable Diffusion 的生成能力，通过注意力引导优化和提示引导精化，在仅需单张正常样本且无需额外训练的条件下，生成多样化逼真的未见异常样本。 视觉异常检测 (AD) 面临异常数据稀缺的核心挑战。现有异常生成方法存在两…
tags:
  - "CVPR 2025"
  - "目标检测"
  - "视觉异常生成"
  - "异常检测"
  - "稳定扩散"
  - "注意力引导"
  - "无需训练"
---

# Unseen Visual Anomaly Generation

**会议**: CVPR 2025  
**arXiv**: [2406.01078](https://arxiv.org/abs/2406.01078)  
**代码**: [GitHub](https://hansunhayden.github.io/AnomalyAny.github.io/)  
**领域**: 图像生成  
**关键词**: 视觉异常生成, 异常检测, 稳定扩散, 注意力引导, 无需训练

## 一句话总结

提出 AnomalyAny 框架，利用预训练 Stable Diffusion 的生成能力，通过注意力引导优化和提示引导精化，在仅需单张正常样本且无需额外训练的条件下，生成多样化逼真的未见异常样本。

## 研究背景与动机

视觉异常检测 (AD) 面临异常数据稀缺的核心挑战。现有异常生成方法存在两类局限：

1. **裁剪粘贴方法**（如 DRAEM、NSA）：从外部数据集或图像本身裁剪粘贴随机模式作为异常，操作简单但生成结果不够真实，缺乏对真实异常的语义理解
2. **生成模型方法**（如 AnomalyDiffusion、RealNet）：使用 GAN 或微调扩散模型生成异常，效果更真实但需要充足的训练样本，在数据有限场景不适用，且只能生成已见过的异常类型

根本矛盾在于：异常的稀有性和多变性使得收集代表性样本困难；工业场景产品变体多样，连正常样本都可能不够充分。需要一种**无需训练、仅需单张正常样本、能生成未见异常类型**的方法。

AnomalyAny 的核心 idea：直接利用预训练 SD 的广泛知识进行异常生成而非微调。关键挑战：(i) 异常在训练数据中很稀少，(ii) 异常通常只占图像小区域容易被忽略。通过测试时正常样本条件化+注意力引导优化来解决。

## 方法详解

### 整体框架

AnomalyAny 由三个核心模块构成，在推理时协同工作：
1. **测试时正常样本条件化**：引导生成过程保持与正常分布一致
2. **注意力引导异常优化**：强制 SD 关注异常概念的生成
3. **提示引导异常精化**：利用详细文本描述进一步提升生成质量

### 关键设计

1. **测试时正常样本条件化 (Test-time Normal Sample Conditioning)**:
    - 功能：使生成的异常图像在外观上与目标正常样本保持一致，同时保留 SD 的多样性
    - 核心思路：给定正常样本 x_normal，通过 VAE 编码并添加噪声。不从纯噪声开始推理，而是从 t_start=T·(1-γ) 步的 z_{t_start}^normal 开始（γ=0.25）。可选使用前景 mask 约束异常位置：z_t = mask⊙z_t + (1-mask)⊙z_t^normal
    - 设计动机：相比微调 SD，测试时条件化保留了模型的泛化能力和多样性

2. **注意力引导异常优化 (Attention-Guided Anomaly Optimization)**:
    - 功能：强制 SD 在生成过程中关注并体现异常语义
    - 核心思路：在每个去噪步 t，聚合 16×16 分辨率的交叉注意力图并 softmax 归一化+高斯平滑。提取异常 token 的注意力图，优化 L_att = 1-max(A_t^j⊙mask)。通过梯度更新 z_t。定位感知调度器：α_t = λ(1+Δt·t)·n_t/n_{t_start}，随注意力聚焦减小步长防止过度优化
    - 设计动机：异常在 SD 训练数据中稀少且占图像比例小，直接生成时异常语义容易被忽略

3. **提示引导异常精化 (Prompt-guided Anomaly Refinement)**:
    - 功能：利用详细文本描述增强异常生成的语义一致性和质量
    - 核心思路：用 GPT-4 生成异常类型的详细描述 c'。在最后 30 个去噪步引入 CLIP 图像损失 L_img = 1-cosine(Φ^T(c'),Φ^V(x̃_t)) 和提示嵌入优化 L_prompt = 1-cosine(τ(c),τ(c'))。联合优化 L = L_img + α_t·L_att
    - 设计动机：注意力优化只处理 1-2 token 的异常描述语义模糊；详细描述提供更丰富的语义引导

### 损失函数 / 训练策略

**无需训练！** 所有模块都在推理时运行：
- L_att：注意力引导损失（异常 token 的最大注意力值）
- L_img：CLIP 图文对齐损失（生成图像 vs 详细异常描述）
- L_prompt：提示嵌入对齐损失（原始提示 vs 详细描述）
- 推理设置：T=100 步，γ=0.25（从第 75 步开始），提示精化在最后 30 步

## 实验关键数据

### 主实验

异常生成质量对比（MVTec AD 平均）：

| 方法 | IS↑ | IC-LPIPS↑ | 需要训练 | 需要异常数据 |
|------|-----|-----------|---------|------------|
| NSA | 1.44 | 0.26 | 否 | 否 |
| RealNet | 1.64 | 0.30 | 是 | 是 |
| AnomalyDiffusion | 1.80 | 0.32 | 是 | 是(1/3测试集) |
| **AnomalyAny** | **2.02** | **0.33** | **否** | **否** |

1-shot 异常检测性能提升：

| 方法 | MVTec I-AUC↑ | MVTec P-AUC↑ | VisA I-AUC↑ | VisA P-AUC↑ |
|------|-------------|-------------|-------------|-------------|
| PaDiM | 76.6 | 89.3 | 62.8 | 89.9 |
| PatchCore | 83.4 | 92.0 | 79.9 | 95.4 |
| WinCLIP+ | 93.1 | 95.2 | 83.8 | 96.4 |
| PromptAD | 94.6 | 95.9 | 86.9 | 96.7 |
| **AnomalyAny** | **94.9** | **95.4** | **89.7** | **97.7** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无正常样本条件化 | 生成图像偏离目标分布 | 缺乏对象外观约束 |
| 无注意力优化 | 异常语义被忽略 | SD 默认忽略小区域异常 |
| 无提示精化 | 异常不够精细 | 缺乏详细语义引导 |
| 无定位调度器 | 出现明显伪影 | 优化步长未递减导致过拟合 |
| 完整 AnomalyAny | 真实且多样的异常 | 三模块协同最优 |

### 关键发现

1. **无需训练即达到最佳生成质量**: IS 和 IC-LPIPS 都超越了需要训练的方法
2. **通用性强**: 可为任意对象类型和异常描述生成样本，不受训练数据分布限制
3. **显著提升下游检测**: 在 VisA 上 I-AUC 从 86.9 提升至 89.7
4. **注意力图可作为 pixel-level 标注**: 最终注意力图可直接定位异常区域
5. **定位调度器避免过度优化**: 按比例减小优化步长有效防止伪影

## 亮点与洞察

1. **零训练范式**: 完全利用预训练 SD 知识，突破了数据稀缺瓶颈
2. **三级递进设计**: 正常条件化→注意力优化→提示精化，层层递进解决 SD 异常生成的三个核心难题
3. **自带标注**: 最终的注意力图可作为 pixel-level anomaly mask
4. **GPT-4 辅助扩展异常类型**: 利用 GPT-4 自动生成可能的异常类型和详细描述，实现开放词表的异常生成
5. **实用价值高**: 工业检测场景的新产品/新缺陷类型缺乏数据的痛点恰好被解决

## 局限与展望

1. **生成质量受 SD 限制**: 对 SD 训练数据中极少出现的异常模式，生成质量可能下降
2. **计算开销**: 每个去噪步都要计算注意力图并进行梯度优化，推理比直接 SD 生成慢
3. **异常位置控制有限**: mask 可约束大致区域，但精细的异常形状和尺度不完全可控
4. **定量评估的局限**: IS 和 IC-LPIPS 不能完全反映异常的真实性
5. 未来方向：更精细的异常位置/形状控制、与更先进扩散模型结合、扩展到 3D 异常生成

## 相关工作与启发

- **vs AnomalyDiffusion**: AnomalyDiffusion 需要在 1/3 异常数据上训练属于已见异常生成；AnomalyAny 不需要任何异常数据生成未见异常
- **vs DRAEM/NSA**: 裁剪粘贴方法缺乏真实感；AnomalyAny 利用 SD 的生成能力产生更真实的异常
- **vs RealNet**: RealNet 也使用 SD 但需要微调受限于训练分布；AnomalyAny 直接推理时使用泛化性更强
- **vs Attend-and-Excite**: 注意力引导优化的思路受其启发但专门适配到异常生成场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 SD 直接用于无训练异常生成的思路新颖，注意力引导+提示精化的组合有效
- 实验充分度: ⭐⭐⭐⭐ MVTec AD + VisA 两个基准，生成质量和下游检测双重评估
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融实验直观，可视化丰富
- 价值: ⭐⭐⭐⭐ 对工业异常检测领域有直接实用价值，零训练范式降低了异常数据合成门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ROICtrl: Boosting Instance Control for Visual Generation](roictrl_boosting_instance_control_for_visual_generation.md)
- [\[CVPR 2025\] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](univad_a_training-free_unified_model_for_few-shot_visual_anomaly_detection.md)
- [\[CVPR 2026\] Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](../../CVPR2026/object_detection/visual_prototype_conditioned_focal_region_generation_for_uav-based_object_detect.md)
- [\[CVPR 2025\] Interpreting Object-level Foundation Models via Visual Precision Search](interpreting_object-level_foundation_models_via_visual_precision_search.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)

</div>

<!-- RELATED:END -->
