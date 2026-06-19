---
title: >-
  [论文解读] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation
description: >-
  [AAAI 2026 Oral][图像生成][扩散模型加速] 提出 SpecDiff，一种基于自推测(self-speculation)的免训练多级特征缓存策略，通过利用少步推测引入**未来信息**辅助token重要性选择，突破了仅依赖历史信息的精度-速度瓶颈，在 Stable Diffusion 3/3.5 和 FLUX 上实现 2.80×/2.74×/3.17× 加速且质量损失可忽略。
tags:
  - "AAAI 2026 Oral"
  - "图像生成"
  - "扩散模型加速"
  - "特征缓存"
  - "自推测"
  - "DiT"
  - "信息利用"
---

# SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation

**会议**: AAAI 2026 Oral  
**arXiv**: [2509.13848](https://arxiv.org/abs/2509.13848)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 扩散模型加速, 特征缓存, 自推测, DiT, 信息利用

## 一句话总结

提出 SpecDiff，一种基于自推测(self-speculation)的免训练多级特征缓存策略，通过利用少步推测引入**未来信息**辅助token重要性选择，突破了仅依赖历史信息的精度-速度瓶颈，在 Stable Diffusion 3/3.5 和 FLUX 上实现 2.80×/2.74×/3.17× 加速且质量损失可忽略。

## 研究背景与动机

### 领域现状

扩散Transformer(DiT)架构（SD3、SD3.5、FLUX）已成为扩散模型的主流设计范式，但随着模型参数量增长（Scaling Law驱动），推理延迟和内存需求成为关键瓶颈。DiT推理中，输入矩阵行数与模型权重行数可比（如SD3中M=4096, N=1536），导致矩阵乘法算术强度高、计算密集(compute-bound)。减少实际计算量是加速推理的直接有效方法。

### 现有方法及局限

**特征缓存**已成为新兴的加速方法，核心是利用迭代推理中特征的相似性，缓存和复用相对不变的特征替代冗余计算：
- **FORA**：直接缓存复用相邻步特征 → 1.8× 加速但 >60% 精度损失
- **ToCa**：基于高注意力分数选择重要token → 2.0× 加速，~30% 损失
- **RAS**：利用两步相邻特征相似度选择 → 2.3× 加速，~20% 损失
- **TaylorSeer**：用所有历史特征差分近似当前特征 → 2.6× 加速，~15% 损失

### 核心矛盾

从信息利用角度分析，所有现有方法**仅依赖历史信息**。DiT推理本质上是马尔可夫过程——下一步输出仅依赖当前步输出。因此，累积历史信息始终被限制在过去步的子集中，**历史信息只能捕获局部变化**，无法预测未来的图像突变（recall分析显示无论1步、2步还是N步历史信息，与理论上限仍有显著差距）。

### 本文切入角度

受LLM推测解码(speculative decoding)启发，作者发现：**同一时步在不同迭代次数间的信息高度相似**。因此可以用少量推测步数提前获取"未来信息"，从全局角度优化token选择。实验显示，历史信息无法预测的重要token中有相当比例可由推测信息捕获。

### 核心 Idea

提出新范式：用原始模型的少量自推测步引入未来信息，以极低开销（<5%额外推理时间，<0.1%额外内存）辅助特征缓存token选择，从局部变化捕获升级到全局变化捕获。

## 方法详解

### 整体框架

SpecDiff的双技术架构：
1. **基于自推测信息的特征选择算法(T1)**：利用未来+历史注意力信息计算token重要性分数
2. **基于特征重要性分数的多级特征分类算法(T2)**：将token分为三级，针对性处理

推理流程：
- 首先执行少步（默认2步）的推测去噪，获取未来时步的注意力信息
- 在正式推理的每一步，结合历史和未来注意力分数计算token重要性
- 按重要性将token分为C1（需计算）、C2（直接复用）、C3（加权近似）三级

### 关键设计

#### 1. **基于自推测信息的特征选择算法**

token $x_i$ 的重要性分数由三部分组成：

$$Score(x_i) = his(x_i) \cdot fut(x_i) \cdot star(x_i)$$

- **历史分数** $his(x_i)$：前一迭代中该token在所有层的注意力分数之和
- **未来分数** $fut(x_i)$：推测步中最近未来时步的注意力分数
- **饥饿分数** $star(x_i) = e^{cf(x_i)}$：$cf(x_i)$ 为被缓存的次数

设计动机：
- 历史×未来的乘积形式确保两方面信息的共同确认
- 饥饿分数解决了分布偏斜问题——实验发现仅选20%重要token时，选择频率前25%的token占据了>75%的选择次数，约40%的token从未被选为重要token，长期缓存导致累积误差
- 由于推测步数少于完整迭代步数，时步无法完全对齐，因此使用最近未来时步的分数

#### 2. **多级特征分类算法**

选出重要token后，如何高效近似其余token的特征？SpecDiff将token分为三级：

**C1级（网络计算）**：重要性分数最高的token（数量由缓存率CR决定，为 $1-CR$ 占比），直接参与UNet/DiT计算

**C2级（直接复用）**：分数最低的token（占总分数的10%），直接复用上一迭代特征：
$$F(x_t^{\text{cached}}) \approx F(x_{t+1}^{\text{cached}})$$

设计动机：这些token的ERROR变异系数低，复用误差小。

**C3级（加权近似）**：不参与计算但在前90%分数分布内的token，用前3步的加权历史近似：

$$F(x_t^{\text{cached}}) = \sum_{i=1}^{3} W_{t+i} \cdot F(x_{t+i}^{\text{cached}})$$

其中权重为时间距离加权的指数衰减：

$$W_{t+i} = \frac{e^{-i}(T_{t+i} - T_t)}{\sum_{i=1}^{3} e^{-i}(T_{t+i} - T_t)}$$

设计动机：
- 分析发现token重要性分数与ERROR变异系数高度相关——高分数token特征变化大，不能简单复用
- 高维空间中cosine相似度低于95%时特征不再高度相似，连续3步即达到此阈值
- 对更早时步降低权重以减少近似误差

### 损失函数 / 训练策略

SpecDiff是**免训练方法**，不需要任何训练过程。所有参数设置：
- 推测步数默认为2
- 推理步数设为28
- SD3/3.5的CFG设为7.0，FLUX设为3.5
- 缓存率CR可变，支持55%到99%的广泛范围

## 实验关键数据

### 主实验

在 Stable Diffusion 3 上的性能（COCO 2014 val，5000对，1024×1024）：

| 方法 | 缓存率 | FID↓ | Clip Score↑ | VQA Score↑ | 加速比 |
|------|--------|------|------------|-----------|--------|
| RFlow(基线) | 0 | 29.31 | 0.3176 | 0.9110 | 1.00× |
| RAS | 50% | 27.26 | 0.3162 | 0.9005 | 1.61× |
| **SpecDiff** | 55% | **27.57** | **0.3168** | **0.9057** | **1.61×** |
| RAS | 75% | 27.38 | 0.3149 | 0.8849 | 2.09× |
| **SpecDiff** | 92% | **29.52** | **0.3160** | **0.8888** | **2.43×** |
| RAS | 87.5% | 40.92 | 0.3044 | 0.8611 | 2.40× |
| **SpecDiff** | 99% | **29.75** | **0.3152** | **0.8822** | **2.80×** |

在 FLUX.1 Dev 上与 TaylorSeer 的比较：

| 方法 | 配置 | FID↓ | Clip↑ | VQA↑ | SSIM↑ | PSNR↑ | 内存(GB)↓ | 加速比 |
|------|------|------|------|------|------|------|----------|--------|
| RFlow | - | 27.68 | 0.3093 | 0.8986 | - | - | 38.36 | 1.00× |
| TaylorSeer | N5O1 | 27.87 | 0.3090 | 0.8909 | 0.7098 | 16.93 | 42.66 | 2.47× |
| **SpecDiff** | 85% | **28.61** | **0.3125** | **0.8925** | **0.7101** | **19.20** | **41.46** | 2.52× |
| TaylorSeer | N6O1 | 28.83 | 0.3107 | 0.8822 | 0.6570 | 16.07 | 42.66 | 2.63× |
| **SpecDiff** | 95% | **29.24** | **0.3124** | **0.8834** | **0.6963** | **19.02** | **41.46** | **3.17×** |

### 消融实验

推测步数的影响（SD3，缓存率99%）：

| 推测步数 | FID↓ | Clip Score↑ | VQA Score↑ | 加速比 |
|---------|------|------------|-----------|--------|
| 2 | 29.75 | 0.3152 | 0.8822 | **2.80×** |
| 3 | 29.87 | 0.3154 | 0.8825 | 2.49× |
| 4 | 29.64 | 0.3157 | 0.8834 | 2.25× |

消融实验（SD3，速度与精度分解）：

| 模块 | FID变化 | Clip变化 | 加速比 |
|------|--------|---------|--------|
| +特征预测(未来信息) | FID降18% | Clip+2.8% | 2.36× |
| +token层次分类 | 再降3.6% | 再+0.35% | 3.17×(总) |

### 关键发现

1. **Pareto前沿突破**：SpecDiff在速度-质量权衡上成功推进了Pareto前沿，在相同计算量下始终优于RAS和TaylorSeer
2. **99%缓存率下仍保持质量**：在SD3上，99%缓存率（即仅1% token参与计算）时FID仅增加0.44，Clip Score仅降0.7%
3. **文图对齐能力保持最好**：相比FID，Clip Score和VQA Score的保持更为出色
4. **2步推测最优**：推测步增多质量略有改善，但速度下降明显（20%/步），综合来看2步是最佳权衡
5. **内存开销极低**：额外<0.1%内存，远低于TaylorSeer的额外内存

## 亮点与洞察

1. **信息利用视角**的分析框架非常新颖——将不同缓存方法统一到信息利用量与性能正相关的分析中
2. **从LLM推测解码借鉴到扩散模型特征缓存**的跨领域迁移很巧妙
3. **三级token分类策略**设计优雅——高分token计算、低分token复用、中间token加权近似，各有理论依据
4. **饥饿分数**防止长尾token被永久忽略的机制简单有效
5. **免训练**特性使其可直接应用于任何DiT模型，无需额外训练成本

## 局限与展望

1. **FID指标不够优**：作者承认高CFG使风格统一化，强化了这一特征可能导致FID偏高
2. **仅支持DiT架构**：未在UNet-based扩散模型上验证
3. **固定推测步数**：2步推测对所有模型和场景可能不是最优的，自适应推测策略值得探索
4. **缓存率需手动设定**：不同图像内容可能需要不同的缓存率，动态缓存率调整是一个方向
5. **未与量化/蒸馏正交方法结合**：特征缓存可与模型压缩技术叠加使用

## 相关工作与启发

- **FORA, ToCa, RAS, TaylorSeer**：特征缓存方法的演进系列，SpecDiff在其基础上引入未来信息
- **Eagle/SpecPIM**（LLM推测解码）：跨领域灵感来源
- **MDTv2**（Gao et al., 2024）：证明DiT的特征关注与时步参数强相关，启发了同时步信息的利用
- **Key Takeaway**：在迭代推理系统中，"偷看未来"（通过廉价的近似推测）是提升效率的普适策略，马尔可夫性质不意味着历史信息足够——同类任务的先验知识（同时步跨迭代）可以补充历史信息的不足

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 信息利用视角和自推测范式是新颖的分析与设计思路
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖SD3/3.5/FLUX，与多个基线全面对比，含消融和Pareto分析
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，图示直观（recall分析、Pareto前沿），分析层次分明
- **价值**: ⭐⭐⭐⭐⭐ — 免训练的3×加速对DiT部署极具实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching](../../CVPR2026/image_generation/sencache_accelerating_diffusion_model_inference_via_sensitivity-aware_caching.md)
- [\[NeurIPS 2025\] Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration](../../NeurIPS2025/image_generation/tortoise_and_hare_guidance_accelerating_diffusion_model_inference_with_multirate.md)
- [\[ICCV 2025\] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](../../ICCV2025/image_generation/from_reusing_to_forecasting_accelerating_diffusion_models_with_taylorseers.md)
- [\[ICLR 2026\] FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference](../../ICLR2026/image_generation/fastflow_accelerating_the_generative_flow_matching_models_with_bandit_inference.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)

</div>

<!-- RELATED:END -->
