---
title: >-
  [论文解读] TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation
description: >-
  [ICML2025][图像恢复][自监督学习] 提出 TimeDART，将自回归建模与去噪扩散过程统一在一个自监督预训练框架中，通过因果 Transformer 编码器捕获长期动态演化、patch 级扩散去噪捕获细粒度局部模式，在预测和分类任务上均超越现有方法。 时间序列自监督学习主要有三类范式，各有短板： 掩码自编码器：（…
tags:
  - "ICML2025"
  - "图像恢复"
  - "自监督学习"
  - "时间序列表示学习"
  - "扩散模型"
  - "Transformer"
  - "预训练"
---

# TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation

**会议**: ICML2025  
**arXiv**: [2410.05711](https://arxiv.org/abs/2410.05711)  
**代码**: [GitHub](https://github.com/Melmaphother/TimeDART)  
**领域**: 图像复原  
**关键词**: 自监督学习, 时间序列表示学习, 扩散模型, 自回归Transformer, 预训练

## 一句话总结
提出 TimeDART，将自回归建模与去噪扩散过程统一在一个自监督预训练框架中，通过因果 Transformer 编码器捕获长期动态演化、patch 级扩散去噪捕获细粒度局部模式，在预测和分类任务上均超越现有方法。

## 研究背景与动机

时间序列自监督学习主要有三类范式，各有短板：

**掩码自编码器**（如 PatchTST、TimeMAE）：预训练与微调存在不一致，因为微调时引入了新的掩码嵌入

**对比学习**（如 CoST、TS2Vec）：侧重序列级建模，难以捕获细粒度时域变化

**自回归方法**：自然对齐从左到右的时间动态，但容易过拟合噪声；MSE 损失隐含假设数据服从高斯分布，导致预测坍缩为以历史值为中心的固定方差高斯分布

**核心动机**：如何在统一框架内同时建模长期动态演化和细粒度局部模式？作者提出在自回归优化框架中嵌入扩散去噪过程，用扩散损失替代简单 MSE，使模型能表达更丰富的多模态分布。

## 方法详解

TimeDART 包含三个核心模块：

### 1. 归一化与 Patch 嵌入

- 对每个样本做 Instance Normalization（零均值、单位方差）
- 将长度为 $L$ 的序列切分为 $N = L/P$ 个不重叠 patch（步长 $S = P$，防止信息泄露）
- 线性嵌入层将 patch 映射到高维表示：$\mathbf{z}_{1:N} = \text{Embedding}(\mathbf{x}_{1:N})$

### 2. 因果 Transformer 编码器

- 在 patch 嵌入序列前拼接可学习的 SOS（Start-of-Sequence）标记，去掉最后一个 patch
- 添加正弦位置编码，并在编码器中施加因果掩码 $M$，限制每个 patch 仅看到其之前的 patch
- 编码器输出：$f(\mathbf{z}_{1:N}^{in}) = \text{Encoder}(\mathbf{z}_{1:N}^{in}, M)$

### 3. Patch 级扩散去噪

**前向过程**：对每个 patch 独立加噪，使用余弦调度器：

$$q(x_j^s | x_j^0) = \mathcal{N}(x_j^s; \sqrt{\gamma(s)} x_j^0, (1 - \gamma(s))I)$$

其中 $\gamma(s) = \prod_{s' \leq s} \alpha(s')$，每个 patch 的加噪步数 $s_j$ 独立随机采样，防止任务过于简单。

**反向过程**：去噪解码器以编码器输出为 key/value、加噪 patch 嵌入为 query 做交叉注意力。解码器使用 self-only 掩码，确保第 $j$ 个输入仅关注第 $j$ 个编码器输出：

$$z_j^{out} = g(\hat{z}_j^{in}, f(\mathbf{z}_{1:j-1}^{in}))$$

### 4. 优化目标：扩散损失替代 MSE

传统自回归 MSE 损失隐含高斯假设，作者证明其等价于：

$$-\log \mathcal{N}(x_j^0; \text{Projector}(f(\mathbf{z}_{1:j-1}^{in})), \sigma^2) + C$$

这假设时间序列服从单峰高斯分布，与现实多模态分布不符。TimeDART 使用扩散损失（等价于 ELBO）：

$$\mathcal{L}_{diff} = \sum_{j=1}^{N} \mathbb{E}_{\epsilon, q(x_j^0)} \left[ \| x_j^0 - g(\hat{z}_j^{in}, f(\mathbf{z}_{1:j-1}^{in})) \|^2 \right]$$

### 5. 下游迁移

预训练完成后丢弃去噪解码器，仅保留嵌入层和编码器，接任务特定头：
- **预测**：flatten head + MSE 损失
- **分类**：max-pooling head + 交叉熵损失

## 实验关键数据

### 预测任务（12 个数据集，MSE/MAE，越低越好）

| 方法 | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Electricity | Traffic | Weather | Exchange |
|------|-------|-------|-------|-------|-------------|---------|---------|----------|
| **TimeDART** | **0.411** | **0.346** | 0.344 | **0.257** | **0.163** | **0.388** | **0.226** | **0.359** |
| SimMTM | 0.409 | 0.353 | 0.348 | 0.263 | 0.162 | 0.392 | 0.230 | 0.451 |
| PatchTST(SSL) | 0.433 | 0.354 | **0.342** | 0.272 | 0.163 | 0.404 | 0.227 | 0.376 |
| TimeMAE | 0.434 | 0.402 | 0.350 | 0.270 | 0.196 | 0.410 | 0.227 | 0.427 |
| Random Init. | 0.439 | 0.358 | 0.351 | 0.269 | 0.177 | 0.410 | 0.231 | 0.440 |

- 在 24 个评估指标中 **83.3%** 取得最优
- 相比随机初始化 MSE 降低 **6.8%**，相比 SOTA 降低 **3%**

### 分类任务（Accuracy）

| 方法 | HAR | Epilepsy | EEG |
|------|-----|----------|-----|
| **TimeDART** | **0.9247** | **0.9712** | **0.8269** |
| SimMTM | 0.9200 | 0.9565 | 0.8165 |
| TimeMAE | 0.9204 | 0.9459 | 0.8148 |
| Random Init. | 0.8738 | 0.9265 | 0.7752 |

- 预训练后平均准确率比随机初始化提升 **5.7%**

### 消融实验

| 设置 | ETTh2 MSE | ETTm2 MSE | Electricity MSE | HAR Acc. |
|------|-----------|-----------|-----------------|----------|
| TimeDART | **0.346** | **0.257** | **0.163** | **0.9247** |
| w/o AR | 0.365 | 0.281 | 0.193 | 0.8966 |
| w/o Diff | 0.352 | 0.265 | 0.164 | 0.9002 |
| w/o AR-Diff | 0.364 | 0.285 | 0.190 | 0.8785 |

移除自回归机制性能甚至差于随机初始化，证明其捕获长期动态的关键作用。

## 亮点与洞察

1. **理论洞察深刻**：从最大似然角度揭示 MSE 损失隐含高斯假设的局限性，用扩散损失自然替代，理论推导完整
2. **两个生成范式的统一**：自回归建模全局时序依赖 + 扩散去噪捕获局部细节，互补性设计精巧
3. **独立加噪策略**：每个 patch 独立随机采样加噪步数，避免任务过于简单化
4. **跨领域迁移能力强**：混合多域预训练后微调效果超越大部分域内预训练方法
5. **骨干网络通用性**：用 TCN 替代 Transformer 编码器仍然有效，验证框架的通用性

## 局限与展望

1. **推理时不使用扩散过程**：预训练的去噪解码器在下游被丢弃，学到的局部模式建模能力未在推理时直接利用
2. **Channel-Independent 策略**：未建模多变量间的相关性，可能在强耦合多变量场景受限
3. **预训练开销**：扩散过程引入额外计算成本，论文未详细分析预训练时间/资源消耗
4. **跨域预训练在 Weather 数据集上略逊于域内**：统一超参数未能适配所有域的特性
5. **分类任务仅 3 个数据集**：评估范围有限，未在更多分类 benchmark 上验证

## 相关工作与启发

- **SimMTM**（Dong et al., 2024）：掩码时间序列建模，流形结构加权
- **PatchTST**（Nie et al., 2023）：patch 级 Transformer 自监督/监督预测
- **TimeMAE**（Cheng et al., 2023）：解耦掩码自编码器
- **CoST**（Woo et al., 2022）：时域+频域对比学习
- **DDPM/扩散模型**（Ho et al., 2020）：去噪扩散概率模型，TimeDART 将其嵌入自回归框架而非直接生成

## 评分
- 新颖性: ⭐⭐⭐⭐ — 自回归+扩散统一框架的设计新颖，理论分析有深度
- 实验充分度: ⭐⭐⭐⭐ — 9 个数据集、消融、超参分析、跨域、few-shot 全面覆盖
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，图示直观，理论推导完整
- 价值: ⭐⭐⭐⭐ — 为时间序列自监督预训练提供了有力的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](../../ICLR2026/image_restoration/skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[CVPR 2026\] Time Without Time: Pseudo-Temporal Representation for Space-Time Super-Resolution](../../CVPR2026/image_restoration/time_without_time_pseudo-temporal_representation_for_space-time_super-resolution.md)
- [\[ICML 2025\] HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration](harmonica_harmonizing_training_and_inference_for_better_feature_caching_in_diffu.md)
- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](../../ICLR2026/image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
