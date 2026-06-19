---
title: >-
  [论文解读] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery
description: >-
  [ICCV 2025][自监督学习][即时类别发现] 提出基于扩散模型的即时类别发现框架 DiffGRE，通过属性组合生成（ACG）合成包含虚拟类别信息的新样本、多样性驱动精炼（DDR）过滤低质量样本、半监督Leader编码（SLE）注入额外类别知识，在 6 个细粒度数据集上显著提升了已有 OCD 方法的性能（平均 ACC-ALL 提升 6.5%）。
tags:
  - "ICCV 2025"
  - "自监督学习"
  - "即时类别发现"
  - "扩散模型"
  - "属性组合生成"
  - "细粒度识别"
  - "在线聚类推理"
---

# Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery

**会议**: ICCV 2025  
**arXiv**: [2507.04051](https://arxiv.org/abs/2507.04051)  
**代码**: [github.com/XLiu443/DiffGRE](https://github.com/XLiu443/DiffGRE)  
**领域**: 其他  
**关键词**: 即时类别发现, 扩散模型, 属性组合生成, 细粒度识别, 在线聚类推理

## 一句话总结

提出基于扩散模型的即时类别发现框架 DiffGRE，通过属性组合生成（ACG）合成包含虚拟类别信息的新样本、多样性驱动精炼（DDR）过滤低质量样本、半监督Leader编码（SLE）注入额外类别知识，在 6 个细粒度数据集上显著提升了已有 OCD 方法的性能（平均 ACC-ALL 提升 6.5%）。

## 研究背景与动机

### 问题定义

即时类别发现（On-the-fly Category Discovery, OCD）是一个实用但充满挑战的任务：
- **训练阶段**：仅使用已标注的已知类别数据（支持集 $\mathcal{D}_S$）训练模型
- **测试阶段**：面对流式输入的数据（查询集 $\mathcal{D}_Q$），数据可能属于已知类别或未知类别
- **目标**：对每个新实例提供即时反馈（实例级推理），而非离线批量聚类

与传统广义类别发现（GCD）的区别：GCD 要求在训练时访问预定义的查询集，且通过离线聚类为批量数据分配标签；OCD 则无需访问查询集，且必须为流式数据提供逐实例的在线反馈。

### 已有方法的不足

**知识不足**：现有 OCD 方法仅从标注数据中挖掘可迁移知识，当标注数据/类别数量稀少时（特别是细粒度场景），已知类别包含的知识往往不足以支撑未知类别的发现

**哈希推理局限**：现有方法（如 SMILE）使用哈希码作为类别描述子，二值化处理不可避免地降低了高维特征的表示能力

**数据增强无效**：传统的类别保持型数据增强（如 T2I 模型生成已知类别图像）对未知类别发现帮助有限——实验表明仅增强已知类别数据难以提升未知类别发现

### 核心动机

**关键洞察**：未知类别可以通过组合已知类别中的语义属性来合成。例如，在两种已知鸟类图像之间进行扩散模型潜空间插值，可以合成与某个未知鸟类类别视觉相似的图像。这种"属性重组"思想为 OCD 任务提供了从已知类别中生成包含额外类别信息的合成数据的可能。

## 方法详解

### 整体框架

DiffGRE 包含三个阶段：
1. **Generate（生成）**：通过属性组合生成（ACG）在扩散模型潜空间中跨类别插值，合成虚拟类别图像
2. **Refine（精炼）**：通过多样性驱动精炼（DDR）过滤与已知类别过于相似的合成图像
3. **Encode（编码）**：通过半监督 Leader 编码（SLE）为合成数据分配可靠的伪标签，并利用 Leader 特征进行在线聚类推理

### 关键设计

#### 1. **属性组合生成（Attribute Composition Generation, ACG）**

- **功能**：从不同已知类别中采样两张图像，在扩散模型的潜空间中进行跨类别插值，合成属于"虚拟类别"的新图像
- **核心思路**：在三个嵌入空间中同时进行球面插值（Spherical Interpolation）——Stable Diffusion 的潜空间、CLIP 视觉空间和 CLIP 文本空间：

$$\bar{z}^* = \frac{\sin((1-\lambda_*)\theta)}{\sin(\theta)} z_1^* + \frac{\sin(\lambda_*\theta)}{\sin(\theta)} z_2^*, \quad * \in \{t, v, l\}$$

其中 $\theta = \arccos(z_1^* \cdot z_2^*)$，$\lambda_* \in [0,1]$ 是插值参数。
- 使用 VAE 编码器提取潜空间嵌入 $z^l$
- 使用 CLIP 视觉编码器提取视觉嵌入 $z^v$
- 使用预训练图像描述模型将图像转为文本，再用 CLIP 文本编码器提取文本嵌入 $z^t$

去噪过程使用标准扩散目标：

$$\mathcal{L}_{DM} = \mathbb{E}_{t, \bar{z}_0^l, \epsilon} \left[\|\epsilon - \epsilon_\theta(\bar{z}^l, t, \bar{z}^t)\|^2\right]$$

- **设计动机**：Stable Diffusion 的潜空间主要捕捉视觉细节，CLIP 空间则优化高层语义理解。将两者结合使得模型能够在保持视觉质量的同时实现语义属性的有效重组，合成视觉上合理且语义上新颖的虚拟类别图像。

#### 2. **多样性驱动精炼（Diversity-Driven Refinement, DDR）**

- **功能**：从大量合成图像中筛选出包含丰富额外类别信息的图像，过滤与已知类别过于相似的图像
- **核心思路**：
  1. 计算合成图像与所有已知类别中心的余弦相似度：$s(z_{\text{gen}}, c_k) = \frac{z_{\text{gen}} \cdot c_k}{\|z_{\text{gen}}\| \|c_k\|}$
  2. 计算每个合成样本与所有类别中心的平均相似度：$s_{\text{mean}}(z_{\text{gen}}) = \frac{1}{K}\sum_{k=1}^{K} s(z_{\text{gen}}, c_k)$
  3. 设定阈值 $\gamma$ 过滤高相似度样本：$\mathcal{F}(I_{\text{gen}}) = \mathbf{1}(s_{\text{mean}}(z_{\text{gen}}) \leq \gamma) \cdot I_{\text{gen}}$

- **设计动机**：扩散过程的随机性导致并非每个合成图像都有助于 OCD。实验表明直接使用全部合成图像会干扰已知类别的判别特征提取。DDR 使用类别中心计算相似度（而非逐图像比较），更稳定且不受长尾分布影响。

#### 3. **半监督 Leader 编码（Semi-supervised Leader Encoding, SLE）**

- **功能**：为合成图像分配可靠的伪标签，并生成类别级 Leader 特征用于在线推理
- **核心思路**：
  1. **虚拟类别分配**：合并标注数据与合成数据构建代理训练集 $\mathcal{D}_A = \mathcal{D}_S \cup \mathcal{D}_G$，使用聚类算法生成初始类别标签，再通过匈牙利算法与已知标签对齐校正
  2. **Leader 特征生成**：计算每个虚拟类别的平均特征作为类别 Leader 特征
  3. **Leader 对比学习**：增大类间距离，缩小类内距离：

$$\mathcal{L}_{sle}(x_n, y_n) = -\log \frac{\exp(f(x_n) \cdot l_{y_n}^T / \tau)}{\sum_{m \neq y_n} \exp(f(x_n) \cdot l_m^T / \tau)}$$

  4. **在线聚类推理（OCI）**：测试时初始化动态 Leader 记忆，通过自适应阈值判断新实例属于已知类别还是创建新类别

- **设计动机**：直接把同对输入混合的图像当作同一新类别会导致严重性能下降。SLE 通过聚类+匈牙利对齐给合成数据分配更合理的类别标签。相比哈希推理，OCI 直接在高维特征空间中操作，避免了二值化带来的信息损失。

### 损失函数 / 训练策略

总损失：

$$\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{reg} + \alpha \cdot \mathcal{L}_{sle} + \beta \cdot \mathcal{L}_{c}$$

- $\mathcal{L}_{sup}$：来自 SMILE 的监督对比学习损失
- $\mathcal{L}_{reg}$：哈希头输出的正则化损失
- $\mathcal{L}_{sle}$：Leader 对比学习损失（$\alpha=0.3$）
- $\mathcal{L}_c$：分类交叉熵损失（$\beta=1.0$）

## 实验关键数据

### 主实验

**哈希推理策略下 DiffGRE 对三个基线的提升（6 个细粒度数据集平均）**：

| 方法 | ACC-ALL | ACC-OLD | ACC-NEW |
|------|---------|---------|---------|
| BaseHash | 25.5 | 36.3 | 19.7 |
| BaseHash + DiffGRE | **32.0** | **47.8** | **23.9** |
| SMILE | 32.7 | 49.1 | 24.3 |
| SMILE + DiffGRE | **36.3** | **56.7** | **25.9** |
| PHE | 38.9 | 61.3 | 26.7 |
| PHE + DiffGRE | **40.0** | **62.9** | **28.0** |

**在线聚类推理（OCI）策略下的对比**：

| 方法 | ACC-ALL | ACC-OLD | ACC-NEW |
|------|---------|---------|---------|
| SMILE + SLE-based | 41.9 | 52.6 | 37.0 |
| SMILE + DiffGRE | **43.4** | **53.4** | **38.7** |
| PHE + SLE-based | 39.7 | 57.1 | 30.8 |
| PHE + DiffGRE | **42.3** | **59.1** | **33.5** |

### 消融实验

**训练组件消融（Arachnida / Mollusca / CUB 数据集，哈希推理）**：

| 配置 | Arachnida-ALL | Mollusca-ALL | CUB-ALL |
|------|--------------|-------------|---------|
| SMILE 基线 | 27.9 | 33.5 | 32.2 |
| w/o $\mathcal{L}_{sle}$ | 29.3 | 33.9 | 33.2 |
| w/o $\mathcal{L}_{c}$ | 34.5 | 36.0 | 33.6 |
| w/o DDR | 33.5 | 34.5 | 32.4 |
| **SMILE + DiffGRE** | **35.4** | **36.5** | **35.4** |

**不同合成方法对比（CUB 数据集）**：

| 合成方法 | 类型 | ACC-ALL |
|---------|------|---------|
| CutMix | 像素混合 | 较低 |
| MixUp | 像素混合 | 较低 |
| Da-Fusion | T2I（已知类别文本） | 有限提升 |
| Diff-Mix | T2I（已知类别文本） | 有限提升 |
| **ACG (Ours)** | **潜空间属性重组** | **最优** |

### 关键发现

1. **DiffGRE 是通用即插即用框架**：在 BaseHash、SMILE、PHE 三个基线上均带来一致提升
2. **DDR 不可或缺**：移除 DDR 导致 ACC-OLD 平均下降 4.6%，证实低质量合成样本会扰乱已知类别的判别特征
3. **SLE 显著优于哈希推理**：在线聚类推理在所有数据集上均优于哈希推理，ACC-NEW 平均提升 16.5%
4. **属性重组优于类别保持型增强**：传统 T2I 方法只能生成已知类别图像，对未知类别发现帮助有限；ACG 通过属性重组可以合成与未知类别相似的图像
5. **合成样本数量应与标注数据量级匹配**：DDR 的最优阈值 $\gamma$ 使剩余合成样本量与标注数据相当时效果最好

## 亮点与洞察

1. **属性重组思想的新颖性**：不是生成已知类别的"忠实"增强，而是通过跨类别潜空间插值合成"虚拟类别"——这是扩散模型在开放世界发现任务中的创新应用
2. **三空间联合插值**：同时在扩散潜空间、CLIP 视觉空间和 CLIP 文本空间进行插值，兼顾视觉细节和高层语义
3. **从哈希到在线聚类的推理范式转变**：SLE 的 Leader 特征保留了高维信息，OCI 的自适应阈值机制使在线推理更加鲁棒
4. **DDR 的简洁有效性**：仅使用类别中心的平均相似度就能有效过滤低质量样本，设计简洁且计算开销低

## 局限与展望

1. **依赖预训练扩散模型质量**：合成图像质量受 Stable Diffusion 性能限制，对低资源域（如医学影像）可能效果有限
2. **插值参数敏感性**：$\lambda_t$, $\lambda_v$, $\lambda_l$ 三个插值参数需要在特定数据集上调优
3. **聚类分配噪声**：SLE 的虚拟类别分配依赖聚类质量，在类别数未知或类别不平衡时可能引入噪声
4. **计算开销**：合成图像需要额外的扩散模型推理时间，特别是大规模数据集上
5. **PHE 基线提升受限**：当基线使用独立优化的特征提取器时（如 PHE），DiffGRE 的提升相对有限

## 相关工作与启发

- 与 GCD/NCD 方法的区别：OCD 不需要训练时访问查询集，且要求逐实例在线推理
- 与 Diff-Mix 的区别：Diff-Mix 在两个已知类别间做混合生成，仍是已知类别增强；DiffGRE 的 ACG 则旨在合成虚拟类别
- 启发性发现：在已知类别间做潜空间插值时，最近邻搜索结果竟然属于未知类别，证实了属性重组合成虚拟类别的可行性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 属性重组合成虚拟类别的思想新颖，三空间联合插值和 OCI 推理设计创新
- **实验充分度**: ⭐⭐⭐⭐ — 6 个细粒度数据集、3 个基线方法的全面评估，消融分析详尽
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，动机图示直观
- **价值**: ⭐⭐⭐⭐ — 即插即用框架具有良好实用性，为扩散模型在类别发现中的应用开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TAR: Token-Aware Refinement for Fine-grained Generalized Category Discovery](../../CVPR2026/self_supervised/tar_token-aware_refinement_for_fine-grained_generalized_category_discovery.md)
- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[NeurIPS 2025\] CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing](../../NeurIPS2025/self_supervised/cleverbirds_a_multiple-choice_benchmark_for_fine-grained_human_knowledge_tracing.md)
- [\[CVPR 2025\] Hyperbolic Category Discovery](../../CVPR2025/self_supervised/hyperbolic_category_discovery.md)
- [\[ICLR 2026\] Adaptive Gaussian Expansion for On-the-fly Category Discovery](../../ICLR2026/self_supervised/adaptive_gaussian_expansion_for_on-the-fly_category_discovery.md)

</div>

<!-- RELATED:END -->
