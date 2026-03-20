# Spectral Attention Steering for Prompt Highlighting

**会议**: ICLR2026  
**arXiv**: [2603.01281](https://arxiv.org/abs/2603.01281)  
**代码**: [waylonli/SEKA](https://github.com/waylonli/SEKA)  
**领域**: llm_nlp  
**关键词**: attention steering, prompt highlighting, spectral decomposition, FlashAttention, key embedding editing  
**作者**: Weixian Waylon Li, Yuchen Niu, Yongxin Yang, Keshuang Li, Tiejun Ma, Shay B. Cohen（University of Edinburgh, RayNeo, Huawei Research, QMUL）

## 一句话总结

提出 SEKA/AdaSEKA，通过对 key embedding 进行谱分解学习"相关性子空间"，在注意力计算前直接编辑 key 向量来实现 prompt highlighting，无需存储完整注意力矩阵，与 FlashAttention 完全兼容，且开销极低（+0.03s/sample）。

## 研究背景与动机

1. **Prompt Highlighting 的实际需求**：在高风险场景中，需要精确引导 LLM 关注 prompt 中用户指定的关键文本（如事实冲突中的新知识、指令跟随中的核心约束），即 attention steering。
2. **现有方法的效率瓶颈**：PASTA 等 SOTA 方法在注意力矩阵计算完成后对其进行后处理修改（post-hoc），必须存储完整的 $T \times T$ 注意力矩阵，与 FlashAttention 等 IO-aware 高效实现不兼容。
3. **额外开销巨大**：PASTA 导致推理延迟增加 +1.03s/sample，内存增加 +23.12 GB；SPA 基于 logit 分布操作，不支持 batch 处理，速度最慢（+5.32s）。
4. **需要昂贵的 head search**：PASTA 还需要针对不同任务做 attention head 搜索来确定应该 steer 哪些 head，增加了部署成本。
5. **Key embedding 的结构化信号**：作者通过对比实验发现，当 prompt 中问题从不相关变为相关时，特定 layer/head 的 key embedding 呈现出一致的方向性偏移（如 PCA 可视化所示），说明"相关性"被编码在 key 表示的结构化子空间中。
6. **Pre-attention 干预的可行性**：注意力分数 $\text{Attn}(i,j) = \frac{\boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d_k}}$ 取决于 query-key 内积，等价的控制可通过编辑 key 端实现，且 key 按 token position 索引，天然适合控制单个 token 被关注的程度。

## 方法详解

### 整体框架

SEKA 分为两个阶段：

- **离线学习阶段**：用合成对比 prompt 构建正/负交叉协方差矩阵，SVD 分解得到"相关性子空间"投影矩阵
- **推理阶段**：对高亮 token 的 key embedding 施加投影变换 $\boldsymbol{k}_j' = \boldsymbol{k}_j + g \boldsymbol{P} \boldsymbol{k}_j$，在注意力计算前完成

### 关键设计 1：谱学习相关性投影（Offline）

构建三类 prompt：neutral（仅上下文）、positive（上下文 + 相关问题）、negative（上下文 + 无关问题），提取同一 token span 在不同条件下的 key embedding $\boldsymbol{h}, \boldsymbol{h}^+, \boldsymbol{h}^-$。

计算交叉协方差矩阵并做 SVD：

$$\boldsymbol{\Omega}_{\ell,h}^{+} = \frac{\boldsymbol{h}^\top \boldsymbol{h}^+}{n}, \quad \boldsymbol{\Omega}_{\ell,h}^{+} = \boldsymbol{U}_{\ell,h}^{+} \boldsymbol{S}_{\ell,h}^{+} \boldsymbol{V}_{\ell,h}^{+\top}$$

正投影取前 $k^+$ 个最大奇异值对应的左奇异向量，负投影取最小的 $k^-$ 个：

$$\boldsymbol{P}_{\ell,h}^{+} = \boldsymbol{U}_{\ell,h,:,:k^+}^{+} (\boldsymbol{U}_{\ell,h,:,:k^+}^{+})^\top, \quad \boldsymbol{P}_{\ell,h}^{-} = \boldsymbol{U}_{\ell,h,:,k^-:}^{-} (\boldsymbol{U}_{\ell,h,:,k^-:}^{-})^\top$$

$k^+, k^-$ 的选取通过累积奇异值比例阈值 $\gamma$ 控制：$\sum_{i=1}^{k^+} S_i^+ / \sum_{i=1}^{d_k} S_i^+ \geq \gamma$。

### 关键设计 2：推理时 Key 编辑

对每个高亮 token 的 key 向量：

$$\boldsymbol{k}_j' = \boldsymbol{k}_j + \frac{g^+ \cdot \boldsymbol{P}_{\ell,h}^+ \boldsymbol{k}_j + g^- \cdot \boldsymbol{P}_{\ell,h}^- \boldsymbol{k}_j}{2}$$

代入注意力公式后等价于在原始注意力分数上加一个低秩偏置：

$$\text{Logits}_{ij} = \underbrace{\frac{\boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d_k}}}_{A_{ij}} + \underbrace{\frac{\boldsymbol{q}_i^\top (g^+ \boldsymbol{P}^+ \boldsymbol{k}_j + g^- \boldsymbol{P}^- \boldsymbol{k}_j) / 2}{\sqrt{d_k}}}_{B_{ij}}$$

由于全程只修改 key 向量，从不接触注意力矩阵，所以天然兼容 FlashAttention。

### 关键设计 3：AdaSEKA 自适应路由

为应对多任务场景，AdaSEKA 学习 $M$ 个领域专家投影。在推理时，提取 prompt 最后一个 token 的 query 向量 $\boldsymbol{q}_{\ell,h}$，计算与各专家主方向的对齐度作为路由权重：

$$\alpha_{m,\ell,h}(\boldsymbol{q}) = \frac{\sum_{k=1}^{K} (\boldsymbol{q}^\top \boldsymbol{u}_{m}^{+(k)}) \cdot \sigma_{m}^{+(k)}}{\max_{m'} |\sum_{k=1}^{K} (\boldsymbol{q}^\top \boldsymbol{u}_{m'}^{+(k)}) \cdot \sigma_{m'}^{+(k)}|}$$

最终投影矩阵为专家投影的加权组合：$\boldsymbol{P}_{\text{dynamic}} = \sum_m \alpha_m \boldsymbol{U}_m^{+} (\boldsymbol{U}_m^{+})^\top$。优势：减少超参调节、模块化部署（新专家即插即用）、路由可解释。

### 关键设计 4：KV Head 筛选

并非所有 head 都对相关性敏感。作者计算正/负 key embedding 的 $\ell_2$ 距离：

$$D_{\ell,h} = \frac{1}{N} \sum_{i=1}^{N} \| \boldsymbol{h}_{\ell,h,i}^+ - \boldsymbol{h}_{\ell,h,i}^- \|_2$$

仅当 $D_{\ell,h} \geq \delta_{\min}$ 时才对该 head 施加投影。可视化显示中后层 head 的区分度明显更大，与 retrieval head 的研究一致。

## 实验

### 主实验：标准 Benchmark

在 CounterFact（知识冲突）、Bias in Bios（职业提取）、Pronoun Changing（代词重写指令跟随）三个任务上评测：

| 模型 | 方法 | CounterFact ES | CounterFact PS | Bias in Bios Acc | Pronoun P.Score | Pronoun A.P.Score |
|------|------|:-:|:-:|:-:|:-:|:-:|
| Qwen3-4B | Original | 45.00 | 45.64 | 79.84 | 93.14 | 90.52 |
| | PASTA | 97.16 | 96.03 | 89.58 | 95.82 | 94.64 |
| | SPA | 65.24 | 57.71 | 68.00 | 80.27 | 78.19 |
| | **SEKA** | **99.02** | **98.61** | **91.02** | 95.18 | 93.26 |
| | **AdaSEKA** | 98.90 | 98.72 | **91.86** | 94.54 | 92.08 |
| Qwen3-8B | Original | 39.04 | 39.59 | 76.08 | 98.00 | 97.84 |
| | PASTA | 92.70 | 91.68 | 86.32 | 98.86 | 98.72 |
| | **SEKA** | **99.08** | **98.96** | **88.74** | 98.56 | 98.26 |
| | **AdaSEKA** | 99.00 | 98.97 | 88.50 | **99.68** | **99.52** |
| Qwen3-14B | Original | 37.56 | 36.12 | 85.22 | 98.42 | 98.22 |
| | PASTA | 76.84 | 66.33 | 88.46 | 90.98 | 90.94 |
| | **SEKA** | **98.92** | **99.02** | 90.28 | 98.66 | 98.54 |
| | **AdaSEKA** | 99.00 | 99.15 | **91.22** | **99.88** | **99.86** |

### 效率对比

| 方法 | 延迟 (s/sample) | 峰值内存 (GB, B=10) | 峰值内存 (GB, B=1) |
|------|:-:|:-:|:-:|
| Original | 0.55 | 27.63 | 16.72 |
| PASTA | 1.58 (+1.03) | 50.75 (+23.12) | - |
| SPA | 5.87 (+5.32) | - | 17.71 (+0.99) |
| **SEKA** | **0.58 (+0.03)** | **27.66 (+0.03)** | **16.75 (+0.03)** |
| AdaSEKA | 0.82 (+0.27) | 43.22 (+15.59) | 18.23 (+1.51) |

SEKA 几乎零开销，PASTA 内存翻倍、延迟翻三倍。

### 消融实验

| 配置 | CounterFact ES (Qwen3-4B) | Bias in Bios Acc | Pronoun A.P.Score |
|------|:-:|:-:|:-:|
| SEKA (完整) | **99.02** | **91.02** | **93.26** |
| w/o learn (随机投影 + head 筛选) | 94.96 | 86.62 | 88.66 |
| w/o learn & filt (随机投影 + 无筛选) | 86.12 | 71.76 | **36.95** |

关键发现：

- 去掉谱学习用随机投影 → 性能明显下降，证明学到的相关性子空间具有实质意义
- 同时去掉 head 筛选 → 灾难性下降（Pronoun 从 90.52 降到 36.95），比不做任何 steering 还差，说明对不敏感的 head 做投影会引入严重干扰

### Lost-in-the-Middle 实验

- **SEKA 对中间段落做 highlighting 可反转 U-shape 性能曲线**：中间位置的 exact match 显著提升
- 对所有段落统一做 highlighting 反而可能加剧 lost-in-the-middle 效应
- 通过调节 $\delta_{\min}$ 控制被 steer 的 head 数量，可以实现 U-shape 曲线的平坦化
- PASTA 在此任务上不如 baseline，说明 post-hoc 方法在长上下文场景的局限性

## 亮点

1. **与 FlashAttention 完全兼容**：这是同类方法中首个做到的，通过 pre-attention key editing 绕开了必须存储注意力矩阵的限制
2. **几乎零开销**：SEKA 仅增加 0.03s/sample 延迟和 0.03 GB 内存，相比 PASTA +1.03s/+23.12 GB 优势极大
3. **几何可解释性强**：$\boldsymbol{k}' = \boldsymbol{k} + g \boldsymbol{P} \boldsymbol{k}$ 的投影变换具有清晰的几何含义——将 key 向相关性子空间方向放大
4. **Training-free**：不需要任何微调，仅依赖少量合成对比 prompt 做离线谱分解
5. **AdaSEKA 的自适应路由机制**减少了跨任务/跨模型的超参调节需求，4 个专家即插即用
6. **Lost-in-the-Middle 的 U-shape 反转**是一个有趣的新发现，展示了 attention steering 对位置敏感性的精准控制能力

## 局限性 / 可改进方向

1. **离线阶段依赖合成数据质量**：对比 prompt triplet 的构建策略会影响学到的投影质量，泛化到新领域需要重新构建
2. **超参数仍需调节**：虽然 AdaSEKA 减少了部分调参，但 $g^+, g^-, \gamma, \delta_{\min}$ 仍需 grid search，且不同模型/任务的最优值不同
3. **仅限 prompt highlighting 场景**：方法聚焦于"让模型关注指定 token"，不涵盖更广泛的 activation steering 目标（如风格控制、安全性）
4. **Lost-in-the-Middle 实验中 highlighting 范围粗糙**：位置 5-25 是人工指定的，实际应用中需要知道哪些段落是 gold passage
5. **AdaSEKA 内存开销非忽略**：batch=10 时 +15.59 GB，主要来自存储多专家的 SVD 分量

## 与相关工作的对比

- **vs PASTA**（Zhang et al., 2024）：PASTA 后处理注意力矩阵，不兼容 FlashAttention，延迟/内存代价高；SEKA 在精度上全面超越且开销可忽略
- **vs SPA**（Tian & Zhang, 2025）：SPA 操作 logit 分布，不支持 batch，速度最慢；在 CounterFact 上远不如 SEKA
- **vs Activation Steering**（SEA, RepE 等）：activation steering 修改 MLP 层的隐状态来控制语义属性，而 SEKA 控制注意力机制决定模型看哪里，二者正交互补
- **与 Retrieval Head 研究的呼应**：Wu et al., 2025; Qiu et al., 2025 发现 retrieval head 集中在中后层，SEKA 的 head 筛选策略与此一致

## 评分

- ⭐ 新颖性: 8/10 — 从 key embedding 端做 pre-attention steering 是新颖且有实际意义的 idea，谱分解 + 自适应路由设计精巧
- ⭐ 实验充分度: 8/10 — 覆盖 5 个模型 × 3 个标准 benchmark + lost-in-the-middle + 消融 + 效率分析，较为全面
- ⭐ 写作质量: 8/10 — 逻辑清晰，可视化（PCA、heatmap）直观，公式推导完整
- ⭐ 综合价值: 8/10 — 解决了 attention steering 与 FlashAttention 不兼容的实际痛点，方法简洁高效，工程落地友好
