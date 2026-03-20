# FastVID: Dynamic Density Pruning for Fast Video Large Language Models

## 基本信息
- **arXiv**: 2503.11187
- **会议**: NeurIPS 2025
- **作者**: Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, Pengzhang Liu, Sicheng Zhao, Guiguang Ding
- **机构**: Tsinghua University, JD.com
- **代码**: https://github.com/LunarShen/FastVID

## 一句话总结
提出 FastVID，通过动态时序分割 (DySeg) + 密度空时剪枝 (STPrune) 从时间和视觉两个维度系统性消除视频 token 冗余，在 LLaVA-OneVision-7B 上剪掉 90.3% 视频 token 后仍保留 98% 精度，LLM prefill 阶段加速 7.1×。

## 背景与动机
Video LLMs 因视频 token 量巨大导致推理成本高昂。现有方法：
- **图像 token 压缩**（FastV, VisionZip, LLaVA-PruMerge）：只处理空间冗余，忽略帧间时间依赖
- **视频压缩**（DyCoke, PruneVID, FrameFusion）：或破坏时间结构，或压缩不够极致，或引入大量延迟

核心洞察：视频 token 冗余需从两个维度分析——
1. **时间上下文**：帧序和连续性影响语义理解（打乱或缺帧导致错误）
2. **视觉上下文**：需要同时保留"代表性"和"独特性"信息

## 核心问题
如何在推理时极端压缩视频 token（>90%）的同时保持时间结构和视觉语义完整？

## 方法详解

### 1. Dynamic Temporal Segmentation (DySeg)
将视频自适应分割为时序有序的高冗余段。

计算相邻帧间余弦相似度 $t_i = \cos(\mathbf{f}_i, \mathbf{f}_{i+1})$，选择分割点：
$$\mathbf{S} = \mathbf{S}_1 \cup \mathbf{S}_2$$
- $\mathbf{S}_1$：最不相似的 $c-1$ 个转换点（确保最低分割数）
- $\mathbf{S}_2$：相似度低于阈值 $\tau$ 的转换点（自适应捕捉场景变化）

简单视频少分段，复杂视频多分段。vs. 固定间隔分割（不保证段内相似）和聚类分割（破坏时间序）。

### 2. Density Spatiotemporal Pruning (STPrune)
在每个高冗余段内进行两阶段剪枝：

**Density-based Token Merging (DTM)**：保留段级视觉上下文
- 每隔 $p$ 帧选取锚帧，利用密度峰值聚类选择锚 token：
$$\rho_i = \exp\left(-\frac{1}{k}\sum_{v_j \in \text{kNN}(v_i)} d(v_i, v_j)^2\right)$$
$$\delta_i = \min_{j: \rho_j > \rho_i} d(v_i, v_j)$$
- 密度峰值 token（$\rho_i \times \delta_i$ 大）作为锚点，兼具代表性和独特性
- 锚中心聚合：$a^* = \beta a + \frac{1-\beta}{n}\sum_{i=1}^n b_i$（$\beta=0.6$）
- 关键：保留锚 token 的原始位置信息，维护 RoPE 编码的空时结构

**Attention-based Token Selection (ATS)**：捕捉显著视觉细节
- 重新引入预训练 SigLIP head 获取 [CLS] 注意力分数
- 选择注意力最高的 token，与 DTM 互补

分配比例 $d=0.4$（DTM 40%, ATS 60%）效果最佳。

## 实验关键数据

### LLaVA-OneVision-7B (32帧)
| 方法 | 保留率 | TFLOPs | MVBench | VideoMME | Avg% |
|---|---|---|---|---|---|
| Vanilla | 100% | 48.82 | 56.9 | 58.6 | 100% |
| VisionZip* | 9.7% | 4.04 | 51.7 | 53.1 | 89.6% |
| PruneVID* | 10.1% | 4.23 | 54.2 | 55.9 | 95.4% |
| **FastVID** | **9.7%** | **4.04** | **55.9** | **57.3** | **98.0%** |

### 效率对比 (LLaVA-OneVision)
| 方法 | Prefill time | 加速比 | Avg Acc |
|---|---|---|---|
| Vanilla | 476.3ms | 1.0× | 100% |
| PruneVID | 101.5ms | 4.7× | 95.4% |
| **FastVID** | **67.2ms** | **7.1×** | **98.0%** |

### 跨模型泛化
- LLaVA-Video-7B (64帧)：保留率 25% → 98.1% 精度
- Qwen2-VL (768帧)：保留率 25% → 96.2% 精度
- Qwen2.5-VL (768帧)：保留率 24.1% → 93.3% 精度

### 长度外推

| 设置 | 帧数 | Token数 | VideoMME |
|---|---|---|---|
| Vanilla | 32 | 6272 | 58.6 |
| FastVID (r=25%) | 128 | 6272 | 60.4 (+1.8) |
| FastVID (r=10%) | 320 | 6080 | 61.4 (+2.8) |

→ 相同 token 预算下采样更多帧反而更好！

## 亮点
1. **系统性分析**：从时间和视觉两个维度剖析视频冗余，设计针对性解决方案
2. **极端压缩仍稳健**：90.3% 剪枝率下保持 98% 精度，远超同类方法
3. **训练无关，即插即用**：纯推理时方法，兼容 FlashAttention、KV cache、多轮对话
4. **广泛泛化性**：在 4 个不同 Video LLM 架构上均 SOTA
5. **密度峰值选择**：锚 token 同时满足代表性和独特性，优于均匀采样和聚类

## 局限性
1. Query-agnostic 剪枝——长视频中特定问题相关帧可能被误剪
2. 长帧模型 (LLaVA-Video 64帧, Qwen2-VL 768帧) 效果不如短帧模型显著
3. 密度分数计算虽做了并行优化，仍引入额外延迟（5.6ms）
4. DySeg 参数 ($c=8, \tau=0.9$) 对所有视频使用固定值

## 与相关工作的对比
- **vs. FastV**：FastV 按 LLM 注意力剪枝单帧 token，不处理帧间冗余
- **vs. VisionZip**：VisionZip 均匀采样+合并，内容无关；FastVID 密度引导更保留语义
- **vs. DyCoke**：DyCoke 跨帧合并+KV cache 压缩，但 prefill 压缩不够极致
- **vs. PruneVID**：PruneVID 用聚类分割+聚类合并，丢失位置信息且速度慢（6.1× slower）
- **vs. LLaVA-PruMerge**：PruMerge 仅空间剪枝，FastVID 空时联合剪枝

## 启发与关联
- **与 Eyes Wide Open 的关联**：Eyes Wide Open 关注流式场景的时序建模，FastVID 关注推理效率——两者可结合实现高效流式视频理解
- **长度外推的启示**：FastVID 证明"更多帧+更少 token per 帧"优于"少帧+全 token"，这为 Video LLM 设计提供了新思路
- **与 LLaVA-PruMerge 的互补**：空间剪枝和时空剪枝可以级联使用

## 评分
- 新颖性：★★★★☆ — 密度峰值在 token 剪枝中的应用新颖
- 技术深度：★★★★☆ — DySeg + STPrune 设计精巧，消融充分
- 实验完整度：★★★★★ — 4个模型 × 4个 benchmark × 多压缩率，极其全面
- 写作质量：★★★★☆ — 结构清晰但略冗长
