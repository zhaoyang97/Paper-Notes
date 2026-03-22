# CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2504.06704](https://arxiv.org/abs/2504.06704)  
**代码**: 无  
**领域**: Transformer效率优化  
**关键词**: 循环卷积、FFT加速、Self-Attention、复杂度优化

## 一句话总结
本文提出CAT（Circular-convolutional Attention），通过FFT计算循环卷积将Self-Attention复杂度从O(N²)降至O(N log N)，同时保持完整的softmax机制和全局注意力。

## 研究背景与动机
尽管Transformers在NLP和CV领域表现卓越，但O(N²)的Self-Attention复杂度仍是制约长序列处理的关键瓶颈。现有解决方案存在问题：
- Linear Transformers通过核方法近似softmax，导致数值不稳定
- 稀疏注意力方法需要长度相关的超参数调优
- Mamba等替代架构改变了Transformer核心机制

本工作引入**Engineering-Isomorphic Transformers (EITs)**框架，在保持全局softmax加权的前提下实现sub-quadratic复杂度。

## 方法详解

### 整体框架
EITs的4个核心要求：
1. **Softmax保持**：输出形式为softmax(F_attn(X)) · F_value(X)
2. **Sub-quadratic复杂度**：严格小于O(N²)
3. **参数效率**：可比标准注意力的参数量
4. **最少超参数**：无长度相关的超参元组

CAT的关键创新：使用单个投影`W_A ∈ ℝ^(D×1)`，而非Q、K、V三个投影。

### 关键设计
**循环卷积机制**：
- 从输入X学习得到向量Z = XW_A ∈ ℝ^(N×1)
- 应用softmax得到Z* = softmax(Z)
- 构造循环矩阵circ(Z*)，使每行都是前一行的一步循环移位
- 利用FFT计算：circ(Z*)·V = IFFT[FFT(Z*) ⊙ FFT(V)]

**理论优势**：
- softmax(circ(Z)) ≡ circ(softmax(Z))：确保exact row-wise softmax
- 从N²个注意力系数降至N个（循环核）
- 支持相对位置编码，符合Transformer早期层的shift模式

| 模型 | 池化方式 | 机制 | 学习参数 | 复杂度 | 精度↑ |
|------|--------|------|---------|--------|-------|
| CLIP-B | token | Attention | 3D² | O(N²) | 0.574 |
| CLIP-B | token | CAT | (D+H)D | O(N log N) | 0.540 |
| CLIP-B | avg | Attention | 3D² | O(N²) | 0.638 |
| CLIP-B | avg | CAT | (D+H)D | O(N log N) | 0.649 |
| CLIP-L | avg | CAT | (D+H)D | O(N log N) | 0.694 |

## 实验关键数据
- ImageNet-1k on ViT：CAT在average pooling下优于标准注意力（0.694 vs 0.646）
- WikiText-103 masked LM：CAT达到8.32 perplexity vs standard 9.82
- Wall-clock speedup：naive PyTorch实现中约10%加速
- 消融：query-value (qv)参数化与averaged-key (qkv)权衡接近（0.694 vs 0.696）

## 亮点与洞察
1. **EIT框架创新**：统一了softmax保持与sub-quadratic复杂度的矛盾
2. **最小设计**：仅需单个投影，显著降低参数
3. **理论严密性**：循环矩阵性质保证了softmax的exact性而非近似
4. **CAT-Alter混合方案**：部分层用CAT、部分用标准注意力，多数场景表现更优
5. **相对位置天然适配**：循环结构与Transformer早期层的局部模式对齐

## 局限性
1. 因果注意力下复杂度回到O(N²)（需显式mask），限制了解码应用
2. 在token pooling下性能略逊于标准注意力
3. FFT实现在中等序列长度(N=256)下开销抵消了理论优势
4. 需要特殊GPU内核充分发挥O(N log N)优势

## 相关工作
- 稀疏注意力：BigBird、Longformer
- 核方法注意力：Performer、Linear Transformer
- 替代架构：S4、Mamba
- 位置编码：相对位置、旋转位置编码

## 评分
⭐⭐⭐⭐ (4/5)
