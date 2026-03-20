# Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)

**会议**: ACL 2025  
**arXiv**: [2501.06892](https://arxiv.org/abs/2501.06892)  
**代码**: [https://github.com/pnborchert/FLARE](https://github.com/pnborchert/FLARE)  
**领域**: LLM效率  
**关键词**: cross-lingual transfer, LoRA, language fusion, adapter bottleneck, multilingual NLU

## 一句话总结
FLARE 在 LoRA 适配器的低秩瓶颈中通过轻量线性/非线性变换融合源语言（英语）和目标语言的逐层表示，无需额外参数即可实现参数高效的跨语言迁移，在 Llama 3.1 上 QA 精确匹配提升 4.9%。

## 研究背景与动机
1. **领域现状**：多语言预训练模型（mPLM）因英语语料占主导地位，非英语语言的表示空间欠训练，下游任务性能明显低于英语
2. **现有痛点**：
   - 输入级融合（拼接源+目标序列）导致序列长度翻倍，attention 计算量平方增长
   - X-Mixup 只在单个 transformer 层进行跨注意力对齐，且需额外参数
   - 翻译后测试（translate-test）丢失文化细节和语义
   - 翻译后训练（translate-train）利用标准 LoRA 未充分利用源语言信息
3. **核心矛盾**：改善跨语言迁移通常需要处理双语输入或额外模块，但这增加计算复杂度，与参数高效微调的目标冲突
4. **切入角度**：LoRA 已经将表示压缩到低秩瓶颈中，可以在这个压缩空间中融合双语信息，几乎不增加额外开销
5. **核心idea一句话**：在 LoRA 的 down-projection 后、up-projection 前，用简单的逐元素操作融合源语言和目标语言表示

## 方法详解

### 整体框架
FLARE 的流程：(1) 先在英语任务数据上标准 LoRA 微调得到 base model；(2) 在目标语言（机器翻译的平行数据）上微调时，将英语（源语言）输入通过 base model（无 fusion adapter）提取各层表示 $V^S$；(3) 目标语言输入通过带 fusion adapter 的模型，在每层 LoRA 瓶颈中将源语言表示 $v_{i+1}^S W^{down}$ 与目标语言表示 $v_i^T W^{down}$ 通过融合函数 $\phi$ 组合；(4) 融合后的低秩表示经 up-projection 加到冻结的 attention 输出上。

### 关键设计

1. **瓶颈内语言融合（Bottleneck Fusion）**:
   - 做什么：在 LoRA 的低秩空间（$r \ll d$）中融合双语表示
   - 核心思路：源语言表示 $S = v^S W^{down}$，目标语言表示 $T = v^T W^{down}$，共享 down-projection 确保在同一空间。融合函数 $\phi$ 包括：加法 $S+T$、乘法 $S \circ T$、ReLU 变体（add+relu、mul+relu）、cross-attention
   - 设计动机：复用 LoRA 已有的 down/up projection，不增加额外参数（除 cross-attention 外）。在低秩空间融合比在原始高维空间融合计算量小得多

2. **逐层表示提取与偏移**:
   - 做什么：源语言用 base model 第 $i+1$ 层表示与目标语言第 $i$ 层表示融合
   - 核心思路：$h = \phi(v_{i+1}^S W^{down}, v_i^T W^{down})$，源语言"领先一层"
   - 设计动机：第 $i+1$ 层已经过该层 transformer block 处理，包含更丰富的任务特定信息，可以"引导"目标语言的第 $i$ 层学习

3. **FLARE MT 变体**:
   - 做什么：用机器翻译模型的 encoder 表示（latent translation）代替 mPLM 的源语言表示
   - 核心思路：直接用 NLLB encoder 将目标语言映射为"潜在翻译"$v^T = \mathcal{M}(x^T)$，经线性投影后融合
   - 设计动机：免去源语言在 mPLM 中的前向传播，进一步减少计算量

### 融合函数对比
| 融合函数 | XNLI | TyDiQA | NusaX | 是否需额外参数 |
|---------|------|--------|-------|-------------|
| add | 80.53 | 40.31 | 78.73 | 否 |
| mul | 79.59 | 36.23 | 77.73 | 否 |
| add+relu | 80.99 | 40.93 | 79.18 | 否 |
| cross-attention | 80.66 | 39.15 | 77.72 | 是（少量） |

add+relu 表现最佳且无额外参数。

## 实验关键数据

### 主实验
四个 mPLM × 三个任务，FLARE vs 各 baseline 的平均性能和排名：

| 方法 | XLM-R Large | mT5-XL | Llama 3.1 8B | Gemma 2 9B |
|------|------------|--------|-------------|-----------|
| LoRA | 65.88 / rank 3.33 | 68.99 / rank 3.67 | 55.55 / rank 3.33 | 52.37 / rank 3.67 |
| X-Mixup | 64.69 / rank 4.67 | 68.82 / rank 4.00 | - | - |
| Input-level fusion | 65.41 / rank 3.00 | 68.84 / rank 4.33 | 55.86 / rank 3.33 | 52.54 / rank 3.00 |
| **FLARE** | **67.03 / rank 1.33** | **69.89 / rank 1.33** | **57.42 / rank 1.33** | **53.54 / rank 1.67** |

FLARE 在所有模型上平均排名第一。

### QA 任务提升（TyDiQA Exact Match）
| 模型 | LoRA | FLARE | 提升 |
|------|------|-------|------|
| Llama 3.1 8B | 15.88 | 20.77 | **+4.9%** |
| Gemma 2 9B | 4.21 | 6.38 | **+2.2%** |
| mT5-XL | 46.76 | 48.94 | +2.2% |
| XLM-R Large | 40.14 | 40.93 | +0.8% |

### 消融实验
| 配置 | Avg Performance | 说明 |
|------|----------------|------|
| FLARE (add+relu) | 67.03 | 完整模型（XLM-R） |
| w/o fusion (标准 LoRA) | 65.88 | 去掉融合掉 1.15 |
| FLARE MT | 65.89 | 用 MT encoder 代替 mPLM 表示，性能接近 |
| Input-level fusion | 65.41 | 拼接输入序列，不如 FLARE |
| X-Mixup | 64.69 | 单层 cross-attention，最差 |

### 关键发现
- **QA 任务获益最大**：decoder-only 模型（Llama、Gemma）在 QA 上提升最明显（+4.9%），因为生成式 QA 更依赖跨语言理解
- **add+relu 是最佳融合函数**：简单且有效，ReLU 帮助过滤不对齐 token 的信息
- **FLARE MT 可行**：用小得多的 MT encoder（600M）代替 mPLM 的源语言表示，性能损失很小，但效率更高
- **低资源语言获益更大**：在 NusaX（11 种印尼语言）上提升最明显

## 亮点与洞察
- **LoRA 瓶颈的新视角**：不只是降维，还可以作为跨语言信息融合的桥梁。这个思路可迁移到其他跨模态融合场景（如视觉-语言），复用 LoRA 基础设施实现轻量级模态融合
- **源语言表示"领先一层"**的设计直觉上类似于 knowledge distillation 中的"教师信号引导学生"，用处理更深的源语言表示引导目标语言学习
- **FLARE MT 展示了一种可能性**：MT encoder 的 latent representation 可以直接注入 LLM，避免翻译中的信息离散化损失

## 局限性 / 可改进方向
- 依赖机器翻译生成平行语料，MT 质量直接影响效果
- decoder-only 模型在分类任务上的提升不如 QA 明显
- 未验证在超低资源场景（无任何平行语料）下的表现
- 融合函数的选择仍是经验性的，可能存在更优的可学习融合方式

## 相关工作与启发
- **vs X-Mixup**: X-Mixup 在单层用 cross-attention 融合，需额外参数且层选择敏感；FLARE 在所有层的 LoRA 瓶颈中融合，无额外参数，更稳定
- **vs Input-level fusion**: 拼接输入导致序列翻倍；FLARE 独立处理后在低秩空间融合，计算量更小
- **vs AdaMergeX**: 合并英语任务 adapter 和目标语言 adapter 的权重；FLARE 在训练时动态融合表示，能学到更细粒度的交互

## 评分
- 新颖性: ⭐⭐⭐⭐ 在 LoRA 瓶颈中融合跨语言表示是简洁有效的新思路
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个模型 × 3 个任务 × 多种 baseline × 融合函数消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观，实验设计合理
- 价值: ⭐⭐⭐⭐ 对跨语言 PEFT 有实际价值，尤其在 QA 等生成任务上

## 相关工作与启发
- 详见论文原文 Related Work 部分的详细对比。

## 评分
- 新颖性: ⭐⭐⭐⭐ LoRA内语言融合
- 实验充分度: ⭐⭐⭐⭐ 多任务多模型
- 价值: ⭐⭐⭐⭐ 实用的参数高效跨语言方法
