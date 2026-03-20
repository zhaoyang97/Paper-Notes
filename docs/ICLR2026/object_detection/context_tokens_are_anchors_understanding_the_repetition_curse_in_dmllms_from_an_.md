# Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective

**会议**: ICLR 2026  
**arXiv**: [2601.20520](https://arxiv.org/abs/2601.20520)  
**代码**: [GitHub](https://github.com/ErikZ719/CoTA)  
**领域**: multimodal_vlm, diffusion_language_model  
**关键词**: 扩散语言模型, 重复生成, 信息流分析, 缓存加速, 注意力机制  

## 一句话总结
通过信息流分析揭示扩散多模态大语言模型(dMLLMs)在使用缓存加速时产生"重复诅咒"的内在机制，并提出 CoTA 方法有效缓解重复问题。

## 背景与动机
1. 扩散式大语言模型(dLLMs)通过迭代去噪并行生成 token，推理延迟高，需依赖缓存加速
2. 缓存机制虽降低延迟，但常导致生成文本大量重复（Repeat Curse）
3. dMLLMs 的黑盒特性阻碍了对重复现象内部机制的理解
4. 作者提出从信息流角度分析，揭示 context token 作为"锚点"聚合语义信息的角色

## 方法
- **CTAE（Context-token Attention Enhancement）**：基于相对距离的高斯衰减项乘以注意力矩阵，增强对 context token 的注意力，保持原有信息流模式。衰减公式 $\mathcal{G}_{i,j} = \gamma_{\min} + (1-\gamma_{\min}) \exp(-(|i-j|/\tau)^2)$
- **CTEV（Context-token Entropy-Guided Voting）**：计算深层（26-30层）context token 的累积熵作为惩罚项加入置信度评分，避免基于不确定 context token 进行解码
- 两个模块均为 plug-and-play，无需训练，计算开销小

## 实验
| 方法 | ARR↓(512) | MRL↓(512) | SRR↓(512) | ARR↓(64) |
|------|-----------|-----------|-----------|----------|
| LLaDA-V | 0.2 | 2.0 | 6.9 | 0.1 |
| +Cache | 14.3 | 11.0 | 82.3 | 7.1 |
| +Cache+CoTA | **1.2** | **1.3** | **6.3** | **1.0** |

- CoTA 将相邻重复率(ARR)降低最多 92%
- 在 DocVQA/ChartQA/MMStar/MME 等 8 个多模态基准上均有一致性能提升
- CTAE 和 CTEV 各自独立也能有效减轻重复

## 亮点
- 首次从信息流视角系统分析 dMLLMs 中缓存导致的重复现象
- 发现 context token 在双向注意力中起"锚点"作用，将注意力逐层集中
- 方法简洁优雅，完全无需训练即可显著改善重复问题

## 局限
- 仅在 LLaDA-V 一个 dMLLM 上验证，泛化性待考察
- 温度参数 τ=5 和深层定义（26-30层）均为经验性设定
- 对更长文本生成和多轮对话场景的效果未充分验证

## 相关工作
- 与 autoregressive 模型中的 attention sink 现象类似，但本文首次在 dMLLMs 中发现
- 相比 dLLM-Cache 等加速方法，CoTA 作为后处理可与之兼容

## 评分
- ⭐⭐⭐⭐ 新颖性: 4/5（信息流分析角度新颖）
- ⭐⭐⭐⭐ 实验充分度: 4/5（消融完整，多基准验证）
- ⭐⭐⭐⭐ 写作质量: 4/5
- ⭐⭐⭐⭐ 价值: 4/5（解决实用问题，方法即插即用）
