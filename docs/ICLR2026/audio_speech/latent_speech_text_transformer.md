# Latent Speech-Text Transformer

**会议**: ICLR 2026 Oral  
**arXiv**: [2510.06195](https://arxiv.org/abs/2510.06195)  
**代码**: 无  
**领域**: 语音-文本多模态  
**关键词**: speech-text modeling, latent patches, autoregressive, ASR, TTS, cross-modal alignment  

## 一句话总结
提出 Latent Speech-Text Transformer (LST)，将离散语音 token 聚合为更高层级的"潜在语音 patch"作为自回归单元，对齐语音和文本的序列建模粒度，在 speech HellaSwag 上获得 +6.5% 绝对提升，且增益随模型规模（420M→7B）持续增长，同时降低 ASR/TTS 推理计算成本。

## 研究背景与动机
1. **领域现状**：语音-文本联合自回归模型面临关键问题——语音 token 序列远长于对应文本，导致计算效率低且跨模态知识迁移困难。
2. **现有痛点**：直接在离散语音 token 上做自回归建模，序列长度可达文本的 10-20×，推理成本极高。
3. **核心idea一句话**：引入潜在语音 patch 作为中间自回归单元，使语音和文本在序列长度上对齐，patch 自然对应文本单元（如音节、单词），促进跨模态知识迁移。

## 方法详解

### 整体框架
在编码器和解码器之间引入 patch 聚合层，将连续的离散语音 token 压缩为潜在 patch，自回归预测在 patch 级别进行，最终由解码器将 patch 展开为语音 token。

### 关键设计
1. **潜在语音 patch**：将多个连续语音 token 聚合为一个 patch，降低自回归序列长度，patch 自然对应文本的语义单元
2. **跨模态对齐**：patch 粒度与文本 token 粒度相近，使模型更容易学习语音-文本的对应关系
3. **可扩展性**：从 420M 到 7B 参数一致有效

## 实验关键数据

| 基准 | 指标 | 提升 |
|------|------|------|
| Speech HellaSwag | 准确率 (compute-controlled) | +6.5% |
| Speech HellaSwag | 准确率 (data-controlled) | +5.3% |
| ASR/TTS | 推理效率 | 显著降低序列长度 |

- 增益从 420M → 1.8B → 7B 持续增长
- ASR 适应更稳定，TTS 推理更高效

## 亮点与洞察
- **语音 patch = 文本 token 的自然对应**：patch 自动学习到与音节/单词的对齐，无需显式对齐监督
- **效率与质量双赢**：降低序列长度既省计算又提升质量（减少长距离依赖的难度）

## 局限性 / 可改进方向
- patch 大小的选择对不同语言/说话速率的鲁棒性未充分验证
- 未与 Moshi 等端到端语音 LLM 做直接对比

## 评分
- 新颖性: ⭐⭐⭐⭐ 潜在 patch 概念简洁有效
- 实验充分度: ⭐⭐⭐⭐ 多尺度验证充分
- 写作质量: ⭐⭐⭐⭐ 清晰易懂
- 价值: ⭐⭐⭐⭐ 对语音-文本联合建模有重要指导意义
