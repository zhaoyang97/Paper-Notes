# GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture

**会议**: ACL 2025  
**arXiv**: [2506.09440](https://arxiv.org/abs/2506.09440)  
**代码**: https://huggingface.co/ai-sage (有，开源模型)  
**领域**: LLM效率  
**关键词**: Mixture of Experts, 俄语LLM, 预训练, tokenizer优化, DPO

## 一句话总结
介绍 GigaChat 系列——首个从头为俄语设计并预训练的 MoE 架构 LLM 家族，包含 20B 总参数/3.3B 激活参数的基座和指令微调模型，在俄语 benchmark 上达到同规模 SOTA，训练速度是同量级 dense 模型的 2 倍，推理延迟降低 40%。

## 研究背景与动机

1. **领域现状**：多语言 LLM（如 Qwen, Mistral）在俄语上的支持主要通过后期 post-training 实现，缺乏从底层为俄语设计的基座模型。现有俄语开源模型（如 ruGPT-3.5）在 MERA 等 benchmark 上表现不佳。
2. **现有痛点**：(a) 从头训练大规模 LLM 需要巨大计算资源，限制了俄语专用模型的发展；(b) 俄语 tokenizer 效率低——通用 tokenizer 对 Cyrillic 字符编码碎片化严重；(c) 俄语专有模型（如 YandexGPT）缺乏透明度，不开源不公开架构。
3. **核心矛盾**：如何在有限资源下训练出性能优异的俄语 LLM？MoE 架构可以在保持模型容量的同时大幅降低计算开销，但对俄语的 MoE 训练经验缺乏。
4. **本文要解决什么？** 构建首个俄语专用 MoE LLM 家族，涵盖预训练、微调、对齐全流程，并开源。
5. **切入角度**：MoE 架构（20B 总参/3.3B 激活）+ 俄语优化 tokenizer + 9.5T tokens 预训练。
6. **核心 idea 一句话**：用 MoE 架构大幅降低训练和推理成本，配合定制 tokenizer 和多源数据，构建俄语专用高效 LLM。

## 方法详解

### 整体框架
GigaChat 家族包含：(1) GigaChat-A3B-base（20B/3.3B 激活 MoE 基座）；(2) GigaChat-A3B-instruct（指令微调版）；(3) GigaChat-A3B-instruct 1.5（含 DPO 对齐版）。此外还有不开源的高级版（Lite、Pro、MAX）通过 API/Telegram Bot/Web 访问。

### 关键设计

1. **MoE 架构设计**:
   - 做什么：用稀疏 MoE 替代 dense MLP，大幅降低每次前向传播的计算量
   - 核心思路：28 层 Transformer，每层含 2 个共享 expert + 64 个路由 expert，16 个注意力头 + 8 个 KV 头（GQA），隐藏维度扩展对齐 Mistral 7B（14,336）。第一层用标准 gated MLP（因为 token 分布问题）。使用 STK Triton kernels 做 block-sparse 计算，免去 expert 并行
   - 设计动机：相比 8B dense 模型（如 Llama 3），训练速度提升 2 倍，推理延迟降低 40%，计算消耗减少 40%；借鉴 DeepSeek MoE 的设计——更多 expert + 更小 expert + 共享 expert

2. **俄语优化 Tokenizer**:
   - 做什么：为西里尔字母、编程语言和 LaTeX 优化 BPE tokenizer
   - 核心思路：使用 HuggingFace BBPE 算法，在含俄语/英语/代码/LaTeX 的混合语料上迭代训练，生成 100+ 候选 tokenizer，选择跨领域平均 token 长度最优的版本。确保 Cyrillic 常用词不被过度切分，编程关键字和 LaTeX 语法完整保留
   - 设计动机：通用 tokenizer 对俄语编码效率低（碎片化），直接影响训练效率和模型容量的利用

3. **预训练数据与策略**:
   - 做什么：收集 9.5T tokens 多源数据，分阶段训练
   - 核心思路：数据包含 4.4T tokens web 数据（俄语 26.5%、英语 63.8%+）、630B tokens 高质量文献、230B tokens 代码、9B tokens 合成数据（数学+代码）。训练用多步常数学习率调度器（warmup 2000 步，在 30%/60%/90%/98% 节点衰减）。之后分两阶段扩展上下文：8K→32K→128K，配合 RoPE ABF 调整
   - 设计动机：合成数据借鉴 Phi-4，对数学和编程能力有显著提升；多阶段上下文扩展是当前长上下文 LLM 的标准做法

4. **改进的 DPO 损失**:
   - 做什么：修改标准 DPO 以减少幻觉和训练不稳定
   - 核心思路：引入非对称权重 $\beta_w$ 和 $\beta_l$，优先提升好回答的得分而非惩罚差回答；额外加入相对于 reference model 的 NLL 正则项以稳定 loss ratio
   - 设计动机：标准 DPO 过度关注拉大好坏差距而非提升绝对质量，且忽略共享前缀的重要性

### 损失函数 / 训练策略
- 预训练：标准 next-token prediction，batch size ~16M tokens
- SFT：约 250K 项人工标注数据，覆盖 10+ 领域
- DPO：非对称加权损失 + NLL 正则

## 实验关键数据

### 主实验（与同规模模型对比）

| Benchmark | GigaChat-A3B-instruct 1.5 | Qwen 2.5 (7B) | Llama 3.1 (8B) | T-Lite |
|-----------|---------------------------|---------------|----------------|--------|
| GSM8K (5-shot) | 0.774 | **0.895** | 0.789 | 0.882 |
| MMLU EN (5-shot) | 0.650 | **0.710** | 0.682 | 0.718 |
| MMLU RU (5-shot) | **0.600** | 0.632 | 0.569 | 0.626 |
| RUBQ (0-shot) | **0.688** | 0.373 | 0.484 | 0.583 |
| WINOGRANDE (4-shot) | **0.762** | 0.636 | 0.624 | 0.670 |
| HumanEval (0-shot) | 0.378 | **0.854** | 0.683 | 0.799 |

GigaChat 在俄语 benchmark (RUBQ, MMLU RU, WINOGRANDE) 上表现优异，但英语和代码任务落后于 Qwen 2.5。

### 效率对比

| 指标 | GigaChat-A3B (MoE) | 同级 Dense 8B |
|------|---------------------|--------------|
| 训练速度 | **2× 更快** | 基准 |
| 推理延迟 | **降低 40%** | 基准 |
| 激活参数 | 3.3B | ~8B |
| 总参数 | 20B | ~8B |

### 关键发现
- **MoE 在中等规模模型上效率优势显著**：3.3B 激活参数对标 8B dense 模型性能，但计算开销大幅降低
- **俄语专精 vs 通用多语言是 trade-off**：GigaChat 在俄语上领先但英语/代码落后于 Qwen 2.5，说明数据配比对多语言能力至关重要
- **改进 DPO 有效**：instruct 1.5（含 DPO）在多数 benchmark 上优于 instruct 版本
- **GigaChat MAX（闭源大版本）**与 Claude 3.7、GPT-4o 竞争力强，在俄语 MERA 上表现优秀

## 亮点与洞察
- **完整的 MoE LLM 构建经验报告**：从架构选择、tokenizer 训练、数据配比到 DPO 改进，是难得的工业级 MoE 训练 technical report，对复现 MoE 训练有重要参考价值
- **俄语 tokenizer 优化**：在多语种 + 代码 + LaTeX 场景下的 tokenizer 选择策略可迁移到其他非英语语言
- **非对称 DPO 损失的实用改进**：解决标准 DPO 过度关注拉大差距而非提升质量的问题，思路可通用

## 局限性 / 可改进方向
- **英语和代码能力较弱**：对于多语言用户吸引力有限
- **规模较小**：只有 3.3B 激活参数，与 70B+ 模型差距明显
- **pre-train 数据英语占比 64%**：对于"俄语专用"模型来说英语占比偏高，可能影响俄语知识密度
- **开源模型与闭源 MAX/Pro 差距大**：GSM8K 0.774 vs 0.956，核心技术可能未完全公开
- 可改进：(a) 可以增大 MoE 规模（如 100B+ 总参数）来对标 top-tier 模型；(b) 可以为更多低资源语言（哈萨克语、乌兹别克语）做专门优化

## 相关工作与启发
- **vs Mixtral (Jiang et al., 2024)**: 都是 MoE 架构，但 Mixtral 面向多语言/英语，GigaChat 专为俄语优化 tokenizer 和数据配比
- **vs DeepSeek MoE (Dai et al., 2024)**: GigaChat 借鉴了 DeepSeek 的更多/更小 expert + shared expert 设计
- **vs ruGPT-3.5**: 之前最好的俄语开源模型，但在 MERA 上表现远不如 GigaChat

## 评分
- 新颖性: ⭐⭐⭐ 架构上无创新（MoE 已成熟），主要贡献在工程和语言针对性应用
- 实验充分度: ⭐⭐⭐⭐ 涵盖俄英双语多个benchmark，有详细训练配置
- 写作质量: ⭐⭐⭐⭐ 技术报告风格，信息量大
- 价值: ⭐⭐⭐⭐ 对俄语NLP社区贡献大，MoE训练经验有参考价值
