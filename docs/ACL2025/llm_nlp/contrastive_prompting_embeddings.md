# Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering

**会议**: ACL 2025  
**arXiv**: [2505.12831](https://arxiv.org/abs/2505.12831)  
**代码**: [GitHub](https://github.com/zifengcheng/CP)  
**领域**: llm_nlp  
**关键词**: sentence embedding, contrastive prompting, activation steering, inference-time, LLM  

## 一句话总结
提出对比提示（Contrastive Prompting, CP）方法，通过引入辅助提示（引导编码非核心信息）并在推理时与正常提示的激活值做对比减法，过滤掉停用词等无关语义，使 LLM 的句子嵌入更聚焦核心语义，在 STS 和分类任务上一致提升现有提示方法。

## 研究背景与动机
1. **领域现状**：从 LLM 提取零样本句子嵌入是实用方向（无需额外数据/微调），主流方法通过 prompt 工程将句子语义压缩到最后一个 token 的隐状态。
2. **现有痛点**：最后一个 token 仍编码了大量非核心信息（停用词等）——即使用 Knowledge 提示强调"主语和动作"，解码概率最高的仍是停用词"a"。
3. **核心矛盾**：prompt 工程只能间接改变表示，无法直接过滤非核心信息。
4. **本文要解决什么？** 如何在推理时直接修改隐状态以增强核心语义、去除非核心信息？
5. **切入角度**：引入辅助提示（"这句话的无关信息是..."）编码非核心信息，用正常提示减去辅助提示的激活值得到纯核心语义向量。
6. **核心 idea 一句话**：用"语义减法"——正常提示的激活值减去辅助提示的激活值，直接过滤非核心信息。

## 方法详解

### 整体框架
三步流程：(1) 辅助提示前向传播到第 $\ell$ 层，提取最后 token 的 contextualized value vector $\mathbf{v}^{\text{aux}}$；(2) 正常提示前向传播到第 $\ell$ 层，用对比向量 $\Delta\mathbf{v} = \mathbf{v}^{\text{nor}} - \mathbf{v}^{\text{aux}}$ 替换最后 token 的 value vector；(3) 调整向量范数后继续前向传播到最后一层，提取句子嵌入。

### 关键设计
1. **辅助提示设计**:
   - 模板："The irrelevant information of this sentence: '[TEXT]' means in one word: "
   - 仅需传播到第 $\ell$ 层（低层），开销极小
   - 可探索不同辅助提示（如"This sentence '[TEXT]' can be ignored"）

2. **对比激活导向（Contrastive Activation Steering）**:
   - 语义激活向量：$\Delta\mathbf{v}^\ell = \mathbf{v}^{\text{nor},(\ell)}_{N_{\text{nor}}} - \mathbf{v}^{\text{aux},(\ell)}_{N_{\text{aux}}}$
   - 仅干预最后一个 token 的 value vector
   - 与 activation steering 方向不同：不需要监督数据的正负样本对

3. **范数调整策略**:
   - 干预后向量范数可能变化，提出两种调整：固定原范数 / 缩放到指定范数
   - 保持表示空间的一致性

### 即插即用特性
CP 可与任意现有提示方法组合：PromptEOL、Pretended CoT、Knowledge、MetaEOL 等，持续提升性能。

## 实验关键数据

### STS 基准（7 任务平均 Spearman 相关系数）
| 方法 | LLM | Avg. 提升 |
|------|-----|----------|
| PromptEOL → +CP | Llama-2-7B | +显著 |
| Knowledge → +CP | Llama-2-7B | +显著 |
| MetaEOL → +CP | Mistral-7B | +显著 |

### 关键发现
- CP 在所有测试的 LLM（Llama-2、Mistral、Qwen 等）和所有基础方法上一致提升
- 辅助提示仅需传播到低层（如第 8 层），额外开销极小
- 对比解码概率验证：CP 后 top-1 概率 token 从停用词变为语义关键词
- 不同辅助提示设计的效果差异较小，方法鲁棒

## 亮点与洞察
- "语义减法"思路简洁优雅——利用辅助提示捕获噪声，正常提示减去噪声 = 纯信号
- 与 activation steering 的联系但又不同：不需要监督数据，每个句子自适应生成控制向量
- 即插即用特性使其可以无缝集成到现有系统中

## 局限性 / 可改进方向
- 辅助提示的设计仍有一定主观性
- 在推理时需要两次前向传播（虽然辅助只到低层）
- 仅在英语 STS 上评估，跨语言效果未知
- 未与微调方法（如 SimCSE 在 LLM 上的版本）对比

## 相关工作与启发
- **vs PromptEOL/MetaEOL**: CP 作为插件在其基础上持续提升
- **vs Activation Steering (Zou et al.)**: 传统方法需要正负监督样本，CP 用辅助提示自生成
- **vs Echo Embeddings**: Echo 重复输入利用 attention，CP 对比两种提示利用语义差

## 评分
- 新颖性: ⭐⭐⭐⭐ 辅助提示+激活对比的思路新颖简洁
- 实验充分度: ⭐⭐⭐⭐ 多 LLM、多方法、STS+分类，对比充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，可视化说明直观
- 价值: ⭐⭐⭐⭐ 对 LLM 句子嵌入研究有直接贡献，即插即用实用性强
