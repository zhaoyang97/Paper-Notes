<!-- 由 src/gen_stubs.py 自动生成 -->
# Vocabulary Customization for Efficient Domain-Specific LLM Deployment

**会议**: NEURIPS2025  
**arXiv**: [2509.26124](https://arxiv.org/abs/2509.26124)  
**代码**: 待确认  
**领域**: llm_efficiency  
**关键词**: vocabulary extension, tokenizer, domain adaptation, inference efficiency, e-commerce, BPE  

## 一句话总结
提出一种保证不增加任何输入 token 数的词表扩展算法，通过向预训练 LLM 的 tokenizer 添加领域特定 token，在电商场景实现输入序列缩短 20%、推理吞吐量提升 20-30%，且不损失模型质量。

## 研究背景与动机
- LLM 在特定领域（电商、医疗、金融）部署时，通用 tokenizer 无法高效编码领域专用术语（品牌名、SKU、多语言描述符）
- 词表不匹配导致 token fertility（每词平均 token 数）偏高，直接增加推理延迟和成本
- 已有 tokenizer 扩展工作主要面向新语言适配（中文、泰语等），领域适配的研究很少
- 此前方法（如 Yamaguchi et al.）将新 merge 操作前置到列表头部，可能导致通用文本的编码效率反而下降
- AdaptiVocab 等方法用 n-gram token 替换已有 token，无法保证编码效率单调不降
- 对扩展后模型是否真正使用新 token 进行生成缺乏系统分析，此前无人研究这一关键问题
- 在 encoder-only 模型上的词表扩展研究较多（BERT 等），但自回归 LLM 需要额外考虑生成端是否采用新 token

## 方法详解

### Tokenizer 扩展算法
1. **训练领域 tokenizer**：在领域数据集上从头训练 BPE tokenizer，获取领域高频 token
2. **扩展原始 tokenizer**：
   - 从领域 tokenizer 中选取不在原词表中的新 token
   - 关键设计：将新 merge 操作**追加**到 merge list 末尾（而非前置），保证原有分词行为不变
   - **保证性质**：任何输入序列经扩展 tokenizer 编码后的 token 数 ≤ 原 tokenizer，因为新 merge 只在原有 merge 都完成后才触发
   - 对比 Yamaguchi et al. 的前置策略：前置会改变已有 merge 的优先级，可能导致通用文本的编码反而变差
3. **Embedding 初始化**：新 token 的 embedding/projection 向量用其组成子 token 的均值初始化（遵循 Yao et al. 2021 的最佳实践）
4. **继续训练**：在混合数据（通用 50% + 领域 50%）上用 cosine schedule 学习率（1e-5 → 5e-7）训练 10K 步
5. **词表大小权衡**：通过扫描不同新增 token 数量（1K-80K），评估编码效率 vs forward pass 速度，找最优平衡点

### 关键保证
- 追加 merge 策略确保向后兼容：不改变任何已有文本的分词结果
- 新 merge 只在原有 merge 规则匹配完毕后才被应用，因此 token 数只可能减少或不变
- 数学上可证：对任意输入 x，|tokens_new(x)| ≤ |tokens_old(x)|
- 这一性质使其特别适合生产环境：无需担心 edge case 导致的性能退化

### 与前置策略的对比
- 前置策略（Yamaguchi et al.）：领域文本压缩快但通用文本编码效率可能下降
- 追加策略（本文）：领域文本压缩稍慢但通用文本编码效率严格不降
- 在 Wikipedia 数据上，前置策略在添加 >20K token 后效率反而下降，本文方法保持稳定
- 考虑到生产中可能遇到的未知分布文本，追加策略的安全性更为重要

### 实验设置
- 基础模型：Llama-3.1 8B（已在电商数据上持续预训练）
- 训练框架：NVIDIA Megatron-LM，60 节点 × 8 H100 GPU
- 推理框架：vLLM，单卡 H100 部署
- 评测：14 个多语言电商任务 + 通用 benchmark（MMLU、NLU 等）

## 实验关键数据

### 推理效率提升

| 指标 | 原始 Llama 3.1 | 扩展后 (+30K tokens) | 变化 |
|---|---|---|---|
| 电商任务平均 token 数 | 基准 | 减少 ~20% | ↓20% |
| Forward pass 时间 | 基准 | +1% | 可忽略 |
| 推理 RPS (300 words) | 29.19 | **35.23** | **+20.7%** |
| 推理 RPS (3000 words) | 1.95 | **2.52** | **+29.2%** |
| Wikipedia token 数 | 基准 | 减少 2-3% | 微降 |
| 新 token 使用率 (≥15 words seq) | - | **~98%** | 模型有效采用 |
| 新 token 使用率 (<15 words seq) | - | ~95.3% | 短序列略低 |

### 模型质量保持

| 模型质量 | 通用 NLU (En) | MMLU | 电商 Avg (En) | 电商 Avg (non-En) |
|---|---|---|---|---|
| 8B LLM (原始) | 71.6 | 63.5 | 60.5 | 47.9 |
| + 词表扩展 | 71.8 | 63.4 | 60.1 | 47.6 |
| 差值 | +0.2 | -0.1 | -0.4 | -0.3 |

## 亮点
- 算法层面保证编码 token 数单调不增，这是相比所有前序工作的核心优势
- 首次系统分析模型是否真正使用新 token：序列 ≥15 words 时新分词被采用率达 98%
- 方法与量化、投机解码、优化 kernel 正交，可叠加获得乘性效率增益
- 工业级验证：eBay 生产环境的 14 个多语言电商任务，480 块 H100 训练
- 词表大小 trade-off 分析实用：30K 新 token 对 8B 模型是最优平衡点
- 长序列（3000 words）场景下吞吐提升更大（29.2%），因为注意力计算占比更高
- 整个流程可复现：仅需领域语料 + 原始模型 + 不到 24h 训练
- 模型质量几乎无损：通用和领域 benchmark 差异均在 0.5% 以内
- 基于 Megatron-LM 的训练框架和 vLLM 的部署验证，贴近生产实际

## 局限性 / 可改进方向
- 追加 merge 策略比前置策略在领域文本上的压缩率提升更慢，需要更多新 token 才达到同等压缩效果
- 实验仅在电商领域验证，医疗/法律/代码等领域的迁移效果未知
- 仅在 Llama 3.1 8B 上实验，作者自己指出 trade-off 是模型规模相关的（越大模型 embedding/projection 占比越小，影响越小）
- 继续训练需要 480 块 H100 × 24h，对中小团队门槛较高
- 未探索在 tokenizer 扩展后是否可以安全剪枝低频原始 token 以进一步优化
- 未与 AdaptiVocab 等方法做直接质量对比（仅与 Yamaguchi et al. 对比）
- 新 token 的 embedding 初始化策略较简单（均值），未尝试更复杂的初始化方法
- 对非英语语言的效率增益未单独报告
- 未讨论 tokenizer 扩展对 KV cache 大小的影响（token 数减少 → KV cache 也相应缩小）
- 缺少对不同初始 tokenizer（如 GPT-4、Mistral）的适用性验证

## 评分
- 新颖性: ⭐⭐⭐ (核心思路"追加而非前置"简洁但增量有限，关键贡献在工程保证和系统分析)
- 实验充分度: ⭐⭐⭐⭐ (14 个生产任务、token 使用率分析、forward pass 速度分析全面，但仅单一领域)
- 写作质量: ⭐⭐⭐⭐ (问题陈述清晰，trade-off 分析可视化好，实验设计系统)
- 价值: ⭐⭐⭐⭐ (工业部署直接可用，20-30% 吞吐提升有实际商业价值，但学术影响力可能有限)
