# Query-Level Uncertainty in Large Language Models

**会议**: ICLR2026  
**arXiv**: [2506.09669](https://arxiv.org/abs/2506.09669)  
**代码**: [GitHub](https://github.com/tigerchen52/query_level_uncertainty)  
**领域**: llm_reasoning  
**关键词**: uncertainty estimation, knowledge boundary, adaptive inference, training-free, internal confidence  

## 一句话总结
提出Query-Level Uncertainty概念，通过Internal Confidence方法在生成前（单次前向传播）估计LLM能否回答给定查询，无需训练即可实现高效的自适应推理（RAG触发/模型级联/弃权）。

## 背景与动机
1. LLM存在知识边界，无法准确回答所有问题；知识边界感知对构建可信高效AI系统至关重要
2. 现有不确定性估计多为answer-level（生成后评估），计算开销大（需完整生成答案）
3. 自适应推理（如RAG、慢思考、模型级联）需要pre-generation信号来决定是否触发额外资源
4. 已有query-level方法需训练探针或微调模型（如IDK token、R-Tuning），泛化性受限
5. LLM内部隐藏状态蕴含丰富的知识可达性信息，跨层一致性可提升输出质量

## 方法
**核心思路**: 用yes/no自评提示让LLM判断能否回答查询，提取P(Yes)作为置信度，无需生成答案。

**Internal Confidence**: 不仅用最后一层最后一个token的P(Yes)，而是跨所有层和token位置计算P(Yes)，然后用Attenuated Encoding加权聚合。权重以"决策中心"（默认为最后层最后token）为中心指数衰减，控制locality参数α=1.0。

**关键公式**: $\text{IC}(\mathbf{h}) = \sum_{n}\sum_{l} w_n^{(l)} P(\text{Yes}|\mathbf{h}_n^{(l)})$，其中权重通过 $\delta_j^{(i)} = \exp(-\alpha|i-j|^2)$ 归一化得到。

**特性**: 完全training-free，仅需单次前向传播，无需生成任何token。

## 实验
| 方法 | Avg AUROC (Qwen-14B) | Avg PRR | 速度提升 |
|------|---------------------|---------|---------|
| Internal Confidence | **67.1** | **31.7** | — |
| P(Yes) top-right | 60.9 | 23.1 | — |
| Predictive Entropy | 59.6 | 19.5 | — |
| Perplexity | 57.7 | 15.3 | — |

**vs Answer-level方法** (GSM8K, Qwen-14B): Internal Confidence AUROC=66.8，耗时0.3s/样本；Semantic Entropy AUROC=60.0，耗时151.8s（506×慢）。

**关键发现**: (1) IC在3个数据集3个模型上一致优于所有baseline; (2) 相比answer-level方法快32×-602×; (3) RAG场景下可在性能几乎不降的情况下减少50%+的RAG调用; (4) 模型越大IC效果越显著; (5) 决策中心附近的层和token信息最有区分力。

## 亮点
- 首次形式化定义query-level uncertainty，将不确定性估计从"后验"推向"先验"
- 完全training-free，单次前向传播，实用性极强
- 跨层跨token的加权聚合策略（Attenuated Encoding）简洁有效
- 在RAG和模型级联中展示了显著的效率-质量权衡优势

## 局限
- 决策中心固定在最后层最后token，非最优（但为保持training-free的权衡）
- 仅在有明确答案的任务（事实QA/数学推理）上验证，开放式生成未涉及
- 贪心解码作为知识边界定义的代理较保守，可能低估模型能力
- 对reasoning-heavy任务（如多步数学推理）的区分能力相对弱于factual QA

## 相关工作
- Answer-level不确定性: Semantic Entropy (Kuhn et al. 2023), P(True) (Kadavath et al. 2022)
- 知识边界检测: IDK token (Cohen et al. 2024), R-Tuning (Zhang et al. 2024a) — 均需训练
- 内部状态探针: Gottesman & Geva 2024训练轻量探针; Semantic Entropy Probes (Kossen et al. 2024)

## 评分
- 新颖性: ⭐⭐⭐⭐ (query-level概念新颖，方法简洁)
- 实验充分度: ⭐⭐⭐⭐ (3模型3数据集，多应用场景)
- 写作质量: ⭐⭐⭐⭐⭐ (问题定义清晰，图示直观)
- 价值: ⭐⭐⭐⭐ (对自适应推理有直接实用价值)
