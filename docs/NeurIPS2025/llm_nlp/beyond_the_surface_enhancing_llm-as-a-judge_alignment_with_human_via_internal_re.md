<!-- 由 src/gen_stubs.py 自动生成 -->
# Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations

**会议**: NeurIPS 2025  
**arXiv**: [2508.03550](https://arxiv.org/abs/2508.03550)  
**代码**: [https://github.com/sustech-nlp/LAGER](https://github.com/sustech-nlp/LAGER)  
**领域**: LLM/NLP  
**关键词**: LLM-as-a-Judge, 隐层表征, 跨层聚合, 评估对齐, 即插即用

## 一句话总结
提出LAGER框架，通过聚合LLM中间层到最终层的score token logits并计算期望分数，无需微调模型即可将LLM评判与人类评分的对齐度提升最高7.5%，且不需要思维链推理步骤就能匹配或超过推理类方法。

## 研究背景与动机
1. **领域现状**：LLM-as-a-Judge已成为自动评估的主流范式，但如何提升其与人类判断的一致性仍是核心挑战。现有方法要么依赖复杂的思维链推理（增加计算成本），要么需要微调（丧失泛化性）。

2. **现有痛点**：标准做法只用最终层的最高概率score token作为评分（vanilla score），(a) 忽略了概率分布中的丰富信息（如4分和5分概率接近但只选5分）；(b) 忽略了中间层可能编码了更好的评判信号。

3. **核心观察**：经验分析发现，中间到上层（middle-to-upper layers）的隐层表征产生的评分与人类判断的一致性常常**高于最终层**——不同层编码了互补的语义和任务信息。

4. **切入角度**：利用所有层的score logits加权聚合得到更好的评分分布，再取期望得到连续细粒度评分。权重通过小规模验证集轻量训练（仅L+1个参数），模型完全冻结。

5. **核心idea一句话**：跨层logits加权聚合 + 概率分布期望 = 比只看最终层argmax更好的评判分数，且即插即用。

## 方法详解

### 整体框架
LLM生成评判时，在score token位置提取所有L+1层（embedding层到最终decoder层）的隐层表征 $\mathbf{h}_n^{(l)}$，通过共享的unembedding矩阵映射为logits，按层加权聚合，对candidate score token做softmax得到概率分布，取期望作为最终评分。

### 关键设计

1. **跨层Logits聚合**：
   - 做什么：$\hat{\mathbf{z}} = \sum_{i=0}^{L} w_i [\mathbf{h}_n^{(i)} \mathbf{W}_{\text{unembd}}]_{\mathcal{M}}$，其中 $\mathcal{M}$ 是candidate score tokens的集合
   - 设计动机：不同层编码不同粒度的信息。底层偏词汇局部信息，中层偏语义，高层偏任务推理。聚合后得到的分数综合了各层视角
   - 关键细节：在聚合**之前**不做softmax归一化（消融证明先归一化会丢失logits的相对尺度信息，性能更差）

2. **期望分数（Expected Score）**：
   - 做什么：$s^* = \sum_{s \in \mathbb{S}} s \times P(s)$，其中 $P(s) = \text{softmax}(\hat{\mathbf{z}})[s]$
   - 设计动机：比argmax更细粒度——如果P(4)=0.45, P(5)=0.55，argmax给5分，期望给4.55分，后者更能区分回复质量
   - 这个简单改动本身就带来显著提升（E-Score基线）

3. **轻量权重训练**：
   - 做什么：在小规模验证集上用CE+MAE联合损失训练L+1个层权重参数（如LLaMA-3.1-8B仅33个参数）
   - 模型backbone完全冻结，不改变next-token prediction
   - 训练一次，跨所有benchmark和下游任务复用
   - 不调权重的均匀聚合版本（LAGER w.o. tuning）也有显著提升

## 实验关键数据

### 主实验（Spearman相关系数，Direct评估即无推理链）

| 模型 | 方法 | Flask | HelpSteer | BIGGen | 平均 |
|------|------|-------|-----------|--------|------|
| LLaMA-3.1-8B | VScore | 0.442 | 0.452 | 0.333 | 0.409 |
| LLaMA-3.1-8B | E-Score | 0.454 | 0.520 | 0.403 | 0.459 |
| LLaMA-3.1-8B | **LAGER** | **0.488** | **0.560** | **0.421** | **0.490** |
| Qwen-2.5-14B | VScore | 0.489 | 0.440 | 0.420 | 0.450 |
| Qwen-2.5-14B | **LAGER** | **0.528** | **0.524** | **0.449** | **0.500** |
| LLaMA-3.3-70B | VScore | 0.501 | 0.508 | 0.445 | 0.485 |
| LLaMA-3.3-70B | **LAGER** | **0.538** | **0.548** | **0.473** | **0.520** |

### 与推理方法对比

| 方法 | Flask | HelpSteer | BIGGen |
|------|-------|-----------|--------|
| VScore+Reasoning | 0.456 | 0.470 | 0.388 |
| LAGER (无推理) | **0.488** | **0.560** | **0.421** |

### 关键发现
- LAGER在三个benchmark上平均提升最高7.5% Spearman相关
- 不需要推理链就能匹配甚至超过显式推理方法——推理链的shallow reasoning反而不可靠
- 均匀聚合（无训练）已有显著提升，训练权重进一步改善
- 先softmax再聚合 < 先聚合再softmax：保留logits尺度信息很重要
- 下游验证：用LAGER选择指令微调数据，在AlpacaEval-2.0上比多个基线更好

## 亮点与洞察
- **"中间层比最终层更懂评判"**的发现很有启发性：最终层可能因为过度拟合next-token prediction目标而丢失了某些评判相关的语义信号
- **极简设计**：仅33个可训练参数（对8B模型），完全即插即用，不改变推理流程——这可能是目前改善LLM-as-a-Judge最轻量的方法
- **期望分数 vs argmax**的改进看似微小但意义重大：它将离散评分转为连续评分，捕捉了模型的"犹豫"信息

## 局限性 / 可改进方向
- 需要访问中间层隐层表征——对API-only模型不适用（虽然退化为E-Score仍有效）
- 权重在验证集上训练——如果验证集与测试分布差异大，可能不最优
- 只关注point-wise评估——pairwise比较场景未探索
- 层数固定的模型间权重不可迁移——每个模型需单独训练权重
- 未分析为什么中间层更好——缺少对"中间层编码了什么评判信号"的深入机制分析

## 相关工作与启发
- **vs G-Eval**：G-Eval用最终层logits的softmax期望（即E-Score），LAGER进一步跨层聚合，效果更好
- **vs Prometheus/TIGERScore**：这些方法微调整个LLM，泛化性受限。LAGER冻结模型、33个参数
- **vs CoT推理方法**：推理方法增加延迟且推理质量不稳定，LAGER不需推理且更快更好

## 评分
- 新颖性: ⭐⭐⭐⭐ 跨层聚合用于评判是新颖视角，但核心技术（加权logits聚合）相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 6个模型、3个benchmark、多个消融、下游应用验证，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰（Figure 2的层级分析很有说服力），方法描述精确
- 价值: ⭐⭐⭐⭐⭐ 即插即用、几乎零成本的改进，对所有使用LLM-as-a-Judge的场景都有直接价值
