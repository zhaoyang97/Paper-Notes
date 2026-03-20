# Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction

**会议**: AAAI 2026  
**arXiv**: [2509.10798](https://arxiv.org/abs/2509.10798)  
**代码**: [GitHub](https://github.com/Mambaaaaaaaaa/Judge-Q)  
**领域**: LLM效率 / KV Cache压缩  
**关键词**: KV Cache剪枝, 可训练soft token, 注意力蒸馏, 长上下文推理, 全局信息保留  

## 一句话总结
提出Judge Q，在模型词表中引入可训练的soft token，训练其注意力模式对齐实际解码token的注意力模式，使其在prefill阶段能替代局部窗口查询来评估KV cache重要性，从而更好地保留全局信息，在LongBench上提升~1分，RULER上提升3+分。

## 研究背景与动机

1. **领域现状**：LLM推理时KV cache随序列长度线性增长，在长上下文场景（>10K token）成为内存瓶颈。现有KV cache剪枝方法（H2O、SnapKV、PyramidKV等）在prefill阶段通过计算注意力分数来评估KV pair重要性，只保留top-k重要的KV pair。

2. **现有痛点**：
   - **过度依赖局部窗口**：现有方法用最后一个窗口的token作为query来计算KV重要性分数。这假设问题出现在输入末尾，但如果问题不在末尾，性能会显著下降
   - **忽略全局信息**：局部窗口只能看到序列末端，无法充分评估序列中远距离位置的KV pair对生成的重要性
   - **理论上界未被逼近**：作者发现直接用真实解码token的注意力来选择KV pair效果最好（理论上界），但prefill阶段还不知道解码token是什么

3. **核心矛盾**：KV cache剪枝的理想目标是保留对未来解码最重要的KV pair，但prefill时无法预知解码内容，而局部窗口是不充分的代理(proxy)。

4. **本文要解决什么？** 设计一种能在prefill阶段近似"用真实解码token选KV"这个理论上界的方法，同时保持低训练成本。

5. **切入角度**：既然理论上界是用解码token的注意力图来选KV，那就训练一组soft token让它们的注意力图逼近解码token的注意力图，作为解码token的代理。

6. **核心idea一句话**：用可训练soft token模拟解码token的注意力分布来指导KV cache剪枝，只训练embedding层参数，开销极低。

## 方法详解

### 整体框架
Judge Q在模型词表末尾添加 $n$ 个soft token（默认 $n=32$），训练时将soft token拼接在prompt后面，训练其对prompt的注意力图对齐真实response的注意力图。推理时将soft token拼接在输入末尾，用它们的注意力分数代替局部窗口来计算KV重要性，完成剪枝后移除soft token，用剪枝后的KV cache继续解码。

### 关键设计

1. **Soft Token注意力蒸馏**:
   - 做什么：训练soft token使其"看"prompt时的注意力分布与真实解码token"看"prompt时的分布一致
   - 核心思路：分别将soft token和response token拼接到prompt后，计算各自对prompt的注意力图，然后在token维度取平均得到 $\mathbf{A}_{\text{soft}}$ 和 $\mathbf{A}_{\text{resp}}$，训练损失为两者的MSE：$\mathcal{L} = \text{MSE}(\mathbf{A}_{\text{soft}}, \mathbf{A}_{\text{resp}})$
   - 设计动机：解码token的注意力天然指向对生成最关键的KV pair，如果soft token能学会相同的注意力模式，就能在prefill阶段代替解码token做出同样好的KV重要性判断
   - 与prompt tuning的区别：传统prompt tuning优化生成质量，Judge Q优化注意力模式对齐，目标完全不同

2. **极低训练成本**:
   - 做什么：只微调embedding层中soft token对应的参数
   - 核心思路：模型所有权重冻结，仅训练新增的32个token的embedding向量。训练数据为ShareGPT的50K样本（45K通用+5K代码），用模型自身生成response（而非用原始数据的response）
   - 设计动机：最小化训练开销，使方法可以轻量适配任何开源模型。用自身生成response是因为注意力图需要与模型自身的解码行为一致

3. **推理时KV Cache剪枝**:
   - 做什么：在prefill阶段用soft token的注意力指导KV剪枝
   - 核心思路：输入末尾追加32个soft token → prefill → 计算soft token对整个输入的注意力图 → 按注意力分数对KV pair排序 → 保留top-k → 移除soft token → 正常解码
   - 设计动机：soft token充当"探针"角色，通过训练获得了全局感知能力，比局部窗口能更好地识别全局关键信息

### 损失函数 / 训练策略
- 损失函数：$\mathcal{L} = \frac{1}{d}\|\mathbf{A}_{\text{soft}} - \mathbf{A}_{\text{resp}}\|_2^2$
- 训练数据：ShareGPT 50K，response由模型自身生成（非原始标注）
- Soft token数量：n=32取得最佳平衡
- 训练仅涉及embedding层的32个新token向量，其余参数完全冻结

## 实验关键数据

### 主实验

LongBench结果（Llama-3.1-8B-Instruct）：

| KV Budget | StreamingLLM | H2O | SnapKV | PyramidKV | **Judge Q** | Full KV |
|-----------|-------------|-----|--------|-----------|------------|---------|
| 128 | 30.50 | 33.67 | 34.31 | 34.08 | **35.90** | 41.23 |
| 256 | 31.79 | 34.37 | 36.56 | 36.00 | **37.69** | 41.23 |
| 512 | 32.64 | 35.30 | 38.31 | 37.58 | **39.17** | 41.23 |

RULER结果（Llama-3.1-8B-Instruct, seq=8192）：

| KV Budget | SnapKV | PyramidKV | **Judge Q** | Full KV |
|-----------|--------|-----------|------------|---------|
| 256 | 57.83 | 56.86 | **63.13** | 87.18 |
| 512 | 62.76 | 61.19 | **69.24** | 87.18 |
| 1024 | 68.21 | 66.30 | **74.12** | 87.18 |

### 消融实验

Critical KV Hit Rate（与理论上界的重合度）：

| KV Budget | SnapKV | Judge Q | 提升 |
|-----------|--------|---------|------|
| 128 | 53.44% | **61.37%** | +7.93% |
| 256 | 55.23% | **62.34%** | +7.11% |
| 512 | 58.46% | **65.06%** | +6.60% |

文本续写任务（DeepSeek-R1-Distill-Llama-8B）：

| 数据集 | SnapKV | Judge Q |
|--------|--------|---------|
| MATH-500 (budget=1024) | 52.4 | **55.0** |
| AIME24 (budget=3072) | 31.1 | **37.8** |

### 关键发现
- **低budget收益更大**：budget=128时提升1.59分，budget=512时提升0.86分。资源越受限方法越有效
- **RULER上提升最显著**：超过3分，峰值接近10分，因为RULER任务更依赖全局检索能力
- **Critical KV Hit Rate稳定高出~8%**：说明soft token学到的注意力模式确实更接近理论上界
- **问题位置不在末尾时优势更大**：prompt调整后baseline性能降~10%，Judge Q仅降<7%
- **模型自生成response比原始response更好**：因为注意力图需要与模型自身的解码行为一致
- **soft token数n=32最优**：过少（16）信息不够，过多（64+）训练难度增加但收益递减

## 亮点与洞察
- **理论上界的发现和逼近思路**：先证明"用真实解码token选KV"是上界，再设计soft token去逼近这个上界。这种"先找上界→再近似"的研究范式非常清晰
- **训练成本极低但效果显著**：只训练32个token的embedding向量，训练数据仅50K样本，就能在LongBench/RULER上持续提升。这使得方法可以轻松适配任何开源模型
- **soft token作为"探针"的思想**：将attention蒸馏与prompt tuning结合，soft token不是为了改变模型输出，而是为了"感知"哪些KV pair重要。这个思路可以迁移到其他需要全局感知的场景（如token pruning、层剪枝等）
- **实验证明局部窗口的局限性**：通过prompt调整实验直接证明了局部窗口方法的脆弱性

## 局限性 / 可改进方向
- **仅在prefill阶段做一次性剪枝**：解码过程中新生成的token也在增长KV cache，Judge Q没有解决decoding阶段的动态剪枝问题（作者在结论中也提到了这个future work）
- **对Full KV的差距仍然明显**：budget=128时Judge Q(35.90) vs Full KV(41.23)，差距5.33分。尤其在极低budget设置下信息丢失仍然严重
- **soft token数量固定**：不同任务可能需要不同数量的soft token来捕获不同粒度的全局信息
- **训练依赖特定数据集**：ShareGPT的数据分布可能不覆盖所有下游任务，虽然实验表明自生成更好，但不同模型需要各自训练
- **未探索与KV cache量化的组合**：KV cache压缩有两条路（剪枝+量化），Judge Q只做剪枝，与量化方法的兼容性未验证

## 相关工作与启发
- **vs SnapKV**: SnapKV用局部窗口+pooling，Judge Q用soft token替代局部窗口。在所有budget设置下Judge Q全面领先，且critical KV hit rate高出~8%
- **vs PyramidKV**: PyramidKV在SnapKV基础上加层级budget分配，但query仍来自局部窗口，Judge Q从根源（query选择）解决问题
- **vs StreamingLLM**: StreamingLLM只保留开头+局部window的KV，信息丢失最严重，在所有任务上均大幅落后
- **vs Lookahead Q-Cache**: 也用伪解码token指导剪枝，但需要额外解码步骤，Judge Q更优雅（训练好的soft token零开销使用）

## 评分
- 新颖性: ⭐⭐⭐⭐ Soft token注意力蒸馏用于KV cache剪枝是原创性想法，理论上界分析有说服力
- 实验充分度: ⭐⭐⭐⭐⭐ 三个benchmark、两个模型、多budget、hit rate分析、prompt扰动、数据消融、token数量消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 观察→动机→方法的推进逻辑清晰，Figure 1的方法对比图直观
- 价值: ⭐⭐⭐⭐ 方法简洁实用，训练成本极低，可直接应用于任何开源LLM
