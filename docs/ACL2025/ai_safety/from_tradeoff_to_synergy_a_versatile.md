# From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2505.09924](https://arxiv.org/abs/2505.09924)  
**代码**: [https://github.com/redwyd/SymMark](https://github.com/redwyd/SymMark)  
**领域**: LLM 水印 / AI 安全  
**关键词**: LLM watermarking, symbiotic watermark, token entropy, semantic entropy, text traceability

## 一句话总结

提出SymMark共生水印框架，融合logits-based和sampling-based两类水印方法（串行/并行/混合三种策略），通过token熵和语义熵自适应选择水印策略，在可检测性、鲁棒性、文本质量和安全性上实现SOTA。

## 研究背景与动机
1. **领域现状**: LLM水印分为logits-based（如KGW修改logit分布）和sampling-based（如AAR修改采样过程）两大类。
2. **现有痛点**: 两类方法各有优劣——logits-based鲁棒但影响文本质量，sampling-based保质但可检测性较弱；且都面临安全性问题（如水印窃取攻击）。
3. **核心矛盾**: 鲁棒性、文本质量和安全性三者之间存在根本性权衡（trade-off），难以同时优化。
4. **本文要解决什么**: 将两类水印方法融合，从trade-off转向synergy。
5. **切入角度**: 借鉴自然界共生关系思想，设计三种融合策略，用熵指标自适应选择。
6. **核心idea一句话**: 用token熵决定是否加logits水印、语义熵决定是否加sampling水印，实现自适应混合水印嵌入。

## 方法详解
### 整体框架
SymMark提供三种共生策略：Serial（串行，每个token同时嵌入两种水印）、Parallel（并行，奇偶位交替嵌入）、Hybrid（混合，基于熵自适应选择），并设计统一检测算法。实验默认使用Unigram作为logits水印、AAR作为sampling水印。在OPT、LLaMA、GPT-J三个模型系列上验证。

### 关键设计
1. **Serial策略**: $y_t = \mathcal{S}_w(\text{softmax}(\mathcal{A}_w(l_t)))$，先修改logits再用水印采样，最大化水印信号但可能影响质量。
2. **Parallel策略**: 奇数位用logits水印+原始采样，偶数位用原始logits+水印采样，独立嵌入减少干扰。
3. **Hybrid策略（核心）**: 引入两个熵判据——token熵 $H_{TE}$ 高于阈值α时加logits水印（模型不确定时修改logits影响小），语义熵 $H_{SE}$ 低于阈值β时加sampling水印（候选语义相似时替换token影响小）。

### 损失函数 / 训练策略
- 非训练方法，直接在推理时嵌入水印
- Hybrid使用K-means聚类（k=64, n=10个聚类）计算语义熵
- 默认超参：token熵阈值α=1.0，语义熵阈值β=0.5
- 检测使用逻辑或：$I = I_l \mid I_s$（任一水印检测到即判定为带水印）

## 实验关键数据
### 主实验（C4数据集 OPT-6.7B 可检测性）

| 方法 | TPR | TNR | F1 | AUC |
|------|-----|-----|-----|-----|
| KGW (logits) | 0.990 | 1.000 | 0.994 | 0.999 |
| Unigram (logits) | 0.995 | 1.000 | 0.997 | 0.998 |
| AAR (sampling) | 0.995 | 1.000 | 0.997 | 0.999 |
| EXP (sampling) | 0.975 | 0.925 | 0.951 | 0.960 |
| **SymMark-Serial** | **1.000** | **1.000** | **1.000** | **1.000** |
| **SymMark-Hybrid** | **1.000** | **1.000** | **1.000** | **1.000** |

### 消融实验（策略对比特性）

| 策略 | 可检测性 | 鲁棒性 | 文本质量 | 安全性 |
|------|---------|--------|---------|-------|
| Serial | 最优 | 最优 | 较差 | 一般 |
| Parallel | 中等 | 中等 | 最优 | 一般 |
| Hybrid | 优秀 | 优秀 | 优秀 | 最优 |

### 关键发现
- SymMark在C4和OpenGen两个数据集上均取得F1=1.000的完美可检测性
- 在3个模型系列（OPT/LLaMA/GPT-J）上均保持稳定优势
- 对比EXP(F1=0.951)和ITS(F1=0.957)等采样方法，优势明显

- Serial在可检测性和鲁棒性上最优（双重水印信号叠加）
- Parallel在文本质量上最优（交替嵌入减少干扰）
- Hybrid综合表现最佳，通过熵自适应平衡各指标
- 语义熵能有效识别何时嵌入sampling水印不影响语义
- 统一检测算法可同时检测三种策略的水印

## 亮点与洞察
- 首次系统性探索logits-based和sampling-based水印的融合，开创了共生水印范式
- 从trade-off到synergy的思路转变具有普适性启发——不同方法的优势可以互补而非相互排斥
- 双熵判据设计巧妙：token熵管logits水印（不确定时改logits影响小），语义熵管sampling水印（语义相似时换词影响小）
- Hybrid策略实现了四维指标（可检测性、鲁棒性、质量、安全性）的平衡最优
- 统一检测算法简洁高效：只要任一水印被检测到即可判定，逻辑或操作低假阳性
- 在11种baseline方法中取得全面领先，实验规模大

## 局限性 / 可改进方向
- 语义熵计算依赖K-means聚类（top-64 token embedding），增加额外计算开销
- 需要与原始模型相同tokenizer的模型进行语义聚类，限制了通用性
- 对长文本的水印检测效果未充分验证，实验文本长度固定在200±30 tokens
- 在对抗性更强的攻击（如模型蒸馏、paraphrase attack）下的鲁棒性待考察
- 超参数α和β的选择对性能有影响，不同场景可能需要重新调优
- 未探索多bit水印场景（当前仅做1-bit检测：有/无水印）

## 相关工作与启发
- 在KGW（Kirchenbauer et al., 2023）和AAR（Aaronson, 2023）两大奠基方法上架桥
- SynthID（Dathathri et al., 2024）的tournament sampling是另一种高质量采样方向
- 从信息论角度（Shannon entropy和semantic entropy）设计水印策略值得借鉴
- SWEET（Lee et al., 2024）和EWD（Lu et al., 2024）从熵角度优化单一方法的思路被本文推广到融合框架
- 对AI生成内容监管和知识产权保护有直接的实际价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 融合两类水印的思路新颖，但单个组件为已有方法
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集多模型多基线对比（11个baseline），涵盖四个维度
- 写作质量: ⭐⭐⭐⭐ 结构清晰，三种策略递进式呈现，易读
- 价值: ⭐⭐⭐⭐ 对LLM水印领域有实际推动作用
- 总评: 工程性强，实用价值高，代码开源便于复现
- 应用场景: AI生成内容监管、版权保护、学术诚信检测
- 复现性: 代码开源（SymMark），可直接集成到现有LLM服务
- 延伸性: 可探索更多类型水印的融合（如sentence-level + token-level）
- 开放问题: 如何在保持水印强度的同时应对更复杂的paraphrase攻击？
- 影响力: 为未来水印方法的设计提供了“融合而非取舍”的新范式
