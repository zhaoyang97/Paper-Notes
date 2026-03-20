# DTR: Dynamic Token Reweighting for Robust Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2505.17132](https://arxiv.org/abs/2505.17132)  
**代码**: [https://github.com/TanqiuJiang/DTR](https://github.com/TanqiuJiang/DTR)  
**领域**: AI安全 / 多模态对抗防御  
**关键词**: VLM jailbreak defense, KV cache optimization, visual token reweighting, refusal direction, inference-time safety  

## 一句话总结
提出DTR——首个通过KV cache优化防御多模态越狱攻击的方法：利用反转安全偏移（Reversal Safety-Relevant Shift）识别对抗性视觉token，通过动态重加权衰减其影响，仅4步优化即可在不依赖图生文转换的前提下，大幅降低攻击成功率（HADES S+T+A: 56.9%→15.9%）同时保持VLM性能和推理效率。

## 背景与动机
VLM（LLaVA、InternVL等）因引入视觉模态而比纯LLM更脆弱，攻击者通过对抗扰动/SD生成/排版嵌入有害内容绕过安全护栏。现有防御方法问题：微调方法（RLHF等）需要人工标注的安全数据且计算昂贵；推理阶段方法（AdaShield迭代prompt、ECSO图转文）开销大或信息损失严重；偏移校准方法（ShiftDC、CoCA）需要图转文参考来精确估计安全偏移，效果受限。核心问题在于：视觉模态引入的安全相关分布偏移如何高效、准确地量化和消除？

## 核心问题
如何在无需安全参考数据、图转文转换或模型微调的前提下，通过推理时干预有效防御多模态越狱攻击？

## 方法详解

### 整体框架
预计算拒绝方向向量 $\mathbf{d}_{ref}$（仅需32对有害/无害prompt各一次）→ 推理时对每个输入优化视觉token缩放向量 $\alpha \in [0,1]^n$（4步梯度下降）→ 应用 $\alpha$ 重加权KV cache → 可选：剔除低权重token加速推理。

### 关键设计
1. **反转安全偏移 (RSS)**：不直接测量安全偏移（需要文本参考 $\tilde{x}$），而是测量通过优化 $\alpha$ 能沿*反转*拒绝方向移动多远。insight：越狱query被优化为误导安全判断，因此可被反向优化回安全区域（RSS大）；良性query本身未偏移，反向优化空间小（RSS小）。实验验证：100个越狱 vs 100个良性query的RSS分布显著分离。

2. **动态token重加权**：优化目标 $\mathcal{L}(\alpha) = f(x(\alpha)) \cdot \mathbf{d}_{ref}/\|\mathbf{d}_{ref}\| + \lambda\|f(x) - f(x(\alpha))\|_2$
   - 第一项：沿拒绝方向最小化安全偏移（对越狱主动修正，对良性无影响）
   - 第二项：保持重加权后的激活与原始激活接近（保留VLM能力）
   - $\lambda=0.1$ 平衡安全与效用

3. **攻击者的根本困境**：DTR创造了攻防博弈的两难——增加对抗token权重（绕过安全护栏）会破坏视觉语义连贯性→ASR-G降低；保持特征token权重→安全偏移被DTR逆转→ASR-R降低。攻击者无法同时优化两个目标。

### 优化策略
- 早停：4步即可，越狱query的loss前4步急剧下降
- Token剔除：$\alpha < \beta$ 的token直接从KV cache移除，20%剔除率平衡效率和性能
- 推理时间：4.01s vs Base 3.65s（仅+10%开销），远优于ShiftDC 10.66s

## 实验关键数据

| LLaVA-Llama2-7b | HADES S+T+A ASR↓ | MM-SafetyBench S+T ASR↓ | JailBreakV Style ASR↓ |
|---|---|---|---|
| Base | 56.9% | 74.5% | 34.0% |
| AdaShield | 17.6% | 13.6% | 8.5% |
| ShiftDC | 16.8% | 13.6% | 25.5% |
| CoCA | 35.7% | 53.6% | 8.5% |
| **DTR** | **15.9%** | **10.0%** | **6.4%** |

- MM-Vet性能保持：DTR在6项能力中5项持平或优于base，Recognition完全不降（50.3→50.3）；CoCA大幅劣化（50.3→28.7）
- 跨模型验证：InternVL-2.5-26b HADES S+T+A 23.1%→3.5%，Llama-4-Scout-17B 11.2%→8.4%
- 自适应攻击+PGD：未防御68% ASR → DTR 18%（仍有效）
- VLGuard文本攻击：safe-image+harmful-text ASR 66.5%→7.4%

### 消融实验要点
- n_ref=32即够（16也接近最优），方向稳定跨数据集/跨领域泛化
- 优化步数m=4即够，loss在前4步急剧下降
- λ=0.1最优平衡安全与效用
- 20%剔除率：推理时间从4.01s降至更低，ASR不变
- 均匀缩放 vs DTR：均匀α=0.3时严重幻觉，DTR精准定位对抗token

## 亮点
- 首次将KV cache优化用于VLM安全——开辟全新防御范式
- RSS的formulation精巧：避免图转文的信息损失，直接在激活空间操作
- 攻击者困境的分析深入：ASR-R和ASR-G的此消彼长使DTR对自适应攻击也鲁棒
- 可解释性强：α热力图直接显示哪些视觉token是对抗性的
- 计算开销极小（仅+10%推理时间），无需训练，即插即用

## 局限性 / 可改进方向
- 对原生多模态VLM（如GPT-4o，视觉和文本不分离处理）需扩展
- 拒绝方向需要少量有害/无害prompt预计算——虽然32个即够但仍非完全零数据
- 强自适应攻击（PGD最小化RSS）仍能达18% ASR——完全消除有挑战
- 层选择（14层最优）可能需要针对不同架构调整

## 与相关工作的对比
- **vs ShiftDC**：ShiftDC需图转文参考+2倍推理时间；DTR无需参考+10%开销
- **vs CoCA**：CoCA在logit层校准，但性能劣化严重（Recognition 50.3→28.7）；DTR在KV cache层操作，性能保持
- **vs AdaShield**：AdaShield迭代精炼prompt，开销高（5.24s）；DTR直接优化α（4.01s）
- **vs KV剪枝方法（MADTP等）**：这些优化效率；DTR首次用KV优化做安全

## 启发与关联
- 拒绝方向的跨域迁移性暗示安全机制是模型层面的本质属性而非数据特定
- KV cache优化用于安全的范式可能扩展到音频/视频多模态

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ KV cache安全优化是全新范式，RSS formulation避免了图转文瓶颈
- 实验充分度: ⭐⭐⭐⭐⭐ 5个VLM、3个攻击基准、自适应攻击、跨域迁移、均匀vs动态分析、11项消融
- 写作质量: ⭐⭐⭐⭐⭐ 动机→观察→方法→理论→实验的逻辑链完整，攻防博弈分析深刻
- 价值: ⭐⭐⭐⭐⭐ 即插即用、开销极小、性能保持——实际部署价值极高
