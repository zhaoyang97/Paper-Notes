# Beyond Linearity in Attention Projections: The Case for Nonlinear Queries

**会议**: ICLR2026 Workshop (GRaM)  
**arXiv**: [2603.13381](https://arxiv.org/abs/2603.13381)  
**代码**: [GitHub](https://github.com/MarkoKarbevski/beyond_query_linearity)  
**领域**: Transformer架构 / Attention机制  
**关键词**: nonlinear query, attention projection, identity prior, bottleneck MLP, transformer architecture  

## 一句话总结
基于 WQ 代数冗余性的理论发现，将线性 Query 投影替换为非线性残差形式 Q(X)=(X+f_θ(X))/2，在相同参数量下超越增加 12.5% 参数的基线。

## 背景与动机
1. Transformer 的注意力计算依赖 WQ、WK、WV 三个线性投影矩阵，但近期代数分析表明存在重参数化不变性
2. 对任意可逆矩阵 Θ，可将 (X, WQ, WK, WV, WO) 映射为 (XΘ, Θ⁻¹WQ, ...)，MHA 输出不变
3. 取 Θ=WQ 可令 WQ→I，说明 WQ 的线性参数是代数冗余的（可被相邻层吸收）
4. 实验验证：WQ=I 的模型与标准基线性能相当，且在 3× 更低 weight decay 下仍稳定
5. 既然线性参数冗余，若要在 query 路径分配参数，就必须引入**非线性**
6. 从单个 token x 生成 q/k/v/残差四个向量全为线性函数，构成信息瓶颈

## 方法详解
- **核心公式**: Q(X) = (X + f_θ(X)) / 2
- **f_θ 结构**: 瓶颈 MLP — f_θ(X) = LN(GELU(RMSNorm(X)·W1)·W2)，W1∈R^{d×r}，W2∈R^{r×d}，r=d/2
- **参数量**: 2dr = d²，与原始 WQ 参数量同阶；归一化层仅增加 O(d) 参数（<0.1%）
- **设计要点**: identity term X 锚定到已知良好先验，保证梯度直通；1/2 缩放因子防止幅度膨胀
- K 和 V 保持标准线性投影不变
- 兼容 GQA/MQA（GQA 共享 WK/WV，只有 WQ 可安全替换）、RoPE、MoE

## 实验关键数据
| 模型 | 非嵌入参数 | Val Loss (59k) | 相对提升 |
|------|-----------|---------------|---------|
| Baseline | 85M | 2.956 | 0 |
| MLP4.75 (宽MLP) | 96M (+12.5%) | 2.927 | 0.98% |
| Res. GELU (本文) | 85M | 2.915 | **1.40%** |

- 训练设置：OpenWebText，单张 RTX 5090，60k steps (~29B tokens)，远超 Chinchilla 最优
- 非线性变体可承受 5× 更高学习率（3e-3 vs 6e-4）和更低 weight decay（0.03 vs 0.1）
- 基线在 weight decay=0.05 时 20k steps 前发散，非线性版本稳定

## 亮点
- 理论动机清晰：从代数冗余性出发，逻辑链完整（WQ 冗余→线性无用→引入非线性）
- 参数中性改进：不增加参数即超越 +12.5% 参数的模型
- 训练稳定性提升：可用更激进的超参数
- 代码和 checkpoint 开源

## 局限性 / 可改进方向
- 仅在 ~124M 单一规模验证，未测试大模型
- 未进行多种子实验（通过固定数据顺序缓解）
- 未测量推理速度（非线性引入串行依赖）
- 超参数搜索不充分，1.40% 可能是下界
- 未评估下游任务表现

## 与相关工作的对比
- **Kernel Attention**: 在 Q=XWQ 之后加非线性特征映射，本文直接替换 WQ
- **MLP-Attention**: 直接用 MLP 生成注意力权重，但增加 ~10% 参数且缺乏理论动机
- **Nonlinear LoRA**: 面向微调场景，本文面向预训练架构设计
- **Gated Attention (Qiu et al.)**: 输出端门控，本文在 query 端引入非线性

## 评分
- 新颖性: ⭐⭐⭐⭐ (理论驱动的架构修改，方向新颖)
- 实验充分度: ⭐⭐⭐ (单一规模，但控制变量严谨)
- 写作质量: ⭐⭐⭐⭐ (Workshop 文章，逻辑清晰)
- 价值: ⭐⭐⭐⭐ (如能在大规模验证将有重要意义)
