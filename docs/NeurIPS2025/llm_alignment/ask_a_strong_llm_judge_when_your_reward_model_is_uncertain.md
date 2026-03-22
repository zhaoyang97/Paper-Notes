# Ask a Strong LLM Judge when Your Reward Model is Uncertain

**会议**: NeurIPS 2025  
**arXiv**: [2510.20369](https://arxiv.org/abs/2510.20369)  
**代码**: [GitHub](https://github.com/zhenghaoxu-gatech/uncertainty-router) (有)  
**领域**: 对齐RLHF  
**关键词**: reward model, LLM-as-judge, 不确定性量化, SNGP, 路由, RLHF

## 一句话总结
提出基于不确定性的路由框架，用SNGP对pairwise reward model做不确定性量化，将高认知不确定性的样本路由到强LLM judge（DeepSeek-R1），在仅调用9.2%~42.5% judge的成本下显著超越随机路由的准确率，且有效改善下游在线RLHF对齐效果。

## 研究背景与动机
1. **领域现状**：RLHF中reward model（RM）是核心组件，但标准RM（pointwise/pairwise）在OOD数据上泛化差，容易被reward hacking。强LLM judge（如DeepSeek-R1、GPT-4）通过CoT推理给出更可靠的偏好判断。
2. **现有痛点**：RM廉价但不可靠——在RM-Bench的hard子集上，SOTA的8B RM准确率仅46.6%（不如随机猜50%）。LLM judge准确但昂贵——长CoT推理导致延迟是标量RM的数十倍，在线RLHF中不可行。
3. **核心矛盾**：如何在有限的LLM judge调用预算下最大化偏好判断准确率？随机路由浪费预算在RM已经判对的样本上。
4. **本文要解决什么？** 设计一个自适应路由策略，精准识别RM不确定的样本（最可能判错），将其路由到强judge，其余用RM快速处理。
5. **切入角度**：从不确定性量化入手——pairwise RM的偏好分类问题天然适合UQ方法（vs pointwise RM在BT模型下不确定性是ill-defined的），采用SNGP（单次推理、无需集成）高效量化认知不确定性。
6. **核心idea一句话**：用SNGP给pairwise RM加不确定性感知，高认知不确定性的配对自动路由到LLM judge，低不确定性的用RM快速处理。

## 方法详解

### 整体框架
输入是prompt $x$ + 两个response $y_1, y_2$。SNGP-PM（pairwise preference model with spectral normalized GP）计算偏好分数 $p$ 和不确定性 $u$。若 $u > \bar{u}$（阈值），路由到DeepSeek-R1 judge获取更可靠判断；否则直接用PM结果。最终偏好差值用于构建RLOO/GRPO的advantage估计，驱动下游policy gradient更新。

### 关键设计

1. **SNGP-PM（不确定性感知的Pairwise RM）**:
   - 做什么：同时输出偏好分数和认知不确定性
   - 核心思路：在LLM backbone上加spectral normalization（保持距离感知）+ GP层（random feature近似），logit $g(h)$ 除以标准差 $u = \sqrt{1 + \lambda \cdot \phi(h)^\top \Sigma \phi(h)}$ 得到校准的偏好分数。$\Sigma$ 是GP后验协方差，在额外一个frozen epoch中计算
   - 设计动机：SNGP只需单模型单次推理，不像MC Dropout或集成需要多次，延迟与普通PM几乎相同。分离了aleatoric uncertainty（BT模型内在噪声，不可减）和epistemic uncertainty（数据覆盖不足，可由judge补充）

2. **不确定性路由策略**:
   - 做什么：根据不确定性阈值 $\bar{u}$ 决定用PM还是judge
   - 核心思路：$u > \bar{u}$ 时路由到DeepSeek-R1，judge返回三类标签（$y_1$更好、$y_2$更好、平局），分别映射为高置信logit或零logit
   - 设计动机：认知不确定性高 = OOD数据 = PM最可能判错，精准投放judge预算

3. **Pairwise Advantage估计（兼容RLOO/GRPO）**:
   - 做什么：将pairwise reward差值转化为policy gradient可用的advantage
   - 核心思路：RLOO的advantage $A_i = \frac{1}{K-1}\sum_{j \neq i}(r(x,y_i) - r(x,y_j))$ 只依赖reward差值，天然适配pairwise PM。无需pointwise reward的绝对值
   - 设计动机：避免了pointwise RM在BT模型下的不确定性ill-defined问题（加任意 $s(x)$ 偏移不改变偏好）

### 训练策略
- 基础模型：Llama-3.1-8B-Instruct
- 训练数据：HelpSteer2-Preference（7118对，含偏好强度标注）
- 数据增强：交换两个response顺序+翻转标签，消除位置偏差
- 训练2个epoch + 1个frozen epoch计算GP协方差矩阵
- Judge：DeepSeek-R1（RM-Bench hard上78.9%准确率）

## 实验关键数据

### 主实验：RewardBench上不同路由策略

| 路由策略 | Judge调用数 | Chat Hard | Reasoning | 总均(vs随机) |
|---------|-----------|-----------|-----------|-------------|
| 无路由 | 0 | 73.8 | 90.0 | 87.3 |
| 不确定性路由 | 274 (9.2%) | **76.8** | **93.7** | **89.2** (+1.7) |
| 随机路由 | 274 (9.2%) | 73.7 | 90.4 | 87.5 |
| 不确定性路由 | 1270 (42.5%) | **81.2** | **97.0** | **91.6** (+2.5) |
| 随机路由 | 1270 (42.5%) | 77.5 | 91.9 | 89.1 |
| DeepSeek-R1 100% | 全量 | 85.8 | 96.9 | 92.3 |

### 消融实验

| 配置 | 说明 |
|------|------|
| SNGP-PM vs 标准PM | 准确率差异<1%，不确定性组件不损害性能 |
| 阈值1.30 vs 1.45 | 更低阈值→更多judge调用→更高准确率，但边际递减 |
| 不确定性路由 vs 随机路由 | 在所有judge调用比例下，不确定性路由均显著优于随机路由（+0.8~+2.5pp） |

### 关键发现
- 不确定性与RM错误率强负相关（Spearman $p < 10^{-29}$），验证了"高不确定性=高出错概率"的假设
- OOD数据（RewardBench、RM-Bench）的不确定性系统性高于ID数据（HelpSteer2验证集）
- 仅9.2%的judge调用即可将RewardBench准确率从87.3%提升到89.2%，cost-effective
- 下游RLHF对齐中，不确定性路由同样优于随机路由，验证了端到端有效性

## 亮点与洞察
- **Pointwise vs Pairwise RM的UQ分析**很深刻——pointwise RM在BT模型下不确定性是ill-defined的（加bias不变），这是选择pairwise PM的理论基础，不只是经验选择
- **SNGP的选择**非常实用——单模型单次推理，不像集成需要N倍开销，延迟几乎不增加，适合在线RLHF
- **Judge的三类标签设计**（好/差/平局）优雅地处理了aleatoric uncertainty——平局返回 $\sigma^{-1}(1/2)=0$，不提供噪声信号
- 路由框架是通用的——可以替换为任何judge和任何UQ方法

## 局限性 / 可改进方向
- Judge（DeepSeek-R1）本身也有偏差（如长度偏好），路由到judge不一定是"ground truth"
- SNGP的GP层是random feature近似，近似质量受特征维度影响
- 仅在8B规模验证，更大模型的PM是否还需要路由？
- 阈值 $\bar{u}$ 需要手动设置，自适应阈值更实用
- 未探索更复杂的路由策略（如active learning式的批量选择）

## 相关工作与启发
- **vs LoRA Ensemble RM**：集成方法需要多模型多次推理，成本高；SNGP单次推理更适合在线场景
- **vs 纯LLM-as-Judge RLHF**：全量judge成本过高（Table 3延迟10x+），本文路由方案是practical compromise
- **vs OAIF (Online AI Feedback)**：OAIF用LLM judge替代RM但延迟大，本文框架可直接改善OAIF的效率

## 评分
- 新颖性: ⭐⭐⭐⭐ Pairwise RM + SNGP UQ + 路由到LLM judge的组合新颖且理论扎实
- 实验充分度: ⭐⭐⭐⭐ RM benchmark + 下游RLHF + 消融充分，但缺乏大模型实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论动机清晰，图表信息密度高，Remark解释到位
- 价值: ⭐⭐⭐⭐ 实用性强——直接插入现有RLHF pipeline，budget-aware的judge调用有工程价值
