# ConfTuner: Training Large Language Models to Express Their Confidence Verbally

**会议**: NeurIPS 2025
**arXiv**: [2508.18847](https://arxiv.org/abs/2508.18847)
**代码**: [GitHub](https://github.com/liushiliushi/ConfTuner)
**领域**: LLM / 不确定性校准
**关键词**: 置信度校准, 语言化置信度, Brier分数, proper scoring rule, LLM过度自信

## 一句话总结
ConfTuner 提出 tokenized Brier score 损失函数（理论证明为 proper scoring rule），仅需 2000 个样本 + 4 分钟 LoRA 微调即可让 LLM 输出校准的语言化置信度（如"我80%确定"），ECE 最大降低 60.9%，支持自我纠错和模型级联等下游应用。

## 研究背景与动机
1. **领域现状**：LLM 在高风险领域（医疗、法律、科学）的可靠部署需要知道"模型有多确定"。现有方法包括 logit-based 校准（但不适用于 API 模型）和 verbalized confidence（让模型说出置信度）。
2. **现有痛点**：LLM 严重过度自信——经常说"我100%确定"即使答案是错的。SaySelf 等方法需要大规模数据（900万样本），LACIE 需要10万样本且训练时间长。关键理论问题：什么样的损失函数能保证置信度校准？
3. **核心矛盾**：语言化置信度是离散的 token（"80%"是一个 token 而非概率值），经典的 Brier score 定义在连续概率上。如何将 proper scoring rule 理论扩展到 token 空间？
4. **本文要解决什么**：(1) 设计理论上有保证的置信度校准损失；(2) 以极低成本（2000样本/4分钟）实现校准。
5. **切入角度**：将 Brier score 从对概率值的评分推广到对"置信度 token 上的概率分布"的评分——如果模型在"80%"上放了更多概率质量，而真实正确率确实接近80%，则损失低。
6. **核心idea一句话**：Tokenized Brier score = 对置信度 token 的 proper scoring rule，迫使模型在最接近真实正确率的 token 上放最多概率。

## 方法详解

### 整体框架
两阶段：(1) 从 SFT 后的模型中提取置信度 token 的 logit 分布 $\mathbf{q}$；(2) 用 tokenized Brier score $\ell(\mathbf{q}, y) = \sum_i q_i(y - i/N)^2$ 做 LoRA 微调（$y$=正确性指标）。

### 关键设计

1. **Tokenized Brier Score（核心贡献）**:
   - 做什么：对置信度 token 集 $\mathcal{T}_N = \{0\%, 10\%, ..., 100\%\}$ 上的 softmax 分布 $\mathbf{q}$ 定义损失
   - 公式：$\ell(\mathbf{q}, y) = \sum_{i=0}^{N} q_i (y - i/N)^2$
   - 理论保证（Theorem 1）：这是 proper scoring rule——模型最小化损失的最优策略是将概率集中在最接近真实条件正确率 $\eta(x)$ 的 token 上
   - 设计动机：经典 Brier score 对单个概率值评分，但 LLM 输出的是 token 上的分布——需要推广

2. **极简 LoRA 微调**:
   - 做什么：rank-8 LoRA 仅在 query/value 投影上微调
   - 仅需 2000 个样本，4 分钟训练（单 GPU）
   - 对比：SaySelf 需 900万样本/120分钟，LACIE 需 10万/26分钟
   - 设计动机：最小改动保留模型原有能力

3. **下游应用（自我纠错 + 级联）**:
   - 自我纠错：低置信度时拒绝回答或重新思考 → 3-10% 准确率提升
   - 模型级联：低置信度时路由到 GPT-4o → HotpotQA 上 +9.3%

### 损失函数 / 训练策略
Tokenized Brier score + LoRA。LLaMA 用 $\mathcal{T}_{100}$（0-100%），Qwen/Ministral 用 $\mathcal{T}_9$（0-9 离散等级）。

## 实验关键数据

### 主实验（5 个推理基准平均 ECE↓ / AUROC↑）

| 模型 | ECE (Base) | ECE (ConfTuner) | 改善 | AUROC (Base) | AUROC (ConfTuner) |
|------|-----------|----------------|------|-------------|------------------|
| LLaMA | 0.2768 | **0.1082** | -60.9% | 0.5923 | **0.6740** |
| Qwen | 0.3781 | **0.2872** | -23.9% | 0.6155 | **0.6861** |
| Ministral | 0.4393 | **0.1884** | -57.1% | 0.5216 | **0.6810** |

### 消融实验
| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 2000 vs 5000 vs 10000 样本 | 2000 已足够 | 极低数据需求 |
| 4分钟 vs SaySelf 120分钟 | 30倍加速 | 计算效率极高 |
| 语言化变体（高/中/低） | 泛化有效 | 不限于数字化置信度 |
| 隐式置信度（CoT中表达） | 也有效 | 非显式"X%"也可校准 |
| 黑盒模型（GPT-4o） | 也改善 | 通过 prompt 框架 |
| 自我纠错 | +3-10% 准确率 | 实用下游应用 |
| 模型级联 | +9.3% (HotpotQA) | 置信度用于路由 |

### 关键发现
- ECE -60.9% 是最突出的数字——说明 LLM 的置信度从接近随机变为有意义的信号
- 2000 样本 + 4 分钟定义了"成本下限"——很难想象更低成本的校准方案
- 级联应用展示了置信度的实用价值——知道何时该请求更强模型

## 亮点与洞察
- **理论的力量**：proper scoring rule 保证模型不能通过"作弊"（如所有回答都说50%）来降低损失——唯一最优策略就是真诚校准。这比临时设计的损失函数有根本优势。
- **"2000样本4分钟"**：极低的成本意味着任何使用 LLM 的团队都可以做置信度校准——民主化了不确定性估计。
- **从校准到决策**：校准本身不是目的，自我纠错和级联才是。论文展示了完整的"校准→利用"链路。

## 局限性 / 可改进方向
- 限于固定的置信度 token 集，灵活的自然语言表达（如"我相当确定但不完全肯定"）未处理
- 理论保证与实践仍有差距——数据质量和优化动态会影响实际校准
- 仅在推理任务上验证，创意写作/翻译等任务的校准未探索
- proper scoring rule 假设独立样本——多轮对话中的依赖关系未考虑

## 相关工作与启发
- **vs SaySelf (Xu et al., 2024)**：SaySelf 用自我训练迭代改善置信度但需900万样本；ConfTuner 理论驱动+2000样本
- **vs LACIE (Band et al., 2024)**：LACIE 训练辅助模型预测置信度；ConfTuner 直接校准模型本身的 token 分布
- **vs Temperature Scaling**：TS 校准 logit 概率但不改变语言化输出；ConfTuner 直接校准语言表达

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Tokenized Brier score 的理论推广优雅且有实际意义
- 实验充分度: ⭐⭐⭐⭐ 3模型×5基准×下游应用×效率对比
- 写作质量: ⭐⭐⭐⭐⭐ 理论和实验叙事结合紧密
- 价值: ⭐⭐⭐⭐⭐ 2000样本+4分钟的校准方案对LLM部署有直接实用价值\n
- **vs CalibrateBeforeUse**: CBU在推理时做温度缩放；ConfTuner直接校准输出token的语义，更适合API场景
- **vs 集成方法 (Self-Consistency)**: 多次采样取一致性做置信度估计成本高（需N次推理），ConfTuner将知识内化为单次输出
- **启发其他模态**: Tokenized proper scoring rule的框架可推广到VLM的视觉置信度、语音识别的可靠度等
- **对RLHF的启示**: ConfTuner的校准可改善RLHF中的奖励模型——校准后的置信度是更好的训练信号
- **对Agent系统的价值**: 自主Agent需要知道何时该求助人类——校准的置信度提供了可靠的决策依据
- **与Bayesian方法的关系**: ConfTuner的校准可视为将LLM输出映射到近似贝叶斯后验概率
- **多语言扩展**: 置信度表达在不同语言中的自然表述不同，跨语言校准是开放问题
- **长文本场景**: 当回答包含多个claim时，每个claim的独立置信度比全局置信度更有价值
- **vs CalibrateBeforeUse**: CBU在推理时做温度缩放；ConfTuner直接校准输出token的语义，更适合API场景
- **vs 集成方法 (Self-Consistency)**: 多次采样取一致性做置信度估计成本高（需N次推理），ConfTuner将知识内化为单次输出
- **启发其他模态**: Tokenized proper scoring rule的框架可推广到VLM的视觉置信度、语音识别的可靠度等
- **对RLHF的启示**: ConfTuner的校准可改善RLHF中的奖励模型——校准后的置信度是更好的训练信号
- **对Agent系统的价值**: 自主Agent需要知道何时该求助人类——校准的置信度提供了可靠的决策依据
- **与Bayesian方法的关系**: ConfTuner的校准可视为将LLM输出映射到近似贝叶斯后验概率
- **多语言扩展**: 置信度表达在不同语言中的自然表述不同，跨语言校准是开放问题
- **长文本场景**: 当回答包含多个claim时，每个claim的独立置信度比全局置信度更有价值
- **vs CalibrateBeforeUse**: CBU在推理时做温度缩放；ConfTuner直接校准输出token的语义，更适合API场景
- **vs 集成方法 (Self-Consistency)**: 多次采样取一致性做置信度估计成本高（需N次推理），ConfTuner将知识内化为单次输出
- **启发其他模态**: Tokenized proper scoring rule的框架可推广到VLM的视觉置信度、语音识别的可靠度等
- **对RLHF的启示**: ConfTuner的校准可改善RLHF中的奖励模型——校准后的置信度是更好的训练信号
- **对Agent系统的价值**: 自主Agent需要知道何时该求助人类——校准的置信度提供了可靠的决策依据
- **与Bayesian方法的关系**: ConfTuner的校准可视为将LLM输出映射到近似贝叶斯后验概率
- **多语言扩展**: 置信度表达在不同语言中的自然表述不同，跨语言校准是开放问题
- **长文本场景**: 当回答包含多个claim时，每个claim的独立置信度比全局置信度更有价值
