# ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment

**会议**: AAAI 2026  
**arXiv**: [2512.06196](https://arxiv.org/abs/2512.06196)  
**代码**: [https://github.com/DeepFlow-research/manager_agent_gym](https://github.com/DeepFlow-research/manager_agent_gym)  
**领域**: LLM推理 / 对齐  
**关键词**: 可解释对齐, rubric学习, 多智能体协作, GSPO, 测试时对齐  

## 一句话总结
提出ARCANE框架，将对齐建模为多智能体协作问题——manager agent通过与stakeholder对话学习生成自然语言rubric（加权可验证准则集），作为worker agent的可解释代理奖励函数，通过SFT+GSPO两阶段训练实现测试时可配置的对齐，在GDPVal基准上GSPO版本的mean return从0.58提升至0.74（N=8）。

## 研究背景与动机

1. **领域现状**：RLHF是当前LLM对齐的主流范式，但它在训练时固化偏好，无法适应stakeholder偏好的变化。测试时奖励模型（如GenRM、GRAM）提供动态评估但不透明——用户不知道评估标准是什么、权重如何分配。

2. **现有痛点**：
   - **RLHF的刚性**：优化固定训练偏好，偏好漂移后需重新训练。在多agent部署中，分布式偏好更难静态捕捉
   - **测试时方法的不透明**：GenRM/GRAM输出标量或文本判断，但不揭示哪些准则驱动评估
   - **现有rubric方法的静态性**：Auto-Rubric、RaR等假设rubric预先给定而非学习得到，无法适应偏好演化

3. **核心矛盾**：对齐需要同时满足可解释性（stakeholder能审计）、可配置性（测试时可调）、和有效性（真正提升输出质量），传统方法最多满足其中两个。

4. **本文要解决什么？** 将rubric生成本身作为策略优化问题，让manager agent学会从stakeholder对话中提炼出可解释、可验证的加权准则集。

5. **切入角度**：借鉴效用理论，将stakeholder的潜在效用函数分解为加权可验证准则的线性组合，通过manager-stakeholder对话交互式地"重建"效用函数。

6. **核心idea一句话**：对齐 = manager学习生成自然语言rubric + worker按rubric执行 + stakeholder可在测试时调整权重。

## 方法详解

### 整体框架
三角色架构：Stakeholder持有真实效用函数 $U^*$ → Manager通过对话提炼rubric $R$ → Worker按rubric执行任务生成输出 $y$。Rubric $R = \{(c_j, w_j)\}_{j=1}^M$ 是加权可验证准则集，每个准则有对应的验证器 $\nu_j$。代理效用 $\hat{u}_\phi(y|x) = \sum_j w_j \nu_j(c_j, x, y)$ 作为 $U^*$ 的可解释近似。

### 关键设计

1. **Rubric表征与验证器**:
   - 做什么：将偏好分解为结构化、可验证的自然语言准则
   - 核心思路：每个准则 $c_j$ 是自然语言描述（如"包含近期实证研究引用"），权重 $w_j \in [0,1]$，$\sum w_j = 1$。验证器可以是规则型（确定性检查）或模型型（LLM/分类器评估语义属性）
   - 设计动机：线性加权使效用可分解、可审计；自然语言准则使非技术stakeholder也能理解

2. **Stakeholder-Manager协作对话**:
   - 做什么：manager向stakeholder提问以揭示潜在偏好，然后合成rubric
   - 核心思路：$R = \mathfrak{D}_\phi(x, q_{1:T}, a_{1:T})$，同时优化包含交互成本的目标：$\max_{\pi_M} \mathbb{E}[U^*(y|x) - \lambda_{\text{clarify}} C_{\text{clarify}} - \lambda_{\text{compute}} C_{\text{compute}}]$
   - 设计动机：建模为"部分可观测下的单次合作博弈"——stakeholder通过语言暴露有限、噪声信息，manager必须推断出忠实的结构化近似

3. **两阶段训练（SFT + GSPO）**:
   - **Stage I (SFT)**：用大推理模型生成合成对话+参考rubric，standard language modeling loss热启
   - **Stage II (GSPO)**：将manager视为随机策略，每个任务采样K个rubric，worker执行后获得stakeholder效用作为回报。使用序列级重要性比率 $s_k(\phi)$（而非token级），加上KL正则、澄清成本 $C_{\text{clarify}}$、计算成本 $C_{\text{compute}}$。引入优先经验回放机制关注低回报episode
   - 设计动机：SFT避免冷启动；GSPO直接优化端到端效用而非模仿参考rubric

4. **测试时Rubric引导**:
   - 做什么：用学到的rubric在测试时引导worker，无需梯度更新
   - 核心思路：支持Best-of-K采样（按rubric得分选最佳）、重要性重采样、树/束搜索等。stakeholder可直接编辑准则 $\{c_j\}$ 和权重 $\{w_j\}$
   - 设计动机：rubric的可解释性使得人类可以在推理时直接修改对齐方向

### 损失函数 / 训练策略
- SFT Loss：标准next-token prediction，mask系统提示和任务输入
- GSPO Loss：PPO风格的clip目标 + KL散度正则 + 澄清成本 + 计算成本（公式12-15）
- 优先经验回放：每个epoch回放回报最低的N-th百分位episode

## 实验关键数据

### 主实验

GDPVal基准，219个任务（175训练+44评估），多步推理+工具使用：

| 方法 | Mean Return (N=1) | Mean Return (N=8) |
|------|-------------------|-------------------|
| No Rubric | 0.58±0.01 | 0.58±0.02 |
| SFT Model | 0.59±0.09 | 0.68±0.03 |
| GSPO Model | 0.62±0.12 | **0.74±0.03** |
| Oracle Rubric | 0.70±0.12 | 0.81±0.03 |

### 消融实验（Faithfulness - NDCG@8）

| 方法 | Mean NDCG@8 | 说明 |
|------|-------------|------|
| No-Conv (Base) | 0.7998 | 无stakeholder对话的基础rubric |
| SFT Rubric | 0.8103 | SFT训练的manager |
| GSPO Rubric | **0.8722** | GSPO训练后排序一致性显著提升 |

### 关键发现
- **GSPO > SFT**：统计显著（Wilcoxon p=0.0182），GSPO rubric在N=8时mean return 0.74 vs SFT 0.68
- **scaling曲线平行**：SFT、GSPO、Oracle三者的best-of-N scaling斜率几乎相同（每倍N约+0.03），说明学到的rubric近似了oracle的评分函数
- **领域差异**：主观/语言密集任务（内容/传播+11.5%、法律+12.5%）提升最大，操作性任务略微下降(-8.1%)
- **可解释性保持**：GSPO rubric约12个准则/rubric，17-18 tokens/准则，与Oracle结构高度一致

## 亮点与洞察
- **将rubric生成建模为策略优化**：不是假设rubric给定，而是让agent学习如何生成rubric。这是对"rubric-as-reward"范式的重要推进
- **GSPO的序列级重要性比率**：比GRPO的token级比率更适合结构化输出（rubric），避免长序列的方差问题
- **成本感知的对齐**：将stakeholder交互成本和计算成本纳入优化目标，避免过度澄清或过复杂的rubric

## 局限性 / 可改进方向
- **仅单worker验证**：框架设计支持多worker但实验只有单worker，未验证多worker协调动态
- **GDPVal是离散episode任务**：缺乏长时间跨度的持续部署评估
- **缺乏因果正则**：manager可能学到与效用相关但非因果的准则（虚假相关）
- **No Rubric baseline使用RLHF模型**：说明训练时对齐不够，但未与GenRM等测试时方法直接比较

## 相关工作与启发
- **vs RLHF/DPO**: 训练时固化偏好，无法测试时配置；ARCANE的rubric可在推理时人工编辑或自动调整
- **vs GenRM/GRAM**: 提供动态评估但不透明；ARCANE的rubric是结构化自然语言，可审计可修改
- **vs Auto-Rubric/RaR**: 假设rubric静态给定，不从stakeholder交互中学习；ARCANE学习生成rubric本身

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将rubric生成建模为multi-agent RL问题是原创贡献，理论框架（效用理论+双层优化）扎实
- 实验充分度: ⭐⭐⭐⭐ RQ1-3结构清晰，有统计显著性检验，但仅44个评估任务，且缺少与GenRM等的直接比较
- 写作质量: ⭐⭐⭐⭐⭐ 数学形式化严谨，从问题定义到方法再到实验的逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 可解释+可配置+有效对齐三位一体，对实际LLM部署有重要指导意义
