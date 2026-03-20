# Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence

**会议**: ICLR 2026  
**arXiv**: [2603.00498](https://arxiv.org/abs/2603.00498)  
**代码**: 待公开  
**领域**: LLM安全 / 微调攻击防御  
**关键词**: 有害微调攻击, 安全对齐, 损失平坦度, 样本加权, FTaaS安全  

## 一句话总结
提出Antibody防御框架：在对齐阶段通过平坦度正则化使模型处于有害损失的平坦区域（梯度小→难被攻击），在微调阶段用基于模型安全知识的样本加权方案（对比目标完成 vs 拒绝的似然比）抑制有害样本的学习，平均Harmful Score从15.29%降至7.04%。

## 研究背景与动机

1. **领域现状**：FTaaS（如OpenAI/Mistral的微调服务）允许用户上传数据微调LLM，但用户提交的数据可能包含有害样本（有意或无意），导致安全对齐被破坏。
2. **现有痛点**：(a) 对齐阶段防御（如Vaccine/Booster）是静态的，无法适应不同的攻击配置（高步数、大学习率）；(b) 微调阶段防御（如Lisa/SafeInstr）要么保护不足要么损害任务性能；(c) 大多数方法在安全性和任务性能之间存在严重tradeoff。
3. **核心矛盾**：标准SFT不区分良性和有害样本——所有梯度都被聚合更新，即使少量有害样本的梯度也能毒化模型。
4. **本文要解决什么**：设计在对齐和微调两个阶段协同工作的防御，既能彻底抑制有害梯度的影响，又不损害良性任务学习。
5. **切入角度**：从梯度影响的角度出发——如果有害样本的梯度在对齐后本来就很小（平坦区域），且在微调时被进一步降权，就能有效消除其影响。
6. **核心idea一句话**：对齐阶段让有害loss平坦（梯度小）+微调阶段用似然比加权（有害样本权重低）→有害梯度被双重抑制。

## 方法详解

### 整体框架
Antibody分两阶段：(1) **对齐阶段**——优化 $\mathcal{L}_{\text{align}}(\theta) + \lambda_t \mathcal{L}_{\text{sharp}}(\theta) + \lambda_{\text{refusal}} \mathcal{L}_{\text{refusal}}(\theta_{\text{pert}})$，使模型在对齐的同时处于有害损失的平坦区域；(2) **微调阶段**——用样本加权更新 $\theta_{t+1} \leftarrow \theta_t - \eta \sum_i w_{\theta_t}(x_i,y_i) \nabla \ell_{\theta_t}(x_i,y_i)$，有害样本权重自动降低。

### 关键设计

1. **平坦度正则化对齐（Robust Alignment via Flatness）**
   - 做什么：使模型处于有害损失 $\mathcal{L}_{\text{harm}}$ 的平坦区域
   - 核心思路：定义sharpness为 $\mathcal{L}_{\text{sharp}}(\theta) = \mathcal{L}_{\text{harm}}(\theta) - \min_{\phi \in \mathcal{B}_\rho(\theta)} \mathcal{L}_{\text{harm}}(\phi)$，即有害损失在ρ邻域内的下降幅度。最小化sharpness→模型处于平坦区域→后续微调中有害样本的梯度自然很小。通过Theorem 4.1（KKT条件）求解双目标优化得到更新方向 $\delta_t^* = \nabla \mathcal{L}_{\text{align}} + \lambda_t \nabla \mathcal{L}_{\text{sharp}}$，$\lambda_t$ 自适应调整
   - 设计动机：平坦区域意味着在θ附近扰动（即微调）不会显著降低有害loss——安全对齐更鲁棒

2. **基于似然比的样本加权微调（Safety Fine-tuning with Weighted Loss）**
   - 做什么：在微调时对每个mini-batch中的样本动态赋权，抑制有害样本
   - 核心思路：对每个样本计算 $r_\theta(x_i,y_i) = \log \frac{\pi_\theta(y_i|x_i)}{\pi_\theta(y_r|x_i)}$（目标完成 vs 拒绝的似然比），然后softmax归一化为权重。安全对齐的模型面对有害prompt时更倾向拒绝→似然比低→权重小→有害梯度被抑制
   - 设计动机：利用对齐阶段已嵌入的安全知识作为隐式有害检测器——无需显式标注哪些样本有害

3. **扰动模型拒绝训练**
   - 做什么：确保即使模型参数发生漂移（被有害样本微调），仍能维持低似然比权重
   - 核心思路：在对齐阶段，模拟微调漂移 $\theta_{\text{pert}} = \theta - \rho \frac{\nabla \mathcal{L}_{\text{harm}}}{\|\nabla \mathcal{L}_{\text{harm}}\|}$，然后训练扰动模型仍能对有害prompt产生高拒绝概率 $\mathcal{L}_{\text{refusal}}(\theta_{\text{pert}})$
   - 设计动机：防止微调过程中有害样本逐渐提高自身权重→权重机制失效

### 理论分析
- Proposition 4.2和4.3提供了mini-batch更新的损失变化分解——通过eNTK分析证明：当batch梯度仅由良性样本贡献时，有害测试样本的loss不变（安全保持），良性测试样本的loss下降（任务学习）

## 实验关键数据

### 主实验（Llama-2-7B, GSM8K+20%有害样本）

| 方法 | HS↓ | FA↑ | 说明 |
|------|-----|-----|------|
| SFT | 23.94 | 10.90 | 无防御 |
| Vaccine | 23.60 | 11.70 | 对齐阶段 |
| Lisa | 5.86 | 9.23 | 微调阶段，任务性能差 |
| Booster | 9.06 | 16.27 | 对齐阶段 |
| **Antibody** | **1.24** | **15.07** | 两阶段协同 |

### 跨数据集平均

| 方法 | 平均HS↓ | 平均FA↑ |
|------|---------|--------|
| Lisa | 15.29 | 60.97 |
| Booster | 19.04 | 65.20 |
| **Antibody** | **7.04** | **竞争性** |

Antibody的HS比次优方法Lisa低8+个百分点。

### 消融实验
- 去掉平坦度正则 → HS升高（有害梯度在微调时不够小）
- 去掉样本加权 → HS升高（有害样本贡献未被抑制）
- 去掉扰动拒绝训练 → 长时间微调后权重机制退化

### 关键发现
- 平坦度正则和样本加权的组合是关键——两者单独使用效果均不如组合
- 似然比权重（Figure 2）在训练过程中自然地将有害和良性样本分离——无需显式标注
- Antibody在大数据量（Figure 1）时尤其有效——其他方法随数据增多安全性恶化，Antibody保持低HS

## 亮点与洞察
- **双重梯度抑制**的设计逻辑极其清晰：第一层（flat region）使梯度天然小 → 第二层（加权）进一步降权 → 有害影响被彻底抑制
- **利用模型自身的安全知识做隐式有害检测**（似然比）非常巧妙——不需要额外的分类器或标注，也不需要知道哪些样本有害
- eNTK的理论分析（Proposition 4.2-4.3）提供了mini-batch加权更新如何选择性影响不同样本的严谨解释
- 与Booster的联系（$\lambda_t$常数时退化为Booster）说明了方法的泛化性

## 局限性 / 可改进方向
- 需要对齐阶段访问有害数据集 $\mathcal{D}_{\text{harm}}$——如果有害类型变化可能需要重新对齐
- LoRA微调场景下验证，全参微调的效果未知
- 拒绝模板 $y_r$ 的选择可能影响似然比计算——不同拒绝风格可能导致不同效果
- 仅测试了20%有害比例，更高比例（50%+）下的鲁棒性待验证
- 计算开销比标准SFT高（需要额外计算似然比和内环扰动步骤）

## 相关工作与启发
- **vs Vaccine**: Vaccine用嵌入扰动增强鲁棒性，Antibody用损失平坦度——后者有更清晰的理论支撑
- **vs Booster**: Booster是Antibody的特例（$\lambda_t$固定）；Antibody的自适应$\lambda_t$和额外的加权机制提供了显著的额外提升
- **vs Lisa**: Lisa交替用安全数据和任务数据，但无法识别批次内的有害样本；Antibody的权重方案做到了sample-wise区分

## 评分
- 新颖性: ⭐⭐⭐⭐ 平坦度正则+似然比加权的组合很有工程智慧，但单项技术较标准
- 实验充分度: ⭐⭐⭐⭐⭐ 4个下游数据集×3个模型+消融+理论分析，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，Figure 2的权重分布可视化极其直观
- 价值: ⭐⭐⭐⭐⭐ 对FTaaS安全有直接的实践意义，HS从15%→7%是显著进步
