# Continual SFT Matches Multimodal RLHF with Negative Supervision

**会议**: CVPR 2025
**arXiv**: [2411.14797](https://arxiv.org/abs/2411.14797)
**代码**: https://github.com/Kevinz-code/nSFT/
**领域**: 对齐RLHF
**关键词**: RLHF, DPO, SFT, 偏好对齐, 多模态大模型, 负监督信号, VLM

## 一句话总结
通过梯度分析发现多模态 RLHF 相比持续 SFT 的核心优势在于 rejected response 中的负监督信号，据此提出 nSFT 方法，用 LLM 从拒绝回复中提取错误信息并构造纠正性对话数据，仅用 SFT loss 就能匹配甚至超越 DPO/PPO 等 RLHF 方法，且只需 1 个模型，显存效率大幅提升。

## 研究背景与动机
1. **领域现状**：VLM 通常经过预训练 → SFT 后，再通过 RLHF（如 DPO、PPO）进行偏好对齐来减少幻觉、提升多模态理解能力。业界普遍认为 RLHF 在偏好对齐阶段严格优于持续 SFT。
2. **现有痛点**：多模态 RLHF 虽然有效，但存在明显的工程瓶颈——DPO 需要同时加载 2 个大模型（policy + reference），PPO 则需要 4 个（policy + reference + reward + critic），显存开销巨大，训练也不稳定。而直接做持续 SFT（用偏好数据的 chosen response 继续训练）效果又明显不如 RLHF。
3. **核心矛盾**：为什么同样的数据，SFT 就不行而 RLHF 就行？到底差在哪里？
4. **本文要解决的问题**：找到 RLHF 优于 SFT 的根本原因，然后把这个优势以 SFT 的方式实现，兼顾效果和效率。
5. **切入角度**：作者从梯度角度分析 DPO loss，发现其梯度本质上就是 chosen SFT 梯度和 rejected SFT 梯度的线性组合。持续 SFT 缺失的恰好就是 rejected response 上的 SFT loss（负监督信号）。
6. **核心idea**：用 LLM 从 rejected response 中提取错误信息并构造纠正性训练数据，使普通 SFT 也能获得 RLHF 的负监督信号优势。

## 方法详解

### 整体框架
nSFT 的流程分为两个阶段：**负监督数据构造** 和 **持续 SFT 训练**。

输入：偏好数据集（包含图片、问题、chosen response $\mathbf{y}_c$ 和 rejected response $\mathbf{y}_r$），与标准 DPO 的输入完全相同。

数据构造阶段：用 LLM（GPT-4）比对 $\mathbf{y}_r$ 和 $\mathbf{y}_c$，借助一个 **视觉错误码表（Vision Error Codebook）** 定位 rejected response 中的具体错误（如对象幻觉、属性错误），然后构造新的纠正性问答对。

训练阶段：用标准 SFT loss 同时训练原始 GT 对话和新构造的纠正性对话，只需要 1 个模型。

### 关键设计

1. **DPO 与 SFT 的梯度关系（理论分析）**：
   - 做什么：从梯度角度揭示 DPO 和 SFT 的本质联系
   - 核心思路：将 DPO loss 中的 logit 展开，忽略 reference model 约束后，DPO logit 变为 $p'_{\text{dpo}} = -(\mathcal{L}_{\text{sft}}(\mathbf{y}_c) - \mathcal{L}_{\text{sft}}(\mathbf{y}_r))$，即两个 SFT loss 之差。进一步对参数求导得到：$\frac{\partial \mathcal{L}_d}{\partial \theta'} = \frac{1}{p_{\text{dpo}}} \left[ \frac{\partial \mathcal{L}_{\text{sft}}(\mathbf{y}_c)}{\partial \theta'} - \frac{\partial \mathcal{L}_{\text{sft}}(\mathbf{y}_r)}{\partial \theta'} \right]$，即 DPO 梯度是 chosen 和 rejected 两个 SFT 梯度的线性组合
   - 设计动机：这直接说明了持续 SFT 不如 RLHF 的原因——缺少了 $\mathcal{L}_{\text{sft}}(\mathbf{y}_r)$ 这个负监督梯度项。进一步分析梯度更新速率比 $|t_2/t_1| < 1$ 说明 DPO 优化偏向于拒绝坏样本

2. **负监督信号解耦与数据构造（$G(\cdot)$ 函数）**：
   - 做什么：将 RLHF 中隐式的负监督信号显式地转化为 SFT 可用的训练数据
   - 核心思路：由于负监督深度纠缠在 DPO 的 pairwise logit 中，无法直接用 SFT 优化。因此引入 LLM 作为构造函数 $G(\cdot)$，输入 rejected response $\mathbf{y}_r$、GT response $\mathbf{y}_c$ 和视觉错误码表 $Q$，输出纠正性对话。最终 nSFT loss 为：$\mathcal{L}_{\text{nSFT}} = \mathcal{L}_{\text{sft}}(\mathbf{y}_c) + \mathcal{L}_{\text{sft}}(G(\mathbf{y}_r; \mathbf{y}_c, Q))$
   - 设计动机：打破了 RLHF 中 chosen-rejected 的 pairwise 耦合关系，使得整个对齐过程只需要 SFT loss，避免了加载 reference model

3. **视觉错误码表（Vision Error Codebook, VEC）**：
   - 做什么：为 LLM 提供全面的图像相关错误类型，指导其精确识别 rejected response 中的幻觉
   - 核心思路：码表覆盖 instance-level（对象存在性、属性、数量）和 image-level（场景、空间关系）两类错误。LLM 先参照 GT 和码表定位错误，再据此构造如"这是一本旅行书吗？是的"/"这是一本食谱书吗？不是"这样的纠正性对话
   - 设计动机：没有 VEC 时，LLM 构造的对话以非实体短语为主（如"might""provide"），引入 VEC 后对话更聚焦于具体物体（如"truck""cup""chair"），关键消融实验证实 VEC 有明显增益

### 损失函数 / 训练策略
- nSFT loss：$\mathcal{L}_{\text{nSFT}} = \mathcal{L}_{\text{sft}}(\mathbf{y}_c) + \mathcal{L}_{\text{sft}}(G(\mathbf{y}_r; \mathbf{y}_c, Q))$
- 消融实验发现加入 per-token KL 约束（类似 RLHF 中的 KL 正则）可以进一步提升效果
- 训练设置：DeepSpeed + ZeRO-3，batch size 128，学习率 2e-6，cosine scheduler
- 不同数据源的构造方式不同：OCRVQA（短回复）手动构造加倍 Q-A 对；TextCaps 和 LLaVA-150k（中/长回复）用 GPT-4 构造 5 轮对话

## 实验关键数据

### 主实验
在 3 种不同数据源（OCRVQA、TextCaps、LLaVA-150k）上用 LLaVA-1.5-7B 做持续对齐，对比 5 种方法：

| 方法 | 数据源 | SQA | GQA | VQAT | MMVet | MME | MMB | POPE | CHAIR↓ | MMHal |
|------|--------|-----|-----|------|-------|-----|-----|------|--------|-------|
| Baseline | — | 66.8 | 62.0 | 58.0 | 30.5 | 1510 | 64.3 | 85.9 | 32.0 | 2.80 |
| Cont. SFT | LLaVA-150k | 67.1 | 60.9 | 57.0 | 31.2 | 1480 | 64.0 | 86.3 | 29.1 | 2.91 |
| GT-DPO | LLaVA-150k | 68.1 | 61.6 | 57.6 | 33.9 | 1497 | 63.9 | 85.9 | 30.7 | 2.80 |
| SeVa | LLaVA-150k | 67.5 | 61.4 | 58.0 | 32.5 | 1490 | 64.7 | 85.6 | 28.2 | 2.94 |
| **nSFT** | LLaVA-150k | **68.4** | **62.3** | **58.4** | **34.2** | **1550** | **65.2** | **87.4** | **25.4** | **3.02** |

SOTA 对比（混合 15k 数据，与 SeVa、SIMA 等方法在其各自最优设置下对比）：

| 方法 | VQA | SQA | GQA | MMB | MME | POPE | SEEDI | SHR↓ | MMVet |
|------|-----|-----|-----|-----|-----|------|-------|------|-------|
| LLaVA-1.5 | 58.2 | 66.8 | 62.0 | 64.3 | 1510 | 85.9 | 65.7 | 36.7 | 30.5 |
| SeVa-7B | 56.2 | 67.5 | 60.7 | 65.6 | 1450 | 86.7 | 65.8 | 34.9 | 36.8 |
| SIMA-7B | 58.3 | 68.1 | 62.2 | 64.9 | 1507 | 86.5 | 65.9 | 34.5 | 32.6 |
| **nSFT** | **58.7** | **68.5** | **62.9** | **67.1** | **1531** | **86.8** | **66.2** | **34.2** | 34.0 |

### 消融实验
| 配置 | MMB | SQA | MME | POPE |
|------|-----|-----|-----|------|
| baseline | 64.3 | 66.8 | 1515 | 85.9 |
| + nSFT (完整) | 65.0 | 68.2 | 1533 | 86.5 |
| + nSFT w/o VEC | 64.4 | 67.6 | 1505 | 86.0 |
| + nSFT w/o chosen | 64.9 | 68.2 | 1523 | 86.4 |
| + nSFT + KL约束 | **65.2** | — | — | — |

### 关键发现
- **负监督信号是核心**：去掉 VEC 后所有指标明显下降，说明精细化的错误识别对负监督质量至关重要
- **chosen response 不是必需的**：去掉 $\mathbf{y}_c$ 后性能几乎不变，因为负监督构造过程已经隐式参考了 GT
- **nSFT 在幻觉指标上提升最大**：LLaVA-150k 数据下 hallucination 总分提升 +11.8（vs GTT-DPO 的+1.3），说明 VEC 引导的纠正性数据特别有效于减少对象级幻觉
- **DPO 更擅长压制最差情况，nSFT 更擅长提升最好情况**：in-domain 评估显示 DPO 的 ACC10w（最差10个样本准确率）更高，nSFT 的 ACC10b（最好10个样本准确率）更高
- **可迁移到更大模型**：在 LLaVA-1.5-13B 和 LLaVA-NeXT-13B 上也能匹配 DPO，并超越纯 SFT
- **训练效率优势明显**：显存约为 DPO 的一半（不需要 reference model），训练时间也更短

## 亮点与洞察
- **理论分析简洁有力**：通过几步推导将 DPO 梯度分解为两个 SFT 梯度的线性组合，直接定位了持续 SFT 失败的原因是缺少 rejected 梯度项。这个分析非常简洁且有说服力。
- **"把 RLHF 的隐式信号显式化"的范式很有启发**：不去优化 pairwise loss，而是先用外部工具（LLM + 错误码表）把信息提取出来再做 SFT，这种思路可以推广到其他 RLHF 场景（如代码生成、数学推理）。
- **视觉错误码表设计巧妙**：通过预定义的细粒度错误类型来引导 LLM 的注意力，避免生成笼统的纠正数据。Wordcloud 可视化很好地说明了有无 VEC 的区别。

## 局限性 / 可改进方向
- **依赖 GPT-4 进行数据构造**：nSFT 的数据构造依赖强大的 LLM（GPT-4），增加了 API 成本，且构造质量受限于 LLM 的能力。可以探索用开源 LLM 或自模型迭代构造。
- **仅验证了多模态场景**：作者自己指出，尚不清楚 nSFT 是否适用于 NLP 领域的 RLHF（如减毒、风格迁移），因为错误类型和构造方式可能完全不同。
- **Vision Error Codebook 需要人工设计**：当前码表覆盖的是视觉理解常见错误，迁移到其他任务需要重新设计。
- **未验证 online/iterative 版本**：当前 nSFT 是一次性构造数据的 offline 方法，是否可以迭代地用模型最新输出来构造负监督数据值得探索。

## 相关工作与启发
- **vs DPO (SeVa/SIMA/GT-DPO)**：DPO 系方法通过 pairwise 优化隐式利用负监督，需要 2 个模型。nSFT 将负监督显式提取后用 SFT 训练，只需 1 个模型，效果相当甚至更好。但 DPO 在最差情况下表现更稳定。
- **vs PPO**：PPO 需要 4 个模型（policy + ref + reward + critic），显存开销最大。nSFT 以远低的资源消耗达到了相似效果。
- **vs 并行工作 (NLP领域)**：NLP 中有类似发现（负反馈在 DPO 中更重要），但未将其与 SFT 关联；本文首次在多模态场景下系统地建立了这种联系并给出实用方案。

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论分析简洁有力，nSFT 范式很有启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 3种数据源×多种基线×多种VLM×9个benchmark，消融非常全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、图表丰富，理论推导易读
- 价值: ⭐⭐⭐⭐ 为偏好对齐提供了一个更高效的替代方案，实用性强
