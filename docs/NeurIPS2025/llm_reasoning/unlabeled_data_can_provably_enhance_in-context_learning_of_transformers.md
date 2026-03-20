# Unlabeled Data Can Provably Enhance In-Context Learning of Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2601.10058](https://arxiv.org/abs/2601.10058)  
**代码**: 无  
**领域**: LLM推理 / 理论  
**关键词**: in-context learning, unlabeled data, semi-supervised learning, EM algorithm, chain-of-thought, transformer theory

## 一句话总结
提出增强型ICL框架，在prompt中同时包含少量标记样本和大量无标记样本，理论证明多层Transformer通过CoT可模拟EM算法从无标记数据中提取信息，将分类excess risk从 $\mathcal{O}(1/\sqrt{N})$ 改进到 $\mathcal{O}(1/\sqrt{N + \text{poly}(M)})$。

## 研究背景与动机

1. **领域现状**：ICL让Transformer无需参数更新就能从prompt中的示例学习新任务，但严重依赖标记示例的数量和质量。高质量标记数据获取成本高昂（如GPT-3.5/4的RLHF数据涉及数千小时专家标注）。
2. **现有痛点**：(a) prompt长度限制了标记示例数量；(b) 现有方法用LLM自身生成伪标签（pseudo-labels），但会继承模型偏差；(c) 海量无标记数据未被ICL利用。
3. **核心矛盾**：存在大量无标记数据但ICL不知道怎么用——传统ICL只处理 $(x,y)$ 对，无标记的 $x$ 被忽略。
4. **本文要解决什么？** 从理论上证明无标记数据可以提升ICL性能，并给出具体的Transformer构造和训练收敛保证。
5. **切入角度**：将增强ICL（labeled + unlabeled in prompt）与经典半监督学习中的EM算法联系——Transformer通过CoT的多步推理可以迭代精化类均值估计。
6. **核心idea一句话**：4层Transformer + CoT推理 = 隐式EM算法，在prompt中同时从标记和无标记数据学习。

## 方法详解

### 整体框架
输入为混合prompt $\mathcal{I} = \mathcal{D}_{label} \cup \mathcal{D}_{unlabel}$，其中 $\mathcal{D}_{label} = \{(\mathbf{x}_j, y_j)\}_{j=1}^N$，$\mathcal{D}_{unlabel} = \{\mathbf{x}_j\}_{j=N+1}^{N+M}$。编码为矩阵 $\mathbf{H}$，包含标记块、无标记块和推理块。Transformer通过CoT在推理块中迭代精化类均值估计 $\hat{\mu}_i^{(t)}$，最终用最近邻分类无标记样本。

### 关键设计

1. **增强ICL的CoT编码**:
   - 做什么：将labeled+unlabeled数据和推理中间状态编码到统一的token序列中
   - 核心思路：推理块 $\mathbf{Q}^{(t)}$ 存储第 $t$ 步的类均值估计 $\hat{\mu}_i^{(t)}$。每步CoT将新产生的 $C$ 个token append到序列末尾
   - 设计动机：利用CoT的自回归特性实现EM的迭代——每步CoT = 一步EM迭代

2. **4层Transformer构造（Theorem 4.1）**:
   - 做什么：显式构造一个4层Transformer实现EM更新
   - 核心思路：更新公式为 $\hat{\mu}_i^{(t+1)} = \hat{\mu}_i^{(t)} - \frac{\eta^{(t)}}{M} \sum_{j} p_{ij}^{(t)}(\hat{\mu}_i^{(t)} - \mathbf{x}_j) + \mathbf{1}_{\{t=0\}} \frac{C}{N} \sum_j (\mathbf{e}_i^\top \mathbf{y}_j) \mathbf{x}_j$。第一层用softmax attention计算E-step（类成员概率 $p_{ij}^{(t)}$），后续层实现M-step（加权均值更新）。初始化用标记数据的类均值
   - 设计动机：精确模拟高斯混合模型的EM算法——E-step计算后验概率，M-step更新参数

3. **收敛性分析（Theorem 4.2）**:
   - 做什么：证明类均值估计在CoT步数增加时收敛到真值
   - 核心思路：在信噪比 $\text{SNR} \geq \Omega(\sqrt{C \log(CM)})$、足够多标记数据 $N$ 和无标记数据 $M$ 的条件下，excess risk为 $\mathcal{O}(1/\sqrt{N + \text{poly}(M)})$，严格优于仅用标记数据的下界 $\mathcal{O}(1/\sqrt{N})$
   - 设计动机：从理论上严格证明无标记数据的价值——不只是经验观察

4. **训练收敛（Theorem 5.1）**:
   - 做什么：证明用teacher forcing训练的Transformer参数线性收敛到目标解
   - 核心思路：将CoT训练损失的梯度分解为两个可分析项，利用involved quantities的各向同性简化分析
   - 设计动机：证明理论构造不只是存在性结果，还能通过标准训练找到

### 损失函数 / 训练策略
Teacher forcing训练：在每个CoT步骤 $t$，监督信号是EM算法的"真实"下一步输出。梯度下降在population loss上以线性速率收敛。

## 实验关键数据

### 主实验
在多类线性分类设置上（$d=20$, $C=$3,5类）：

| 方法 | 均值估计误差 | 分类准确率 |
|------|------------|-----------|
| 传统ICL (仅N个标记) | 基线 | 基线 |
| 增强ICL (N标记 + M无标记) | **显著降低** | **显著提升** |
| 仅标记数据的Bayes最优 | 低于增强ICL | 低于增强ICL |

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| M增加 (更多无标记) | 性能持续提升 | 验证 $\mathcal{O}(1/\sqrt{N+\text{poly}(M)})$ |
| T增加 (更多CoT步) | 性能提升后饱和 | EM收敛特性 |
| N固定, 仅增加M | 性能从N-only水平提升 | 无标记数据独立贡献 |

### 关键发现
- 增强ICL显著超越传统ICL，且超越了仅用标记数据的Bayes最优分类器——证明了无标记数据的真实价值
- 性能增益随M增加而持续增长，与理论预测一致
- CoT步数T=5-10步通常足够收敛

## 亮点与洞察
- **ICL与半监督学习的理论桥梁**：首次从理论上建立了"Transformer的CoT推理 = EM算法"的等价关系，为理解ICL的推理机制提供了新视角
- **精确的excess risk改进量化**：$\mathcal{O}(1/\sqrt{N}) \to \mathcal{O}(1/\sqrt{N + \text{poly}(M)})$ 是clean的理论结果，清晰展示了无标记数据的边际贡献
- **存在性+可学习性的完整理论**：不仅构造了理想Transformer（存在性），还证明了训练收敛（可学习性），理论链条完整

## 局限性 / 可改进方向
- 理论仅限于多类线性分类+高斯混合模型——是否能推广到非线性分类、回归任务？
- 假设各向同性协方差 $\Sigma$——非各向同性情况下理论分析困难
- 构造的4层Transformer是"理想化"的——实际预训练的LLM是否隐式学到了类似的EM策略？
- 实验仅在合成数据上验证——真实NLP/CV任务上的增强ICL效果未测试
- 无标记数据需要与任务相关——如果无标记数据分布与任务不匹配，效果可能退化

## 相关工作与启发
- **vs Unsupervised ICL (Gupta et al. 2024)**: 纯无监督ICL只用无标记数据,本文用labeled+unlabeled的半监督框架，有更强的理论保证
- **vs Pseudo-label方法 (Wan et al. 2023)**: 伪标签继承模型偏差；增强ICL直接在prompt中推理，不生成伪标签
- **vs Li et al. 2025 (concurrent)**: 他们用线性Transformer + 二分类 + 渐近分析；本文用softmax attention + 多分类 + 非渐近收敛

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从理论上回答"无标记数据能否帮助ICL"，建立了ICL-EM的等价性
- 实验充分度: ⭐⭐⭐ 仅合成数据实验，缺乏真实任务验证——理论工作的标准做法但仍是局限
- 写作质量: ⭐⭐⭐⭐ 理论论文写作规范，符号系统清晰，但技术密度高
- 价值: ⭐⭐⭐⭐ 对理解ICL机制有重要贡献，但从线性分类到实际LLM的gap较大
