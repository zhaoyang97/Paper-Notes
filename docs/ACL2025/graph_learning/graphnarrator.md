# GraphNarrator: Generating Textual Explanations for Graph Neural Networks

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2410.15268](https://arxiv.org/abs/2410.15268)  
**代码**: 无  
**领域**: 图神经网络 / 可解释AI / 自然语言解释  
**关键词**: GNN Explainability, Natural Language Explanation, Expert Iteration, Pseudo-Label, Text-Attributed Graph  

## 一句话总结
提出GraphNarrator——首个为图神经网络生成自然语言解释的方法，通过将显著性图解释"语言化"为文本段落、用Expert Iteration迭代优化伪标签质量、最终蒸馏到端到端解释器模型，在三个数据集上生成的解释在忠实度、简洁性和人类偏好上均优于GPT-4o零样本解释。

## 背景与动机
GNN在推荐系统、社交网络、分子图等领域广泛应用，但其决策过程不透明。现有GNN解释方法（如GNNExplainer、PGM-Explainer）提供节点/边级别的重要性分数，但当图节点关联了文本特征（Text-Attributed Graph, TAG）时，这些细粒度的token重要性分数难以被人类理解——它们冗余、分散、没有整合。自然语言解释能把分散的重要性信号整合成连贯、人类友好的文本，但目前没有为GNN生成自然语言解释的方法。

## 核心问题
如何在没有ground truth解释标签的情况下，为GNN的预测生成忠实、简洁、人类可读的自然语言解释？

## 方法详解

### 整体框架
GraphNarrator分三步：
1. **显著性解释生成与语言化**：用post-hoc显著性方法获取节点/token重要性，将图结构通过BFS+层次组织转化为"显著性段落"（Saliency Paragraph）
2. **Expert Iteration优化伪标签生成器**：基于三个信息论指标（输入忠实度、输出忠实度、简洁性），迭代筛选高质量伪标签并微调LLM
3. **知识蒸馏到端到端解释器**：将优化后的伪标签用于训练只需原始输入即可生成解释的LLaMA模型

### 关键设计

1. **显著性图语言化（Saliency Verbalization）**
   - 以目标节点为根做BFS构建树结构，剪去不重要节点
   - Pre-Order遍历将树转为层次化文档（节点→章节，子节点→子章节）
   - 跨分支边通过引用句维护图结构信息
   - 每个token后附上重要性分数，如 `probabilistic(5.11)`

2. **信息论解释质量指标**
   - **输入忠实度 $f_S$**：解释与重要输入token之间的PMI，通过掩码token预测用语言模型估计
   - **输出忠实度 $f_F$**：解释与预测标签之间的PMI
   - **简洁性 $f_B$**：解释长度与输入长度之比
   - 采样不同阈值 $\tau$ 定义"重要token"，使模型学习灵活的重要性判断

3. **Expert Iteration闭环训练**
   - 每轮迭代：生成候选解释→三指标评分→筛选top-50%高质量解释→微调生成器LLM
   - 每轮仅需50个高质量样本即可稳步提升
   - GPT-4o-mini作为初始生成器，Gemma-2B作为PMI估计模型

### 训练策略
最终使用LLaMA-3.1-8B + LoRA微调（rank=16, alpha=16），通过知识蒸馏从优化后的伪标签生成器学习，产出端到端explainer（无需显著性输入即可生成解释）。

## 实验关键数据

| 数据集 | 方法 | Simulatability↑ | PMI-10%↑ | Brevity↓ |
|--------|------|----------------|----------|----------|
| DBLP | GPT-4o zero-shot | 0.82 | 0.142 | 0.385 |
| DBLP | **GraphNarrator** | **0.95** | **0.155** | **0.354** |
| Cora | GPT-4o zero-shot | 0.95 | 0.414 | 0.357 |
| Cora | **GraphNarrator** | **0.97** | **0.418** | **0.315** |
| Book-History | GPT-4o zero-shot | 0.89 | 0.456 | 0.768 |
| Book-History | **GraphNarrator** | **0.96** | **0.533** | **0.506** |

人类评估中，GraphNarrator在结构信息保持上比GPT-4o提升33.7%，语义信息保持提升23.9%。

### 消融实验要点
- 去掉$f_S$（输入忠实度）→PMI分数下降
- 去掉$f_F$（输出忠实度）→Simulatability下降
- 去掉$f_B$（简洁性）→解释变长但其他指标提升（三指标存在trade-off）
- Expert Iteration过程中三指标随迭代稳步提升

## 亮点
- **首创将GNN解释转化为自然语言**：从散乱的token重要性到连贯的文本解释，对GNN可解释性是质的飞跃
- **BFS+层次文档的图语言化方案**：巧妙将图结构转为LLM可理解的层次化文档，跨分支边用引用句处理
- **信息论指标+Expert Iteration的无监督优化**：在没有ground truth的情况下，用PMI度量忠实度、迭代筛选高质量样本，优雅解决了"没有标注怎么训练"的问题
- **模型无关性**：可适配任何GNN架构和显著性方法

## 局限性 / 可改进方向
- 推理成本较高（LLM骨干），对极大子图推理可能超过2分钟
- 仅在节点分类任务上验证，未测试图分类、链接预测等其他任务
- 三个数据集规模有限（Cora仅2708节点），大规模TAG上的表现未知
- 伪标签的质量上限受初始显著性方法的质量影响

## 与相关工作的对比
- **vs GNNExplainer/PGM-Explainer**：后者输出节点/边重要性分数，不可读；GraphNarrator输出自然语言，人类友好
- **vs SMV（Saliency Map Verbalization）**：SMV仅针对文本分类，不处理图结构；GraphNarrator处理TAG的图结构和跨节点关系
- **vs GPT-4o零样本解释**：GPT-4o不了解GNN的内部决策过程；GraphNarrator通过显著性信号将模型内部信息传递给解释生成器

## 启发与关联
- "显著性→语言化→LLM理解"的pipeline可扩展到其他非文本模态的可解释性（如视觉模型explain）
- Expert Iteration + 信息论指标的无监督优化思路可用于其他缺乏ground truth标签的文本生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次提出为GNN生成自然语言解释，图语言化和Expert Iteration组合巧妙
- 实验充分度: ⭐⭐⭐⭐ 自动评估+人类评估+消融，但数据集较小且仅测试节点分类
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但符号较多，部分公式理解门槛高
- 对我的价值: ⭐⭐⭐ Expert Iteration的无监督优化思路有启发，但具体应用场景（GNN解释）较窄
