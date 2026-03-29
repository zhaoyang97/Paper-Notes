# C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.19674](https://arxiv.org/abs/2509.19674)  
**代码**: https://github.com/zhoujiahuan1991/NeurIPS2025-C2Prompt  
**领域**: LLM/NLP  
**关键词**: 联邦持续学习, prompt学习, 类感知聚合, 分布补偿, 知识冲突

## 一句话总结
针对联邦持续学习中prompt通信时的类级知识不一致问题，提出C²Prompt方法，通过局部类分布补偿（LCDC）和类感知prompt聚合（CPA）两个机制显式增强跨客户端的类级知识一致性，在ImageNet-R上Avg准确率达87.20%，超出SOTA Powder 2.51%。

## 研究背景与动机

1. **领域现状**：联邦持续学习（FCL）需要分布式客户端在隐私保护下从持续到达的任务数据中学习。基于prompt的方法（如CODAPrompt + FedAvg）通过维护任务特定的prompt并冻结预训练backbone，在FCL中表现较好。
2. **现有痛点**：现有prompt-based FCL方法在服务器端聚合prompt时忽略了**类级知识一致性**问题：(a) 不同客户端对同一类别的数据分布不同（intra-class分布差距），导致学到的语义不一致；(b) 不同prompt之间的类级关联性（inter-prompt class-wise relevance）未被利用，导致聚合时无关甚至冲突的知识被融合。
3. **核心矛盾**：prompt通信中缺乏类级一致性 → 新prompt之间产生知识冲突 → 还干扰旧prompt → 同时加剧空间遗忘（跨客户端）和时间遗忘（跨任务）。
4. **本文要解决什么？** (a) 如何在客户端本地弥补非IID带来的类内分布偏差？(b) 如何在服务器端根据类级相关性精确聚合prompt？
5. **切入角度**：从类级knowledge coherence的角度切入——既在数据输入层面做分布补偿（LCDC），又在参数聚合层面做类感知加权（CPA）。
6. **核心idea一句话**：通过估计全局类分布来补偿本地分布偏差 + 通过prompt-类亲和度矩阵实现类感知聚合，双管齐下解决FCL中的知识冲突。

## 方法详解

### 整体框架

C²Prompt基于CODAPrompt架构，在冻结的ViT-B/16上学习两类prompt：
- **局部类分布补偿prompt** $\mathcal{P}^c_{t,k}$：每个类别一个，将本地特征对齐到全局类分布
- **局部判别性prompt** $\mathcal{P}^d_{t,k}$：标准的CODAPrompt prompt，用于学习分类知识

训练流程分两阶段：Round 0 做全局分布估计 + LCDC训练；Round 1~$N_r$ 做判别性prompt学习 + CPA聚合。

### 关键设计

1. **全局类分布估计（Global Distribution Estimation）**:
   - 做什么：在服务器端聚合各客户端的类别分布统计量，估计每个类的全局分布
   - 核心思路：假设每个客户端k上类i的特征服从高斯分布 $\mathcal{N}(\mu^t_{i,k}, (\sigma^t_{i,k})^2)$，利用混合高斯的矩估计得到全局均值 $\mu^g_i = \sum_k \mu^t_{i,k} p^t_{k,i}$ 和方差 $(\sigma^g_i)^2 = \sum_k ((\mu^t_{i,k})^2 + (\sigma^t_{i,k})^2) p^t_{k,i} - (\mu^g_i)^2$
   - 设计动机：只传输均值和方差（不传数据），保护隐私的同时获得全局分布视角。通信开销极小（稀疏分布参数）

2. **局部类分布补偿（LCDC）**:
   - 做什么：学习类特定补偿prompt，使本地特征对齐到全局分布
   - 核心思路：对每个类i学习一个补偿prompt $\mathbf{p}^c_i \in \mathbb{R}^{L_c \times d}$，拼接到输入token后送入冻结的ViT。使用分布对齐损失 $\mathcal{L}_c = -\frac{1}{2}(f_{x,p} - \mu^g_i)^\top (\Sigma^g_i)^{-1} (f_{x,p} - \mu^g_i)$ 最大化输出特征在全局高斯分布下的似然
   - 设计动机：不需要生成数据或共享原始数据，仅通过prompt调节特征表示来弥补本地非IID偏差。训练后冻结，作为后续判别性学习的"分布校正器"

3. **类感知prompt聚合（CPA）**:
   - 做什么：在服务器端根据prompt与类别的亲和度进行加权聚合，而非简单平均
   - 核心思路：训练时在线记录每个prompt与每个类的累积匹配分数（client histogram $H^i_k$），上传到服务器后构成矩阵 $\mathbf{H}^t_g \in \mathbb{R}^{KN \times |\mathcal{C}_t|}$。计算inter-prompt相关性矩阵 $W^t_g = \text{softmax}(\mathbf{H}^t_g (\mathbf{H}^t_g)^\top / \tau)$，用 $\mathbf{P}^{t*}_g = W^t_g \mathbf{P}^t_g$ 做加权聚合
   - 设计动机：类别亲和度相似的prompt应该获得更大的聚合权重，减少不相关类别知识的干扰。histogram在在线学习中几乎零额外开销

4. **判别性学习 + 知识蒸馏**:
   - 做什么：标准的分类学习 + 跨轮次知识保留
   - 总损失 $\mathcal{L}_d = \mathcal{L}_{ce} + \beta \mathcal{L}_{kd}$，其中 $\mathcal{L}_{kd}$ 是来自Powder的蒸馏损失
   - 补偿prompt以50%概率使用（p=0.5），兼顾原始数据和补偿后数据信息

### 训练策略
- Backbone: ViT-B/16 (ImageNet-21k预训练，冻结)
- 判别性prompt: N=8, $L_p$=10, d=768；补偿prompt: $L_c$=3
- 客户端数K=5，每任务通信轮次$N_r$=3
- 优化器: Adam, lr=0.01

## 实验关键数据

### 主实验

| 方法 | 发表 | ImageNet-R Avg↑ | ImageNet-R AIA↑ | DomainNet Avg↑ | DomainNet AIA↑ |
|------|------|----------------|-----------------|----------------|----------------|
| FedWEIT | ICML2021 | 71.10 | 74.30 | 67.84 | 69.63 |
| GLFC | CVPR2022 | 72.96 | 75.21 | 69.75 | 70.34 |
| Fed-CODAP | CVPR2023 | 79.65 | 75.14 | 72.47 | 72.84 |
| Powder | ICML2024 | 84.69 | 84.08 | 75.98 | 77.28 |
| **C²Prompt** | **本文** | **87.20** | **85.93** | **78.88** | **77.55** |

ImageNet-R上Avg超Powder 2.51%，DomainNet上超2.90%。

### 消融实验

| 配置 | ImageNet-R Avg | 说明 |
|------|---------------|------|
| Baseline (Powder) | 84.69 | 基线 |
| + LCDC only | 86.57 (+1.88) | 分布补偿有效 |
| + CPA only | 86.02 (+1.33) | 类感知聚合有效 |
| + LCDC + CPA (Full) | 87.20 (+2.51) | 两者互补 |

### 关键发现
- LCDC和CPA分别在输入层面和参数层面解决知识不一致，两者互补性强
- C²Prompt是唯一在DomainNet大规模数据集上实现负遗忘率（FM<0）的方法
- 前向迁移（FT）提升最大：ImageNet-R +3.15%，DomainNet +2.59%——说明全局分布估计有效帮助了新任务学习
- 通信开销仅比Powder增加0.6%，推理时无额外参数/计算

## 亮点与洞察
- **全局分布估计的隐私-效率权衡**设计巧妙：只传均值+方差，不传数据不传梯度，通信增量极小却能有效弥补非IID差距
- **类感知聚合的"免费"实现**：client histogram在训练过程中在线累积，零额外计算，但提供了精确的prompt-类亲和度信息作为聚合权重
- prompt注意力可视化（Figure 5）直观展示了CPA使prompt更关注判别性区域，而Powder的prompt注意力散漫

## 局限性 / 可改进方向
- 高斯假设可能对复杂多模态分布不准确，特别是类别内部有子簇结构时
- 仅在ViT-B/16 + 图像分类上验证，未扩展到更大backbone或NLP任务
- 客户端数量固定为5，对更大规模（如50+客户端）的扩展性未验证
- 补偿prompt的使用概率p=0.5是固定的，可以探索自适应概率策略

## 相关工作与启发
- **vs Powder (ICML2024)**: Powder引入了知识蒸馏做跨轮次保留，本文在其基础上增加了类级分布补偿和类感知聚合。C²Prompt保留了Powder的蒸馏损失
- **vs CODAPrompt**: C²Prompt使用CODA作为基础prompt架构，区别在于聚合方式从简单FedAvg变为类感知加权
- **vs PILoRA/LoRM**: LoRA-based FCL方法效果远差于prompt-based方法（PILoRA仅45.43%），说明prompt学习在FCL中有天然优势

## 评分
- 新颖性: ⭐⭐⭐⭐ 全局分布估计+类感知聚合的组合思路清晰且有效，但单个模块都有先例
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、12个对比方法、消融+可视化充分，但缺少大规模客户端实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，但符号较多，公式推导放附录
- 价值: ⭐⭐⭐⭐ FCL是实际需求驱动的方向，方法实用且开销小
