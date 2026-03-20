# Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization

**会议**: ICLR 2026 / **arXiv**: [2509.22115](https://arxiv.org/abs/2509.22115)  
**代码**: 已公开（补充材料）  
**领域**: llm_alignment / 强化学习  
**关键词**: GRPO, policy optimization, down-sampling, advantage variance, token selection, curriculum learning  

## 一句话总结
提出**D3S**（Dynamic Dual-Level Down-Sampling）框架，在sample层最大化advantage方差、在token层优先选取高熵+高advantage的token，配合动态调度策略，用不到20% token实现更快收敛和更优性能。

## 背景与动机
1. Critic-free方法（GRPO/GSPO）通过group相对奖励估计advantage，消除了critic网络的显存负担
2. 但大group稀释关键学习信号——大量uninformative样本（全对/全错）淹没少数有价值梯度
3. Razin等发现提高奖励方差$\text{Var}(R)$可加速收敛，但GRPO归一化后advantage方差固定为1
4. 理论分析表明：最大化**advantage方差**$\text{Var}(A')$能提高梯度范数上界，而最大化$\text{Var}(R)$无法改变此上界
5. Token层面同样存在大量低信息量token稀释梯度信号的问题

## 方法
- **Sample-level下采样**: 对整个batch的rollout计算group-relative advantage，选择最大化$\text{Var}(A_{\hat{S}})$的子集，跨group操作保留全局高方差样本
- **Token-level选择**: 用$|A_{i,t}| \times H_{i,t}$（advantage绝对值×策略熵）排序，只保留top-K%的token参与梯度更新
- **动态调度**: 线性插值 $(N_{init}, K_{init}) \to (N_{final}, K_{final})$，早期激进下采样加速学习，后期逐渐放宽防止过拟合
- **理论保障**: Proposition 2证明梯度范数上界与$(\text{Var}(A'))^{1/3}$正相关；Lemma 1证明从标准化集合中总能抽出方差$\geq 1$的子集

## 实验
| 模型/方法 | AIME24 | AIME25 | AMC23 | MATH | 平均 |
|-----------|--------|--------|-------|------|------|
| Qwen2.5-Math-7B + GRPO | 13.2/37.6 | 5.5/21.6 | 47.0/83.5 | 48.5/70.2 | 29.8/53.1 |
| + PODS (max Var(R)) | 16.1/40.5 | 7.8/24.5 | 52.8/81.5 | 53.0/71.1 | 34.1/54.4 |
| + **D3S** | **20.3/48.2** | **7.9/25.8** | **54.4/87.1** | 52.2/71.5 | **34.3/56.8** |
| Qwen2.5-Math-7B + GSPO + **D3S** | 18.3/43.3 | **8.3/26.9** | 53.2/83.8 | **54.9/71.4** | **35.8/56.2** |
| Llama3.1-8B + GRPO + D3S | Pass@1 +3.3, Pass@8 +7.8 | — | — | — | — |

| 分析 | 发现 |
|------|------|
| Token消耗 | D3S仅使用原始GRPO不到20%的token |
| 梯度范数 | D3S产生更高的策略梯度，收敛更快 |
| 动态调度 | 固定高强度下采样后期过拟合；动态调度保持持续提升 |
| Entropy管理 | D3S更好地控制熵波动，训练更稳定 |

## 亮点
- 理论清晰：从梯度范数上界出发推导出advantage方差最大化优于reward方差最大化
- Token-level选择的$|A| \times H$度量直觉合理——同时关注"影响"和"不确定性"
- D3S作为即插即用模块，与GRPO/GSPO均兼容，跨模型（Qwen/Llama）一致有效
- 动态调度借鉴课程学习思想，优雅平衡效率与泛化

## 局限性
- 仅在数学推理任务验证，代码生成/通用对话等任务的效果未知
- Sample-level跨group操作可能在分布式训练中引入通信开销
- 动态调度的线性插值策略较简单，非线性schedule可能更优
- 未深入分析D3S在不同reward分布（稀疏/密集）下的行为差异

## 相关工作
- GRPO(Shao,2024)/GSPO(Zheng,2025): 基线critic-free方法 → D3S在此基础上提升效率
- PODS(Xu,2025): 最大化Var(R)选样本 → D3S证明最大化Var(A)上界更紧
- Razin(2024,2025): 奖励方差加速收敛 → D3S将此insight拓展到advantage方差并扩展到token级别

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐
