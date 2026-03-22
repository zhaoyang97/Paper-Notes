# Do Different Prompting Methods Yield a Common Task Representation in Language Models?

**会议**: NeurIPS 2025  
**arXiv**: [2505.12075](https://arxiv.org/abs/2505.12075)  
**代码**: https://github.com/...  
**领域**: ICL机制与Task表示  
**关键词**: Function Vectors、上下文学习、指令跟随、任务表示

## 一句话总结
本文扩展函数向量方法至指令提示，发现演示和指令诱发的任务表示主要不同，仅部分重叠，解释了为何结合两者效果更优。

## 研究背景与动机
尽管演示（demonstrations）和指令（instructions）都能指导LLM执行任务，但它们是否激活相同的内部表示机制仍不清楚。理解这一问题对以下方面关键：
- 解释为何结合演示和指令时模型性能更优
- 指导prompt设计以实现更可靠的任务推断
- 识别任务表示的脆弱性和改进点

本工作基于Function Vectors (FVs)方法，首次系统比较演示vs指令的内部机制。

## 方法详解

### 整体框架
**Function Vectors核心**：识别对特定任务执行有因果贡献的注意力头集合{a^D}，计算其激活均值向量v_t = Σ_{a_lj∈A^D} ā^t_{lj}

**扩展至指令**：
- 生成多个指令变体（短/长）
- 构造对比基线（3种策略：等概率序列、真实文本、其他任务指令）
- 计算因果间接效应(CIE)识别关键头
- 在zero-shot评估中应用FV

### 关键设计

**baseline构造三策略**：
1. **等概率令牌**：采样与原始任务说明概率分布匹配但内容无关的序列
2. **真实文本**：从WikiText-103采样长度/概率匹配但无任务信息的文本
3. **其他任务指令**：采样来自不同任务的指令

**异构FV构造**（新颖）：
- Demo-localized heads + Instruction activations（反向组合）
- 测试不同定位和激活的交叉兼容性
- 量化不对称性（Demo→Instruction vs 反向）

## 实验关键数据

| 模型 | 演示FV | 指令FV | 联合 | 控制（同FV 2x） |
|------|--------|--------|------|-----------------|
| Llama-3.2-3B | 58% | 48% | 63%↑ | 60% |
| Llama-3.1-8B | 42% | 22% | 51%↑ | 45% |
| Llama-3.2-3B | 55% | 39% | 65%↑ | 62% |
| OLMo-2-1124-7B | 68% | 51% | 76%↑ | 72% |

关键发现：联合干预相比单独使用性能提升清晰，且不全归因于放大效应。

**共享头数量**：
- Llama-3.2-3B：20个顶头中仅7个共享
- Llama-3.1-8B：20个顶头中仅4个共享
- 指令头因果影响分数明显低于演示头（更分散的表示）

**异质FV性能**：
- Demo头+Instruction激活 → Demo头+Demo激活：下降幅度小
- Instruction头+Demo激活 → 更大下降
- 结论：Instruction头编码了Demo不用的信息

## 亮点与洞察

1. **首次系统比较**：在控制条件下直接对比演示vs指令的神经表示
2. **Function Vectors创新扩展**：成功泛化至多种任务表示形式
3. **发现表示非共有**：大多数头仅由一种提示形式激活，反驳"通用表示"假设
4. **不对称性洞察**：Instruction转Demo容易，Demo转Instruction困难，反映了学习优先级（演示更原始）
5. **可迁移性**：Post-trained模型的Instruction FV可引导Base模型，跨模型表示有通用性

**演示ICL的优先性**：
- Base模型已支持演示学习
- Instruction学习需后训练才能启用
- SFT > DPO阶段启用Instruction FV

## 局限性

1. **任务范围有限**：50个相对简单数据集，遗漏复杂/开放式任务
2. **表示方法选择**：仅研究Function Vectors，未比对Task Vectors等替代方法
3. **硬件约束**：未探索扩展到更大模型（70B+）的表现
4. **干预深度固定**：未充分调研不同层数干预的差异（虽有附录探讨）
5. **重复有限**：各条件仅单次运行，未量化方差

## 相关工作

- 上下文学习机制：Grokking、ICL学习算法分析
- 指令跟随：FLAN、prompt工程研究
- 任务表示提取：Task Vectors、Latent CoT

## 评分
⭐⭐⭐⭐ (4/5)
