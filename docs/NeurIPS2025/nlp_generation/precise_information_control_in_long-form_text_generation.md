# Precise Information Control in Long-Form Text Generation

**会议**: NeurIPS 2025  
**arXiv**: [2506.06589](https://arxiv.org/abs/2506.06589)  
**代码**: 无  
**领域**: NLP生成 / 忠实性  
**关键词**: 精确信息控制, 忠实性幻觉, 可验证声明, 偏好学习, 长文生成  

## 一句话总结
提出Precise Information Control (PIC)任务——要求LLM生成的长文严格基于给定声明集合（不遗漏不添加），构建PIC-Bench评测8个任务发现SOTA模型70%以上生成包含忠实性幻觉，通过弱监督偏好数据构建+DPO训练的PIC-LM将8B模型F1从69.1%提升至91.0%。

## 研究背景与动机

1. **领域现状**：LLM在长文本生成中的幻觉问题分为事实性幻觉（与真实世界知识矛盾）和忠实性幻觉（与提供的输入上下文矛盾）。大量工作关注事实性幻觉，但忠实性幻觉同样关键——即使给了正确的上下文信息，模型仍可能添加未支持的内容或遗漏关键信息。
2. **现有痛点**：（a）忠实性评估通常是binary的（"忠实/不忠实"），粒度太粗；（b）没有标准化任务来量化LLM在给定明确声明时的信息控制精度；（c）现有模型在用户明确指定应包含哪些信息时仍严重幻觉。
3. **核心矛盾**：理论上，忠实性幻觉应该是可以完全消除的（因为正确答案就在输入中），但实际上SOTA模型的完美忠实率不超过30%。
4. **本文要解决什么**：（a）形式化定义和评测长文生成中的信息控制精度；（b）训练能精确控制输出信息的模型。
5. **切入角度**：以"可验证声明"为粒度单位，把忠实性分解为precision（不多说）和recall（不少说）。
6. **核心idea**：PIC = 给定声明集合C，生成的文本中每个声明都能被C支持（precision），且C中每个声明都出现在生成中（recall）。

## 方法详解

### 整体框架
两阶段：（1）PIC-Bench评测——将8个长文生成任务转换为PIC格式，用声明提取+验证评估模型表现；（2）PIC-LM训练——SFT+弱监督偏好数据DPO后训练Llama 3.1 8B Instruct。

### 关键设计

1. **PIC任务形式化**
   - 做什么：将长文生成重新定义为声明级别的精确控制问题。
   - 核心思路：输入 = 指令 $\mathcal{I}$ + 声明集 $C = \{c_1, \ldots, c_n\}$。生成响应 $\theta(\mathcal{I}, C)$，从中提取声明 $C' = \{c'_1, \ldots, c'_m\}$。**Full PIC**：$C' = C$（不遗漏不添加），用F1评价。**Partial PIC**：$C' \subseteq C$（可以选择子集但不能添加），用precision评价。
   - 设计动机：Full PIC适用于需要完整覆盖的场景（传记、改写），Partial PIC适用于允许选择性引用的场景（摘要、RAG QA）。两种模式覆盖了实际应用的核心需求。

2. **弱监督偏好数据构建**
   - 做什么：自动生成PIC导向的偏好对用于DPO训练，无需人工标注。
   - 核心思路：对每个样本 $(\mathcal{I}, C_{orig}, y_{orig})$，随机删除部分声明得到 $C_{perturb} \subset C_{orig}$，用SFT模型基于 $C_{perturb}$ 生成 $y_{perturb}$。得到两个偏好对：以 $C_{orig}$ 为上下文时 $y_{orig}$ 优于 $y_{perturb}$（正确答案应包含完整信息），反之亦然。用归一化对数概率差作为指令跟随代理信号来自适应选择：$\sigma(\frac{\log p_\theta(y_{orig})}{L} - \frac{\log p_\theta(y_{perturb})}{L})$ 超过阈值 $\tau$ 则选第一种构造。
   - 设计动机：随机采样两种构造的equal probability可能损害指令跟随能力（删掉太多声明后的response可能无法充分回答指令），自适应选择在PIC和指令跟随之间取得平衡。

3. **声明提取与验证pipeline**
   - 做什么：自动将长文分解为可验证声明并检查支持关系。
   - 核心思路：用LLM-based claim extractor将输出分解为独立的可验证声明（介于句子级和原子级之间的粒度），用claim verifier检查语义等价性。人类一致性验证确认pipeline的可靠性。
   - 设计动机：可验证声明granularity是关键——句子级可能混合支持和不支持的信息，原子级缺乏独立验证的上下文。

### 训练策略
- 基座：Llama 3.1 8B Instruct
- SFT阶段：在PIC格式的高质量数据上微调（No Robots + FLAN + CNN + EntityBios + long-form QA）
- DPO阶段：Length-normalized DPO on 弱监督偏好数据

## 实验关键数据

### 主实验（PIC-Bench Full Setting，F1）

| 模型 | 平均F1 | 完美F1比例 | 最难任务(PopBios-CF) |
|------|--------|-----------|-------------------|
| Llama 3.1 8B Inst. | 69.1 | 3.7% | 23.7 |
| Tulu 3 8B | 76.9 | 4.9% | 51.3 |
| Llama 3.3 70B Inst. | 78.9 | 11.3% | 72.5 |
| QwQ 32B | 84.5 | 18.0% | 67.5 |
| GPT-4o | 83.1 | 17.0% | 71.3 |
| Claude 3.5 Sonnet | 87.1 | 24.7% | 82.6 |
| **PIC-LM (8B, Ours)** | **91.0** | **43.9%** | **84.2** |

### 下游应用

| 任务 | Baseline (Llama 8B) | PIC-LM | 提升 |
|------|-------|--------|------|
| ASQA (RAG QA) EM Recall | 52.5% | 61.5% | +17.1% |
| Birthplace Factual Precision | 65.9% | 86.0% | +30.5% |
| QAMParI F1@5 | 13.5% | 22.6% | +67.4% |

### 关键发现
- **SOTA模型70%+生成含忠实性幻觉**：即使是Claude 3.5 Sonnet，完美F1比例也仅24.7%，说明精确信息控制远未解决。
- **反事实场景最难**：PopBios-CF（将实体替换为另一个知名人物）上所有模型表现最差，说明参数化知识和上下文信息冲突时模型倾向于跟随参数化记忆。
- **8B PIC-LM超越所有开源和闭源模型**：91.0% vs Claude 3.5的87.1%，证明targeted post-training的有效性。
- **忠实性提升可传导为事实性提升**：PIC-LM在RAG和事实检查pipeline中显著提升准确率，说明精确的上下文跟随能力是减少事实幻觉的关键基础设施。

## 亮点与洞察
- **PIC任务的优雅简洁**：把复杂的"幻觉检测"简化为"声明集的precision/recall"，使问题well-defined且可量化。两种设置（full/partial）覆盖了核心场景。
- **弱监督偏好数据的巧妙构造**：通过随机删除声明创建 $C_{perturb}$，自动获得偏好对，无需人工标注但效果显著。用对数概率差作为指令跟随的代理signal也很优雅。
- **忠实性→事实性的传导效应**：论文证明精确的上下文跟随不仅减少忠实性幻觉，还间接提升事实准确率——这为"先检索/验证，再忠实生成"的pipeline提供了理论和实验支持。

## 局限性 / 可改进方向
- 声明提取和验证依赖LLM，本身可能引入误差。
- 完美PIC（零幻觉）即使对PIC-LM也仅43.9%，说明问题仍远未解决。
- Partial PIC中precision很高但可能牺牲了有用性（不敢多说）。
- 训练数据包含in-domain任务，OOD泛化性仍可提升。

## 相关工作与启发
- **vs FActScore**: FActScore评估facts vs Wikipedia的事实性，PIC评估outputs vs inputs的忠实性，方向互补。
- **vs Conformal Importance Summarization**: 该工作保留重要句子的recall保证，PIC同时要求precision和recall在声明级别的精确控制，更严格。
- **vs RLHF for hallucination**: 一般RLHF用reward model学偏好，PIC-LM用自动构建的PIC-specific偏好数据，更有针对性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ PIC任务形式化和弱监督偏好构建都是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 8个任务、13+模型、下游应用验证、大量消融
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义精确，实验全面，论文结构清晰
- 价值: ⭐⭐⭐⭐⭐ 对长文生成忠实性问题提出了可操作的解决方案
