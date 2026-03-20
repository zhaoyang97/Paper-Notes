# AlignTree: Efficient Defense Against LLM Jailbreak Attacks

**会议**: AAAI 2026  
**arXiv**: [2511.12217v1](https://arxiv.org/abs/2511.12217v1)  
**代码**: [https://github.com/Gilgo2/AlignTree](https://github.com/Gilgo2/AlignTree) (有)  
**领域**: AI Safety  
**关键词**: LLM安全, 越狱攻击防御, 随机森林分类器, Refusal Direction, SVM  

## 一句话总结

AlignTree 利用 LLM 内部激活特征（线性 refusal direction + 非线性 SVM 信号）训练轻量级随机森林分类器，在几乎不增加计算开销的情况下高效检测越狱攻击，实现了 SOTA 的攻击成功率（ASR）降低效果。

## 背景与动机

LLM 面临严重的越狱攻击（jailbreak）威胁：攻击者通过精心构造的 prompt 绕过安全对齐机制，诱导模型生成有害内容。现有防御方法存在明显的**效率与鲁棒性矛盾**：

- **预处理防御**（如 LlamaGuard、ShieldGemma）：需要额外部署一个安全 LLM 做输入过滤，计算开销大；
- **过程中防御**（如 SmoothLLM）：需要多次重复推理或生成多个 prompt 副本，延迟高；
- **后处理防御**（如 SelfDefense、AutoDefense）：需要 LLM 对自身输出做二次审查，计算量加倍甚至翻倍。

更关键的是，先前基于激活空间的防御方法主要依赖**单一线性 refusal direction**来判别有害 prompt，但近期研究揭示 LLM 中的拒绝行为在几何上并非完全线性，单一线性信号不足以捕获所有恶意模式。

## 核心问题

如何设计一个**计算高效、无需额外模型、无需额外推理**的 LLM 越狱防御机制，既能有效降低攻击成功率，又不会导致对正常 prompt 的过度拒绝？

## 方法详解

### 整体框架

AlignTree 是一个**过程中（in-process）防御**方法，在 LLM 推理过程中监控内部激活状态。其流程：
1. 对输入 prompt 做一次前向传播，提取各层隐藏状态
2. 从隐藏状态中提取两类特征：(i) 线性 refusal activations，(ii) 非线性 SVM 概率特征
3. 将两类特征拼接输入随机森林分类器，输出有害性置信度
4. 与阈值 $\tau$ 比较，决定放行或拦截

### 关键设计

1. **Refusal Activations（线性拒绝信号）**: 采用 difference-in-means 方法，在有害样本集 $D_{\text{harmful}}$ 和无害样本集 $D_{\text{harmless}}$ 上计算每个 token 位置、每层的平均激活向量差：$r_i^{(l)} = \mu_i^{(l)} - v_i^{(l)}$。然后通过验证集评估各向量的拒绝引导/消除效果，选出最优的单一 refusal direction $r^*$。对每层最后一个 token 的隐藏状态 $h$ 计算投影标量：$\text{proj}_{r^*}(h) = \frac{h \cdot r^*}{\|r^*\|}$，得到各层的 refusal activation 标量特征。

2. **SVM 非线性恶意信号提取**: 针对模型每层 $l$ 和选定的 8 个 token 位置（前3后5），训练独立的 RBF 核 SVM 分类器 $\text{SVM}_i^{(l)}$，共 $8 \times L$ 个。在验证集上评估后保留准确率最高的 $L/2$ 个。通过 Platt scaling 将 SVM 决策值映射为校准的有害性概率 $P_{\text{harmful}}(x_i^{(l)})$，形成非线性特征向量。

3. **随机森林分类器**: 将上述两类特征拼接为完整输入：

$$F(t) = [\text{proj}_{r^*}(x_{-1}^{(l)}(t))]_{l=1}^{L} \oplus [P_{\text{harmful}}(x_i^{(l)}(t))]_{(i,l) \in \mathcal{S}}$$

使用浅层随机森林（50 棵树，最大深度 6，最小分裂样本数 5）进行有害/无害分类。

### 损失函数 / 训练策略

- SVM 使用 RBF 核，通过 5-fold 交叉验证生成 out-of-fold 概率
- 随机森林超参数：`n_estimators=50, max_depth=6, min_samples_split=5`（grid search 验证了超参数不敏感性）
- **阈值选取**：使用 $F_\beta$ score（$\beta=0.2$，偏重 precision）在验证集上选取最优阈值 $\tau$：

$$F_\beta = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- 训练数据：refusal/SVM 训练集来自 AdvBench、MaliciousInstruct、TDC2023、StrongReject、HarmBench（有害）和 ALPACA（无害）；随机森林训练集使用 JailbreakBench、PAIR、AutoDAN 攻击样本 + ALPACA、XSTest
- 训练时间约 3 分钟（单张 RTX 6000 Ada，最大模型）

## 实验关键数据

| 数据集 | 指标 | AlignTree | 之前SOTA | 提升 |
|--------|------|-----------|----------|------|
| MalwareGen (Qwen2.5-0.5B) | ASR↓ | **4.0** | 5.0 (AutoDefense) | 降低 1pp |
| PAIR (Qwen2.5-0.5B) | ASR↓ | **6.0** | 8.0 (SelfDefense-Input) | 降低 2pp |
| AutoDAN (Qwen2.5-0.5B) | ASR↓ | **0** | 0 (AutoDefense) | 持平 |
| PromptInject (Llama-3.1-8B) | ASR↓ | **18.0** | 28.0 (SelfDefense) | 降低 10pp |
| PAIR (Gemma-3-12b) | ASR↓ | **10.0** | 19.0 (AutoDefense) | 降低 9pp |
| 白盒自适应攻击（3个模型） | ASR↓ | **0** | 0 (AutoDefense) | 持平但快60x+ |
| PIQA/ARC 等无害数据集 | 误拒率↓ | **0-1%** | 0-8% (AutoDefense) | 最低误拒 |

**效率对比**：AlignTree 执行时间接近 baseline（无防御），比 AutoDefense 快约 10-50 倍，比 SmoothLLM 快 5-20 倍。在白盒攻击实验中，AlignTree (2.40s) vs AutoDefense (140.74s) 快约 58 倍，且两者 ASR 均为 0。

### 消融实验要点

- **RefusalClassifier**（仅线性信号）：在对齐较好的模型（Llama）上有效，但在弱对齐模型（Qwen）上近乎无效（ASR 89.0），说明线性信号依赖基础模型的对齐质量
- **SVMClassifier**（仅非线性信号）：泛化性差，在部分数据集上误拒率过高
- **MultiRefusalsClassifier**（多refusal方向）：优于单方向，证实拒绝机制的多维性
- **AlignTreeLinear**（线性SVM替代RBF）：在个别模型上表现好（Gemma-3-12b），但整体不一致（Qwen2.5-0.5B 上 ASR 61.0 vs AlignTree 4.0）
- **完整 AlignTree**：在所有模型上表现最稳定一致

## 亮点

- **极低计算开销**：不需要额外 LLM、不需多次推理、不需 prompt 变体，仅在已有前向传播的激活上做轻量分类
- **线性+非线性信号互补**：首次在防御框架中结合线性 refusal direction 和非线性 SVM 特征，消融实验有力证明了非线性信号的重要性
- **低误拒率**：在4个常识推理数据集上几乎零误拒，实用性强
- **广泛评估**：9个 LLM（3个家族 × 3个规模）+ 多个攻击基准 + 白盒自适应攻击，实验覆盖面非常全

## 局限性 / 可改进方向

- 需要为每个模型**单独训练一个分类器**，无法跨模型复用
- 在弱对齐模型上，refusal direction 信号几乎失效，完全依赖 SVM 信号
- PromptInject 数据集上 ASR 仍然较高（如 Qwen2.5-0.5B 上 41.0），说明"忽略前置指令"类攻击仍有挑战
- ASR 评估依赖 ChatGPT-4o 判断，可能引入评估偏差
- 仅使用浅层分类器，未探索更复杂模型是否能进一步提升（如 MLP）
- 未来可引入"可疑"阈值区间，把不确定的 prompt 交给更强防御进一步处理

## 与相关工作的对比

| 方法 | 额外模型 | 额外推理次数 | 计算开销 | ASR表现 |
|------|---------|-------------|---------|--------|
| LlamaGuard | 需要 | 1次（guard LLM） | 高 | 低 |
| AutoDefense | 需要 | 20次 | 极高 | 低 |
| SmoothLLM | 不需要 | 10次 | 中高 | 中 |
| SelfDefense | 不需要 | 2次 | 中 | 中 |
| PerplexityDefense | 不需要 | 0次 | 极低 | 高（防御弱） |
| **AlignTree** | **不需要** | **0次** | **极低** | **低** |

AlignTree 是唯一在不引入额外模型、不增加推理次数的情况下，同时实现低 ASR 和低误拒率的方法。

## 启发与关联

- **拒绝行为的非线性本质**：论文实验充分验证了 LLM 中拒绝行为不是简单的线性现象，未来的对齐研究应更多关注激活空间中的非线性结构
- **级联轻量分类器思路**：用 SVM 提取概率特征再喂入 RF 的两阶段设计，在其他需要在推理过程中做实时决策的场景（如幻觉检测、毒性过滤）中也可借鉴
- **阈值自适应策略**：$F_\beta$ score 选阈值的方法可泛化到任何需要 precision-recall 权衡的安全决策场景
- 与 JBShield 等激活空间防御工作可进一步融合，探索更丰富的特征组合

## 评分

- 新颖性: ⭐⭐⭐⭐ 线性+非线性信号组合的思路有新意，但基于 refusal direction 和 SVM/RF 的技术本身并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 9个模型、3个家族、多种攻击、白盒自适应攻击、详尽消融、超参数敏感性分析，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详实，但表格数量过多稍显冗长
- 价值: ⭐⭐⭐⭐ 实用性强，计算开销极低是真正的亮点，可直接部署到生产环境
