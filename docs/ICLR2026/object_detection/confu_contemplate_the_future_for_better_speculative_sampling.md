# ConFu: Contemplate the Future for Better Speculative Sampling

**会议**: ICLR2026  
**arXiv**: [2603.08899](https://arxiv.org/abs/2603.08899)  
**代码**: 待确认  
**领域**: LLM推理加速 / 推测解码  
**关键词**: speculative decoding, contemplate tokens, future prediction, MoE, draft model, EAGLE  

## 一句话总结
提出 ConFu 框架，通过 contemplate tokens 让 draft model 预见 target model 的未来生成方向，结合 MoE 动态机制和锚点采样训练，在 EAGLE-3 基础上提升 8-11% 的接受率和速度。

## 背景与动机
1. 推测解码用轻量 draft model 提议候选 token，由 target model 验证，避免逐 token 推理
2. EAGLE 系列是当前 SOTA，利用 target model 隐状态训练单层 Transformer draft head
3. 核心问题：现有 draft model 仅基于当前前缀生成，误差随步数累积，接受率下降
4. 直觉：如果 draft model 能预知 target model 的"思路方向"，生成的候选更可能被接受
5. Latent reasoning（连续思考 token）已证明可提升推理，但需多次前向传播，代价高

## 方法详解
**核心创新 1: Contemplate Tokens + Soft Prompts**
- 在 target model 输入前插入可学习 soft prompt tokens，末尾附加 contemplate token
- Contemplate token 的隐状态编码 target model 的"中间思想"→作为 future token f 提供给 draft model
- 仅 contemplate tokens 可 attend to soft prompts，不影响原始前缀表征
- 推理时在 draft tree 每个节点插入一个 contemplate token，并行验证+生成未来预测

**核心创新 2: MoE 动态 Contemplate Token**
- 用 MoE 参数化 contemplate token embedding 而非固定学习向量
- 以最新接受 token 的隐状态为输入，线性 router 选择 top-K experts 的加权组合
- 不同上下文（数学推理 vs 写作）自适应选择不同"指令"

**核心创新 3: 训练框架**
- Anchor Token Sampling: 随机采样 K_train 个锚点 token 插入 contemplate token，避免序列翻倍
- Future Prediction Replication: 锚点的 future prediction 复用给临近 token，增强鲁棒性
- 损失：KL 散度对齐 target 和 draft 分布，无需额外辅助损失

## 实验关键数据
**Llama-3.2-3B on SpecBench (T=0.0, 30 nodes)**:
| 方法 | 平均接受长度 τ | 加速比 SR |
|------|-----------|------|
| EAGLE-3 | 4.00 | 1.83× |
| **ConFu** | **4.41** | **2.11×** |

- 在所有任务类型（写作/QA/代码/数学等）均一致提升
- 不同温度（0.0/0.7/1.0）和预算（30/60 nodes）下均有效
- 8B 模型同样有 8-11% 提升
- 8×H100 训练，单 H100 推理

## 亮点
- 首次将连续推理 token 与推测解码桥接
- Contemplate token 利用 pause token 机理，推理开销极小（仅增 2T 个验证 token）
- MoE 动态 token 是 pause token 设置中首次引入的动态性
- 建立在 EAGLE-3 之上，从检查点初始化，训练高效

## 局限性 / 可改进方向
- 仅在 Llama-3 3B/8B 测试，更大模型（70B+）待验证
- Soft prompt 和 contemplate token 的超参数调优空间未充分探索
- MoE expert 数量和 top-K 选择的影响未详细分析
- 推理时额外 2T 个 token 的延迟在极端场景下可能非微不足道

## 与相关工作的对比
- **EAGLE-1/2/3**: 逐步改进 draft 架构和训练，但无未来预测机制；ConFu 为正交改进
- **Medusa/HASS**: 不如 EAGLE 系列，ConFu 以 EAGLE-3 为基线
- **BiTA**: 用 soft prompt 直接解码未来 token；ConFu 用其引导 draft model
- **Latent Reasoning (COCONUT等)**: 需多步前向传播；ConFu 用 pause token 并行计算

## 评分
- 新颖性: ⭐⭐⭐⭐ (概念新颖：future prediction + speculative decoding)
- 实验充分度: ⭐⭐⭐⭐ (多任务/多温度/多预算的全面评测)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图示直观)
- 价值: ⭐⭐⭐⭐ (为推测解码开辟新的改进方向)
