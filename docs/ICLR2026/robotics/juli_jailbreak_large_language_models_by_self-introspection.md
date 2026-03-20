# JULI: Jailbreak Large Language Models by Self-Introspection

**会议**: ICLR 2026  
**arXiv**: [2505.11790](https://arxiv.org/abs/2505.11790)  
**代码**: 无  
**领域**: AI安全 / Jailbreak 攻击  
**关键词**: jailbreak, logit bias, API attack, token log probability, BiasNet  

## 一句话总结
揭示对齐 LLM 的 top-k token log probability 中仍包含有害信息，提出 JULI——仅用不到目标模型 1% 参数的 BiasNet 插件操纵 logit bias，在仅访问 top-5 token 概率的 API 场景下成功越狱 Gemini-2.5-Pro（harmfulness 4.19/5），比 SOTA 快 140 倍。

## 研究背景与动机
1. **领域现状**：LLM 越狱攻击分为需要模型权重的白盒攻击和仅通过 API 的黑盒攻击。API 场景下的攻击极具挑战——无法访问梯度、完整 logits 或生成过程。
2. **现有痛点**：LINT（当前 API 攻击 SOTA）需 top-500 token 访问（多数 API 不提供）、推理需 99.7 秒、harmfulness 仅 2.25/5。
3. **核心矛盾**：对齐训练应该消除有害知识的表达，但 LLM API 返回的 top-k token 概率中是否仍泄露有害信息？
4. **本文要解决什么？** 能否仅用 API 返回的少量 token 概率（如 top-5）高效越狱主流 LLM？
5. **切入角度**：发现 >85% 的有害 response token 出现在 top-5 概率中——对齐只是压低了它们的概率而非消除。
6. **核心idea一句话**：用轻量 BiasNet 学习 logit bias 来提升有害 token 概率，仅需 100 条有害数据训练。

## 方法详解

### 整体框架
BiasNet $F_\theta$ 接收目标 LLM 的 log probability 输出 $\log p_\alpha(x_n)$，计算 logit bias $B = F_\theta(\log p_\alpha(x_n))$，修正后的概率 $\tilde{p}_\alpha(x_n) = p_\alpha(x_n) + B$。

### 关键设计

1. **BiasNet**：<1% 目标模型参数（~$10^7$），投影层复用 LLM head（白盒）或随机正交矩阵（黑盒/API）。
2. **Token 泄露发现**：>85% 有害 token 在 top-5 中，对齐未消除有害知识，仅降低概率。
3. **Padding 机制**：API 仅返回 top-k（如 top-5）时，用零 padding 填充剩余位置。
4. **训练**：仅 100 条有害数据，15 epochs。

## 实验关键数据

| 设置 | 模型 | JULI Harmful Score | SOTA Baseline |
|------|------|-------------------|---------------|
| API (top-5) | Gemini-2.5-Pro | **4.19/5** | FLIP: 2.09 |
| 开源 | Llama3-8B | **3.44/5** | ED: 3.02 |
| 推理时间 | - | **0.71s** | LINT: 99.7s |

### 关键发现
- 对齐 LLM 的 top-5 token 概率足以恢复有害输出——对齐是概率压低而非知识擦除。
- 仅 100 条训练数据 + <1% 参数的插件即可攻破 SOTA 防御。
- 比 LINT 快 140 倍，harmfulness 提升 ~2 倍。

## 亮点与洞察
- **"知识泄露" vs "知识擦除"**：与 Erase or Hide 的"浅层对齐"发现一致——对齐后有害知识仍存在于模型中，只是被概率性地抑制。JULI 证明这种抑制可以被外部插件轻松逆转。
- **API 安全的红旗**：现实中的 LLM API（如 Gemini API）返回 top-k 概率，JULI 证明这本身就是一个攻击面。

## 局限性 / 可改进方向
- BiasNet 需要少量有害数据训练，限制了完全零知识攻击。
- 防御方案未深入讨论——如限制 API 返回的 token 数或对概率加噪。

## 相关工作与启发
- **vs ChatInject**：ChatInject 利用 chat template 结构越狱，JULI 利用概率分布越狱。攻击面不同但互补。
- **vs Inoculation Prompting**：Inoculation 在训练时抑制特征，JULI 在推理时恢复被抑制的特征。两者共同说明当前对齐的"浅层"本质。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个仅用 top-5 API 概率的实用越狱，BiasNet 概念新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型(含闭源) × 多场景
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 对 API 安全设计有直接启示
