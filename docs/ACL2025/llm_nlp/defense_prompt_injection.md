# Defense Against Prompt Injection Attack by Leveraging Attack Techniques

**会议**: ACL 2025  
**arXiv**: [2411.00459](https://arxiv.org/abs/2411.00459)  
**代码**: [GitHub](https://github.com/LukeChen-go/pia-defense-by-attack)  
**机构**: NUS & HKUST & HIT-Shenzhen
**领域**: AI安全  
**关键词**: prompt injection, defense, attack techniques, shield prompt, fake completion, LLM security

## 一句话总结

本文提出一种"以攻为防"的 prompt injection 防御策略：将已有的攻击技术（ignore、escape、fake completion）反转用于防御，在被注入的数据内容后追加 shield prompt + 原始指令，使 LLM 忽略注入指令而执行原始指令，在多种攻击场景下将 ASR 降至接近零。

## 研究背景与动机

### 现状

- LLM 被广泛集成到 Microsoft Copilot、Perplexity.ai 等应用中，通过外部工具检索数据
- 这些应用中，攻击者可将恶意指令注入到检索结果等外部数据内容中（间接 prompt injection）
- LLM 由于强大的指令跟随能力和无法区分原始指令与注入指令，容易被误导执行恶意操作
- OWASP 已将 prompt injection 列为 LLM 应用的 #1 安全风险

### 痛点

- 现有防御方法分为 fine-tuning 和 prompt engineering 两类
- Fine-tuning 方法需要标注数据和大量计算资源（如 StruQ、Instruction Hierarchy）
- Prompt engineering 方法（如 Sandwich、Instructional reminder）虽然免训练，但防御效果有限
- 攻击者利用 ignore prompt、escape characters、fake completion 等手段可轻松突破现有防御

### 核心洞察

- 攻击和防御有着相似的设计目标：都是让 LLM 忽略不想要的指令、执行想要的指令
- 攻击方让 LLM 忽略原始指令、执行注入指令；防御方让 LLM 忽略注入指令、执行原始指令
- 因此，高效的攻击技术可以被"反转意图"，直接用于设计更强的防御方法

## 方法详解

### 整体框架

给定输入指令 I、干净数据 D 和被注入的恶意 prompt P，防御方法在 poisoned data (D⊕P) 之后追加一个 shield prompt S 和原始输入指令 I 的副本，使得 M(I⊕D⊕P⊕S⊕I) = R^b（正常响应），同时不干扰干净数据的推理。

### 关键设计 1：Ignore Defense

- 灵感来源：Ignore Attack 通过 "Ignore all previous instructions" 让 LLM 忽略原始指令
- 防御策略：在 poisoned data 后也追加 ignore prompt 作为 shield prompt，指示 LLM 忽略所有先前指令（包括原始和注入的），然后附上原始输入指令
- shield prompt 可以设计得比示例更有说服力

### 关键设计 2：Escape Defense

- 灵感来源：Escape-Deletion Attack 使用 "\b"、"\r" 等特殊字符模拟删除先前内容
- 防御策略：在 poisoned data 后追加 "\b" 和 "\t" 字符模拟删除注入指令，然后附上原始输入指令
- 如果删除模拟生效，可以有效"擦除"注入的恶意指令

### 关键设计 3：Fake Completion Defense（含模板变体）

- 基础版本：伪造一个对最后指令的响应（如 "### Response: OK"），让 LLM 以为注入指令已执行完毕，然后只跟随附加的原始输入指令
- 模板增强版（Fakecom-t）：利用对话模板知识进行多轮对话模拟
  - 先模拟 assistant 角色报告检测到 prompt injection 攻击
  - assistant 拒绝并不信任所有先前指令
  - 再模拟 user 角色确认原始输入指令
  - 这是最强的防御变体，因为利用了真实的对话结构

### 训练策略

- 所有防御方法均为 training-free，不需要微调任何模型
- 仅需知道使用的攻击方法类型来选择对应的防御策略
- 可以与任何 LLM 直接配合使用

## 实验关键数据

### 主实验：直接 Prompt Injection 防御（Table 1）

| 防御方法 | Llama3（5种攻击平均 ASR↓） | Qwen2（平均 ASR↓） | Llama3.1（平均 ASR↓） |
|---------|-------------------------|-------------------|---------------------|
| None | ~63% | ~90% | ~70% |
| Sandwich | ~30% | ~45% | ~29% |
| Instructional | ~36% | ~88% | ~50% |
| Ours-Ignore | ~15% | ~11% | ~10% |
| Ours-Fakecom-t | **~5%** | **~7%** | **~5%** |

- Fakecom-t 在 Qwen2 上将 Combined Attack ASR 从 100% 降至 2.40%
- Fakecom-t 在 Fakecom Attack 上将 ASR 降至 0.0%（Llama3）

### 消融实验：间接 Prompt Injection 防御

| 防御方法 | Llama3 平均 ASR↓ | Qwen2 平均 ASR↓ | Llama3.1 平均 ASR↓ |
|---------|-----------------|----------------|-------------------|
| None | ~28% | ~44% | ~35% |
| Ours-Fakecom-t | **~5%** | **~5%** | **~5%** |

### 关键发现

1. **最强攻击 → 最强防御**：基于最有效攻击技术（Fake Completion with Template）设计的防御方法表现最好
2. 在某些场景下 ASR 降至 0%，远超现有 training-free 方法
3. 防御方法对模型常规任务的准确率影响极小（QA 任务和情感分析准确率基本保持）
4. 方法对未知攻击类型也有较好的泛化能力

## 亮点与洞察

- **哲学层面的创新**：揭示了攻击和防御的对偶关系——相同的技术手段可以通过反转意图实现截然不同的目标
- **极简但有效**：不需要训练、不需要额外数据、不需要修改模型，仅靠在 prompt 中追加防御文本就能大幅提升安全性
- 多轮对话模拟防御（Fakecom-t）特别巧妙：利用 LLM 对对话结构的敏感性来强化防御
- 为 LLM 应用的部署提供了即插即用的安全方案

## 局限性 / 可改进方向

- 防御方法的选择依赖于对攻击类型的先验知识（虽然有泛化能力，但最优防御需要匹配）
- Escape Defense 的效果不稳定，依赖于 LLM 对特殊字符的处理方式
- 面对基于梯度优化的攻击（GCG 等），纯 prompt engineering 方法可能存在天花板
- 未考虑攻击者也可能感知到防御并进行对抗性调整的场景

## 相关工作与启发

- 与 StruQ (Chen et al., 2024) 的 fine-tuning 防御方法互补：本文方法免训练但依赖 prompt 设计，StruQ 需要训练但更鲁棒
- Ignore Attack (Perez & Ribeiro, 2022) 和 Fake Completion Attack (Willison, 2023) 是本文防御方法的直接灵感来源
- 启发：其他安全领域（如对抗攻击防御）是否也能通过"反转攻击技术"来设计防御？

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 攻防对偶的视角非常新颖
- **技术深度**: ⭐⭐⭐ — 方法本身较简单，但实验全面
- **实用性**: ⭐⭐⭐⭐⭐ — 即插即用，直接可部署
- **实验充分度**: ⭐⭐⭐⭐ — 多模型、多攻击类型、直接+间接场景
