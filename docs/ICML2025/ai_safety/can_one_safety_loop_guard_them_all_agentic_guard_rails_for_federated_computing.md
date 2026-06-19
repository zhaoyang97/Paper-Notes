---
title: >-
  [论文解读] Can One Safety Loop Guard Them All? Agentic Guard Rails for Federated Computing
description: >-
  [ICML 2025][AI安全][federated computing] 提出 Guardian-FC——首个后端无关的联邦计算统一安全框架，通过 Agentic-AI 控制平面的有限状态安全循环（Sense→Predict→Act→Prove）统一监管 FHE、DP、MPC 等异构隐私机制，实现一套 guard-rail 逻辑跨所有隐私后端的一致性安全执行。
tags:
  - "ICML 2025"
  - "AI安全"
  - "federated computing"
  - "guard rails"
  - "privacy-preserving"
  - "FHE"
  - "DP"
  - "MPC"
  - "finite-state machine"
---

# Can One Safety Loop Guard Them All? Agentic Guard Rails for Federated Computing

**会议**: ICML 2025  
**作者**: Narasimha Raghavan Veeraragavan, Jan Franz Nygård  
**arXiv**: [2506.20000](https://arxiv.org/abs/2506.20000)  
**代码**: 未公开  
**领域**: AI 安全, 联邦计算, 隐私保护  
**关键词**: federated computing, guard rails, privacy-preserving, FHE, DP, MPC, finite-state machine  

## 一句话总结

提出 Guardian-FC——首个后端无关的联邦计算统一安全框架，通过 Agentic-AI 控制平面的有限状态安全循环（Sense→Predict→Act→Prove）统一监管 FHE、DP、MPC 等异构隐私机制，实现一套 guard-rail 逻辑跨所有隐私后端的一致性安全执行。

## 研究背景与动机

**领域现状**：联邦计算允许多机构在不共享原始数据的情况下协作训练和分析，是隐私保护的核心范式。常用的隐私技术包括全同态加密（FHE，跟踪噪声预算）、差分隐私（DP，跟踪 ε 预算）和安全多方计算（MPC，跟踪份额完整性），每种技术有各自独立的安全指标和检查机制。

**现有痛点**：当前每种隐私后端都需要定制化的安全监控、控制逻辑和审计工具，导致：(1) **碎片化**——开发团队需要为每种后端分别构建和维护安全框架，工作大量重复；(2) **互操作性差**——混合隐私后端的工作流（如 FHE→MPC→DP 的联邦 Kaplan-Meier 生存曲线计算）缺乏统一安全框架，无法跨阶段协调安全策略；(3) **一致性风险**——异构系统中安全策略执行不一致可能导致隐私泄露或系统故障。

**核心矛盾**：隐私机制的多样性要求灵活性，但安全执行需要一致性——如何在不修改核心安全逻辑的情况下支持新的隐私后端？

**切入角度**：将安全 guard-rail 与隐私机制解耦，在两者之间插入一层后端无关的领域特定语言（DSL），使同一套安全循环逻辑可跨所有隐私后端运行。

**核心 idea**：用 DSL 抽象隐私操作、用 Agentic-AI 安全循环执行统一 guard-rail，实现"一个安全环路守护所有隐私后端"。

## 方法详解

### 整体框架

Guardian-FC 采用两层架构：上层为 **Agentic-AI 控制平面**（只处理签名元数据，不接触原始数据），下层为**联邦数据平面**（节点和中央聚合器执行隐私保护计算）。两层通过认证通道交互——上行传输签名遥测帧，下行传输签名控制命令。联邦计算逻辑用后端无关的 DSL 编写为 plug-in，运行时动态绑定到可互换的执行提供者（EP）。

### 关键设计

1. **后端无关的领域特定语言（DSL）与执行提供者（EP）**:

    - 功能：解耦计算逻辑与隐私实现
    - 核心思路：数据科学家用 DSL 编写 plug-in（模块化计算单元），定义"做什么"但不涉及隐私机制细节。EP 实现 DSL 操作的具体语义——FHE-EP 做加密计算、DP-EP 做噪声添加、MPC-EP 做秘密共享。同一 plug-in 代码通过更换 EP 可在不同隐私后端上运行而无需修改。编译时生成 manifest（JSON 结构化描述文件），指定 plug-in 名、DSL 操作、选择的 EP、启用的 guard-rail 谓词
    - 设计动机：后端无关性是实现统一安全循环的前提——只有当计算描述与隐私实现分离，安全逻辑才能不关心底层技术细节

2. **Agentic-AI 有限状态安全循环**:

    - 功能：以固定频率（如 1Hz）持续监控和执行安全策略
    - 核心思路：安全循环由四个阶段构成，周期性执行：
        - **Sense**：遥测收集器从所有节点和聚合器收集签名度量帧（噪声预算、隐私损失、延迟等）
        - **Predict**：哨兵对齐遥测流并在当前和预测度量上评估 guard-rail 谓词（布尔函数 $p_i: \{\mathbf{m}, \hat{\mathbf{m}}\} \to \{true, false\}$），预测即将到来的安全违规
        - **Act**：控制引擎通过密码编排器发出签名 A-command（如 A-BOOTSTRAP 重置噪声预算、A-ABORT_JOB 终止作业、A-ISOLATE_PARTY 隔离故障节点）
        - **Prove**：审计引擎将所有命令和确认追加到仅追加 Merkle 账本，提供防篡改的合规审计证据
    - 设计动机：有限状态安全循环是形式化验证的基础——状态空间足够小可以进行模型检查，确保安全不变量可被证明

3. **Manifest 驱动的准入验证**:

    - 功能：在作业执行前快速拒绝不兼容的配置
    - 核心思路：作业提交时附带 manifest，控制引擎执行 fail-fast 准入检查：验证 plug-in 的每个 DSL 操作码都被选定 EP 实现，确认所有启用的 guard-rail 谓词引用的度量键由 EP 提供。通过检查后，manifest 广播到所有节点，遥测收集器拒绝任何偏离 manifest 约定的遥测帧
    - 设计动机：前置验证防止资源浪费和运行时安全违规；manifest 作为权威契约确保所有组件共享一致配置

### 损失函数

本工作为系统框架设计，不涉及模型训练。

## 实验

### 主实验——三个定性安全场景

| 场景 | 隐私后端 | 触发谓词 | 安全行动 | 结果 |
|------|---------|---------|---------|------|
| A: CKKS 噪声耗尽 | FHE | $p_1$: noiseBits < $\theta_{fhe}$ | A-BOOTSTRAP（重置噪声预算） | 本地修复，计算继续 |
| B: DP 预算溢出 | DP | $p_2$: $\varepsilon_{spent}$ > $\varepsilon_{max}$ | A-ABORT_JOB（终止作业） | 安全清除种子和份额 |
| C: 恶意 MPC 份额 | MPC | $p_3$: shareAuthFail ≥ 2 | A-ISOLATE_PARTY（隔离故障方） | 剩余份额合并完成作业 |

### 形式化安全属性验证

| 属性 | 形式化表述 | 含义 |
|------|-----------|------|
| 安全性 | Aggregator=FINALIZE ⟹ (∀i: Node[i]∈$S_{ok}$) ∧ (∀p∈P: ¬p) | 作业完成时所有节点安全且无谓词违规 |
| 活性 | ◇(μ=0)，ranking 函数 μ 单调递减 | 系统不会永远停留在中间状态，必然到达终止 |

### 关键发现

- 三个场景中安全循环的核心逻辑（Sense→Predict→Act→Prove）、组件边界和审计语义完全不变——只有触发的谓词和相应 A-command 不同
- manifest 驱动的准入验证实现了编译时安全检查，将不兼容问题前移到执行前
- 有限状态机的状态空间被刻意控制在较小范围，为模型检查（model checking）创造了可行条件
- 扩展新隐私后端只需实现新 EP 模块——无需修改控制平面或 guard-rail 逻辑

## 亮点与洞察

- 首次提出跨 FHE/DP/MPC 的统一安全框架，解决了联邦计算中安全执行碎片化的根本问题
- 两层架构设计——控制平面只处理签名元数据不接触原始数据，本身满足隐私约束
- DSL + EP 的解耦设计兼具灵活性和一致性，为新隐私技术（如函数加密、可信执行环境）的集成预留了简洁的扩展路径
- 形式化安全属性（安全性 + 活性）为未来的模型检查验证奠定了理论基础

## 局限性

- 目前仅通过定性场景演示而非定量实验验证，缺乏原型系统的性能评测数据
- 1Hz 遥测频率是否足以捕获所有安全事件未经验证——高吞吐场景可能需要更高频率
- DSL 的表达能力和具体规范尚未完成定义，仍处于概念阶段
- 假设所有节点诚实地发送签名遥测——恶意节点伪造遥测的问题未讨论
- guard-rail 谓词的阈值（如 $\theta_{fhe}$、$\varepsilon_{max}$）固定为静态值，缺乏自适应调节策略
- 未提供运行时开销估计或与现有安全框架的对比

## 相关工作与启发

- **联邦学习安全**：现有工作多关注单一隐私后端的安全（如 DP 的隐私会计、FHE 的噪声管理），Guardian-FC 试图在更高抽象层统一管理
- **Agentic AI 安全**：将 AI agent 的"感知-决策-行动"循环应用于系统安全监控是一种新颖方向
- **启发**：框架的 DSL 设计可能影响联邦计算标准化的发展方向——如果社区能统一 DSL 规范，将大幅降低联邦计算系统的开发成本
- **开放研究方向**：论文提出了 RL 自适应阈值调节、多 EP 组合的类型化 (ε,δ,λ)-calculus、人机交互覆盖等有价值的未来方向

## 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次提出后端无关的统一联邦计算安全框架 |
| 技术深度 | ⭐⭐⭐ | 形式化模型有基础但无实现验证，DSL 规范不完整 |
| 实验完整度 | ⭐⭐ | 仅定性场景演示，无定量性能评测 |
| 写作质量 | ⭐⭐⭐⭐ | 架构描述清晰，形式化表述规范 |
| 实用性 | ⭐⭐⭐ | 概念前瞻但离实际部署尚远，需要原型验证 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](../../ICML2026/ai_safety/one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)
- [\[ICML 2026\] OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL](../../ICML2026/ai_safety/omnivl-guard_towards_unified_vision-language_forgery_detection_and_grounding_via.md)
- [\[ICML 2025\] Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)
- [\[ICML 2025\] Generalization in Federated Learning: A Conditional Mutual Information Framework](generalization_in_federated_learning_a_conditional_mutual_information_framework.md)
- [\[ICML 2025\] Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models](theoretically_unmasking_inference_attacks_against_ldp-protected_clients_in_feder.md)

</div>

<!-- RELATED:END -->
