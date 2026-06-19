---
title: >-
  [论文解读] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents
description: >-
  [NeurIPS 2025][多模态VLM][adversarial attack] 揭示针对多模态OS Agent的新型对抗攻击MIP(Malicious Image Patches)：在屏幕截图中嵌入人眼不可察觉的对抗性扰动图像块(约占屏幕1/7面积)，当OS Agent截屏捕获后会输出预定义的恶意API调用序列；通过联合优化实现跨用户指令和屏幕布局的Universal泛化，攻击成功率高达100%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "adversarial attack"
  - "OS agent"
  - "malicious image patch"
  - "VLM security"
  - "computer worm"
---

# MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents

**会议**: NeurIPS 2025  
**arXiv**: [2503.10809](https://arxiv.org/abs/2503.10809)  
**代码**: [GitHub](https://github.com/AIchberger/mip-against-agent)  
**领域**: 机器人  
**关键词**: adversarial attack, OS agent, malicious image patch, VLM security, computer worm

## 一句话总结

揭示针对多模态OS Agent的新型对抗攻击MIP(Malicious Image Patches)：在屏幕截图中嵌入人眼不可察觉的对抗性扰动图像块(约占屏幕1/7面积)，当OS Agent截屏捕获后会输出预定义的恶意API调用序列；通过联合优化实现跨用户指令和屏幕布局的Universal泛化，攻击成功率高达100%。

## 研究背景与动机

**领域现状**：OS Agent（如Claude Computer Use、Windows Agent Arena中的Agent）将VLM从被动文本输出者升级为主动计算机控制者——可以执行鼠标点击、键盘输入、文件操作、网络请求等。这种能力转变使VLM的安全风险从"输出有害文本"升级为"执行有害操作"。

**现有痛点**：已有研究发现OS Agent可被文本注入(prompt injection)或弹窗(pop-up)攻击。但这些方法需要直接访问Agent的文本输入管道，且容易被现有过滤机制检测和阻断。OS Agent依赖**截屏**进行导航的特性提供了全新的视觉域攻击面——目前对此研究极为有限。

**核心矛盾**：OS Agent必须通过截屏观察环境→安全风险；攻击者只需控制屏幕上一小块区域（社交媒体图片、桌面壁纸）就可能劫持整个Agent→检测困难。

**本文目标** 系统研究视觉域对OS Agent的攻击：能否通过操控屏幕上的小区域图像就劫持Agent执行任意恶意行为？这种攻击能否跨场景泛化？

**切入角度**：将传统VLM对抗攻击扩展到OS Agent的多组件pipeline,处理screen parser不可微、图像resize、离散像素等特有约束。

**核心 idea**：MIP将完整的恶意程序指令编码在一个视觉上不可察觉的图像patch中，OS Agent截屏处理后会直接输出并执行该恶意程序——无需依赖Agent自身的推理能力来组装攻击。

## 方法详解

### 整体框架

攻击pipeline：(1) 对目标VLM用PGD优化图像patch区域内的像素扰动 → (2) 将优化好的MIP嵌入桌面壁纸或社交媒体帖子 → (3) OS Agent截屏时捕获MIP → (4) VLM处理含MIP的截图后输出预定义的恶意target $\mathbf{y}$（包含完整API调用） → (5) Agent执行恶意行为。

### 关键设计

1. **多约束PGD优化**:
    - 功能：在OS Agent的多组件pipeline约束下优化对抗扰动
    - 核心思路：目标函数 $\boldsymbol{\delta}^* = \arg\min_{\mathcal{R}, \boldsymbol{\delta} \in \Delta_\mathcal{R}^\epsilon} \mathcal{L}(f_{\boldsymbol{\theta}}(\mathbf{p}_{txt}, q(l(\mathbf{s}, \mathbf{s}_{som}) + \boldsymbol{\delta})), \mathbf{y})$，需满足：扰动限制在patch区域$\mathcal{R}$内、$\ell_\infty \leq \epsilon=25/255$、不改变screen parser的SOM检测结果、离散整数像素、图像resize用可微近似替代。每步PGD后投影回合法集合
    - 设计动机：OS Agent不是简单的VLM——它有screen parser(不可微)、resize(损失信息)、API解析等额外组件，攻击必须全部绕过

2. **Universal MIP泛化优化**:
    - 功能：让单个MIP对多种用户指令和屏幕布局都有效
    - 核心思路：从targeted（单一prompt+screenshot对优化）扩展到universal——每步PGD随机采样batch=8对$(p,s) \sim \text{Uniform}(\mathcal{P}_+ \times \mathcal{S}_+)$做联合更新，优化到所有训练对上malicious target概率超99%。评估在未见的prompt集$\mathcal{P}_-$和screenshot集$\mathcal{S}_-$上的泛化
    - 设计动机：现实中攻击者无法预知受害者的具体指令和屏幕状态，Universal MIP是实际可部署的前提

3. **直接编码 vs 间接引导**:
    - 功能：将完整恶意程序直接编码到MIP中，而非间接引导Agent推理
    - 核心思路：target output $\mathbf{y}$包含完整的API调用序列（33-52 token），如打开终端执行内存溢出或导航到恶意网站。一旦VLM输出$\mathbf{y}$，Agent立即通过API执行，不依赖Agent自身能力
    - 设计动机：间接攻击（让Agent"组装"恶意行为）引入额外失败点——Agent可能被劫持但无法正确构造恶意程序；直接编码保证一旦触发就必然执行

## 实验关键数据

### 主实验

| 设置 | 恶意行为 | 训练对ASR | 未见prompt ASR | 未见屏幕ASR |
|------|---------|----------|---------------|------------|
| Desktop Targeted | 内存溢出(33 token) | 1.00 | 0.91 | 0.00 |
| Desktop Universal | 内存溢出 | ~1.00 | ~0.90 | ~0.80 |
| Desktop Universal | 恶意网站(52 token) | ~0.90 | ~0.80 | ~0.70 |
| Social Media Universal | 内存溢出 | ~1.00 | ~0.85 | ~0.75 |
| Social Media Universal | 恶意网站 | ~0.90 | ~0.70 | ~0.60 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Targeted vs Universal | ASR on unseen | Universal显著提升跨场景泛化 |
| 不同Screen Parser | OmniParser vs GroundingDINO | MIP可跨parser泛化(ASR ~0.5-0.7) |
| 不同VLM (11B vs 90B) | 跨模型ASR | 在11B上优化的MIP可攻击90B模型 |
| Agent执行中捕获 | 多步交互后ASR | MIP在Agent执行正常任务时仍有效 |

### 关键发现
- Targeted MIP可跨prompt泛化但不跨screenshot泛化(ASR=0)，Universal解决此问题
- MIP可跨screen parser和VLM大小泛化——在Llama-11B上优化的MIP能攻击90B模型
- 即使Agent已执行若干正常步骤后才遇到MIP，攻击仍然有效
- Desktop场景比Social Media场景更易攻击（SOM元素更少，文本上下文更短）

## 亮点与洞察
- "OS Agent计算机蠕虫"概念：如果恶意行为包括分享含MIP的帖子，MIP可自传播——首次提出这种自动传播的OS Agent攻击
- 直接编码策略绕过了Agent推理能力的不确定性，一旦触发就保证执行
- 跨VLM大小的泛化(11B→90B)暗示VLM视觉处理存在共享的系统性弱点
- 攻击检测极其困难——MIP视觉上与正常图像无异，且不依赖文本管道

## 局限与展望
- 需要白盒访问VLM参数（PGD需要梯度），黑盒迁移攻击未充分探索
- $\epsilon=25/255$的扰动在放大观察时可能可见，更隐蔽的攻击有待研究
- 仅在Windows Agent Arena上评估，其他OS Agent框架（如Claude Computer Use）未测试
- 防御方法（如对抗训练、输入净化）的有效性未充分讨论

## 相关工作与启发
- **vs Pop-up攻击(Zhang et al.)**: Pop-up依赖可见弹窗且文本可被过滤；MIP视觉不可察觉且绕过文本过滤
- **vs VLM对抗攻击(Bailey et al.)**: Bailey等人攻击有tool-use能力的VLM但直接输入对抗图像；MIP需要通过截屏间接传递，面临更多约束
- **vs Wu et al.**: 他们攻击captioning模型间接引导Agent；MIP直接攻击VLM决策，更可靠

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究视觉域OS Agent攻击，提出"Agent蠕虫"概念
- 实验充分度: ⭐⭐⭐⭐ 两种设置、两种行为、跨parser/模型/prompt泛化的全面评估
- 写作质量: ⭐⭐⭐⭐ 形式化清晰，约束描述精确
- 价值: ⭐⭐⭐⭐⭐ 对OS Agent安全有重大警示意义，在Agent大规模部署前必须解决

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](../../ACL2025/multimodal_vlm/agent_rewardbench.md)
- [\[ACL 2025\] MMINA: Benchmarking Multihop Multimodal Internet Agents](../../ACL2025/multimodal_vlm/mmina_benchmarking_multihop_multimodal_internet_agents.md)
- [\[ICML 2025\] Robust Multimodal Large Language Models Against Modality Conflict](../../ICML2025/multimodal_vlm/robust_multimodal_large_language_models_against_modality_conflict.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[CVPR 2025\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](../../CVPR2025/multimodal_vlm/from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)

</div>

<!-- RELATED:END -->
