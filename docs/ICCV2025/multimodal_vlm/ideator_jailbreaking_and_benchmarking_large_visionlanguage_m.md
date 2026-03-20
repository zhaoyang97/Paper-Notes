# IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves

**会议**: ICCV 2025  
**arXiv**: [2411.00827](https://arxiv.org/abs/2411.00827)  
**代码**: [https://github.com/roywang021/IDEATOR](https://github.com/roywang021/IDEATOR)  
**领域**: 多模态VLM / AI安全 / 对抗攻击  
**关键词**: jailbreak attack, VLM safety, red teaming, multimodal attack, safety benchmark  

## 一句话总结
提出IDEATOR，首个用VLM自身做红队攻击VLM的黑盒越狱框架——利用一个弱安全对齐的VLM（MiniGPT-4）作为攻击者，结合Stable Diffusion生成语义丰富的图文越狱对，通过breadth-depth探索策略迭代优化，在MiniGPT-4上达94%攻击成功率（平均5.34次查询），迁移到LLaVA/InstructBLIP/Chameleon达75-88%，并构建VLJailbreakBench（3654样本）揭示11个VLM的安全漏洞。

## 研究背景与动机
1. **领域现状**：VLM越狱攻击主要分为白盒（GCG、VAJM等需要梯度访问）和黑盒（MM-SafetyBench等依赖手工模板）。白盒方法不实际（无法访问商业模型内部），黑盒方法依赖人工设计的攻击模板（如typographic attack），缺乏多样性和灵活性。
2. **现有痛点**：(1) 白盒攻击生成的对抗图像无语义（噪声pattern），易被安全机制检测；(2) MM-SafetyBench等黑盒方法需要人工设计pipeline，可扩展性差；(3) 现有安全benchmark多用显式有害内容，很少测试复杂多模态越狱场景；(4) 缺少能自动化、大规模生成多样越狱样本的工具。
3. **核心矛盾**：有效的越狱需要"上下文丰富且语义隐蔽"的图文组合，但自动生成这样的多模态攻击极其困难——既要有攻击性又要有隐蔽性。
4. **本文要解决什么**：构建一个完全自动化的黑盒VLM越狱框架，无需白盒访问/人工模板/训练，能生成语义丰富的多模态越狱样本并大规模评估VLM安全性。
5. **切入角度**：VLM本身就有强大的内容理解和生成能力——如果解除安全约束，VLM可以成为最强的红队工具。用MiniGPT-4（安全约束较弱的开源VLM）做攻击者，迭代分析受害VLM的响应并优化攻击策略。
6. **核心idea一句话**：用VLM攻击VLM——弱安全对齐的VLM作为红队模型自主生成图文越狱对，通过breadth-depth迭代探索多种攻击策略。

## 方法详解

### 整体框架
攻击者VLM $\mathcal{M}_\mathcal{A}$（MiniGPT-4）接收越狱目标$\mathcal{G}$ → 生成JSON输出{analysis, text_prompt, image_prompt} → Stable Diffusion 3根据image_prompt生成图像 → 图文组合发给受害VLM $\mathcal{M}_\mathcal{V}$ → 受害者响应$\mathcal{R}$回传给攻击者 → 攻击者分析响应并优化策略（CoT推理） → 迭代直到成功或达到最大轮次。Breadth探索多条独立攻击路径，Depth在每条路径上迭代优化。

### 关键设计

1. **VLM作为红队模型**:
   - 做什么：MiniGPT-4（Vicuna-13B）作为攻击者VLM，通过精心设计的system prompt模拟对抗者行为。
   - 核心思路：系统prompt指定三个角色：(1) 红队助手——生成越狱prompt；(2) JSON格式约束——输出analysis/image_prompt/text_prompt三个字段；(3) 上下文学习——提供攻击范例指导策略。
   - 设计动机：Vicuna比LLaMA更宽松（更少safety拒绝），MiniGPT-4的开源性允许自定义system prompt。VLM的预训练知识让它能生成语义丰富、上下文合理的攻击——远比模板化攻击更隐蔽。

2. **Breadth-Depth探索策略**:
   - 做什么：Breadth=$N_b$条独立攻击路径（不同初始策略），每条Depth=$N_d$轮迭代优化。
   - 核心思路：Breadth保证策略多样性（角色扮演、情感操纵、学术场景等），Depth保证每种策略充分优化（根据victim反馈调整）。默认$N_b=7, N_d=3$，即21次查询。
   - 效果：$N_b=1,N_d=1$ → 45% ASR；$N_b=7,N_d=3$ → 94% ASR。单增breadth或depth效果有限，联合提升效果最显著。
   - 设计动机：单一攻击策略容易被特定防御机制阻挡。多策略并行+迭代优化能更全面地探索VLM的漏洞空间。

3. **Chain-of-Thought攻击分析**:
   - 做什么：在JSON的analysis字段中，攻击者VLM分析上一轮victim的拒绝原因并提出改进策略。
   - 核心思路：CoT让攻击者能学习受害者的"拒绝模式"——比如"这种直接请求会被拒绝，改用角色扮演场景"或"文字攻击被检测到，将有害内容转移到图片中"。
   - 设计动机：模拟人类红队测试者的思维过程——分析失败原因、调整策略、尝试新角度。这是IDEATOR能后续优化攻击的核心机制。

4. **VLJailbreakBench构建**:
   - 3654个多模态越狱样本，覆盖12个安全主题+46个子类别
   - Base set（916样本）：MiniGPT-4攻击LLaVA-1.5
   - Challenge set（2738样本）：Gemini-1.5-Pro攻击GPT-4o-mini（更强攻击者+更强防御者=更高质量样本）
   - 11个VLM评估结果：Claude-3.5-Sonnet最安全（19.65% ASR），GPT-4o Mini最易攻破（72.21%）

## 实验关键数据

### 攻击效果（MiniGPT-4为受害模型，AdvBench 100样本）

| 方法 | 黑盒 | 免训 | ASR% |
|------|------|------|------|
| 无攻击 | - | - | 35.0 |
| GCG（白盒文本） | ✗ | ✗ | 50.0 |
| GCG-V（白盒视觉） | ✗ | ✗ | 85.0 |
| UMK（白盒多模态） | ✗ | ✗ | 94.0 |
| MM-SafetyBench（黑盒） | ✓ | ✓ | 66.0 |
| **IDEATOR（黑盒）** | **✓** | **✓** | **94.0** |

### 跨模型迁移

| ASR% | LLaVA | InstructBLIP | Chameleon |
|------|-------|-------------|----------|
| 无攻击 | 7.0 | 12.0 | 16.0 |
| MM-SafetyBench | 46.0 | 29.0 | 22.0 |
| **IDEATOR** | **82.0** | **88.0** | **75.0** |

### VLJailbreakBench Challenge Set (11模型)

| 模型 | 平均ASR% | 类型 |
|------|---------|------|
| GPT-4o Mini | 72.21 | 商业 |
| Gemini-2.0-Flash-Think | 71.44 | 商业 |
| Qwen2-VL | 71.40 | 开源 |
| GPT-4o | 46.31 | 商业 |
| **Claude-3.5-Sonnet** | **19.65** | 商业 |

### 消融

| Nb | Nd=1 | Nd=3 |
|----|------|------|
| 1 | 45% | 68% |
| 7 | 85% | **94%** |

| 攻击模态 | ASR% | 平均查询数 |
|---------|------|---------|
| 仅图像 | 85% | 5.84 |
| 仅文本 | 86% | 7.46 |
| **图文联合** | **94%** | **5.34** |

### 关键发现
- **黑盒IDEATOR达到白盒SOTA水平**（94% vs UMK 94%），远超其他黑盒方法（+28% vs MM-SafetyBench）。
- **迁移性极强**：从MiniGPT-4上生成的越狱样本直接迁移到LLaVA达82%——比MM-SafetyBench高36%。
- **图文联合攻击最有效且最高效**：94% ASR仅需5.34次查询（<1分钟），而文本-only需要7.46次。图像有两个独特价值：隐藏有害内容+增强角色扮演场景。
- **现有防御极度不足**：AdaShield-S对IDEATOR的防御效果有限（ASR 94%→84%），而对FigStep/MM-SafetyBench很有效（-32%/-29%）。IDEATOR攻击策略的多样性使其天然抗防御。
- **Claude-3.5-Sonnet最安全但仍有19.65% ASR**——约每5-6次尝试就有一次成功。这比之前benchmark报告的安全率低得多，说明需要adversarial benchmark评估。
- IDEATOR能自动产生比MM-SafetyBench更丰富的攻击策略谱（typographic、roleplay、emotional manipulation等），本质上$\mathcal{A}_{IDEATOR} \supseteq \bigcup_i \mathcal{A}_i$。

## 亮点与洞察
- **"用VLM攻击VLM"是一个深刻的安全洞察**：VLM的强大能力是双刃剑——同样的多模态理解和生成能力可以被用来构造越狱攻击。最弱的安全对齐成为最大的风险——任何开源的弱对齐VLM都可能成为攻击工具。
- **Breadth-Depth探索的博弈论视角**：将越狱攻击建模为攻击者与防御者的multi-round game，攻击者通过分析防御者的拒绝模式来调整策略。这比一次性攻击更贴近真实的安全威胁场景。
- **VLJailbreakBench的差异化价值**：之前的benchmark多用显式有害文本+模板图像，IDEATOR生成的样本在语义上更隐蔽——揭示了现有safety alignment在面对复杂场景时的脆弱性。GPT-4o Mini 72%的ASR消除了商业模型"足够安全"的假象。

## 局限性 / 可改进方向
- 攻击者VLM的选择受限——需要安全约束较弱的模型，但随着safety alignment提升，合适的攻击者模型会减少。
- VLJailbreakBench规模较小（3654样本），扩展需要更多计算资源。
- 未探索对视频VLM和多轮对话VLM的攻击。
- 可能被用于生成有害内容——论文已添加Disclaimer和ethical考虑。

## 相关工作与启发
- **vs MM-SafetyBench**: MM-SafetyBench依赖手工模板（query-relevant image + typographic），IDEATOR完全自动化且ASR高28%。迁移性差距更大——82% vs 46%在LLaVA上。
- **vs UMK (白盒)**: UMK需要梯度访问且对抗perturbation无语义，IDEATOR是黑盒且图像有丰富语义。ASR相当但适用范围完全不同。
- **vs Arondight**: Arondight需要训练红队LLM，IDEATOR是training-free——直接通过system prompt和in-context learning让现有VLM充当攻击者。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个VLM-as-red-team框架，breadth-depth探索策略新颖，VLJailbreakBench填补多模态安全benchmark空白
- 实验充分度: ⭐⭐⭐⭐⭐ 5种baseline comparison，4个受害模型，11个benchmark模型，消融（breadth/depth/模态），防御评估（AdaShield），大量可视化
- 写作质量: ⭐⭐⭐⭐ 清晰，threat model明确，Figure 1的对比直观
- 价值: ⭐⭐⭐⭐⭐ 对VLM安全领域有重大意义——证明了VLM自身可以成为最强的红队工具，VLJailbreakBench揭示了商业模型的真实安全水平
