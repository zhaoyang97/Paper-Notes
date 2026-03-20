# Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models

**会议**: ACL 2025  
**arXiv**: [2507.18263](https://arxiv.org/abs/2507.18263)  
**代码**: [https://github.com/DeepLearnXMU/Locate_and_Focus_ST](https://github.com/DeepLearnXMU/Locate_and_Focus_ST)  
**领域**: 语音翻译 / 术语翻译  
**关键词**: speech translation, terminology translation, speech LLM, sliding retrieval, multi-modal knowledge

## 一句话总结

提出Locate-and-Focus方法用于语音LLM的术语翻译：先用滑动窗口检索定位语音中包含术语的片段，再通过音频替换和Tag Cue引导模型聚焦翻译知识，在英中/英德方向上术语翻译成功率大幅提升。

## 研究背景与动机
1. **领域现状**: 端到端语音翻译（ST）在术语翻译上表现不佳，术语（人名、药名等）的正确翻译对信息传递至关重要。
2. **现有痛点**: Collect-and-Integrate范式引入所有语料术语导致大量无关信息；Retrieve-and-Demonstrate范式检索的示例包含与术语翻译无关的句子部分。
3. **核心矛盾**: 外部翻译知识虽有帮助，但引入方式粗糙——文本模态与音频模态不匹配，检索示例来自不同说话人，导致ST模型难以充分利用。
4. **本文要解决什么**: 精确定位语音中的术语片段，并有效引导语音LLM聚焦翻译知识。
5. **切入角度**: 两步走——先"定位"（滑动窗口检索术语对应的语音片段），再"聚焦"（音频替换建立共享锚点+Tag标签提醒翻译）。
6. **核心idea一句话**: 在语音中定位术语片段并替换到翻译知识中建立共享音频锚点，再用\<Term\>标签提醒模型聚焦术语翻译。

## 方法详解
### 整体框架
分为术语片段定位（Terminology Clip Localization）和术语聚焦翻译（Terminology-Focused Translation）两个步骤。前者通过滑动窗口检索在语音中定位术语对应的音频片段，后者通过音频替换和Tag Cue两种策略引导语音LLM聚焦翻译知识。语音编码器用对比学习训练，翻译模型用LoRA微调。

### 关键设计
1. **Sliding Retrieval（滑动检索）**: 用语音编码器对术语片段c和语音u编码，以c的长度为窗口大小、步长为1滑动u，计算每个子序列与c的max-pooling余弦相似度，取最大值作为术语出现概率，同时定位对应语音片段s。
2. **Audio Replacement（音频替换）**: 将检索到的翻译知识三元组中的语音片段c替换为定位到的片段s，使语音和翻译知识共享相同的声学特征锚点，引导模型关注翻译知识。
3. **Tag Cue（标签提示）**: 在训练数据中术语翻译前插入\<Term\>特殊标签，推理时模型预测\<Term\>即触发对翻译知识的关注。

### 损失函数 / 训练策略
- 定位步骤：对比学习训练语音编码器 $\mathcal{L}_{SE} = -\log \frac{e^{sim(u, c^+)}}{e^{sim(u, c^+)} + \sum e^{sim(u, c_i^-)}}$
- 翻译步骤：标准next token prediction损失 + LoRA微调
- 两步顺序训练：先训练定位再训练翻译
- 使用CosyVoice2 TTS生成术语语音片段，SenseVoice ASR验证质量

## 实验关键数据
### 主实验（Oracle Knowledge Setting, EN→ZH）

| 方法 | CoVoST2 TSR | CoVoST2 BLEU | MuST-C TSR | MuST-C BLEU |
|------|------------|-------------|------------|-------------|
| Base Model | 24.12 | 35.82 | 27.61 | 25.73 |
| SALM | 76.53 | 55.97 | 69.01 | 32.10 |
| Retrieve-and-Demo | 60.88 | 50.22 | 58.87 | 30.18 |
| **Locate-and-Focus** | **90.13** | **58.49** | **94.09** | **34.52** |

### 消融实验（Oracle Setting, EN→ZH CoVoST2）

| 配置 | TSR | BLEU |
|------|-----|------|
| Locate-and-Focus (full) | 90.13 | 58.49 |
| w/o Audio Replacement | 89.67 | 58.37 |
| w/o Tag Cue | 89.00 | 58.25 |
| w/o Both | 88.59 | 58.32 |

### 关键发现
- 数据集规模：训练集10K语音+14K术语对，CoVoST2/MuST-C/MSLT三个测试集
- 对比Translation Training（仅做翻译训练无术语知识），TSR仅27.30% vs 90.13%
- 对比Base Model（无任何术语增强），TSR仅24.12%，差距巨大

- Locate-and-Focus在术语翻译成功率（TSR）上大幅领先：CoVoST2 EN→ZH 90.13% vs SALM 76.53% vs R&D 60.88%
- EN→DE方向提升更显著：CoVoST2 TSR 96.35% vs SALM 85.91%
- Audio Replacement和Tag Cue均有正贡献，但定位本身是最关键的组件
- 端到端设置下仍有显著提升（65.53% TSR），但不如Oracle设置
- 一般翻译质量（BLEU）不受影响甚至略有提升

## 亮点与洞察
- 首个在语音LLM中利用多模态细粒度翻译知识的端到端术语翻译方法
- 滑动窗口检索设计简洁有效，可并行计算，仅轻微增加推理延迟
- 音频替换策略巧妙地解决了跨模态/跨说话人差异问题——让语音和翻译知识共享相同的声学特征
- \<Term\>标签作为自提醒机制设计轻量实用，不需要额外模块
- 英→德方向TSR高达96.35%，表明方法在形态丰富的语言上同样有效
- 消融实验表明定位本身是最关键组件，Audio Replacement和Tag Cue各有约1-2%的增量贡献
- 自行构建的术语翻译数据集（CoVoST2/MuST-C/MSLT + TTS/ASR管线）方法论可复用

## 局限性 / 可改进方向
- 端到端设置与Oracle设置的性能差距仍较大（TSR相差约25%），检索精度是瓶颈
- 数据集自行收集，术语对由LLM提取后人工检查，可能有遗漏或错误
- TTS生成的术语语音可能与真实语音存在域差异（合成vs自然），影响定位精度
- 仅支持英→中和英→德两个方向，多语言（如低资源语言）扩展性待验证
- 滑动窗口步长为1可能在长语音上开销较大；可探索多尺度或分层检索
- 未考虑一个语音中包含多个术语且术语距离很近的情况

## 相关工作与启发
- SALM（Gaido et al., 2023）和Retrieve-and-Demonstrate（Li et al., 2024a）是主要对比范式，本文取两者之长
- CB-Whisper（Li et al., 2024b）在ASR领域的术语识别思路可借鉴，但未解决跨模态问题
- 音频替换策略可推广到其他需要多模态对齐的场景（如视频翻译中的visual grounding）
- 对口译辅助系统有直接应用价值——精确定位并提示专业术语翻译
- CosyVoice2和SenseVoice的TTS-ASR管线为数据构建提供了可扩展的方法论

## 评分
- 新颖性: ⭐⭐⭐⭐ 定位+聚焦的两步设计和音频替换策略新颖，解决了跨模态对齐难题
- 实验充分度: ⭐⭐⭐⭐ 多数据集多方向验证，消融完整，但仅支持两个翻译方向
- 写作质量: ⭐⭐⭐⭐ 图示清晰，方法描述详尽，数学公式规范
- 价值: ⭐⭐⭐⭐ 对语音翻译中的术语问题有实际推动作用
- 总评: 工程性强且实用，数据集构建方法论可复用，对口译辅助系统有直接应用价值
- 复现性: 代码开源，数据集构建管线可复用
- 延伸性: 可推广到多语言方向、低资源语言场景
- 开放问题: 如何在实时场景下平衡定位精度和推理速度？
- 影响力: 为语音翻译中的术语处理提供了新的多模态知识利用范式
- 发展方向: 可组合CTC或维特比解码进一步改进术语定位精度
