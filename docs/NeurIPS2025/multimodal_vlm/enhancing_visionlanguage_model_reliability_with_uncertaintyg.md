# Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2412.06474](https://arxiv.org/abs/2412.06474)  
**代码**: [https://github.com/kigb/DropoutDecoding](https://github.com/kigb/DropoutDecoding)  
**领域**: 多模态VLM / AI安全  
**关键词**: VLM幻觉, 不确定性量化, 视觉token, Dropout解码, 认知不确定性  

## 一句话总结
提出Dropout Decoding——量化视觉token的认知不确定性(epistemic uncertainty)，选择性遮掩高不确定性token，通过集成多个遮掩后的解码结果做多数投票，无需训练即在InstructBLIP上CHAIR_I降低16%、CHAIR_S降低12%。

## 背景与动机
LVLM的视觉幻觉源于对视觉token的误解——某些视觉token携带的信息被模型错误理解。已有方法要么需要训练（微调/RLHF），要么基于启发式（如OPERA修改beam search、VCD对比解码），缺乏对"哪些视觉token不可靠"的原理性度量。

## 核心问题
如何在推理时识别哪些视觉token不可靠，并通过选择性遮掩来提升LVLM输出的可靠性？

## 方法详解

### 整体框架
两阶段推理方法：(1) 解码前——量化每个视觉token的不确定性；(2) 解码时——基于不确定性做token dropout + 集成投票。

### 关键设计

1. **视觉token的文本空间投影**: 用logit lens将视觉token投影到文本词表空间：q_proj_i = softmax(W_V · h_v_i)。投影后可以看到每个视觉token"像什么文本"——信息丰富的patch投影出"Berlin"/"computer"，无信息的patch投影出"a"/"the"。

2. **认知不确定性分解**: 将总不确定性分解为偶然不确定性（数据固有）和认知不确定性（模型知识不足）。关键发现：**认知不确定性**与视觉token的信息量正相关——高认知不确定性对应信息丰富但可能被误解的patch（如城市名、特定物体）。

3. **不确定性引导的Token Dropout**: 
   - 标记高认知不确定性的视觉token
   - 生成多组随机dropout掩码（每组遮掩不同的高不确定性token子集）
   - 用每组掩码单独解码
   - 多数投票聚合结果

### 类比原理
传统Dropout对模型参数做随机遮掩防止过拟合→Dropout Decoding对输入视觉token做选择性遮掩防止"过度依赖"不可靠视觉信息。

## 实验关键数据

| 模型 | 方法 | CHAIR_S↓ | CHAIR_I↓ | THRONE F1↑ |
|------|------|---------|---------|-----------|
| InstructBLIP | Greedy | 27.87 | 7.90 | 0.809 |
| InstructBLIP | OPERA | 28.07 | 8.23 | 0.805 |
| InstructBLIP | VCD | 39.33 | 19.10 | 0.737 |
| InstructBLIP | **Dropout Decoding** | **24.53** | **6.63** | **0.814** |
| LLaVA-1.5 | Greedy | 42.20 | 12.83 | 0.795 |
| LLaVA-1.5 | **Dropout Decoding** | **39.80** | **11.73** | **0.804** |
| LLaVA-NEXT | Greedy | 28.80 | 8.10 | 0.815 |
| LLaVA-NEXT | **Dropout Decoding** | **26.26** | **7.39** | **0.821** |

- VCD在InstructBLIP上反而大幅恶化，Dropout Decoding一致有效
- 在THRONE上precision和F1同时提升

### 消融实验要点
- **认知 vs 偶然不确定性**: 认知不确定性引导效果远优于偶然不确定性
- **候选数量**: 5-10组dropout掩码最优
- **dropout比例**: 遮掩top-30%高不确定性token最优

## 亮点
- **原理清晰**: 从贝叶斯不确定性分解到token dropout的推导链完整
- **无需训练**: 推理时即插即用，兼容任何LVLM
- **认知不确定性的发现**: 高认知不确定性token = 信息丰富但可能被误解的关键token
- **一致有效**: 在3个不同LVLM、多个benchmark上都稳定提升

## 局限性
- 多次解码+投票增加推理延迟（~5-10x）
- 依赖logit lens的投影质量
- 对开放式生成（非分类/描述）的适用性未验证

## 启发与关联
- 与FlowCut互补：FlowCut用CLS注意力识别冗余token并剪枝（效率导向），Dropout Decoding用不确定性识别不可靠token并集成（可靠性导向）
- 认知不确定性可以指导VHR——不确定性高的token区域需要更强的视觉注意力增强
- 与REVERSE结合：Dropout Decoding在token级检测不确定性，REVERSE在phrase级检测幻觉

## 评分
- 新颖性: ⭐⭐⭐⭐ 将Dropout从参数空间迁移到输入token空间是巧妙的
- 实验充分度: ⭐⭐⭐⭐ 3个模型、CHAIR+THRONE双benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1的投影可视化和不确定性分解极其直观
- 价值: ⭐⭐⭐⭐ 推理时VLM可靠性增强的实用方法
