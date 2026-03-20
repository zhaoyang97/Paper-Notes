# InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention

**会议**: NeurIPS 2025  
**arXiv**: [2509.16691](https://arxiv.org/abs/2509.16691)  
**代码**: [https://github.com/FireRedTeam/InstanceAssemble](https://github.com/FireRedTeam/InstanceAssemble)  
**领域**: 图像生成 / 布局控制  
**关键词**: layout-to-image, instance assembling attention, DiT, bounding box control, LoRA, DenseLayout benchmark  

## 一句话总结
提出InstanceAssemble，通过实例组装注意力机制（instance-assembling attention）实现layout条件的精确控制——支持bbox位置控制和多模态内容控制（文本+视觉内容），作为轻量LoRA模块适配到现有DiT模型，同时提出DenseLayout benchmark（5K图像90K实例）和Layout Grounding Score评估指标。

## 背景与动机
Layout-to-Image（L2I）生成是可控图像生成的重要任务——给定物体的位置（bbox）和描述，在指定位置生成指定物体。现有L2I方法在复杂布局条件下表现次优，且与DiT架构的适配不够灵活。同时缺乏针对密集布局的评估benchmark和精确的评估指标。

## 核心问题
如何在DiT-based T2I模型中高效注入Layout控制，实现精确的位置和内容控制，同时保持与各种风格LoRA的兼容性？

## 方法详解

### 关键设计
1. **Instance-Assembling Attention**：核心机制——将每个实例的bbox区域对应的attention token进行分组，在组内进行instance-specific的注意力计算。这使得每个实例区域的生成独立受其对应的文本/视觉描述控制，避免不同实例之间的特征串扰。

2. **多模态内容控制**：不仅支持文本描述控制每个实例的内容（如"红色汽车"），还支持视觉内容控制（如使用参考图像指定物体外观）。这比纯文本描述更精细。

3. **轻量LoRA适配**：通过LoRA模块注入到现有DiT模型中，不需要全量微调。这使得InstanceAssemble可以与其他风格LoRA兼容——用户可以同时使用布局控制LoRA和风格LoRA。

4. **DenseLayout Benchmark + LGS指标**：构建了包含5K图像、90K实例的密集布局benchmark。提出Layout Grounding Score (LGS)作为更精确的L2I评估指标——衡量生成的物体是否出现在指定的bbox位置且类别正确。

## 实验关键数据
- 在复杂布局条件下达到SOTA性能
- 与多种风格LoRA兼容
- DenseLayout benchmark提供了更challenging的评估场景
- LGS比现有指标更准确地反映布局控制质量

## 亮点
- **Instance-assembling attention**优雅地解决了多实例特征串扰问题
- **多模态控制（文本+视觉）**使控制更灵活
- **LoRA适配**保持了与style LoRA生态的兼容性——实用性强
- **DenseLayout benchmark和LGS指标**为L2I社区提供了更好的评估工具
- 代码和模型开源

## 局限性 / 可改进方向
- 在极密集布局（大量重叠bbox）下性能可能下降
- 仅在DiT架构上验证
- LGS指标的鲁棒性需要更多验证

## 与相关工作的对比
- **vs. GLIGEN**：GLIGEN也做layout-guided生成但基于UNet；InstanceAssemble基于DiT+LoRA——更现代
- **vs. VisualCloze**：VisualCloze统一多种图像生成任务；InstanceAssemble专注于精确的layout控制

## 评分
- 新颖性: ⭐⭐⭐⭐ instance-assembling attention和多模态控制组合有新意
- 实验充分度: ⭐⭐⭐⭐ 有新benchmark+新指标+SOTA对比
- 写作质量: ⭐⭐⭐⭐ 方法清晰
- 价值: ⭐⭐⭐⭐ 为DiT时代的Layout-to-Image提供了实用方案
