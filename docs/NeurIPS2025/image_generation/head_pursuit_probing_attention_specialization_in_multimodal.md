# Head Pursuit: Probing Attention Specialization in Multimodal Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2510.21518](https://arxiv.org/abs/2510.21518)  
**代码**: [https://github.com/lorenzobasile/HeadPursuit](https://github.com/lorenzobasile/HeadPursuit)  
**领域**: 多模态VLM / 可解释性  
**关键词**: 注意力头特化, Matching Pursuit, 可解释性, 模型编辑, 稀疏分解  

## 一句话总结
用信号处理中的Simultaneous Orthogonal Matching Pursuit (SOMP)算法分解注意力头在unembedding矩阵上的稀疏表示，揭示注意力头的语义特化现象（如政治/国籍/月份/数字等），仅编辑1%的头即可可靠地抑制或增强特定概念——在语言和视觉-语言模型上均验证有效。

## 背景与动机
已有注意力头分析工具（如Logit Lens）通常只对单个样本做启发式分析，难以跨样本泛化。作者将Logit Lens重新解释为"Matching Pursuit的单步单样本特例"，然后用完整的SOMP算法做多样本、多方向的稀疏分解，系统性地揭示每个注意力头的语义角色。

## 核心问题
注意力头是否在语义层面存在特化？如果存在，能否利用这种特化来可控地编辑模型行为？

## 方法详解

### 核心方法：SOMP稀疏分解
将每个注意力头的输出H ∈ R^{n×d}在unembedding矩阵D ∈ R^{v×d}上做稀疏分解：H ≈ W*D。SOMP迭代选择与残差最大相关的dictionary atom（即unembedding中的token向量），得到每个头的"top-k语义方向"。

### 关键发现
- **Mistral-7B的头特化**：L18.H27特化于政治（COVID/Soviet/Obama/Biden/Clinton），L24.H20特化于国籍（British/American/European），L25.H14特化于月份，L30.H28特化于数字
- **概念特定头选择**：用受限字典（仅包含目标概念的token）做SOMP，以方差解释比排序头的相关性
- **编辑1%头即生效**：翻转sign或缩放top-k头的输出，可显著抑制/增强目标概念

### 应用验证
1. **QA任务**：翻转"国籍"相关头→国籍类问题准确率暴降，其他类别不受影响
2. **毒性缓解**：翻转"毒性"相关头→生成内容毒性显著降低
3. **VLM图像分类**：翻转"颜色"相关头→颜色识别任务降级
4. **VLM图像描述**：增强颜色头(α>1)→生成描述中颜色词增多；抑制(α=-1)→颜色消失

## 实验关键数据
- Mistral-7B TriviaQA：翻转"国籍"头后国籍问题准确率降~15%，随机头仅降~2%
- LLaVA图像分类：翻转颜色头后颜色识别准确率下降明显
- 毒性生成：头编辑有效降低毒性指标
- 所有场景中随机控制组（同数量不同头）效果微弱——证明特化是真实的

## 亮点
- **信号处理×可解释性**: 将Matching Pursuit引入Transformer分析是新颖的跨学科bridging
- **Logit Lens的泛化**: 从"单步单样本"泛化到"多步多样本"稀疏分解
- **编辑仅需1%头**: 极其高效的模型行为控制——不需要训练
- **跨模态验证**: LLM和VLM上都展示了头特化现象

## 局限性 / 可改进方向
- 需要预定义target概念的token列表
- SOMP的计算开销随模型规模增长
- 头特化模式在不同模型间可能不一致
- 未测试在更大规模模型（70B+）上的效果

## 与相关工作的对比
- **vs VHD/VHR**: VHD用视觉有/无两条件的欧氏距离分类"视觉敏感头"，Head Pursuit用稀疏分解分析"语义特化头"——方法论互补
- **vs Logit Lens/Tuned Lens**: Head Pursuit是Logit Lens的多样本多方向泛化版本
- **vs MANU(模态感知遗忘)**: MANU基于激活统计剪枝神经元，Head Pursuit可以提供更精准的头级别指导

## 启发与关联
- SOMP选择的特化头可以指导VLM的模型压缩——保留重要概念头、剪枝冗余头
- 与VHR结合：VHR增强视觉头+Head Pursuit控制语义头=多维度精细控制
- 可用于Agent安全——识别和抑制与有害行为相关的特化头

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Matching Pursuit在Transformer可解释性中的创新应用
- 实验充分度: ⭐⭐⭐⭐ 语言+视觉语言，QA+毒性+分类+描述多任务
- 写作质量: ⭐⭐⭐⭐⭐ Table 1的头特化示例极其直观
- 价值: ⭐⭐⭐⭐⭐ 为模型可控编辑提供了数学原理性的工具
