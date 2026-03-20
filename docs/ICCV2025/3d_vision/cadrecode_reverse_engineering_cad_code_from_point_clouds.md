# CAD-Recode: Reverse Engineering CAD Code from Point Clouds

**会议**: ICCV 2025  
**arXiv**: [2412.14042](https://arxiv.org/abs/2412.14042)  
**代码**: 未知  
**领域**: 3D视觉 / CAD逆向工程 / 3D理解  
**关键词**: CAD逆向工程, 点云, Python代码生成, LLM, sketch-extrude序列  

## 一句话总结
将CAD sketch-extrude序列表示为Python代码，利用轻量级点云投影器 + 预训练LLM解码器将点云翻译为可执行Python代码来重建CAD模型，在DeepCAD/Fusion360/真实世界CC3D数据集上显著超越现有方法，且输出代码可被通用LLM理解用于CAD编辑和问答。

## 背景与动机
3D CAD逆向工程旨在从点云等3D表示恢复构成CAD模型的sketch和操作序列。传统方法将CAD序列表示为参数化token序列，需要专用tokenizer和复杂的序列预测网络。LLM天然擅长生成结构化代码，而Python是LLM最熟悉的编程语言之一。

## 核心问题
如何利用LLM的代码生成能力来解决CAD逆向工程问题？

## 方法详解

### 整体框架
输入点云 → 轻量级点云投影器(编码几何信息) → 预训练LLM解码器 → 输出Python代码(sketch绘制+extrude操作) → 执行代码重建3D CAD模型

### 关键创新
1. **Python代码作为CAD序列表示**: 将传统的参数化sketch-extrude序列重新表示为Python代码，利用LLM对Python语法的先验知识
2. **点云投影器 + LLM解码器**: 用小型LLM作为解码器，轻量级projector连接点云特征和LLM输入空间
3. **百万级程序化合成数据集**: 100万个CAD序列的训练数据
4. **LLM可解释输出**: 生成的Python代码可被通用LLM理解，支持CAD编辑和基于点云的CAD问答

## 实验关键数据
- 在DeepCAD、Fusion360和真实世界CC3D数据集上显著超越现有方法
- 输出代码可被通用LLM解释用于下游任务(编辑/问答)

## 亮点 / 我学到了什么
- **代码即表示**: 用编程语言表示3D建模序列，巧妙利用LLM的代码先验
- **跨模态桥接**: 从点云到代码的翻译，连接3D感知和程序化建模
- **可解释性**: Python代码输出天然可读可编辑，比参数序列更实用

## 局限性 / 可改进方向
- 限于sketch-extrude这类基本CAD操作
- 复杂CAD模型可能需要更长的代码序列

## 评分
- 新颖性: ⭐⭐⭐⭐ Python代码表示CAD + LLM解码器的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 三个数据集 + 下游应用(编辑/问答)
- 写作质量: ⭐⭐⭐⭐ 摘要清晰，贡献明确
- 对我的价值: ⭐⭐⭐ CAD领域，代码生成范式有参考价值
