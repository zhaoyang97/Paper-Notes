# MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents

**会议**: NeurIPS 2025  
**arXiv**: [2503.10809](https://arxiv.org/abs/2503.10809)  
**代码**: [https://github.com/AIchberger/mip-against-agent](https://github.com/AIchberger/mip-against-agent)  
**领域**: 多模态VLM / AI安全 / Agent攻击  
**关键词**: adversarial attack, OS agent, malicious image patch, VLM security, computer worm  

## 一句话总结
揭示针对多模态OS Agent的新型攻击向量——Malicious Image Patches (MIPs)：在屏幕截图中嵌入人类不可察觉的对抗性扰动图像块，当OS Agent截屏时自动触发恶意行为（如数据泄露、内存溢出），且可跨用户指令、屏幕布局和屏幕解析器泛化，甚至具备"计算机蠕虫"般的自传播潜力。

## 背景与动机
OS Agent（如Claude Computer Use、Windows Agent Arena）让VLM直接控制用户电脑——执行鼠标点击、键盘输入等操作。与被动输出文本的传统VLM不同，OS Agent的失败或被操控会产生直接的现实后果（文件删除、数据泄露、金融损失）。现有攻击（pop-up、prompt injection）依赖文本管道访问且易被过滤检测，而OS Agent依赖**截屏**导航的特性提供了全新的**视觉域攻击面**。

## 核心问题
能否通过仅操控屏幕上的一个小图像区域（如桌面壁纸或社交媒体图片），就劫持OS Agent执行任意恶意行为？这种攻击能否跨指令、跨屏幕布局泛化？

## 方法详解

### 整体框架
MIP攻击流程：
1. 攻击者用PGD在VLM参数上优化图像patch内的对抗性扰动
2. 将MIP嵌入桌面壁纸或社交媒体帖子
3. 当OS Agent截屏导航时捕获MIP
4. VLM处理包含MIP的截图后，输出恶意API调用序列（如打开终端执行恶意命令）

### 关键设计

1. **形式化约束处理**：OS Agent有多组件pipeline（screen parser→VLM→APIs），攻击需满足：
   - 扰动限制在图像patch区域ℛ内（约1/7屏幕面积）
   - ε=25/255的∞范数约束确保人眼不可察觉
   - 不得改变screen parser的SOM检测结果
   - 离散整数像素约束
   - 需适配VLM的图像resize操作

2. **Universal MIPs**：从targeted（单一prompt+截图对优化）扩展到universal（对多个prompt×多个截图联合优化）。方法：每步随机batch采样8对(prompt, screenshot)做PGD更新，直到所有对上的malicious target token概率超99%。

3. **两种攻击载体**：
   - Desktop setting：MIP嵌入桌面壁纸，1000×1000 pixels，边角衰减减少视觉伪影
   - Social media setting：MIP嵌入社交媒体帖子图片，900×900 pixels
   
4. **两种恶意行为**：
   - ym：33-token程序，打开终端造成内存溢出
   - yw：52-token程序，打开浏览器导航到恶意网站

5. **OS Agent蠕虫概念**：如果恶意行为包括分享/转发含MIP的帖子，MIP可自传播——首次提出"OS Agent计算机蠕虫"概念

### 实验环境
- Microsoft Windows Agent Arena (WAA)，154个预定义任务
- VLM：Llama 3.2 Vision系列（11B PT/IT、90B PT/IT）
- Screen parser：OmniParser和GroundingDINO+TesseractOCR
- 硬件：8×A100

## 实验关键数据

| 设置 | 目标 | Seen prompts ASR | Unseen prompts ASR | Unseen screens ASR |
|------|------|-------------------|--------------------|--------------------|
| Desktop targeted | yw | 1.0 | 0.3+ | 0.0 |
| Desktop universal | yw | ~0.9 | ~0.8 | ~0.7 |
| Desktop universal | ym | ~1.0 | ~0.9 | ~0.8 |
| Social universal | yw | ~0.9 | ~0.7 | ~0.6 |

关键发现：
- Universal MIPs可以泛化到未见过的prompts、screens和parsers
- 跨parser迁移：OmniParser→GroundingDINO仍然有效
- 跨执行步骤：Agent完成多步benign操作后遇到MIP仍被劫持
- 跨VLM：单个MIP可同时攻击11B PT + 11B IT + 90B IT，ASR≥90%
- 未见VLM的迁移性差——与已有adversarial image研究一致

## 亮点 / 我学到了什么
- **OS Agent的视觉攻击面**比文本攻击面更危险——MIP难以被过滤检测，可广泛传播
- 1/7屏幕面积、25/255扰动就足以编码完整的恶意程序——信息编码能力惊人
- **计算机蠕虫概念**极具前瞻性——当MIP导致Agent分享含MIP的帖子时，攻击自我传播
- 对screen parser不变性的处理（确保SOM不被扰动改变）是工程上的关键约束
- universal攻击通过多样化训练集实现泛化——与NoisyRollout的"diversity→generalization"思路类似但用于攻击

## 局限性 / 可改进方向
- 需要白盒访问VLM参数来做PGD——黑盒setting下效果未知
- 跨VLM系列迁移性差（Llama→其他模型系列）
- 仅在WAA环境测试，real-world OS（macOS、Linux）未验证
- 防御策略仅初步讨论（stochastic augmentation、verifier module）未实现
- ε=25/255在社交媒体的JPEG压缩后是否仍有效？

## 与相关工作的对比
- vs **Agent Smith (ICML 2024)**：Agent Smith对抗全图攻击多Agent网络，MIP只需控制一个小patch且针对OS Agent pipeline
- vs **Pop-up attacks**：Pop-up可见且易被过滤，MIP人眼不可见且编码在图像像素中
- vs **Prompt injection**：需要文本管道访问，MIP完全在视觉域操作
- vs **Wu et al. (ICLR 2025)**：他们通过误导captioning模型间接攻击，MIP直接操控VLM输出

## 与我的研究方向的关联
- 对VLM safety研究极其重要——OS Agent如果要大规模部署，MIP是必须解决的安全问题
- 与SAE for VLM (2504.02821)互补——SAE的concept-level filtering可能是一种防御MIP的方式
- 启发：能否用adversarial training或vision encoder robustness来防御MIP？这是一个重要的研究方向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究OS Agent的视觉对抗攻击，computer worm概念极具前瞻性
- 实验充分度: ⭐⭐⭐⭐⭐ Targeted/Universal、跨prompt/screen/parser/VLM/execution step的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 形式化清晰，Figure 1-2极具说服力，constraints处理严谨
- 对我的价值: ⭐⭐⭐⭐⭐ Agent安全是未来关键方向，MIP定义了需要防御的威胁模型
