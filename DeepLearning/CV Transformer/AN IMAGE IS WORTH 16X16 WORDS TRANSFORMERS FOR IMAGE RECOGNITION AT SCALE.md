# 论文信息
- 时间：2021
- 期刊：ICLR
- 网络名称：ViT
- 意义：Transformer杀入CV界
- 作者：Alexey Dosovitskiy∗,†, Lucas Beyer∗, lexander Kolesnikov∗, Dirk Weissenborn∗,Xiaohua Zhai∗, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby∗,†;∗equal technical contribution, †equal advising; Google Research, Brain Team
- 实验环境：TPUv3 hardware
- 数据集：ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images 

# 一、解决的问题
1. >Transformer applications to computer vision remain limited
2. 随着模型和数据集的增长，仍然没有出现性能饱和的迹象
# 二、做出的创新
1. >We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks
2. >our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive with (or even better than) state-of-the-art CNNs
3. 与先前在计算机视觉中使用自我注意的工作不同，我们没有在除了初始补丁提取步骤之外的体系结构
# 三、设计的模型

![ViT Model](../pictures/ViT%20model.png)

- >The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). 
- >Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019)
- >The MLP contains two layers with a GELU non-linearity

![ViT Equation](../pictures/ViT%20Equation.png)

- >Inductive bias 归纳偏置
- >Hybrid Architecture 混合体系

- >fine-tuning and high resolution
# 四、实验结果

## 1、比之前模型的优势
1. >Vision Transformer models pre-trained on the JFT-300M dataset outperform ResNet-based baselines on all datasets, while taking substantially less computational resources to pre-train
2. >ViT pre-trained on the smaller public ImageNet-21k dataset performs well too
3. >we note that pre-training efficiency may be affected not only by the architecture choice, but also other parameters, such as training schedule, optimizer, weight decay, etc
## 2、有优势的原因

## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题
1. 成功在CV中引入了Transformer
## 2、模型是否遗留了问题
1. 需要大训练集才能发挥优势
2. CNN和Transformer的混合算法在较小的计算量下略优于ViT，但是在大计算量的情况下，差异消失
## 3、模型是否引入了新的问题

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论