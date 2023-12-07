# 论文信息
- 时间：2021
- 期刊：ICLR
- 网络名称：ViT
- 意义：Transformer杀入CV界
- 作者：Alexey Dosovitskiy∗,†, Lucas Beyer∗, lexander Kolesnikov∗, Dirk Weissenborn∗,Xiaohua Zhai∗, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby∗,†;∗equal technical contribution, †equal advising; Google Research, Brain Team
- 实验环境：TPUv3 hardware
- 数据集：ImageNet-21k JFT-300M
- [返回上一层 README](../README.md)

[ppt](../ppt/ViT/VIT.pdf)

[python code](../code/ViT/VIT.py)

# 一、解决的问题
1. 挑战了CNN在CV界的绝对地位，足够多的数据上做预训练
2. 打破了NLP和CV的界限，为多模态做出了杰出贡献
3. 遮挡、数据偏移、对抗性的Patch去除和打乱Patch的图片，处理的很好
4. 需要更少的资源->2500 days TPU v3 core
5. 之前都是CNN+self attention -> self attention 取代 CNN -> 直接平移Transformer

# 二、做出的创新
1. 把图片分成16X16的Patches，解决视觉2D拉长，导致序列过长的问题
2. 网络中间的特征图，当Transformer的输入
3. 直接迁移NLP的Transformer，尽量少做修改
4. 以往的降维工作，方法奇特，没有得到良好的硬件加速

# 三、设计的模型

![ViT Model](../pictures/ViT/ViT%20model.png)

- 图片预处理的过程：
>- 把一张图（224·224·3）打成Patch（16·16·3）
>- 把patch变成了一个序列（196个=224/16,16·16·3=768维）
>- 通过一个线性投射层（全连接层768·768），变成了一个特征，并加上了(是Sum，不是拼接concatenation)位置编码
>- 加上了cls（借鉴BERT）

- 进入标准的Transformer Encoder
>- Embedded Patches(197·768)
>- Multi-HeadAttention k、q、v（如果是12个头，12个kqv，768/12=64，每个都是197·64）
>- MLP会放大4倍（197·3072），再缩小投射回去

![ViT Equation](../pictures/ViT/ViT%20Equation.png)

- 用公式表述了上述过程

- >Inductive bias 归纳偏置
- >Hybrid Architecture 混合体系

- >fine-tuning and high resolution

# 四、实验结果

## 1、比之前模型的优势
- ViT在没有强约束的数据集上，没有ResNet好，缺少归纳偏置，少了先验知识
- CLS和全局平均池化（GAP）得好好调参，可以获得不错的效果，**炼丹技术要过硬**
- 相对，绝对位置编码效果差不多
## 2、有优势的原因
- 大规模预训练之后，再做下游任务，效果就是很好
- ViT在中小型数据集上，远不如ResNet，在用超大数据集上，ViT远高ResNet
- 在计算量增大的情况下，混合模型还不如Transformer了
- 能学到基函数、能学到位置编码
- 能学到语义距离，在模型较浅的位置，真的能注意到全局的信息
## 3、改进空间
- 做的任务可以扩充
- 模型架构可以转换
- 有监督，自监督
- 挖了一个多模态的大坑
# 五、结论

## 1、模型是否解决了目标问题
1. 成功在CV中引入了Transformer
## 2、模型是否遗留了问题
1. 需要大训练集才能发挥优势
2. CNN和Transformer的混合算法在较小的计算量下略优于ViT，但是在大计算量的情况下，差异消失
## 3、模型是否引入了新的问题
- Transfomer用到视觉中的困难
1. 自注意力是最关键的，输入两两相互互动，算得Attention图，硬件能支持的也就是几百上千
2. 怎么把2D图片变成一维，做起来很难，而且拉直之后非常长
# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论