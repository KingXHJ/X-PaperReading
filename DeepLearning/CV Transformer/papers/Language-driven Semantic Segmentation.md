# 论文信息
- 时间：2022
- 期刊：ICLR
- 网络名称：Lseg
- 意义：CLIP用在分割领域
- 作者：Boyi Li, Kilian Q. Weinberger, Serge Belongie, Vladlen Koltun, René Ranftl
- 实验环境：
- 数据集：7个分割数据集

# 前言
1. 分割和分类的任务很像。无非是把图片上的分类，变成了像素级别的分类。往往在图片方面的分类技术，都能直接用到像素级别分类上来，这就是过去几年分割领域卷的原因
    - 分类有自主意力，分割也有自主意力
    - 分类有对比学习，分割有密集对比学习
    - 分类用Transformer，分割有SETR segformer、Mask former
    - 分类有pesudo-labeling，分割也很快应用了
    - 分类有了CLIP，分割就出了zero-shot
2. 拿CLIP的预训练模型，做language-driven的语义分割
# 一、解决的问题


# 二、做出的创新

# 三、设计的模型
1. 模型总览图

    ![Lseg model](../pictures/Lseg%20model.png)
    
    ![Lseg dimension](../pictures/Lseg%20dimension.png)
    
    - 和CLIP模型总览图非常像
    - 图片->分割的模型->得到一个特征图->upscaling放大（保证输出和原图一致）
    - 图像编码器是DPT结构：ViT + decoder（本文作者之前的工作），decoder的作用就是把一个bottleneck feature，Upscale上去，就像psp和aspp这种层
    - 最后和ground truth去算cross entropy loss，不是CLIP的对比学习的目标函数
    - 虽然用了CLIP，但是还是有监督训练
2. 意义
    - 就是把文本的这一支，加入到了传统的，有监督分割的pipeline里
    - 文本和图像结合，就能学到language aware的特征
    - 这样在做推理的时候，就可以通过文本的prompt得到分割的效果
3. 和CLIP的关系
    - 作者说图像编码器和文本编码器可以用任意的模型和预训练模型的参数
    - 但是这篇论文中，为了达到最好的效果
        - 文本编码器：用了CLIP的文本编码器，是锁住的，训练也没动
        - 图像编码器：ViT + decoder，ViT要么用CLIP的参数，要么用之前别的工作的参数，原始ViT或者DiT；最后发现CLIP原始参数不太行，ViT和DiT参数更好，也没给解释
4. Spetial regularization block
    - 不知道是增加novelty，还是增加些内容
    - 这些block也就是一些conv或者depthwise conv层
    - 作者应该是觉得，在文本和视觉特征进行相乘之后，应该还有一些可学习的参数在后面，多理解文本和视觉该如何交互，从而达到最好的效果
    - 作者没有过多强调，消融实验也做得比较短
    - 2个block很好，加到4个就崩了，作者也没说为什么
# 四、实验结果
1. 效果

    ![Lseg result](../pictures/Lseg%20result.png)
    
    - 分割的非常的好
    - 没有的类，就不做检测
2. 用的不是常见的数据集
    - 用了之前做few-shot的数据集
3. 效果
    - 比之前zero-shot的效果提升不上，但是比one-shot的效果都差了很多
## 1、比之前模型的优势

## 2、有优势的原因

## 3、改进空间

# 五、结论

## 1、模型是否解决了目标问题

## 2、模型是否遗留了问题
1. 现在的zero-shot分割比有监督的分割还是差了很多，还是有提升空间的
2. zero-shot其实没有真的学到语义，他是去算相似度，谁和它最接近，就去选谁，但是并不能知道，这个词是不是真的和图片是对应的，因此提升空间还是很大的
## 3、模型是否引入了新的问题

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
