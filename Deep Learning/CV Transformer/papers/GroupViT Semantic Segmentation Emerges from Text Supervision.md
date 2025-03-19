# 论文信息
- 时间：2022
- 期刊：CVPR
- 网络名称：GroupViT
- 意义：CLIP用在分割领域
- 作者：Jiarui Xu1*, Shalini De Mello2, Sifei Liu2, Wonmin Byeon2, Thomas Breuel2, Jan Kautz2, Xiaolong Wang1; 1UC San Diego; 2NVIDIA
- 实验环境：
- 数据集：
- [返回上一层 README](../README.md)
# 一、解决的问题
1. Lseg虽然用了CLIP的预训练参数，图画的也很像CLIP，但是目标函数终究不是对比学习，也不是无监督学习的框架，并没有把文本当作一个监督信号来使用，导致它还是依赖于手工标注的segmentation mask，这就会导致能用的数据集太小了，而且对于分割来说，手工标注分割的mask是非常贵的一件事
2. 如何摆脱掉手工标注，如何真的做到用文本当作监督信号，从而达到无监督的训练
3. GroupViT就是利用文本当作监督信号，摆脱手工标注，从而进行无监督的训练
# 二、做出的创新
1. GroupViT贡献：就是在已有的ViT框架中，加入了grouping block，同时加入了可学习的group tokens
# 三、设计的模型
1. 模型结构

    ![GroupViT model](../pictures/GroupViT/GroupViT%20model.png)
    
    1. 视觉很早之前做无监督分割的时候，经常用一类方法，叫做grouping
        - 类似说，有一些聚类的中心点，从这个点开始发散，把周围相似的点逐渐扩充成一个group，这个group相当于一个segmentation mask，是一种自下而上的方式
    2. 这里作者重新审视了grouping的方法，发现能把grouping完美的用在当前的框架中，提出了一个计算单元（图右），叫做grouping block，还有一些可学习的group tokens，主要目的是想让这个模型在初始学习的时候，能慢慢的把相邻相近的像素点group起来，变成一个又一个segmentation mask
2. 模型细节

    ![GroupViT dimension](../pictures/GroupViT/GroupViT%20dimension.png)
    
    1. 12层的Vision Transformer（12层Transformer）
    2. 图像编码器输入有两个
        1. 一个是来自于原始图像的patch embedding
        2. 另一个就是group tokens
        3. 224x224->patch size 16x16->14x14的序列->就是196长度->经过Linear projection就得到了一些patch embedding，维度是196x384，用的是ViT small的缘故
        4. group tokens（cls token，用于代表整个图片）作者设的是64x384，384是为了保持维度不变，可以和patch embedding 做拼接；64是希望一开始就有足够的聚类中心，反正到时候还可以合并
            - 之前用一个token，是因为用一个类别代表图片，现在是要做分割
    3. 利用Grouping block把之前的patch embedding直接assign到64个group token上，相当于做了一次聚类的分配，那么segment token就是64x384
        - Grouping block还有另外一个好处，变相的相当于把序列长度降低了，模型的计算复杂度和计算时间就下降了
        - 和Swin Transformer一样，是一个层级式的网络结构
    4. Grouping block的操作
        1. 用类似自注意力机制的方式，先算了一个相似度矩阵，用相似度矩阵去帮助原来的image token做聚类中心的分配，从而完成了把这个输入从196x384，降维到64x384的过程
        2. 当然这么做聚类中心的分配过程是不可导的，因此作者用了一个小trick，一个gumbel softmax，从而把模型变成可导的，那么模型就可以端到端的进行训练了
    5. 到此用了6层Transformer的第一阶段结束了，作者想让维度进一步降低
    6. 新加了8个group tokens，也就是8x384，希望通过进一步的Transformer学习，能把64个segment tokens映射到8个聚类中心上。作者在第9层Transformer后加了一个grouping block，让64->8
3. 训练
    1. 和CLIP都是通过图片文本对，去算一个对比学习的loss
    2. 问题是，原来的CLIP是一个图片有一个特征，一个文本一个特征；但是GroupViT是一个文本特征，有8个图像特征，怎么把8个特征融合成一个，变成整个图像的Image level特征？
        - 作者用最简单的Avg pooling得到了一个1x384
4. 优势
    - 保持了CLIP的特征，scale能力很好

5. 如何做zero-shot推理

    ![GroupViT zero-shot](../pictures/GroupViT/GroupViT%20zero-shot.png)
    
    1. 和CLIP一样
    2. 给一个图片，经过GroupViT，得到8个Group embedding
    3. 有可能的标签，通过文本编码器，得到一系列的文本特征
    4. 只要去算图像的Group embedding和文本特征之间的相似度，就知道每个Group embedding之间对应什么样一个class
    5. 很明显的一个局限性，因为最后模型只有8个这个Group embedding，最多只能检测8类，不过这是一个超参数，8个效果最好
    
        ![GroupViT output tokens](../pictures/GroupViT/GroupViT%20output%20tokens.png)
# 四、实验结果

![GroupViT group token](../pictures/GroupViT/GroupViT%20group%20token.png)

- 真的学到了东西

![GroupViT result](../pictures/GroupViT/GroupViT%20result.png)

- Baseline中用文本做监督信号的第一个工作
- 还可以做zero-shot的inference
- 比有监督还是差很多
## 1、比之前模型的优势

## 2、有优势的原因

## 3、改进空间
- 两个局限性
    1. GroupViT结构还是偏向于一个图像的编码器，没有很好的利用dense prediction的特性，比如分割中很火的Dilated Convolution，或者pyramid pooling，或者U-net这种结构，从而能够获得更多的上下文信息，能获得更多尺度的信息，帮助去做更好的分割任务
    2. 分割方向存在一个背景类
        - GroupViT考虑背景类的方法：在做zero-shot推理的时候，不光是选择最大的相似度，因为有时候最大的相似度也比较小，作者为了尽可能提高前景类的分割性能，设置了一个相似度阈值，相似度必须超过阈值，且是最大的一个，才能说Group embedding属于这一类；如果都没有超过阈值，GroupViT认为这是一个背景类
        - 对PASCAL VOC还行，用PASCAL Context和COCO就不太好，类别太多，导致阈值难以设置
        - group token学的很好，也就是真正的分割做得很好，但是分类分错了。作者做了上线的实验，如果给GroupViT一个输出的prediction mask，拿这个mask去跟所有的ground truth mask做对比，一旦发现哪个iou最大，直接把ground truth label给GroupViT prediction mask当成它的标签，这样一来只要分割做的好，分类就不会错。最后发现，整体数据集效果（mIoU）往上提了很多点，跟有监督那边的最高分差不多，分割做的很好，***语义分割做的不好***，分类错了很多
            - 这个只能怪CLIP的训练方式，这种方式只能学到语义信息非常明确的东西，学不到模糊意义的东西
            - 解决方案如阈值怎么设，是不是根据类别去设，是不是做一个可学习的阈值，或者改整个zero-shot的推理过程，或者在训练的时候加入一种约束，把背景融入到训练之中

# 五、结论

## 1、模型是否解决了目标问题

## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题

# 六、代码

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
