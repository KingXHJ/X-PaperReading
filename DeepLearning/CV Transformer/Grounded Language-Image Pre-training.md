# 论文信息
- 时间：2022
- 期刊：CVPR
- 网络名称：Glip
- 意义：CLIP在目标检测的应用，对标分割的GroupViT
- 作者：Liunian Harold Li∗1†, Pengchuan Zhang∗2♠, Haotian Zhang∗3†, Jianwei Yang2, Chunyuan Li2, Yiwu Zhong4†, Lijuan Wang5, Lu Yuan5, Lei Zhang6, Jenq-Neng Hwang3, Kai-Wei Chang1, Jianfeng Gao2; 1UCLA, 2Microsoft Research, 3University of Washington, 4University of Wisconsin-Madison, 5Microsoft Cloud and AI, 6International Digital Economy Academy
- 实验环境：
- 数据集：
# 一、解决的问题
- 研究动机：
- 跟分割一样，精心标注好的数据很难得，而且生活中边边角角的类和层出不穷的新类，没有办法训练一个模型，把这些都做得很好，只能依赖这种做open vocabulary的detection模型，去把这些corner case处理好
- 想训练一个好的open vocabulary的detection模型，就要像CLIP一样，能有一个好的预训练数据集，最好的规模就是上千万上亿，能把文本和图像之间的关系学得很好，同时把定位学的也很好
- 怎么利用更多的数据？怎么能把图像文本对用上（图像文本对数据很好收集，很轻松就上亿了）？
- 作者发现在Vision Language的下游任务里面，还有一类任务叫Vision grounding（就是给一句话，去把这句话里的一些物体，在当前的图片里，把它定位出来），和目标检测在图片中找bounding box很像，作者打算把它们结合起来
- 真的可以把detection和phrase grounding两个任务结合起来变成统一的框架，数据集也可以大大扩充；如果再把伪标签那一系列东西加进来（self-training），这样模型就可以在没有标注过的图像文本对上去生成这些bounding box的标签，从而扩大整个训练数据集的数量，把模型训练的更好
- 在没看过数据集的情况下，可以在COCO和LVIS上分别达到49.8AP和26.9AP，有监督的基线也就40多AP，Glip这种zero-shot已经将近50AP了，性能非常强
# 二、做出的创新
1. zero-shot推理过程

    ![Glip zero-shot](../pictures/Glip%20zero-shot.png)
    
    1. 不论给出物体的标签，把它变成一句话，把这句话扔给Glip模型，Glip就能把这些类别都检测出来了
    2. 或者直接给一句话，也可以检测出想要的东西
2. 两个任务结合

    $$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{loc}$$
    
    1. 对目标检测来说，训练的目标函数一般就是一个分类的loss，加上一个定位的loss
    2. 定位部分主要是根据模型不同，选择怎么生成定位框
    3. 区别就在于怎么算分类的loss
    4. 对于detection来说，它的标签是一两个单词，是one-hot标签；而对Vision grounding来说，他的标签是一个句子
3. loss该如何算
    1. Detection：
        - 给定一个图片，有这个图像的backbone，就可以得到region embedding。假如有n个bounding box，每个bounding box的维度是d，接下来要做的就是接一个分类头，看一下每个bounding box里面的物体到底是哪一类。分类头的矩阵就是W，维度是cxd，c就是有多少个类别。所以把region embedding的O和W一乘，就得到了这个分类最后的logic。再用mns把bounding box筛选一下，再去跟ground truth去算cross entropy loss，就能得到最终的分类loss
        ![Glip detection loss](../pictures/Glip%20detection%20loss.png)
        
    2. Vision grouding
        - 算了一个匹配的分数 $S_{ground}$ ，就是想看看图像中的区域，和句子里的单词是怎么匹配上的。图像的处理还是一样的，有image backbone，得到了一些region feature，但是接下来不是分类头，而是像ViLD一样，换成了一个文本编码器，通过已知的prompt，就能得到文本的embedding，和图像的embedding去算u你similarity，就可以得到region word alignment score $S_{ground}$ ，如果画成图，就是ViLD里的ViLD text分支，一模一样
        
        ![Glip vision ground](../pictures/Glip%20vision%20ground.png)
        
    3. 结合
        - 只需要做小小的改动，就可以结合在一块
        - 改动就是，什么时候算是一个positive match，什么时候算是一个negative match
        - 通过实验验证，这个模型是完全可以迁移到任何一个目标检测的数据集上的
        - 下面就是怎么扩大数据集，怎么把预训练模型做的更好
4. 数据集
    - Glip的几个变体
    - 表格上面是用已有的数据集，做有监督的训练
    - 将detection和grouding数据集合并，获得更大的数据集
    - 但是这样也逃不过corner case
    - 又引入了图像文本对，用伪标签（即GLIP-T（C）直接在Cap4M上做推理，把推理得到的bounding box当作ground truth，虽然有错误，但是不影响伪标签的有效性，有利于模型的效果和稳健性）
    
    ![Glip dataset](../pictures/Glip%20dataset.png)     
    
# 三、设计的模型

![Glip model](../pictures/Glip%20model.png)

1. 图片经过图像编码器，得到region embedding
2. 文本经过文本编码器，得到文本的embedding
3. 跳过中间部分到目标函数
4. 因为这是一个有监督学习的模型，时时刻刻都是有bounding box annotation，所以当抽出来region O的时候，是知道和单词是如何一一对应的，这样在算完O和P的相似度点乘之后呢，就可以和ground truth去算alignment loss，这样就完成了图像和文本的融合，之后去做zero-shot
5. 对于定位loss来说，当有了ground truth，所以算一个最基本的L1 loss即可
6. 看中间的deep fusion
    - 作者的意思和Lseg作者一样，当抽出文本和图像的特征之后，理论上是可以去计算相似度矩阵的，但是直接这么算，图像和文本的joint embedding space还没有学得很好，如果多加一些层数，让他们去融合一下，可能他们就会学得更好，相似的会拉进一些，不相似的会拉远一些。总之能让最后的图像和文本的特征更加的强，而且更加的有关联性，这样再算相似度更有针对性
    - 结构可选，有很多方法，这里就是做了交叉。也可以用到groundViT
    - **图像分割和目标检测都是稠密型预测（dense prediction）任务的一种，都需要分类和定位，方法都是可以借鉴的**
# 四、实验结果
1. 能够做出没有标注的物体检测，并且正确的检测出数量

2. 数值结果

    ![Glip result](../pictures/Glip%20result.png)
    
    - zero-shot很强
    - 不算是完全公平的比较，训练数据集和trick都不一样
## 1、比之前模型的优势

## 2、有优势的原因

## 3、改进空间
1. 出现了GLIPv2（另一篇论文GLIPv2:Unifying Localization and VL Understanding）
    1. 融合了更多的数据集和更多的任务
    2. 把所有带定位的任务，如分割、检测和Vision Language的任务：vqa，vision grounding，vision captioning都放在了一起
    3. 文本端丰富了更多，图片没变
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
