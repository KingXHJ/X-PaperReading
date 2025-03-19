# 论文信息
- 时间：2022
- 期刊：ICLR
- 网络名称：ViLD
- 意义：CLIP在目标检测领域的应用
- 作者：Xiuye Gu1, Tsung-Yi Lin2, Weicheng Kuo1, Yin Cui1; 1Google Research, 2Nvidia∗
- 实验环境：21年4月就上传到arXive上了，也就是从CLIP出来到工作结束，前后2个月，训练了460个epoch，没有大量TPU根本做不出来这么高质量工作，不愧是Google+Nvidia
- 数据集：
- [返回上一层 README](../README.md)
***附录有很多好玩的，记得去看***

# 引言写法
1. 上来一张图，然后作者提出一个问题，直接把这篇文章要做什么引了出来
    - 一句话把文章的研究动机说的明明白白
    
    ![ViLD example](../pictures/ViLD/ViLD%20example.png)
    
# 一、解决的问题
1. 就是要做一个Open vocabulary的目标检测，从而能够检测到任意的新的物体类别
2. 现在目标检测数据集标注的类别都太有限了(有限的类别叫做base category，基础类)，只能检测泛泛的内容，给不出具体的东西特征
3. 能不能在现有的数据集基础之上，不去做额外标注，模型就可以具有检测Novel categories，新的类别的能力（有点像生成模型？？）

# 二、做出的创新
1. CLIP当teacher，去蒸馏自己的网络，从而能达到zero-shot去做目标检测的目的

# 三、设计的模型

![ViLD model](../pictures/ViLD/ViLD%20model.png)

- (a)是baseline，有监督的方法
- (b)(c)(d)都是ViLD的方法，(b) + (c) = (d)，最后还提出了一个能让ViLD推理更快的ViLD ensemble

1. baseline和ViLD-text
    1. baseline是一个maskRCNN，两阶段的分类器
        - 第一阶段会出一些region Proposal（N proposal）
        - 第二阶段就是根据n个proposal，过一个detection head，得到一些region embeddings，在通过一个分类头，得到抽取出来的Bounding box是什么类
        - 目标检测一般分为两块：怎么定位和怎么分类，定位就是bounding box画的准不准，分类就是bounding box里面的物体判断的准不准
        - 这篇论文有点把它们解耦开来的意思，所有的框架图几乎都是从第二个阶段才开始，输入都是n个proposal，第一阶段都没有画
    2. ViLD text
        - 想做open vocabulary，或者是zero-shot这种object detection，就要和文本联合起来
        - 根据n个proposal，过一个detection head，经过一些操作之后就得到了一些region embeddings，然后就是算一些文本的embedding，把物体的类别拿过来，然后给一些prompt，生成一个句子，然后把句子扔给任何一个文本编码器就可以了
            - 注意：文本来自物体的类别，做的还是有监督的学习，类别还是base category（数据集里的基础类），因此在这个阶段，ViLD的text模型，只是把图像的特征和文本的特征联系到了一起，它的open vocabulary/zero-shot的性能还有待加强
        - Text Embedding作者标成了蓝色，代表它是被锁住的，至始至终没有改变过参数，没有参加过预训练
        - 有了图像特征N region embeddings和文本特征Text Embeddings，就可以做一个点乘，相似度就是最后分类的logics，就可以去做cross entropy loss，进行模型的训练
        - background：由于做的是有监督的训练，用的都是基础类，不在基础类里的所有别的类别，只能全部归为背景类。因此背景类的学习非常的关键，专门有一个学习背景的embedding，需要在模型训练的时候，把它学好。和Text Embeddings一样，直接和N region embeddings做点乘
    3. ViT训练的数学公式
    
        ![ViLD function](../pictures/ViLD/ViLD%20function.png)
        
        1. 假设有一个图像I，$\phi(I,r)$ 就是去抽取一下图像的特征，r就是提前知道的proposal，就是抽取出来的bounding box candidate，经过额外层R的计算，就得到了region embedding $e_r$
        2. 接下来定义一个 $e_{bg}$ background embedding，还有文本特征 $t_1 \quad t_2 ... t_{|C_B|}$，数据集里有多少个基础类，也就是有多少个 $C_B$，也就是这个text有多少个embedding，然后region embeddings就分别和background以及所有的文本类去做点乘相似度计算，得到了ViLD的text模型的输出，类似于logitics
        3. 再去做softmax，就可以和ground truth去算cross entropy loss
2. ViLD image
    1. 想法很简单：
        - CLIP做得很好
        - 那么自己设计的图像编码器M region embeddings的输出，尽可能地和CLIP的M region embeddings的输出尽可能的一致就好了
        - 方法就是知识蒸馏
    2. 蒸馏
        - Teacher网络：
            1. 有一些抽好的proposal，也就是那些bounding box
            2. 把它们抠出来（cropping），做一些resize的操作，比如变成224x224
            3. 然后扔给CLIP预训练好的图像编码器（锁住的，从头到尾都没有训练过，保证抽出来的特征和CLIP一样好）
            4. 然后就可以得到图像的特征了
        - Student网络：
            1. 就是之前一直用的目标检测的框架
            2. 先有proposal
            3. 然后过检测头
            4. 然后通过Projection layer和L2 Normalization抽一些特征
            5. 希望抽出来的M region embeddings和CLIP尽可能地接近
        - 最后用简单的L1 loss做一下蒸馏就可以了
        - 监督信号是CLIP带来的图像编码，就不再受基础类的限制了
        - 利用CLIP蒸馏大大增加了open vocabulary的能力
        - 小小的弊端：
            - baseline和Text都是N个proposal，到了ViLD image变成了M个proposal？
            - 主要是为了让训练变得更加的快速，在训练开始之前，就抽好了M个pre-computed proposal，等训练的时候，直接去加载M image embeddings就好了。而N proposal是随时可以改变的
3. ViLD
    1. ViLD Text和ViLD image的合体
    2. 两个目标函数
    3. 整个模型框架没有改变
    4. 右边只有训练的时候才会用到
    5. 为了方便，N和M一块给，等到时候再劈开，N个去算cross entropy loss，M个去算L1 loss
4. 模型总览图

    ![ViLD overview](../pictures/ViLD/ViLD%20overview.png)
    
    - 上面的部分是模型的训练
    - 下面是推理过程
# 四、实验结果

![ViLD LVIS](../pictures/ViLD/ViLD%20LVIS.png)

1. LVIS是一个非常长尾的一个目标检测数据集，一共有1203类，类别非常多，但是图片还是COCO的图片，因此会有很多非常不常见的物体，只标注了一次或者两次
2. （rare），针对这几个分别去算AP
3. 作者主要去体现open vocabulary/zero-shot，因此把frequent和common类算成了base category，这样就一共有866类，这是在训练的时候可见的类
4. 剩下的337就当作了Novel category，去做zero-shot
5. 有监督训练的网络，还用了RFS（repeated factor sampling 一种数据采样的方式，能够帮助你解决数据长尾问题，就尽可能的多去采一些尾部的类别，少采一些头部的类别，尽量让他均衡一些），这是一个很强的基线模型，网络用的是Res50，在LVIS结果只有12.3
6. 但是ViLD text就很接近这个数字了。而ViLD作为一个zero-shot，大幅度超过了有监督训练模型的结果。不过这也是利用了LVIS数据集的特性，标注的太少了，用RFS也没啥用，而且会导致有监督模型越训越差，所以才会有这么大差距
7. ViLD实验做得很全，把backbone换了，数据集换了，都可以发现ViLD都可以直接用过来，而且性能都是大幅度提升
8. 可以直接拓展到其他数据集上

![ViLD dataset](../pictures/ViLD/ViLD%20dataset.png)

## 1、比之前模型的优势

## 2、有优势的原因

## 3、改进空间

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
