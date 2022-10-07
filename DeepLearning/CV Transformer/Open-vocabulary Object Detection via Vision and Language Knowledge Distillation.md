# 论文信息
- 时间：2022
- 期刊：ICLR
- 网络名称：ViLD
- 意义：CLIP在目标检测领域的应用
- 作者：Xiuye Gu1, Tsung-Yi Lin2, Weicheng Kuo1, Yin Cui1; 1Google Research, 2Nvidia∗
- 实验环境：21年4月就上传到arXive上了，也就是从CLIP出来到工作结束，前后2个月，训练了460个epoch，没有大量TPU根本做不出来这么高质量工作，不愧是Google+Nvidia
- 数据集：

# 引言写法
1. 上来一张图，然后作者提出一个问题，直接把这篇文章要做什么引了出来
    - 一句话把文章的研究动机说的明明白白
    ![ViLD example](../pictures/ViLD%20example.png)
# 一、解决的问题
1. 就是要做一个Open vocabulary的目标检测，从而能够检测到任意的新的物体类别
2. 现在目标检测数据集标注的类别都太有限了(有限的类别叫做base category，基础类)，只能检测泛泛的内容，给不出具体的东西特征
3. 能不能在现有的数据集基础之上，不去做额外标注，模型就可以具有检测Novel categories，新的类别的能力（有点像生成模型？？）

# 二、做出的创新
1. CLIP当teacher，去蒸馏自己的网络，从而能达到zero-shot去做目标检测的目的

# 三、设计的模型
![ViLD model](../pictures/ViLD%20model.png)
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
            - 注意：文本来自物体的类别


# 四、实验结果

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