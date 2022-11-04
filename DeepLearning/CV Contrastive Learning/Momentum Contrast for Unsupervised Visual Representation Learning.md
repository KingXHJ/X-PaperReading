# 论文信息
- 时间：2019
- 期刊：CVPR
- 网络名称：MoCov1
- 意义：无监督训练效果也很好
- 作者：Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick; Facebook AI Research (FAIR)
- 实验环境：8 GPUs
- 数据集：PASCAL VOC, COCO

# 一些前瞻知识
- ***对比学习的灵活性就在于，只要找到一个方法去定义正样本和负样本就可以实现***
- ***无监督学习和自监督学习（无监督学习的一种）一般有两个方向可以做——在代理任务上做文章，和在目标函数上做文章***
- 代理任务：一般是指大家不太感兴趣的任务，并不是分类、分割、检测有实际应用场景的任务。代理任务主要是为了学习一个好的特征
- 目标函数：它的研究可以和代理任务分开。MoCo就是这么做的，它提出的又大又一致的字典，主要影响Info NCE这个目标函数的计算
    - 生成式网络的目标函数：原图和新建图之间的差异， $L_1$ 和 $L_2$ loss
    - 判别式网络的目标函数：cross-entropy/margin-based
    - 对比学习目标函数：去特征空间里，衡量各个样本对之间的相似性，让相似物体的特征拉的尽量近，不相似的物体特征推的尽量远。
        - 与上述两种的区别就是：
        - 上述两种最终的目标是固定的
        - 但是对比学习的目标是在训练中不断变化的，由编码器抽取的数据特征决定
    - 对抗性的目标函数：衡量的是两个概率分布之间的差异（GAN），主要做无监督数据生成的
# 一、解决的问题
1. 无监督预训练在NLP很成功，但是在CV不成功。作者认为是信号空间的不同，NLP是离散的字典，但是CV是在一个连续的、高维的空间。
2. 无监督学习就该训练一些编码器，实现字典的查找
# 二、做出的创新
1. 无监督学习的表征工作，在分类任务上逼近有监督基线模型，在主流的视觉任务上，检测、分割、人体关键点检测都超越了有监督预训练模型
2. 不需要大规模的标好的数据集做预训练
3. 利用了动量对比学习的方法： $$y_t = my_{t-1}+(1-m)x_t$$ 动量的本质就是不希望当前的输出完全依赖于之前的输出或当前的输入
4. 把对比学习看成是字典查询的任务。做了一个动态的字典：
    - 有一个队列：队列里的样本不需要做梯度回传，所以就可以往队列里放很多负样本，从而使字典变得很大
    - 还有一个移动平均的编码器：是想让字典里的特征尽量保持一致
5. `linear protocol`：先预训练好了一个骨干网络，把它用到不同数据集上的时候，先把骨干网络冻住（back bone freeze），只去学习最后的那个全连接层（分类头）。这就相当于，我们把提前学好的预训练模型当作了一个特征提取器，只抽特征，可以间接证明预训练的特征到底学的好不好。
6. MoCo学好的特征是可以迁移到下游任务的
# 三、设计的模型
1. 为什么是动态字典：
    ![MoCo Contrast work](../pictures/MoCo%20contrast%20work.png)
    - $x_1$ 经历了不同的变化得到了 $x^1_1$ 和 $x^2_1$ ，一个正样本对
    - 我们称 $x^1_1$ 为anchor（锚点基准）， $x^2_1$ 相对于锚点为 $x_1$ 的正样本（positive），剩下的图片都是负样本（negative）
    - 样本输入编码器，得到特征输出。 $x^1_1$ 和 $x^2_1$ 的编码器可以一样，也可以不一样。但是我们通常让positive和negative输入一样的编码器，获得同样的特征空间
    - 对比学习就是让正样本对在空间里的距离尽可能的接近，并远离负样本对
    - 动态字典：如果把特征 $f_{11}$ 当作一个`query`（ $x^q$ ），把特征 $f_{12}, f_2, f_3,...$ 看作一组`key`（ $x^k$ ），那么这种寻找匹配，就是一个查询字典的过程
    - 本质就是要训练一些编码器，编码的特征可以进行字典查找

2. 效果好的条件
    - 必须要大：字典越大，越可以从高维的视觉空间做抽样。`key`越多，表征的视觉信息越多，越丰富
    - 训练的时候要尽可能的保持一致性：字典里的`key`都应该用相同或者相似的编码器去产生，这样和`query`做对比的时候，才能保证尽可能的一致。如果用不同编码器，`query`很可能找到的是和自己用相同编码器的`key`，而不是相同语义的`key`，变相的引入了一个捷径（`ShortCut solution`）

3. MoCo总览图：
    ![MoCo figure1](../pictures/MoCo%20figure1.png)
    - MoCo的贡献就是在`momentum encoder`生成的`queue`。
    - 字典太大，显卡内存肯定吃不消。所以要想一个办法，能让字典的大小，和每次模型去做前向过程时的batch size大小剥离开
    - 解决方案：用队列这种数据结构。移除老的mini-batch，添入新的mini-batch
    - 但是这样难以保证一致性
    - 于是引入了`momentum encoder` $$\theta_k=m\theta_{k-1}+(1-m)\theta_q$$ 选择一个大的动量 $m$ ，使得 $\theta_k$ 变化缓慢，受 $\theta_q$ 影响较小，就可以尽最大可能保证相对的一致性

4. 选择代理任务：
    - 一个简单的`instance discrimination`任务（个体判别任务）
    - 是什么呢：如果一个`query`和一个`key`，是同一个图片的不同视角（不同的随机裁剪得到的），就说这个`query`和`key`能配上对。

5. Info NCE：
    - $\frac{exp(z_k)}{\sum^{k}_ {i=0}exp(z_ {i})}$ 这是softmax
    - $-log \frac{exp(z _ {k})}{\sum^{k}_ {i=0}exp(z_ {i})}$ 这是cross entropy loss
    - 这里的 $k$ 在有监督学习下，指的是这个数据集里***有多少类别***。在对比学习中，理论上好用，实际行不通。如果用`instance discrimination`当自监督信号的话，那么这里的 $k$ 将是非常巨大的数字，那就是有多少图片，就有多少类了。代表了***负样本数的多少***。softmax在有这么多类别的情况下工作不了
    - 所以引进了NCE（noise contrastive estimation） loss：把这么多类简化成二分类任务，data sa
    - 但是二分类可能对模型学习不友好，在那么多噪声中，大家可能不是一个类，看成多分类比较好，NCE loss变成了Info NCE loss $$\mathcal{L} _ {q} = -log \frac{exp(q \cdot k _ {+} / \tau)}{\sum^{K}_ {i=0}exp(q \cdot k_ {i} / \tau)}$$
        - $\tau$ 是温度的超参数，控制分布形状
        - $q\cdot k$ 类比于softmax用的 $z$ ，是logit

6. Momenum Contrast
    - 对比学习在高维的、连续的输入上打造一个离散的、动态的字典
    - 动态的原因：字典里的`key`都是随机取样的；给`key`做编码的编码器也是不断在训练中变化

7. Dictionary as a queue:
    - 核心就是把字典用一个队列表示出来，***FIFO***
    - 整个队列就是一个字典，里面的元素就是放进去的`key`
    - 每一次mini-batch都有新`key`进去，老的`key`出来
    - 用队列的好处就是可以重复使用之前编码好的、老的`key`
    - 这样就可以把mini-batch的大小和字典大小剥离开了
    - 字典一直是数据的子集
8. Momentum update：
    - 用了太大的队列，没有办法做梯度回传
    - 也就是`key`的编码器没有办法通过反向传播的方式去更新参数了
    - 有一个方法是每个iteration后，把`query`学习的参数复制给`key`，但是这会降低所有`key`的一致性
    - 于是引入了`momentum encoder` $$\theta_k=m\theta_{k-1}+(1-m)\theta_q$$ 选择一个大的动量 $m$ ，使得 $\theta_k$ 变化缓慢，受 $\theta_q$ 影响较小，就可以尽最大可能保证相对的一致性。用了 $m=0.999$ 。使用大动量比小动量效果更好
9. Relation to previous mechanisms:
    ![MoCo comparison](../pictures/MoCo%20comparison.png)
    - 之前的方法：
        - end-to-end：q和k都是一个mini-batch来的，encoder也可以梯度学习了，但是受限于字典大小
        - memory bank：只有一个q的encoder，把所有特征都存在了memory bank，但是特征一致性不好，每个epoch会导致一致性差的更多；扩展性不好
        - MoCo：也只有一个编码器更新

10. shuffle BN
    - 用了BN以后，很有可能造成当前样本信息的泄露，模型可能通过泄露的信息很容易找到了正样本，而不会去好好学习模型
    - 在多卡训练中，把样本打乱，送到不同的GPU上，在集中回来算loss，这样BN就存在与各个GPU上了

# 四、实验结果
1. Linear Classification Protocol:
    - 居然找到学习率**30**比较好！！！
2. 消融实现
    - queue：
        - 受到硬件限制
        - 一致性不好，性能不好
    - momentum：
        - 动量0.999最好，0.9和0.9999都没0.999好
3. Transferring Features:
    - ***无监督就是要学习一个可迁移的模型***
    - 最有影响力的就是下游任务做微调
    - 在检测上做了实验
        - 归一化解决学习率30的问题
        - 短期学习，防止COCO这种超大数据集，不需要预训练也可以表现得很好
## 1、比之前模型的优势

## 2、有优势的原因

## 3、改进空间
- 从ImageNet 100M的数据集，换到自己的大数据集instagram 10个亿的时候，模型提升只有零点几个点或者一个点。作者认为是应该换个代理任务，来更好的利用起来大数据集
- 有没有可能仿照BERT，把masked auto-encoding结合起来。这不就是MAE哈哈哈哈哈哈哈哈哈哈
# 五、结论

## 1、模型是否解决了目标问题

## 2、模型是否遗留了问题

## 3、模型是否引入了新的问题

# 六、代码
![MoCo code](../pictures/MoCo%20code.png)
```
# f_q, f_k: encoder networks for query and key
# queue: dictionary as a queue of K keys (CxK)
# m: momentum
# t: temperature
f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples
    x_q = aug(x) # a randomly augmented version
    x_k = aug(x) # another randomly augmented version
    q = f_q.forward(x_q) # queries: NxC
    k = f_k.forward(x_k) # keys: NxC
    k = k.detach() # no gradient to keys
    # positive logits: Nx1
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))
    # negative logits: NxK
    l_neg = mm(q.view(N,C), queue.view(C,K))
    # logits: Nx(1+K)
    logits = cat([l_pos, l_neg], dim=1)
    # contrastive loss, Eqn.(1)
    labels = zeros(N) # positives are the 0-th
    loss = CrossEntropyLoss(logits/t, labels)
    # SGD update: query network
    loss.backward()
    update(f_q.params)
    # momentum update: key network
    f_k.params = m*f_k.params+(1-m)*f_q.params
    # update dictionary
    enqueue(queue, k) # enqueue the current minibatch
    dequeue(queue) # dequeue the earliest minibatch
```
# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论
