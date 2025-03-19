# 对比学习串烧
> 从2018年开始

## 目录
- [百花齐放（2018-2019）](#百花齐放2018-2019)
    - [InstDisc](#instdisc)
        - [InstDisc研究动机](#instdisc研究动机)
        - [InstDisc所用方法](#instdisc所用方法)
        - [InstDisc研究贡献](#instdisc研究贡献)
    - [InvaSpread](#invaspread)
        - [InvaSpread研究动机](#invaspread研究动机)
        - [InvaSpread所用方法](#invaspread所用方法)
        - [InvaSpread研究贡献](#instdisc研究贡献)
    - [CPC(Contrastive Predictive Coding)](#cpccontrastive-predictive-coding)
        - [CPC研究动机](#cpc研究动机)
        - [CPC所用方法](#cpc所用方法)
        - [CPC研究贡献](#cpc研究贡献)
    - [CMC(Contrastive Multiview Coding)](#cmccontrastive-multiview-coding)
        - [CMC研究动机](#cmc研究动机)
        - [CMC所用方法](#cmc所用方法)
        - [CMC研究贡献](#cmc研究贡献)
- [CV 双雄（19年中-20年中）](#cv-双雄19年中-20年中)
    - [MoCo（讲过了）->学习写作思路](#moco讲过了-学习写作思路)
        - [MoCo和其他工作的区别](#moco和其他工作的区别)
        - [MoCo和其他工作的联系](#moco和其他工作的联系)
    - [SimCLR](#simclr)
        - [SimCLR研究动机](#simclr研究动机)
        - [SimCLR所用方法](#simclr所用方法)
        - [SimCLR研究贡献（看看人家的数据增强方法，和消融实验）](#simclr研究贡献看看人家的数据增强方法和消融实验)
    - [MoCov2](#mocov2)
        - [MoCov2研究动机](#mocov2研究动机)
        - [MoCov2所用方法](#mocov2所用方法)
        - [MoCov2研究贡献](#mocov2研究贡献)
    - [SimCLRv2（半页讲升级，大部分篇幅讲怎么做半监督训练）](#simclrv2半页讲升级大部分篇幅讲怎么做半监督训练)
        - [SimCLRv2研究动机](#simclrv2研究动机)
        - [SimCLRv2所用方法](#simclrv2所用方法)
        - [SimCLRv2研究贡献](#simclrv2研究贡献)
    - [SwAV（Swap assignment views）](#swavswap-assignment-views)
        - [SwAV研究动机](#swav研究动机)
        - [SwAV所用方法](#swav所用方法)
        - [SwAV研究贡献](#swav研究贡献)
    - [CPCv2](#cpcv2)
    - [InfoMin](#infomin)
    - [模型大一统](#模型大一统)
- [不用负样本](#不用负样本)
    - [BYOL（Latent就是特征的意思）](#byollatent就是特征的意思)
        - [BYOL研究动机](#byol研究动机)
        - [BYOL所用方法](#byol所用方法)
        - [BYOL研究贡献](#byol研究贡献)
    - [SimSiam（Kaiming He）](#simsiamkaiming-he)
        - [SimSiam研究动机](#simsiam研究动机)
        - [SimSiam所用方法](#simsiam所用方法)
        - [SimSiam代码](#simsiam代码)
        - [SimSiam研究贡献](#simsiam研究贡献)
- [Transformer](#transformer)
    - [MoCov3（半页讲升级，大部分篇幅讲怎么做自监督训练，使ViT训练稳定）](#mocov3半页讲升级大部分篇幅讲怎么做自监督训练使vit训练稳定)
        - [MoCov3研究动机](#mocov3研究动机)
        - [MoCov3所用方法](#mocov3所用方法)
        - [MoCov3代码](#mocov3代码)
        - [MoCov3研究贡献](#mocov3研究贡献)
    - [DINO](#dino)
        - [DINO研究动机](#dino研究动机)
        - [DINO所用方法](#dino所用方法)
        - [DINO代码](#dino代码)
        - [DINO研究贡献](#dino研究贡献)
- [回顾](#回顾)
- [返回上一层 README](../README.md)



# 百花齐放（2018-2019）
## InstDisc
### InstDisc研究动机
1. 代理任务：个体判别任务instance discrimination
2. memory bank
3. InstDisc这种无监督的学习方式：
    ![InstDisc research motivation](../pictures/Overview%20of%20Comparative%20Learning/InstDisc/InstDisc%20research%20motivation.png)
    - 把按照类别分类的有监督信号推到了极致。把每一个instance（每一张图片）都看成是一个类别，目标就是学习一种特征，把每一个图片都区分开来

### InstDisc所用方法
![InstDisc model](../pictures/Overview%20of%20Comparative%20Learning/InstDisc/InstDisc%20model.png)
- 通过一个卷积神经网络，把所有图片都编码成一个特征，他们希望这个特征，在最后的特征空间里，能够尽可能地分开
- 正样本就是图片本身，可能会经过一些数据增强；负样本自然是数据集里其他的图片
- 大量的负样本特征存在于memory bank中，所有图片的特征都存进去
- 对ImageNet这个数据集来说，他有128万张图片，就意味着memory bank里要存128万行，意味着每个特征的维度不能太高，不然存储代价就太大了，本文选择128维
- 前向过程：
    - 假设现在batch size是256，即有256张图片进入编码器
    - 通过一个Res 50，最后的维度是2048维，然后把它降到128维
    - batch size是256，证明我们有256个正样本，负样本就是从memory bank中随机的抽一些负样本出来。文中抽了4096个负样本出来
    - 有了正、负样本，就可以用NCE loss去算对比学习的目标函数了
    - 一旦更新完了网络，就可以把这个mini-batch的数据样本，它所对应的那些特征，到memory bank里更换掉，memory bank就得到了更新
    - 接下来就反复这个过程，不停的更新编码器和memory bank，让学到的特征尽可能有区分性
- 论文细节：
    - Proximal Regularization：给模型加了约束，从而让memory bank里面的特征进行动量式的更新
    - 实验设置中的超参数设置

### InstDisc研究贡献
1. 提出了个体判别的任务
2. 用这个代理任务和NCE loss去做对比学习，从而取得了不错的无监督表征学习效果
3. 同时提出了用别的数据结构去存大量的负样本
4. 以及如何对这个特征进行动量的更新

## InvaSpread
- CVPR 2019
- SimCLR的一个前身

### InvaSpread研究动机
- 没有使用额外的数据结构去存储负样本，它的正负样本就是来自于同一个mini batch
- 只使用一个编码器进行端到端的学习
- 想法就是最基本的对比学习

![InvaSpread research motivation](../pictures/Overview%20of%20Comparative%20Learning/InvaSpread/InvaSpread%20research%20motivation.png)
- 对相似的图片、相似的物体，它的特征应该保持不变；对不相似、不同的物体，应该尽可能的分开
### InvaSpread所用方法
![InvaSpread model](../pictures/Overview%20of%20Comparative%20Learning/InvaSpread/InvaSpread%20model.png)
- 代理任务上选取了个体判别任务
- 前向过程：
    - batch size 256，经过数据增强，又得到了256张图片
    - 对于 $x_1$ 来说，它的正样本就是 $\hat{x}_1$ ；负样本是其他所有图片，包括原始图片及数据增强后的图片。即正样本数目是256，负样本数目是 $(256-1)\cdot 2$ 
    - 过CNN，再降维到128
    - 目标函数是NCE loss的一个变体
- 为什么要从一个mini-batch里抽正负样本：这样就可以用一个编码器，去做端到端的训练
### InvaSpread研究贡献
1. 为什么没有SinCLR效果好？：
    - 是因为字典不够大
    - 做对比学习的时候，负样本数量最好足够多

## CPC(Contrastive Predictive Coding)
- 前两个都是判别式任务，这个是生成式任务
- CPC结构通用，还可以处理音频、图片、文字和强化学习
### CPC研究动机
1. 全能结构
### CPC所用方法
![CPC model](../pictures/Overview%20of%20Comparative%20Learning/CPC/CPC%20model.png)
- 用音频信号做了输入
- 模型思想：
    - 有一个持续的序列，把之前时刻的序列都扔给一个编码器
    - 编码器返回一些特征
    - 把特征喂给一个自回归的模型 $g_{ar}$ (auto-regressive)
        - 一般常见的自回归模型就是RNN或者LSTM模型
    - 每一步的输出就会得到 $c_t$ (context representation)代表上下文的一个特征模式
    - 用 $c_t$ 预测未来，与正确的输出做对比，得到正样本
    - 负样本定义广泛，可以给任意输出去和 $c_t$ 编码，就是负样本
### CPC研究贡献
1. Info NCE loss

## CMC(Contrastive Multiview Coding)
- 数据集：NYU RGBD：这个数据集有4个视角
### CMC研究动机
1. 人身上的器官相当于很多的传感器，每一个视角都有噪声，而且可能是不完整的；最重要的是那些信息，是在视角中共享的
2. 要学一个非常强大的特征，抓住所有视角下的互信息（关键信息）
### CMC所用方法
![CMC model](../pictures/Overview%20of%20Comparative%20Learning/CMC/CMC%20model.png)
- 4个视角：
    - 原始图像
    - 图像的深度信息
    - SwAV ace normal
    - 物体分割图像
- 虽然不同的输入来自不同的传感器，但是都是一个物体，都是正样本（来自不同视角）
### CMC研究贡献
1. 第一个做多视角的对比学习，证明对比学习的灵活性，多模态的潜力
2. 局限性：不同的视角需要不同的编码器，计算量会比较大


# CV 双雄（19年中-20年中）
## MoCo（讲过了）->学习写作思路
### MoCo和其他工作的区别
1. 把之前的工作都总结成了一个字典查询的问题
2. 提出了队列，实现更大的字典
3. 提出了一个动量编码器，解决字典特征不一致的问题
### MoCo和其他工作的联系
1. 超参数几乎照搬InstDisc

## SimCLR
### SimCLR研究动机
- 和InvaSpread的区别：
    - 用了数据增强
    - 加了MLP层
    - 用了更大的batch-size
### SimCLR所用方法
![SimCLR model](../pictures/Overview%20of%20Comparative%20Learning/SimCLR/SimCLR%20model.png)
- 有一个mini-batch的图片 $x$ ，对这个mini batch所有的图片做数据增强，不同的数据增强得到 $x_i$ 和 $x_j$ ，这就是正样本，数量为N；负样本就是 $(2N-1)$
- 进行编码，两个编码器共享参数
- projection head $g(\cdot)$ 是一个MLP（降维），只有训练的时候用，下游任务的时候就给他扔掉。这是提点的关键，非常惊喜，非常正经，非常简单
- 最后衡量正样本之间是否能达到最大一致性，用normalized temperature-scaled的交叉熵函数（L2归一化，loss上乘 $\tau$ ），和Info NCE loss非常接近
### SimCLR研究贡献（看看人家的数据增强方法，和消融实验）
1. 优势不在于单一的工作，而是把之前所有的工作结合在一起
2. 做了非常精细的消融实验

## MoCov2
### MoCov2研究动机
1. 看到SimCLR好的结果之后，发现他们的技术，都是即插即用，用到了MoCov2
### MoCov2所用方法
![MoCov2 model](../pictures/Overview%20of%20Comparative%20Learning/MoCov2/MoCov2%20model.png)
- 加了MLP层
- 加了数据增强
- 训练的时候用了cosine learning rate schedule
- 训练更长的epoch 200->800
### MoCov2研究贡献
![MoCov2 result1](../pictures/Overview%20of%20Comparative%20Learning/MoCov2/MoCov2%20result1.png)

![MoCov2 result2](../pictures/Overview%20of%20Comparative%20Learning/MoCov2/MoCov2%20result2.png)

- 效果的原因：就是硬件条件
    - SimCLR：8台8卡机，64张GPU
    - MoCov2：1台8卡机

## SimCLRv2（半页讲升级，大部分篇幅讲怎么做半监督训练）
### SimCLRv2研究动机
1. 与MoCo融合
### SimCLRv2所用方法
![SimCLRv2 model](../pictures/Overview%20of%20Comparative%20Learning/SimCLRv2/SimCLRv2%20model.png)
- 自监督训练一个大的模型出来
- 有了这么好的模型之后，用一小部分有标签数据做有监督微调
- 微调结束了，相当于获得了一个teacher模型，生成很多标签，这样就可以在无标签数据上去做自学习了

- v1->v2：
    - 用更大的模型，152层残差网络
    - 2层MLP
    - 使用了动量编码器：因为负样本够多，所以提升较少
### SimCLRv2研究贡献

## SwAV（Swap assignment views）
### SwAV研究动机
1. 给一张图片生成不同视角，用一个视角特征，预测另一个视角
2. 对比学习和聚类融合
### SwAV所用方法
![SwAV model](../pictures/Overview%20of%20Comparative%20Learning/SwAV/SwAV%20model.png)
- 特征做对比，做近似，浪费
- 不和负样本比，和更简单的东西——聚类中心比（prototype），维度 $D\cdot K$ ，d就是特征的维度，k是聚类的数量
- $q_1$ 和 $q_2$ 可以互相预测
- 一个小trick->multi crop（对比学习和聚类结合没有太多用处，multi-crop是关键）
    - 相比于之前工作的两个crop，使用多个crop
    - 增加了正样本数量
    - 为了不增加太多计算量，使用了小crop
    - 关注了更多的全局信息，而不是中心信息
### SwAV研究贡献
1. 聚类中心数量（3000）比负样本数量（几万）少了很多
2. 聚类中心比负样本有语义含义
3. CNN刷榜最高

## CPCv2 
- 融合了更多技巧
- 用了更大模型
- 用了更大的图像块
- 做了更多方向上的预测任务
- batch norm换layer norm
- 使用了更多的数据增强
## InfoMin
- 最小化互信息
- 不多不少的互信息
- 拿到很好的数据视角

## 模型大一统
- NCE loss
- 模型就是编码器+MLP和ReLU
- 用更多的数据增强
- 动量编码器
- 训练的更久

# 不用负样本

## BYOL（Latent就是特征的意思）
### BYOL研究动机
1. 自己跟自己学
2. 负样本是一个约束，给模型学习动力，防止模型学到捷径
### BYOL所用方法
![BYOL model](../pictures/Overview%20of%20Comparative%20Learning/BYOL/BYOL%20model.png)
- 一个mini batch的输入，经过两次数据增强，得到了 $v$ 和 $v^{'}$
- 通过编码器得到特征，两个编码器结构相同，更新不同，下面的是动量更新
- projection head
- 有趣的地方：
    - 做一个预测
    - 没有像SwAV有聚类中心
    - 用自己一个视角特征，预测另一个特征
- 训练完成后，只有上方的编码器留下了，剩下的都拿掉，再去做下游任务
- 目标函数不一样：
    ![BYOL loss](../pictures/Overview%20of%20Comparative%20Learning/BYOL/BYOL%20loss.png)
- batch norm带来的问题（论文关键，作者没发现，是别人发现的）
    ![SimCLR struct](../pictures/Overview%20of%20Comparative%20Learning/SimCLR/SimCLR%20struct.png)
    - MLP有两个BN操作
    ![MoCov2 struct](../pictures/Overview%20of%20Comparative%20Learning/MoCov2/MoCov2%20struct.png)
    - MLP没有BN
    ![BYOL struct](../pictures/Overview%20of%20Comparative%20Learning/BYOL/BYOL%20struct.png)
    - MLP第一个全连接层后面有BN
    
    - 博主没用BN，效仿MoCov2，然后训练坍塌了
        ![BN experiment](../pictures/Overview%20of%20Comparative%20Learning/BYOL/BN%20exprienment.png)
        - 只要放了BN就好使
        - BN：把一个batch里的所有样本特征，算一下它们的均值和方差（running mean\running variance），用整个batch算的均值和方差，去做归一化
        - 这也证明了，当在算某一个正样本的loss时，也看到了其他样本的特征，有信息泄漏的
        - 因为有信息泄露，可以把其他样本的信息想象成隐式负样本。当有了BN时，BYOL不光自己跟自己学，其实也在做对比，做的对比任务是：
            - 当前正样本图片和平均图片差别

- **但是这样会导致失去创新型，还是做了对比学习**

- 作者写了另一篇文章回应：BYOL无BN也可以工作
![BYOL without BN](../pictures/Overview%20of%20Comparative%20Learning/BYOL/BYOL%20without%20BN.png)
- batch norm确实关键
- 特例：
    - 即使MLP有BN，BYOL还是训练失败了，就不能解释BN很关键了
    - 当encoder和MLP都没有BN的时候，SimCLR也失败了
    - 证明了BN不提供隐式负样本
- 达成一致：
    - BN还是之前的作用
    ![BYOL new intro](../pictures/Overview%20of%20Comparative%20Learning/BYOL/BYOL%20new%20inro.png)

### BYOL研究贡献
1. 不用负样本

## SimSiam（Kaiming He）
### SimSiam研究动机
1. 更简单的结构
2. 不用负样本
3. 不需要大的batch size
4. 不用动量编码器
### SimSiam所用方法
![SimSiam model](../pictures/Overview%20of%20Comparative%20Learning/SimSiam/SimSiam%20model.png)
- 和BYOL唯一区别，没用动量编码器

### SimSiam代码
```
# f: backbone + projection mlp
# h: prediction mlp
for x in loader: # load a minibatch x with n samples
x1, x2 = aug(x), aug(x) # random augmentation
z1, z2 = f(x1), f(x2) # projections, n-by-d
p1, p2 = h(z1), h(z2) # predictions, n-by-d
L = D(p1, z2)/2 + D(p2, z1)/2 # loss
L.backward() # back-propagate
update(f, h) # SGD update
def D(p, z): # negative cosine similarity
z = z.detach() # stop gradient
p = normalize(p, dim=1) # l2-normalize
z = normalize(z, dim=1) # l2-normalize
return -(p*z).sum(dim=1).mean()
```

### SimSiam研究贡献
1. SimSiam能成功训练，不会坍塌，是因为有stop gradient的操作
2. 提出假设，因为有stop gradient，可以归结为EM算法
3. 一个训练过程或者说一套模型参数，被认为劈成了两份，相当于我们在解决两个子问题，更新交替进行
4. 也可以理解为k-means聚类问题

![Kaiming He conclu](../pictures/Overview%20of%20Comparative%20Learning/SimSiam/Kaiming%20He%20conclu.png)


![SimSiam result](../pictures/Overview%20of%20Comparative%20Learning/SimSiam/SimSiam%20result.png)
- batch size只有何凯明大佬能用256
- 动量编码器真的很好用
- 只从分类任务来看，BYOL就好
- 物体检测、实力分割：MoCov2和SimSiam更好
- MoCov2对于下游任务来说，更快更稳

# Transformer
## MoCov3（半页讲升级，大部分篇幅讲怎么做自监督训练，使ViT训练稳定）
### MoCov3研究动机
1. 自监督的训练
### MoCov3所用方法
- MoCov2和SimSiam的自然延伸工作
- 骨干网络，从残差换成了ViT
    ![res-trans](../pictures/Overview%20of%20Comparative%20Learning/MoCov3/res-tans.png)
    - 大batch size反倒不好
- 小trick：
    - 去看每次回传的梯度情况
    - loss大幅度震动，梯度也有波峰，产生在第一层的patch projection
    - patch projection：ViT，如何把图片打成patch，可训练的全连接层
    - 初始化知乎，冻住，不让它训练就解决了
    - ***tokenization的重要性***
### MoCov3代码
```
# f_q: encoder: backbone + proj mlp + pred mlp
# f_k: momentum encoder: backbone + proj mlp
# m: momentum coefficient
# tau: temperature
for x in loader: # load a minibatch x with N samples
x1, x2 = aug(x), aug(x) # augmentation
q1, q2 = f_q(x1), f_q(x2) # queries: [N, C] each
k1, k2 = f_k(x1), f_k(x2) # keys: [N, C] each
loss = ctr(q1, k2) + ctr(q2, k1) # symmetrized
loss.backward()
update(f_q) # optimizer update: f_q
f_k = m*f_k + (1-m)*f_q # momentum update: f_k
# contrastive loss
def ctr(q, k):
logits = mm(q, k.t()) # [N, N] pairs
labels = range(N) # positives are in diagonal
loss = CrossEntropyLoss(logits/tau, labels)
return 2 * tau * loss
```
### MoCov3研究贡献
1. 解决了训练震荡的问题

## DINO
### DINO研究动机
1. ViT在自监督训练的过程中有非常多有趣的特性
2. 不用任何标签训练的ViT，把自注意力图拿出来，它们非常准确的抓住了每个物体的轮廓
### DINO所用方法
![DINO model](../pictures/DINO/DINO%20model.png)
- 自蒸馏框架，和BYOL一样，自己跟自己学
- 每个部分换了个名字，过程和BYOL以及SimSiam类似
- 避免模型坍塌，做了一个centering的操作：把整个batch里的样本都算一个均值，然后减掉这个均值，就算是centering
- 很像BYOL对BN的讨论
### DINO代码
``` 
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params
for x in loader: # load a minibatch x with n samples
x1, x2 = augment(x), augment(x) # random views
s1, s2 = gs(x1), gs(x2) # student output n-by-K
t1, t2 = gt(x1), gt(x2) # teacher output n-by-K
loss = H(t1, s2)/2 + H(t2, s1)/2
loss.backward() # back-propagate
# student, teacher and center updates
update(gs) # SGD
gt.params = l*gt.params + (1-l)*gs.params
C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)
def H(t, s):
t = t.detach() # stop gradient
s = softmax(s / tps, dim=1)
t = softmax((t - C) / tpt, dim=1) # center + sharpen
return - (t * log(s)).sum(dim=1).mean()
```
- 和MoCov3太像了，前向过程几乎一模一样
### DINO研究贡献

# 回顾
![contra history](../pictures/Overview%20of%20Comparative%20Learning/History/contra%20history.png)
1. 第一阶段
    - InstDisc：提出了个体判别的任务；memory bank外部数据结构去存储负样本，从而达到一个又大又一致的字典
    - InvaSpread：不用外部结构，端到端学习；只用了一个编码器，从而可以端到端学习；受限于batch size大小，性能不太好
    - CPC v1：提出Info NCE loss；CPC v1是一个预测型代理任务，不仅可以做图像，还可以做音频、视频、文字和强化学习
    - CMC：把两个视角的任务，扩展到了多个视角，从而给之后的多视角，多模态的工作打下了铺垫
    - Deep cluster：没有讲；是基于聚类学习的，当时还没有用对比学习
2. 第二阶段
    - MoCo v1：InstDisc延伸性工作：
        - 把memory bank比那成一个队列
        - 把动量更新特征，变成了动量更新编码器，从而能预训练一个很好的模型；第一个在很多视觉任务上，让一个无监督预训练模型比有监督预训练模型表现好的方法
        - 属于使用外部数据结构的
    - SimCLR v1：InvaSpread的延伸性工作；用了很多的技术：
        - 加大了batch size
        - 用了更多的数据增强
        - 加了projection head
        - 训练了更长的时间
        - 在ImageNet上取得了很好的结果
    - CPC v2：把上述好的方法拿来用了一遍，比v1在ImageNet上高了30多个点
    - CMC：把这些分析了一边，提出了InfoMin的原则：
        - 两个样本\两个视角之间的互信息，要不多不少才是最好的
    - MoCo v2：看到SimCLR的技巧都很有用，把即插即用的技术拿过来用了一遍
        - 比MoCo v1和SimCLR v1都要好
    - SimCLR v2：SimCLR作者对模型也做了一些改动
        - 主要去做半监督学习了
    - SwAV：借用了聚类思想（提Deep cluster是为了引出SwAV），把聚类学习和对比学习结合起来的工作
        - 核心源于multi crop技术
        - 否则跟SimCLR和MoCo v2结果差不多

3. 第三阶段
    - BYOL：处理负样本太麻烦，不要负样本了
        - 自己跟自己学
        - 编成预测任务
        - 目标函数也很简单，不使用Info NCE了，用简单的MSE loss就可以训练出来
        - 但是大家觉得不可思议，于是出来一个博文，假设说BYOL能够工作，主要因为有BN提供了隐式负样本，所以BYOL可以正常训练不会坍塌。
        - BYOL v2：做了一系列式样，发现BN只是帮助了模型的训练；用另外一种方式提供更好的初始化，不需要BN提供的batch统计量，照样能工作
    - SimSiam：跟着BYOL出来了。
        - 把之前的工作都总结了一下
        - 感觉大家都在堆技术，堆的太多了就不好分析了，领域不好推进了
        - 化繁为简，提出了孪生网络的学习方法
            - 既不需要用大的batch size
            - 也不需要用动量编码器
            - 也不需要负样本
            - 照样取得不错的结果
        - stop grandient的操作是至关重要的
        - 因为这个操作的存在，SimSiam可以看作是一种EM算法，通过逐步更新的方式，避免模型坍塌
    - BarlosTwins：换了一个目标函数，把之前大家做的对比、预测，变成了两个矩阵之间去比相似性。由于在21年3月提出，很快就湮没在了ViT之中
4. 第四阶段（ViT）
    - MoCo v3：把骨干网络从残差换成了ViT，但是训练不稳定
        - 把patch projection layer冻住
        - 能够提高模型训练的稳健性
    - DINO：把骨干网络从残差换成了ViT，但是训练不稳定
        - 把teacher网络的输出，先归一，即centering
        - 能够提高模型训练的稳健性

5. MAE火了，大家去掩码学习了
