# 论文信息
- 时间：2021.09.02
- 期刊：一篇博客(https://distill.pub/2021/gnn-intro/)
- 网络名称：对图神经网络一个简单容易的介绍
- 意义：GNN可视化介绍
- 作者：Benjamin Sanchez-Lengeling, Emily Reif, Adam Pearce, Alexander B. Wiltschko; Google Research
- 实验环境：
- 数据集：
- [返回上一层 README](../README.md)
# 一、解决的问题
![How GNN work](../pictures/GNN/How%20GNN%20work.png)
1. 神经网络被用于处理图的结构和性质上面
2. 图神经网络足够深的时候，顶部一个节点是由上一层的几个节点计算而来，可能一个节点能处理到比较大范围里面的信息
3. 刚刚开始，还有很多可以研究
4. 内容结构：
    - 用什么数据结构表示一张图
    - 图跟别的数据有什么不一样的地方，为什么要用图，而不是卷积神经网络或是别的
    - 构建一个GNN，看看它长什么样子
    - GNN的playground（*往往有这个词，作者都是花了很多心思的，值得去玩一下*）
# 二、图是什么
1. 图是表示实体(entity)之间的一些关系
    - 所谓的实体就是一些点(node)
    - 关系就是一个边(edge)
    - V Vertex (or node) attributes  e.g., node identity, number of neighbors
    - E Edge (or link) attributes and directions  e.g., edge identity, edge weight
    - U Global (or master node) attributes(信息或者属性)  e.g., number of nodes, longest path
2. 图中信息均可以用向量来表示
    ![GNN information](../pictures/GNN/GNN%20information.png)
    - embedding的参数能不能通过学习来学到
3. 图的方向性
    ![GNN direction](../pictures/GNN/GNN%20direction.png)
4. 把图片表示成图
    ![GNN picture to graph](../pictures/GNN/GNN%20picture%20to%20graph.png)
    - 图片中每一个像素是一个点
    - 边用邻接矩阵表示，这个矩阵通常很大很稀疏
    ![GNN sentence to graph](../pictures/GNN/GNN%20sentence%20to%20graph.png)
    - 每一个词表示一个顶点
    - 词和下一个词有一个有向边
    ![graph can represent](../pictures/GNN/graph%20can%20respresent.png)
    - 图可以表示很多实体之间的关系

# 三、图存在的问题
1. 图层面
    - 对图进行分类  eg.识别环
2. 顶点层面
    - 判断关系分裂
3. 边层面
    - 分析关系

# 四、图用在机器学习上的挑战
1. 最核心的问题是：如何把图表示在神经网络上，并且和神经网络是兼容的
    - 四种信息：
        - 顶点
        - 边
        - 全局
        - 图的连接性（每条边到底连接哪两个点）
    - 前三个都可以用向量来表示，连接性可以用连接矩阵表示
2. 问题：连接性矩阵特别大
    - 稀疏，难以处理
    - 连接矩阵的行列交换都是一样的
    - 需要存储高效且排序不影响结果
        - 维护邻接列表
        ![GNN Adjacency List](../pictures/GNN/GNN%20Adjacency%20List.png)
        - 长度和边数一样
        - 第i项是第i条边连接的两个点
# 五、GNN
1. 定义：GNN是一个对图上所有的属性，包括顶点、边和全局的上下文，进行的一个可以优化的变换，这个变换是可以保持住图的对称信息的
    - 用信息传递的神经网络
    - GNN的输入输出都是一个图
    - GNN会对顶点、边和全局的向量进行变化，但是不会改变图的连接性
2. 构建一个GNN
    ![GNN construction](../pictures/GNN/GNN%20construction.png)
    - 对顶点，边和全局分别构造一个MLP
    - MLP的输入输出大小是一样的，取决于输入向量
    - 三个MLP组建了一个GNN
    - 满足了输出的图比输入的图，属性更新了，但是结构没变
3. 最后一层的输出怎么得到预测值
    - 加一个输出维度为分类数量的全连接层，和一个softmax
    ![GNN output](../pictures/GNN/GNN%20output.png)
    - 在这个例子里，所有顶点共享一个全连接层的参数
    - 稍微复杂的情况：想对顶点做分类，但是没有顶点向量怎么办
        - 采用pooling（汇聚）
            - 把和当前点相连的点边的向量拿出来，再拿出全局的向量
            - 把向量都加起来，得到该点的向量
            ![GNN pooling](../pictures/GNN/GNN%20pooling.png)
            ![GNN pooling](../pictures/GNN/GNN%20pooling2.png)
            - 同样的，如果只知道顶点的向量不知道边的向量，操作是类似的
            ![GNN pooling edge](../pictures/GNN/GNN%20pooling%20edge.png)
            - 同样的，如果只知道顶点的向量不知道全局的向量，操作是类似的，把所有的顶点向量加起来
            ![GNN pooling global](../pictures/GNN/GNN%20pooling%20global.png)
4. GNN结构
![GNN construction all](../pictures/GNN/GNN%20construction%20all.png)
- 缺失信息就适当加入汇聚层
- 问题是，并没有通过汇聚，引入图的信息
5. 信息传递（改进pooling）
    - 进入MLP之前，把一个顶点和他的邻居顶点的向量汇聚在一起，进入MLP
    ![GNN information pooling](../pictures/GNN/GNN%20information%20pooling.png)
    - 和卷积有点像
        - 卷积的同一个窗口里是存在相同的加权和
        - 这里没有加权
        - 相同的就是，越深的网络，顶点掌握的信息越多
        - 消息传递的表示方法
        ![GNN information pooling represent](../pictures/GNN/GNN%20information%20pooling%20represent.png)
    - 提前做属性汇聚
        - 顶点信息融合到边中
        - 再把更新后的边的信息融合给顶点
        ![GNN prevoius pooling](../pictures/GNN/GNN%20previous%20pooling.png)
        - 换一个方向或者交替更新
        ![GNN prevoius pooling2](../pictures/GNN/GNN%20previous%20pooling2.png)

6. 全局信息作用
    - master node or context node
    - 虚拟的点，与所有点相连，与所有的边相连
    - 解决图非常大，消息传递较远的时候，信息补充

![GNN predict](../pictures/GNN/GNN%20predict.png)

# 六、GNN playground
1. 模型参数越大，模型准确率的上限越高
2. 向量越大，精度中值越高
3. 层数较高，中值增加
4. **耦合现象严重**
5. 消息传递越多，准确率中值越高

# 七、GNN讨论
1. 图中顶点间，有多种边（有向边/无向边）
2. 图可能是分层的，里面是有子图的（hypernode）
![GNN other possible](../pictures/GNN/GNN%20other%20possible.png)
3. 最后的顶点，看到的图范围越大，但是计算量很大，难以BP，因此要采样，进行分块BP
![GNN sample](../pictures/GNN/GNN%20sample.png)
4. Inductive biases
    - 不对世界做假设，什么东西都学不出来 
    - 卷积神经网络假设的是空间变换的不变性
    - 循环神经网络假设的是时序的连续性
    - 图神经网络的假设，就是之前的两大特点
        - 保持了图的对称性，不管怎么交换顶点顺序，GNN对它的作用都保持不变
        - 比较不同汇聚的操作
            - 求和
            - 求平均
            - 求max
            - 没有一种是特别理想的
        ![GNN test pooling](../pictures/GNN/GNN%20test%20pooling.png)
        - 实际应用中去尝试
5. GCN作为一个子图函数的近似
    - GCN：图卷积神经网络，就是带了汇聚的那一个
    - 还有一个叫MPNN的东西
    - 如果有k层，每层都看它的一个邻居的话，相当于看到一个子图，大小是k
    - 可以看作是以当前点为中心，周围距离为k的范围子图的汇聚
    ![GNN GCN](../pictures/GNN/GNN%20GCN.png)

6. 点和边做对偶
7. 图卷积作为一个矩阵的乘法，和矩阵乘法，和walks on a graph的关系
    - 核心思想是在图上做卷积或者做random walk，等价于把它的邻接矩阵拿出来，然后做一个矩阵的乘法
    - page rank就是在一个很大的图上面做一个随机的游走
        - 拿一个邻接矩阵出来，不断地和一个向量做乘法
    - 图卷积和矩阵的乘法，是高效实现的关键点
8. graph attention network
    - 卷积的权重是和位置相关的
    - 但是对于图来说，不需要有这个位置信息
        - 因为每个顶点的邻居个数不变
        - 可以随意打乱顶点顺序的
        - 权重对位置不敏感
    - 参考注意力机制：
        - 权重是两个顶点向量之间的关系
        - 而不是顶点所在的位置
9. 图的可解释性
    - 抓出来中间过程，看看图都在学什东西

10. generative modeling
    - 图神经网络不改变图结构
    - 想要生成图出来
    - 核心内容是对图的top结构进行有效的建模

# 读者角度（挖掘文章中没有提到的）：
1. 总结文章发现问题的思路
2. 总结文章改进的思想
3. 总结文章还存在或者可以改进的问题
4. 提出对模型参数和细节的一些思考和讨论