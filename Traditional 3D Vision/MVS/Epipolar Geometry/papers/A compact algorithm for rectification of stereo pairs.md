# 论文信息
- 时间：2000
- 期刊：Machine Vision and Applications
- 算法名称：极线校正
- 意义：将对极几何的倾斜平面修正为平行关系
- 作者：Andrea Fusiello, Emanuele Trucco, Alessandro Verri
- 实验环境：
# 补充知识
1. 相机模型
    - 针孔相机由其光学中心 $C$ 和视平面（或图像平面） $\mathcal{R}$ 建模。3D点 $W$ 被投影到由 $\mathcal{R}$ 与包含 $C$ 和 $W$ 的线的交点给出的图像点 $M$ 中。包含 $C$ 且与 $\mathcal{R}$ 正交的线称为光轴，其与 $\mathcal{R}$ 的交点为主点。 $C$ 和 $\mathcal{R}$ 之间的距离是焦距
    - 设 $\mathbf{w}=[x \quad y \quad z]^{T}$ 是 $W$ 在世界参考系中的坐标（任意固定）， $\mathbf{m}=[u \quad v]^{T}$ 是 $M$ 在图像平面（像素）中的坐标。从3D坐标到2D坐标的映射是透视投影，它由齐次坐标中的线性变换表示。设 $\tilde{\mathbf{m}}=[u \quad v \quad 1]^{T}$ 和 $\tilde{\mathbf{w}}=[x \quad y \quad z \quad 1]^{T}$ 分别为 $M$ 和 $W$ 的齐次坐标；则透视变换由矩阵 $\tilde{\mathbf{P}}$ ： $$\begin{equation} \tilde{\mathbf{m}} \cong \tilde{\mathbf{P}} \tilde{\mathbf{w}} \end{equation}$$
    其中， $\cong$ 等于比例因子。
    - 因此，相机由其透视投影矩阵（perspective projection matrices，此后称为PPM）P建模，该矩阵可以使用QR分解，分解为乘积： $$\begin{equation} \tilde{\mathbf{P}}=\mathbf{A}[\mathbf{R}|\mathbf{t}] \end{equation}$$ 其中，矩阵A仅取决于固有参数，并具有以下形式： $$\begin{equation} \mathbf{A}=
            \begin{bmatrix} 
            \alpha _{u} & \gamma & u_ {0} \\ 
            0 & \alpha _{v} &  v_ {0} \\
            0 & 0 & 1
            \end{bmatrix}
            \end{equation}$$ 其中， $\alpha _{u} = - f k_ {u}, \alpha _{v} = − f k_ {v}$ 分别是水平和垂直像素的焦距（ $f$ 是以毫米为单位的焦距， $k_ {u}$ 和 $k_ {v}$ 是沿 $u$ 和 $v$ 轴的每毫米有效像素数）， $(u_ {0}, v_ {0})$是由光轴与视平面的交点给出的主点坐标， $\gamma$ 是模拟非正交 $u - v$ 轴的歪斜因子
    - 摄像机的位置和方向（外部参数）由3×3旋转矩阵 $\mathbf{R}$ 和平移向量 $\mathbf{t}$ 编码，表示将摄像机参考系带入世界参考系的刚性变换
    - 让我们将PPM写为： $$\begin{equation} \tilde{\mathbf{P}} = \begin{bmatrix} \begin{array}{c|c} \mathbf{q}^{T}_ {1} & q_ {14} \\ \mathbf{q}^{T}_ {2} & q_ {24} \\ \mathbf{q}^{T}_ {3} & q_ {34} \end{array} \end{bmatrix} = [\mathbf{Q} | \tilde{\mathbf{q}}] \end{equation}$$
    - 在笛卡尔坐标中，投影（公式(1)）写道： $$\begin{equation} \left\{ \begin{aligned} u & = & \frac{\mathbf{q}^{T}_ {1} \mathbf{w} + q_ {14}}{\mathbf{q}^{T}_ {3} \mathbf{w} + q_ {34}} \\ v & = & \frac{\mathbf{q}^{T}_ {2} \mathbf{w} + q_ {24}}{\mathbf{q}^{T}_ {3} \mathbf{w} + q_ {34}} \end{aligned} \right. \end{equation}$$
    - 焦平面是与包含光学中心 $C$ ，与视平面平行的平面。 $\mathbf{c}$ 的坐标 $C$ 由下式给出： $$\begin{equation} \mathbf{c} = -\mathbf{Q}^{-1} \tilde{\mathbf{q}} \end{equation}$$
    ![change axis](../pictures/compact%20algorithm%20for%20rectification%20of%20stereo%20pairs/change%20axis.png)
    
    - 因此， $\tilde{\mathbf{P}}$ 可以被写为： $$\begin{equation} \tilde{\mathbf{P}} = [\mathbf{Q} | - \mathbf{Q} \mathbf{c}] \end{equation}$$
    - 与图像点 $M$ 相关联的光线是线 $MC$ ，即3D点的集合 $\lbrace \mathbf{w} : \tilde{\mathbf{m}} \cong \tilde{\mathbf{P}} \tilde{\mathbf{w}} \rbrace$ 。参数形式： $$\begin{equation} \mathbf{w} = \mathbf{c} + \lambda \mathbf{Q}^{-1} \tilde{\mathbf{m}}, \quad \quad \quad \quad \lambda \in \mathbb{R} \end{equation}$$
2. 对极几何(Epipolar geometry)
    ![Epipolar geometry](../pictures/compact%20algorithm%20for%20rectification%20of%20stereo%20pairs/Epipolar%20geometry.png)
    - 专有名词
        - 极点 $E_1$ ：右相机坐标原点 $C_2$ 在左像平面上的像
        - 极点 $E_2$ ：左相机坐标原点 $C_1$ 在右像平面上的像 
        - 极平面 $R_1, R_2$ ：由两个相机坐标原点、和物点 组成的平面 
        - 极线 $l_1, l_2(M_{1} E_{1}, M_{2} E_{2})$ ：极平面与两个像平面的交线 
        - 极线约束：给定图像上的一个特征，它在另一幅图像上的匹配视图一定在对应的极线上，即已知 $M_{1}$ ，则它对应在右图的匹配点一定在极线 $l_2$ 上；反之亦然
        - 极线约束给出了对应点重要的约束条件，它将对应点匹配从整幅图像中查找压缩到一条线上查找，大大 减小了搜索范围，对对应点的匹配起指导作用
    
    ![Rectified cameras](../pictures/compact%20algorithm%20for%20rectification%20of%20stereo%20pairs/Rectified%20cameras.png)
    - 当 $C_1$ 位于右相机的焦平面中时，右对极位于无穷远处，对极线在右图像中形成一束平行线。一种非常特殊的情况是当两个极点都在无穷远处时，当线 $C_1, C_2$（基线）包含在两个焦平面中时，即视平面平行于基线时，就会发生这种情况。然后，极线在两幅图像中形成一束平行线。任何一对图像都可以被变换，使得对极线在每个图像中是平行和水平的。此过程称为校正(rectification)
   
# 一、解决的问题
1. 给定一对立体图像，校正确定每个图像平面的变换，使得成对的共轭对极线变得共线并平行于其中一个图像轴（通常是水平轴）。校正后的图像可以被认为是通过旋转原始相机获得的新立体设备获取的。校正的重要优点是计算立体对应关系变得更简单，因为搜索是沿着校正图像的水平线进行的
2. 此前的计算机视觉算法太少，而且对约束限制非常严格

# 二、做出的创新
1. 提出了一种用于一般无约束立体相机的线性校正算法：采用原始相机的两个透视投影矩阵，并计算一对校正投影矩阵
2. 代码非常简单，只有22行matlab代码
# 三、设计的模型
1. 思路：
    - 我们假设立体相机已校准，即PPM(perspective projection matrices) $\hat{p}_{o1}$ 和 $\hat{p}_{o2}$ 已知。校正背后的想法是定义两个新的PPM $\hat{p}_{n1}$ 和 $\hat{p}_{n2}$ ，通过将旧的PPM围绕其光学中心旋转，直到焦平面变得共面，从而包含基线。这确保了极点是无限的；因此，极线是平行的。要具有水平极线，基线必须平行于两个相机的新 $X$ 轴。此外，为了进行适当的校正，共轭点必须具有相同的垂直坐标。这是通过要求新相机具有相同的固有参数来实现的。注意，由于焦距相同，视平面也共面
2. 条件：
    - 新PPM的位置（即光学中心）与旧相机相同，而新方向（两个相机相同）与旧方向的不同之处在于适当的旋转
    - 两个相机的固有参数相同。因此，得到的两个PPM仅在其光学中心上有所不同，它们可以被认为是沿着其参考系的 $X$ 轴平移的单个相机
    
3. 让我们根据因式分解来编写新的PPM。根据等式2和7 $$\begin{equation} \tilde{\mathbf{P}}_ {n1} = \mathbf{A}[\mathbf{R} | - \mathbf{R} \mathbf{c}_ {1}], \quad \quad \tilde{\mathbf{P}}_ {n2} = \mathbf{A}[\mathbf{R} | - \mathbf{R} \mathbf{c}_ {2}] \end{equation}$$ 两个PPM的固有参数矩阵 $A$ 相同，可以任意选择（参见matlab代码）。光学中心 $\mathbf{c}_ {1}$ 和 $\mathbf{c}_ {2}$ 由用等式(6)计算的旧光学中心给出。给出相机姿态的矩阵 $\mathbf{R}$ 对于两个PPM是相同的。它将通过其行向量来指定： $$\begin{equation} \mathbf{R} = \begin{bmatrix} \mathbf{r}^{T}_ {1} \\ \mathbf{r}^{T}_ {2} \\ \mathbf{r}^{T}_ {3} \end{bmatrix} \end{equation}$$ 分别是相机参考系的 $X$ 、 $Y$ 和 $Z$ 轴，以世界坐标表示

4. 根据之前的评论，我们采取：
    1. 平行于基线的新 $X$ 轴： $\mathbf{r}_ {1} = (\mathbf{c}_ {1} - \mathbf{c}_ {2}) / ||\mathbf{c}_ {1} - \mathbf{c}_ {2}||$
    2. 与 $X$ （强制）和 $\mathbf{k}$ 正交的新 $Y$ 轴： $\mathbf{r}_ {2} = \mathbf{k} \wedge \mathbf{r}_ {1}$
    3. 与XY正交的新Z轴（强制）： $\mathbf{r}_ {3} = \mathbf{r}_ {1} \wedge \mathbf{r}_ {2}$
    - 在第二点中， $\mathbf{k}$ 是一个任意单位向量，它固定了新 $Y$ 轴在与 $X$ 正交的平面中的位置。我们将其等于旧左矩阵的 $Z$ 单位向量，从而将新 $Y$ 轴约束为与新 $X$ 和旧的 $Z$ 都正交

5. 注意：当光轴平行于基线时，即当存在纯向前运动时，该算法失败

6. 在Fusiello等人（1998）中，我们对整改要求进行了形式化分析，并证明本节给出的算法满足了这些要求

7. 整流变换(The rectifying transformation)
    - 为了校正（比方说）左图像，我们需要计算将 $\tilde{\mathbf{P}}_ {o1} = [\mathbf{Q}_ {o1} | \tilde{\mathbf{o1}}]$ 的图像平面映射到 $\tilde{\mathbf{P}}_ {n1} = [\mathbf{Q}_ {n1} | \tilde{\mathbf{n1}}]$ 的图平面上的变换。我们将看到，所寻求的变换是由3×3矩阵 $\mathbf{T}_ {1} = \mathbf{Q}_ {n1} \mathbf{Q}^{-1}_ {o1}$ 给出的共线性。同样的结果也适用于右图
    - 对于任何3D点 $\mathbf{w}$ ，我们可以写： $$\begin{equation} \left\{ \begin{aligned} \tilde{\mathbf{m}}_ {o1} & \cong & \tilde{\mathbf{P}}_ {o1} \tilde{\mathbf{w}} \\ \tilde{\mathbf{m}}_ {n1} & \cong & \tilde{\mathbf{P}}_ {n1} \tilde{\mathbf{w}} \end{aligned} \right. \end{equation}$$ 
    - 根据等式8，光线方程如下（因为整流不会移动光学中心）： $$\begin{equation} \left\{ \begin{aligned} \mathbf{w} &= \mathbf{c}_ {1} + \lambda _{o} \mathbf{Q}^{-1}_ {o1} \tilde{\mathbf{m}}_ {o1}, \quad \quad \quad \quad \lambda _{o} \in \mathbb{R} \\ \mathbf{w} &= \mathbf{c}_ {1} + \lambda _{n} \mathbf{Q}^{-1}_ {n1} \tilde{\mathbf{m}}_ {n1}, \quad \quad \quad \quad \lambda _{n} \in \mathbb{R} \end{aligned} \right. \end{equation}$$ 因此 $$\begin{equation} \tilde{\mathbf{m}}_ {n1} = \lambda \mathbf{Q}_ {n1} \mathbf{Q}^{-1}_ {o1} \tilde{\mathbf{m}}_ {o1} \quad \quad \quad \lambda \in \mathbb{R} \end{equation}$$
    - 然后，将变换 $\mathbf{T}_ {1}$ 应用于原始左图像以生成校正图像

    ![rectified image](../pictures/compact%20algorithm%20for%20rectification%20of%20stereo%20pairs/rectified%20image.png)
    - ***请注意，校正图像的像素（整数坐标位置）通常对应于原始图像平面上的非整数位置。因此，通过双线性插值(bilinear interpolation)计算校正图像的灰度级***
    - 通过三角测量重建3D点（Hartley和Sturm，1997）可使用Pn1、Pn2直接从校正图像中进行

8. 整流算法总结(Summary of the rectification algorithm)
    - 鉴于立体在研究和应用中的高度扩散，我们努力使我们的算法尽可能容易地再现和使用。为此，我们给出了算法的工作matlab代码；代码简单紧凑（22行），所附注释使其在不了解matlab的情况下可以理解。整流函数（参见matlab代码）的用法如下
        - 给定一对立体图像I1、I2和PPM Po1、Po2（通过校准获得）;
        - 计算[T1, T2, Pn1, Pn2]=整流（Po1, Po2）；
        - 通过应用T1和T2校正图像

        ```
        function [T1,T2,Pn1,Pn2] = rectify(Po1,Po2)

        % RECTIFY: compute rectification matrices

        % factorize old PPMs
        [A1,R1,t1] = art(Po1);
        [A2,R2,t2] = art(Po2);

        % optical centers (unchanged)
        c1 = - inv(Po1(:,1:3))*Po1(:,4);
        c2 = - inv(Po2(:,1:3))*Po2(:,4);

        % new x axis (= direction of the baseline)
        v1 = (c1-c2);
        % new y axes (orthogonal to new x and old z)
        v2 = cross(R1(3,:)’,v1);
        % new z axes (orthogonal to baseline and y)
        v3 = cross(v1,v2);

        % new extrinsic parameters
        R = [v1'/norm(v1)
             v2'/norm(v2)
             v3'/norm(v3)];
        % translation is left unchanged

        % new intrinsic parameters (arbitrary)
        A = (A1 + A2)./2;
        A(1,2)=0; % no skew

        % new projection matrices
        Pn1 = A * [R - R*c1];
        Pn2 = A * [R - R*c2];

        % rectifying image transformation
        T1 = Pn1(1:3,1:3) * inv(Po1(1:3,1:3));
        T2 = Pn2(1:3,1:3) * inv(Po2(1:3,1:3));

        % ------------------------

        function [A,R,t] = art(P)
        % ART: factorize a PPM as P=A*[R;t]

        Q = inv(P(1:3, 1:3));
        [U,B] = qr(Q);

        R = inv(U);
        t = B*P(1:3,4);
        A = inv(B);
        A = A ./A(3,3);
        ```
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