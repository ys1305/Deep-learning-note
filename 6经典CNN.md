# CNN：图像分类

1. `ImageNet` 数据集：一个开源的图片数据集，包含超过 1400 万张图片和图片对应的标签，包含 2 万多个类别。

    自从`2010` 年以来，`ImageNet` 每年举办一次比赛，即：`ImageNet` 大规模视觉识别挑战赛`ILSVRC` ，比赛使用 1000 个类别图片。

    2017 年 7 月，`ImageNet` 宣布`ILSVRC` 于`2017` 年正式结束，因为图像分类、物体检测、物体识别任务中计算机的正确率都远超人类，计算机视觉在感知方面的问题基本得到解决，后续将专注于目前尚未解决的问题。

2. `ImageNet` 中使用两种错误率作为评估指标：

    - `top-5` 错误率：对一个图片，如果正确标记在模型输出的前 5 个最佳预测（即：概率最高的前 5 个）中，则认为是正确的，否则认为是错误的。

      最终错误预测的样本数占总样本数的比例就是 `top-5` 错误率。

    - `top-1` 错误率：对一个图片，如果正确标记等于模型输出的最佳预测（即：概率最高的那个），则认为是正确的，否则认为是错误的。

      最终错误预测的样本数占总样本数的比例就是 `top-1` 错误率。

3. 注：`feature map` 的描述有两种：`channel first`，如`256x3x3`；`channel last`，如`3x3x256` 。这里如果未说明，则默认采用`channel last`描述。另外也可以显式指定，如：`3x3@256` 。

## 一、LeNet

1. 1998 年`LeCun` 推出了`LeNet` 网络，它是第一个广为流传的卷积神经网络。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/lenet.jpg)

2. `LeNet` 网络包含了卷积层、池化层、全连接层，这些都是现代`CNN` 网络的基本组件。

    - 输入层：二维图像，尺寸为`32x32`。

    - `C1、C3、C5` 层：二维卷积层。

      其中`C5` 将输入的 `feature map`（尺寸 `16@5x5` ）转化为尺寸为`120x1x1` 的 `feature map`，然后转换为长度为`120` 的一维向量。

      这是一种常见的、将卷积层的输出转换为全连接层的输入的一种方法。

    - `S2、S4` 层：池化层。使用`sigmoid` 函数作为激活函数。

      > 后续的 `CNN` 都使用`ReLU` 作为激活函数。

    - `F6` 层：全连接层。

    - 输出层：由欧式径向基函数单元组成。

      > 后续的`CNN` 使用`softmax` 输出单元。

      下表中，`@` 分隔了通道数量和`feature map` 的宽、高。

      | 网络层 | 核/池大小 | 核数量 | 步长 | 输入尺寸 | 输出尺寸 |
      | ------ | --------- | ------ | ---- | -------- | -------- |
      | INPUT  | -         | -      | -    | -        | 1@32x32  |
      | C1     | 5x5       | 6      | 1    | 1@32x32  | 6@28x28  |
      | S2     | 2x2       | -      | 2    | 6@28x28  | 6@14x14  |
      | C3     | 5x5       | 16     | 1    | 6@14x14  | 16@10x10 |
      | S4     | 2x2       | -      | 2    | 16@10x10 | 16@5x5   |
      | C5     | 5x5       | 120    | 1    | 16@5x5   | 120@1x1  |
      | F6     | -         | -      | -    | 120      | 84       |
      | OUTPUT | -         | -      | -    | 84       | 10       |

## 二、AlexNet

1.  2012 年`Hinton` 和他的学生推出了`AlexNet` 。在当年的`ImageNet` 图像分类竞赛中，`AlexeNet` 以远超第二名的成绩夺冠，使得深度学习重回历史舞台，具有重大历史意义。

### 2.1 网络结构

1. `AlexNet` 有 5 个广义卷积层和 3 个广义全连接层。

    - 广义的卷积层：包含了卷积层、池化层、`ReLU`、`LRN` 层等。
    - 广义全连接层：包含了全连接层、`ReLU`、`Dropout` 层等。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/alexnet.jpg)

2. 网络结构如下表所示：

    - 输入层会将`3@224x224` 的三维图片预处理变成`3@227x227` 的三维图片。

    - 第二层广义卷积层、第四层广义卷积层、第五层广义卷积层都是分组卷积，仅采用本`GPU` 内的通道数据进行计算。

      第一层广义卷积层、第三层广义卷积层、第六层连接层、第七层连接层、第八层连接层执行的是全部通道数据的计算。

    - 第二层广义卷积层的卷积、第三层广义卷积层的卷积、第四层广义卷积层的卷积、第五层广义卷积层的卷积均采用`same` 填充。

      > 当卷积的步长为 1，核大小为`3x3` 时，如果不填充 0，则`feature map` 的宽/高都会缩减 2 。因此这里填充 0，使得输出`feature map` 的宽/高保持不变。

      其它层的卷积，以及所有的池化都是`valid` 填充（即：不填充 0 ）。

    - 第六层广义连接层的卷积之后，会将`feature map` 展平为长度为 4096 的一维向量。

    | 编号  | 网络层     | 子层    | 核/池大小 | 核数量 | 步长 | 激活函数 | 输入尺寸  | 输出尺寸  |
    | ----- | ---------- | ------- | --------- | ------ | ---- | -------- | --------- | --------- |
    | 第0层 | 输入层     | -       | -         | -      | -    | -        | -         | 3@224x224 |
    | 第1层 | 广义卷积层 | 卷积    | 11x11     | 96     | 4    | ReLU     | 3@227x227 | 96@55x55  |
    | 第1层 | 广义卷积层 | 池化    | 3x3       | -      | 2    | -        | 96@55x55  | 96@27x27  |
    | 第1层 | 广义卷积层 | LRN     | -         | -      | -    | -        | 96@27x27  | 96@27x27  |
    | 第2层 | 广义卷积层 | 卷积    | 5x5       | 256    | 1    | ReLU     | 96@27x27  | 256@27x27 |
    | 第2层 | 广义卷积层 | 池化    | 3x3       | -      | 2    | -        | 256@27x27 | 256@13x13 |
    | 第2层 | 广义卷积层 | LRN     | -         | -      | -    | -        | 256@13x13 | 256@13x13 |
    | 第3层 | 广义卷积层 | 卷积    | 3x3       | 384    | 1    | ReLU     | 256@13x13 | 384@13x13 |
    | 第4层 | 广义卷积层 | 卷积    | 3x3       | 384    | 1    | ReLU     | 384@13x13 | 384@13x13 |
    | 第5层 | 广义卷积层 | 卷积    | 3x3       | 256    | 1    | ReLU     | 384@13x13 | 256@13x13 |
    | 第5层 | 广义卷积层 | 池化    | 3x3       | -      | 2    | -        | 256@13x13 | 256@6x6   |
    | 第6层 | 广义连接层 | 卷积    | 6x6       | 4096   | 1    | ReLU     | 256@6x6   | 4096@1x1  |
    | 第6层 | 广义连接层 | dropout | -         | -      | -    | -        | 4096@1x1  | 4096@1x1  |
    | 第7层 | 广义连接层 | 全连接  | -         | -      | -    | ReLU     | 4096      | 4096      |
    | 第7层 | 广义连接层 | dropout | -         | -      | -    | -        | 4096      | 4096      |
    | 第8层 | 广义连接层 | 全连接  | -         | -      | -    | -        | 4096      | 1000      |

3. 网络参数数量：总计约 6237 万。

    - 输出`Tensor size` 采用`channel last` 风格描述。即`227x227x3` 等价于前文的 `3@227x227` 。

    - 第 6 层广义连接层的卷积的参数数量最多，约 3770 万，占整体六千万参数的 60%。

      原因是该子层的卷积核较大、输入通道数量较大、输出通道数量太多。该卷积需要的参数数量为：$256\times 6\times 6\times 4096 = 37,748,736 。​$

    | 编号  | 网络层     | 子层    | 输出 Tensor size | 权重个数 | 偏置个数 | 参数数量   |
    | ----- | ---------- | ------- | ---------------- | -------- | -------- | ---------- |
    | 第0层 | 输入层     | -       | 227x227x3        | 0        | 0        | 0          |
    | 第1层 | 广义卷积层 | 卷积    | 55x55x96         | 34848    | 96       | 34944      |
    | 第1层 | 广义卷积层 | 池化    | 27x27x96         | 0        | 0        | 0          |
    | 第1层 | 广义卷积层 | LRN     | 27x27x96         | 0        | 0        | 0          |
    | 第2层 | 广义卷积层 | 卷积    | 27x27x256        | 614400   | 256      | 614656     |
    | 第2层 | 广义卷积层 | 池化    | 13x13x256        | 0        | 0        | 0          |
    | 第2层 | 广义卷积层 | LRN     | 13x13x256        | 0        | 0        | 0          |
    | 第3层 | 广义卷积层 | 卷积    | 13x13x384        | 884736   | 384      | 885120     |
    | 第4层 | 广义卷积层 | 卷积    | 13x13x384        | 1327104  | 384      | 1327488    |
    | 第5层 | 广义卷积层 | 卷积    | 13x13x256        | 884736   | 256      | 884992     |
    | 第5层 | 广义卷积层 | 池化    | 6x6x256          | 0        | 0        | 0          |
    | 第6层 | 广义连接层 | 卷积    | 4096×1           | 37748736 | 4096     | 37752832   |
    | 第6层 | 广义连接层 | dropout | 4096×1           | 0        | 0        | 0          |
    | 第7层 | 广义连接层 | 全连接  | 4096×1           | 16777216 | 4096     | 16781312   |
    | 第7层 | 广义连接层 | dropout | 4096×1           | 0        | 0        | 0          |
    | 第8层 | 广义连接层 | 全连接  | 1000×1           | 4096000  | 1000     | 4097000    |
    | 总计  | -          | -       | -                | -        | -        | 62,378,344 |

### 2.2 设计技巧

1.  `AlexNet` 成功的主要原因在于：

    - 使用`ReLU` 激活函数。
    - 使用`dropout`、数据集增强 、重叠池化等防止过拟合的方法。
    - 使用百万级的大数据集来训练。
    - 使用`GPU`训练，以及的`LRN` 使用。
    - 使用带动量的 `mini batch` 随机梯度下降来训练。

#### 2.2.1 数据集增强

1. `AlexNet` 中使用的数据集增强手段：

    - 随机裁剪、随机水平翻转：原始图片的尺寸为`256xx256`，裁剪大小为`224x224`。

      - 每一个`epoch` 中，对同一张图片进行随机性的裁剪，然后随机性的水平翻转。理论上相当于扩充了数据集$ (256-224)^2\times 2 = 2048​$ 倍。

      - 在预测阶段不是随机裁剪，而是固定裁剪图片四个角、一个中心位置，再加上水平翻转，一共获得 10 张图片。

        用这 10 张图片的预测结果的均值作为原始图片的预测结果。

    - `PCA` 降噪：对`RGB`空间做`PCA` 变换来完成去噪功能。同时在特征值上放大一个随机性的因子倍数（单位`1` 加上一个$ \mathcal N(0,0.1)$ 的高斯绕动），从而保证图像的多样性。

      - 每一个`epoch` 重新生成一个随机因子。
      - 该操作使得错误率下降`1%` 。

2. `AlexNet` 的预测方法存在两个问题：

    - 这种`固定裁剪四个角、一个中心`的方式，把图片的很多区域都给忽略掉了。很有可能一些重要的信息就被裁剪掉。
    - 裁剪窗口重叠，这会引起很多冗余的计算。

    改进的思路是：

    - 执行所有可能的裁剪方式，对所有裁剪后的图片进行预测。将所有预测结果取平均，即可得到原始测试图片的预测结果。
    - 减少裁剪窗口重叠部分的冗余计算。

    具体做法为：将全连接层用等效的卷积层替代，然后直接使用原始大小的测试图片进行预测。将输出的各位置处的概率值按每一类取平均（或者取最大），则得到原始测试图像的输出类别概率。

    下图中：上半图为`AlexNet` 的预测方法；下半图为改进的预测方法。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/fc_detection.png)

#### 2.2.2 局部响应规范化

1. 局部响应规范层`LRN`：目地是为了进行一个横向抑制，使得不同的卷积核所获得的响应产生竞争。

    - `LRN` 层现在很少使用，因为效果不是很明显，而且增加了内存消耗和计算时间。
    - 在`AlexNet` 中，该策略贡献了`1.2%` 的贡献率。

2. `LRN` 的思想：输出通道 i 在位置 (x,y) 处的输出会受到相邻通道在相同位置输出的影响。

    为了刻画这种影响，将输出通道 i 的原始值除以一个归一化因子。

    $$
    \hat a_i^{(x,y)}=\frac{a_i^{(x,y)}}{\left(k+\alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)}(a_j^{(x,y)})^2\right)^\beta},\quad i=0,1,\cdots,N-1
    $$
    其中：$a_i^{(x,y)} ​$为输出通道 i 在位置 $(x,y)​$ 处的原始值$，\hat a_i^{(x,y)} ​$为归一化之后的值。$n ​$为影响第 $i ​$通道的通道数量（分别从左侧、右侧 $n/2 ​$个通道考虑）。$\alpha,\beta,k​$ 为超参数。

    一般考虑$ k=2,n=5,\alpha=10^{-4},\beta=0.75 。$

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/alexnet_LRN.png)

#### 2.2.3 多 GPU 训练

1.  `AlexNet` 使用两个`GPU`训练。网络结构图由上、下两部分组成：一个`GPU`运行图上方的通道数据，一个`GPU` 运行图下方的通道数据，两个`GPU` 只在特定的网络层通信。即：执行分组卷积。

    - 第二、四、五层卷积层的核只和同一个`GPU` 上的前一层的`feature map` 相连。
    - 第三层卷积层的核和前一层所有`GPU` 的`feature map` 相连。
    - 全连接层中的神经元和前一层中的所有神经元相连。

#### 2.2.4 重叠池化

1. 一般的池化是不重叠的，池化区域的大小与步长相同。`Alexnet` 中，池化是可重叠的，即：步长小于池化区域的大小。

    重叠池化可以缓解过拟合，该策略贡献了`0.4%` 的错误率。

2. 为什么重叠池化会减少过拟合，很难用数学甚至直观上的观点来解答。一个稍微合理的解释是：重叠池化会带来更多的特征，这些特征很可能会有利于提高模型的泛化能力。

#### 2.2.5 优化算法

1. `AlexNet` 使用了带动量的`mini-batch` 随机梯度下降法。

2. 标准的带动量的`mini-batch` 随机梯度下降法为：

    $$
    \mathbf{\vec v}\leftarrow \alpha\mathbf{\vec v}-\epsilon\nabla_{\vec\theta} J(\vec\theta)\\ \vec\theta\leftarrow \vec\theta+\mathbf{\vec v}
    $$
    而论文中，作者使用了修正：

    $$
    \mathbf{\vec v}\leftarrow \alpha\mathbf{\vec v}-\beta\epsilon\vec\theta-\epsilon\nabla_{\vec\theta} J(\vec\theta)\\ \vec\theta\leftarrow \vec\theta+\mathbf{\vec v}
    $$
    

    - 其中$ \alpha=0.9 ， \beta=0.0005 ，\epsilon $为学习率。
    - $-\beta\epsilon\vec\theta​$ 为权重衰减。论文指出：权重衰减对于模型训练非常重要，不仅可以起到正则化效果，还可以减少训练误差。

## 三、VGG-Net

1.  `VGG-Net` 是牛津大学计算机视觉组和`DeepMind`公司共同研发一种深度卷积网络，并且在 2014 年在`ILSVRC`比赛上获得了分类项目的第二名和定位项目的第一名。

2.  `VGG-Net` 的主要贡献是：

    - 证明了小尺寸卷积核（`3x3` ）的深层网络要优于大尺寸卷积核的浅层网络。
    - 证明了深度对网络的泛化性能的重要性。
    - 验证了尺寸抖动`scale jittering` 这一数据增强技术的有效性。

3.  `VGG-Net` 最大的问题在于参数数量，`VGG-19` 基本上是参数数量最多的卷积网络架构。

### 3.1 网络结构

1. `VGG-Net` 一共有五组结构（分别表示为：`A~E` ）， 每组结构都类似，区别在于网络深度上的不同。

    - 结构中不同的部分用黑色粗体给出。

    - 卷积层的参数为`convx-y`，其中`x` 为卷积核大小，`y` 为卷积核数量。

      如：`conv3-64` 表示 `64` 个 `3x3` 的卷积核。

    - 卷积层的通道数刚开始很小（64 通道），然后在每个池化层之后的卷积层通道数翻倍，直到 512。

    - 每个卷积层之后都跟随一个`ReLU`激活函数，表中没有标出。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/vgg_net.png)

2. 通用结构：

    - 输入层：固定大小的`224x224` 的`RGB` 图像。

    - 卷积层：卷积步长均为 1。

      - 填充方式：填充卷积层的输入，使得卷积前后保持同样的空间分辨率。

        - `3x3` 卷积：`same` 填充，即：输入的上下左右各填充 1 个像素。
        - `1x1` 卷积：不需要填充。

      - 卷积核尺寸：有`3x3` 和`1x1` 两种。

        - `3x3` 卷积核：这是捕获左右、上下、中心等概念的最小尺寸。

        - `1x1` 卷积核：用于输入通道的线性变换。

          在它之后接一个`ReLU` 激活函数，使得输入通道执行了非线性变换。

    - 池化层：采用最大池化。

      - 池化层连接在卷积层之后，但并不是所有的卷积层之后都有池化。
      - 池化窗口为`2x2`，步长为 2 。

    - 网络最后四层为：：三个全连接层 \+ 一个`softmax` 层。

      - 前两个全连接层都是 4096 个神经元，第三个全连接层是 1000 个神经元（因为执行的是 1000 类的分类）。
      - 最后一层是`softmax` 层用于输出类别的概率。

    - 所有隐层都使用`ReLU` 激活函数。

3. `VGG-Net` 网络参数数量：

    其中第一个全连接层的参数数量为：`7x7x512x4096=1.02亿` ，因此网络绝大部分参数来自于该层。

    > 与`AlexNet` 相比，`VGG-Net` 在第一个全连接层的输入`feature map` 较大：`7x7 vs 6x6`，`512 vs 256` 。

    | 网络     | A , A-LRN | B      | C      | D      | E    |
    | -------- | --------- | ------ | ------ | ------ | ---- |
    | 参数数量 | 1.13亿    | 1.33亿 | 1.34亿 | 1.38亿 | 1.44 |

### 3.2 设计技巧

1. 输入预处理：通道像素零均值化。

    - 先统计训练集中全部样本的通道均值：所有红色通道的像素均值 $\overline {Red} 、$所有绿色通道的像素均值$ \overline {Green} 、$所有蓝色通道的像素均值$ \overline {Blue} 。$

      $\overline {Red} = \sum*{n}\sum*{i}\sum*{j} I*{n,0,i,j}\\ \overline {Green} = \sum*{n}\sum*{i}\sum*{j} I*{n,1,i,j}\\ \overline {Blue} = \sum*{n}\sum*{i}\sum*{j} I*{n,2,i,j}$

      其中：假设红色通道为通道`0`，绿色通道为通道`1`，蓝色通道为通道`2` ；n 遍历所有的训练样本$，i,j $遍历图片空间上的所有坐标。

    - 对每个样本：红色通道的每个像素值减去$ \overline {Red} ，$绿色通道的每个像素值减去$ \overline {Green} ，$蓝色通道的每个像素值减去 $\overline {Blue} 。$

2. 多尺度训练：将原始的图像缩放到最小的边 $S>=224 ，$然后在整副图像上截取`224x224` 的区域来训练。

    有两种方案：

    - 在所有图像上固定$ S $：用 $S=256 $来训练一个模型，用 $S=384 $来训练另一个模型。最后使用两个模型来评估。

    - 对每个图像，在 $[S*{\min},S*{\max}]$ 之间随机选取一个$ S$ ，然后进行裁剪来训练一个模型。最后使用单个模型来评估。

      - 该方法只需要一个单一的模型。
      - 该方法相当于使用了尺寸抖动(`scale jittering`) 的数据增强。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/vgg_net_result.png)

3. 多尺度测试：将测试的原始图像等轴的缩放到预定义的最小图像边，表示为 $Q （Q$ 不一定等于$ S ），$称作测试尺度。

    在一张测试图像的几个归一化版本上运行模型，然后对得到的结果进行平均。

    - 不同版本对应于不同的$ Q ​$值。
    - 所有版本都执行通道像素归一化。注意：采用训练集的统计量。

    该方法相当于在测试时使用了尺寸抖动。实验结果表明：测试时的尺寸抖动导致了更好的性能。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/vgg_net_result2.png)

4. 评估有三种方案：

    - `single-crop`：对测试图片沿着最短边缩放，然后选择其中的 `center crop` 来裁剪图像，选择这个图像的预测结果作为原始图像的预测结果。

      该方法的缺点是：仅仅保留图片的中央部分可能会丢掉图片类别的关键信息。因此该方法很少在实际任务中使用，通常用于不同模型之间的性能比较。

    - `multi-crop`：类似`AlexNet` 的做法，对每个测试图像获取多个裁剪图像，平均每个裁剪图像的预测结果为原始图像的预测结果。

      该方法的缺点是：需要网络重新计算每个裁剪图像，效率较低。

    - `dense`：将最后三个全连接层用等效的卷积层替代，成为一个全卷积网络。其中：第一个全连接层用`7x7` 的卷积层替代，后面两个全连接层用`1x1` 的卷积层替代。

      该全卷积网络应用到整张图片上（无需裁剪），得到一个多位置的、各类别的概率字典。通过原始图片、水平翻转图片的各类别预测的均值，得到原始图片的各类别概率。

      该方法的优点是：不需要裁剪图片，支持多尺度的图片测试，计算效率较高。

    实验结果表明：`multi-crop` 评估方式要比`dense` 评估方式表现更好。另外，二者是互补的，其组合要优于任何单独的一种。下表中，`S=[256;512]`，Q={256,384,512} 。

    > 还有一种评估策略：`ensemble error` 。即：同时训练同一种网络的多个不同的模型，然后用这几个模型的预测结果的平均误差作为最终的 `ensemble error` 。
    >
    > 有一种术语叫`single-model error`。它是训练一个模型，然后采用上述的多种`crop/dense` 评估的组合，这些组合的平均输出作为预测结果。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/vgg_net_result3.png)

5. 权重初始化：由于网络深度较深，因此网络权重的初始化很重要，设计不好的初始化可能会阻碍学习。

    - 论文的权重初始化方案为：先训练结构`A` 。当训练更深的配置时，使用结构`A` 的前四个卷积层和最后三个全连接层来初始化网络，网络的其它层被随机初始化。
    - 作者后来指出：可以通过 `Xavier`均匀初始化来直接初始化权重而不需要进行预训练。

6. 实验结果表明：

    - 分类误差随着网络深度的增加而减小。
    - 从`A-LRN` 和 `A` 的比较发现：局部响应归一化层`LRN` 对于模型没有任何改善。

## 四、Inception

1. `Inception` 网络是卷积神经网络的一个重要里程碑。在`Inception` 之前，大部分流行的卷积神经网络仅仅是把卷积层堆叠得越来越多，使得网络越来越深。这使得网络越来越复杂，参数越来越多，从而导致网络容易出现过拟合，增加计算量。

    而`Inception` 网络考虑的是多种卷积核的并行计算，扩展了网络的宽度。

2. `Inception Net` 核心思想是：稀疏连接。因为生物神经连接是稀疏的。

3. `Inception` 网络的最大特点是大量使用了`Inception` 模块。

### 4.1 Inception v1

#### 4.1.1 网络结构

1. `InceptionNet V1` 是一个 22 层的深度网络。 如果考虑池化层，则有 29 层。如下图中的`depth` 列所示。

    网络具有三组`Inception` 模块，分别为：`inception(3a)/inception(3b)`、`inception(4a)/inception(4b)/inception(4c)/inception(4d)/inception(4e)`、`inception(5a)、inception(5b)`。三组`Inception` 模块被池化层分隔。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v1_all.jpg)

2. 下图给出了网络的层次结构和参数，其中：

    - `type` 列：给出了每个模块/层的类型。
    - `patch size/stride` 列：给出了卷积层/池化层的尺寸和步长。
    - `output size` 列：给出了每个模块/层的输出尺寸和输出通道数。
    - `depth`列：给出了每个模块/层包含的、含有训练参数层的数量。
    - `#1x1`列：给出了每个模块/层包含的`1x1` 卷积核的数量，它就是`1x1` 卷积核的输出通道数。
    - `#3x3 reduce`列：给出了每个模块/层包含的、放置在`3x3` 卷积层之前的`1x1` 卷积核的数量，它就是`1x1` 卷积核的输出通道数。
    - `#3x3`列：给出了每个模块/层包含的`3x3` 卷积核的数量，它就是`3x3` 卷积核的输出通道数。
    - `#5x5 reduce`列：给出了每个模块/层包含的、放置在`5x5` 卷积层之前的`1x1` 卷积核的数量，它就是`1x1` 卷积核的输出通道数。
    - `#5x5`列：给出了每个模块/层包含的`5x5` 卷积核的数量，它就是`5x5`卷积核的输出通道数。
    - `pool proj`列：给出了每个模块/层包含的、放置在池化层之后的`1x1` 卷积核的数量，它就是`1x1` 卷积核的输出通道数。
    - `params`列：给出了每个模块/层的参数数量。
    - `ops`列：给出了每个模块/层的计算量。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v1_struct.png)

3. `Inception V1` 的参数数量为 697.7 万，其参数数量远远小于`AlexNet`（6 千万）、`VGG-Net`（超过 1 亿）。

    `Inception V1` 参数数量能缩减的一个主要技巧是：在`inception(5b)`输出到`linear`之间插入一个平均池化层`avg pool`。

    - 如果没有平均池化层，则`inception(5b)` 到 `linear` 之间的参数数量为：`7x7x1024x1024`，约为 5 千万。
    - 插入了平均池化层之后，`inception(5b)` 到 `linear` 之间的参数数量为：`1x1x1024x1024`，约为 1 百万。

#### 4.1.2 Inception 模块

1. 原始的`Inception` 模块对输入同时执行：3 个不同大小的卷积操作（`1x1、3x3、5x5`）、1 个最大池化操作（`3x3` ）。所有操作的输出都在深度方向拼接起来，向后一级传递。

    - 三种不同大小卷积：通过不同尺寸的卷积核抓取不同大小的对象的特征。

      使用`1x1、3x3、5x5` 这些具体尺寸仅仅是为了便利性，事实上也可以使用更多的、其它尺寸的滤波器。

    - 1 个最大池化：提取图像的原始特征（不经过过滤器）。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v1.png)

2. 原始`Inception` 模块中，模块的输出通道数量为四个子层的输出通道数的叠加。这种叠加不可避免的使得`Inception` 模块的输出通道数增加，这就增加了`Inception` 模块中每个卷积的计算量。因此在经过若干个模块之后，计算量会爆炸性增长。

    解决方案是：在`3x3` 和 `5x5` 卷积层之前额外添加`1x1` 卷积层，来限制输入给卷积层的输入通道的数量。

    注意：

    - `1x1` 卷积是在最大池化层之后，而不是之前。这是因为：池化层是为了提取图像的原始特征，一旦它接在`1x1` 卷积之后就失去了最初的本意。
    - `1x1` 卷积在`3x3`、`5x5` 卷积之前。这是因为：如果`1x1` 卷积在它们之后，则`3x3` 卷积、`5x5` 卷积的输入通道数太大，导致计算量仍然巨大。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v1_2.png)

#### 4.1.3 辅助分类器

1. 为了缓解梯度消失的问题，`InceptionNet V1` 给出了两个辅助分类器。这两个辅助分类器被添加到网络的中间层，它们和主分类器共享同一套训练数据及其标记。其中：

    - 第一个辅助分类器位于`Inception(4a)` 之后，`Inception(4a)` 模块的输出作为它的输入。

    - 第二个辅助分类器位于`Inception(4d)` 之后，`Inception(4d)` 模块的输出作为它的输入。

    - 两个辅助分类器的结构相同，包括以下组件：

      - 一个尺寸为`5x5`、步长为`3`的平均池化层。
      - 一个尺寸为`1x1`、输出通道数为`128` 的卷积层。
      - 一个具有`1024` 个单元的全连接层。
      - 一个`drop rate = 70%`的 `dropout` 层。
      - 一个使用`softmax` 损失的线性层作为输出层。

2. 在训练期间，两个辅助分类器的损失函数的权重是 0.3，它们的损失被叠加到网络的整体损失上。在推断期间，这两个辅助网络被丢弃。

    在`Inception v3` 的实验中表明：辅助网络的影响相对较小，只需要其中一个就能够取得同样的效果。

    事实上辅助分类器在训练早期并没有多少贡献。只有在训练接近结束，辅助分支网络开始发挥作用，获得超出无辅助分类器网络的结果。

3. 两个辅助分类器的作用：提供正则化的同时，克服了梯度消失问题。

### 4.2 Inception v2

1. `Inception v2` 的主要贡献是提出了`Batch Normalization` 。论文指出，使用了`Batch Normalization` 之后：

    - 可以加速网络的学习。

      相比`Inception v1`，训练速度提升了 14 倍。因为应用了`BN` 之后，网络可以使用更高的学习率，同时删除了某些层。

    - 网络具有更好的泛化能力。

      在`ImageNet` 分类问题的`top5` 上达到`4.8%`，超过了人类标注 `top5` 的准确率。

2. `Inception V2` 网络训练的技巧有：

    - 使用更高的学习率。
    - 删除`dropout`层、`LRN` 层。
    - 减小`L2` 正则化的系数。
    - 更快的衰减学习率。学习率以指数形式衰减。
    - 更彻底的混洗训练样本，使得一组样本在不同的`epoch` 中处于不同的`mini batch` 中。
    - 减少图片的形变。

3. `Inception v2` 的网络结构比`Inception v1` 有少量改动：

    - `5x5` 卷积被两个`3x3` 卷积替代。

      这使得网络的最大深度增加了 9 层，同时网络参数数量增加 25%，计算量增加 30%。

    - `28x28` 的`inception` 模块从 2 个增加到 3 个。

    - 在`inception` 模块中，有的采用最大池化，有的采用平均池化。

    - 在`inception` 模块之间取消了用作连接的池化层。

    - `inception(3c),inception(4e)` 的子层采用步长为 2 的卷积/池化。

    > `Pool+proj` 列给出了`inception` 中的池化操作。
    >
    > - `avg+32` 意义为：平均池化层后接一个尺寸`1x1`、输出通道`32` 的卷积层。
    > - `max+pass through` 意义为：最大池化层后接一个尺寸`1x1`、输出通道数等于输入通道数的卷积层。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v2.png)

4. `Inception V2` 的网络参数约为`1126` 万。

    | 层           | 参数数量 |
    | ------------ | -------- |
    | conv1        | 9408     |
    | conv2        | 114688   |
    | inception-3a | 218094   |
    | inception-3b | 259072   |
    | inception-3c | 384000   |
    | inception-4a | 608193   |
    | inception-4b | 663552   |
    | inception-4c | 912384   |
    | inception-4d | 1140736  |
    | inception-4e | 1447936  |
    | inception-5a | 2205696  |
    | inception-5b | 2276352  |
    | fc           | 1024000  |
    | 共           | 11264111 |

5. `Inception V2` 在`ImageNet` 测试集上的误差率：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v2_result.png)

### 4.3 Inception v3

1. 虽然`Inception v1` 的参数较少，但是它的结构比较复杂，难以进行修改。原因有以下两点：

    - 如果单纯的放大网络（如增加`Inception` 模块的数量、扩展`Inception` 模块的大小），则参数的数量会显著增长，计算代价太大。
    - `Inception v1` 结构中的各种设计，其对最终结果的贡献尚未明确。

    因此`Inception v3` 的论文重点探讨了网络结构设计的原则。

#### 4.3.1 网络结构

1. `Inception v3` 的网络深度为 42 层，它相对于`Inception v1` 网络主要做了以下改动：

    - `7x7` 卷积替换为 3 个`3x3` 卷积。

    - 3 个`Inception`模块：模块中的`5x5` 卷积替换为 2 个`3x3` 卷积，同时使用后面描述的网格尺寸缩减技术。

    - 5 个`Inception` 模块：模块中的`5x5` 卷积替换为 2 个`3x3` 卷积之后，所有的`nxn` 卷积进行非对称分解，同时使用后面描述的网格尺寸缩减技术。

    - 2 个`Inception` 模块：结构如下。它也使用了卷积分解技术，以及网格尺寸缩减技术。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v3_4.png)

2. `Inception v3` 的网络结构如下所示：

    - `3xInception` 表示三个`Inception` 模块，`4xInception` 表示四个`Inception` 模块，`5xInception` 表示五个`Inception` 模块。

    - `conv padded` 表示使用 0 填充的卷积，它可以保持`feature map` 的尺寸。

      在`Inception` 模块内的卷积也使用 0 填充，所有其它的卷积/池化不再使用填充。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v3.png)

3. 在`3xInception` 模块的输出之后设有一个辅助分类器。其结构如下：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v3_5.png)

4. `Inception v3` 整体参数数量约 23,626,728 万（论文`Xception: Deep Learning with Depthwise Separable Convolutions`）。

#### 4.3.2 设计技巧

1. `Inception v3` 总结出网络设计的一套通用设计原则：

    - 避免`representation` 瓶颈：`representation` 的大小应该从输入到输出缓缓减小，避免极端压缩。在缩小`feature map` 尺寸的同时，应该增加`feature map` 的通道数。

      `representation` 大小通常指的是`feature map` 的容量，即`feature map` 的`width x height x channel` 。

    - 空间聚合：可以通过空间聚合来完成低维嵌入，而不会在表达能力上有较大的损失。因此通常在`nxn` 卷积之前，先利用`1x1` 卷积来降低输入维度。

      猜测的原因是：空间维度之间的强相关性导致了空间聚合过程中的信息丢失较少。

    - 平衡网络的宽度和深度：增加网络的宽度或者深度都可以提高网络的泛化能力，因此计算资源需要在网络的深度和宽度之间取得平衡。

##### 4.3.2.1 卷积尺寸分解

1. 大卷积核的分解：将大卷积核分解为多个小的卷积核。

    如：使用 2 个`3x3` 卷积替换`5x5` 卷积，则其参数数量大约是 1 个`5x5` 卷积的 72% 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v3_1.png)

2. `nxn` 卷积核的非对称分解：将`nxn` 卷积替换为`1xn` 卷积和`nx1` 卷积。

    - 这种非对称分解的参数数量是原始卷积数量的 $\frac 2n $。随着`n` 的增加，计算成本的节省非常显著。
    - 论文指出：对于较大的`feature map` ，这种分解不能很好的工作；但是对于中等大小的 `feature map` （尺寸在`12～20` 之间），这种分解效果非常好。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v3_2.png)

##### 4.3.2.2 网格尺寸缩减

1. 假设输入的`feature map` 尺寸为`dxd`，通道数为`k`。如果希望输出的`feature map` 尺寸为`d/2 x d/2`，通道数为`2k`。则有以下的两种方式：

    - 首先使用`2k` 个 `1x1` 的卷积核，执行步长为 1 的卷积。然后执行一个`2x2` 的、步长为 2 的池化操作。

      该方式需要执行 $2d^2k^2 $次`乘-加`操作，计算代价较大。

    - 直接使用`2k` 个`1x1` 的卷积核，执行步长为 2 的卷积。

      该方式需要执行$ 2(\frac{d}{2})^2k^2 ​$次`乘-加`操作，计算代价相对较小。但是表征能力下降，产生了表征瓶颈。

    事实上每个`Inception` 模块都会使得`feature map` 尺寸缩半、通道翻倍，因此在这个过程中需要仔细设计网络，使得既能够保证网络的表征能力，又不至于计算代价太大。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v3_6.png)

2. 解决方案是：采用两个模块`P` 和 `C` 。

    - 模块`P` ：使用`k` 个`1x1` 的卷积核，执行步长为 2 的卷积。其输出`feature map` 尺寸为`d/2 x d/2`，通道数为`k`。
    - 模块`C`：使用步长为 2 的池化。其输出`feature map` 尺寸为`d/2 x d/2`，通道数为`k`。

    将模块`P` 和模块`C` 的输出按照通道数拼接，产生最终的输出`feature map` 。

    ![1557575141544](C:\Users\Administrator.PC-201805312002\AppData\Roaming\Typora\typora-user-images\1557575141544.png)

##### 4.3.2.3 标签平滑正则化

1. 标签平滑正则化的原理：假设样本的真实标记存在一定程度上的噪声。即：样本的真实标记不一定是可信的。

    对给定的样本 $\mathbf{\vec x} ，$其真实标记为 $y 。$在普通的训练中，该样本的类别分布为一个 $\delta $函数：$P(Y=k\mid X=\mathbf{\vec x})=\delta(y-k),\quad k=1,2,\cdots,K。$记做$ \delta_{k,y} 。$

    采用标签平滑正则化（`LSR:Label Smoothing Regularization`）之后，该样本的类别分布为：

    $P(Y=k\mid X=\mathbf{\vec x})=(1-\epsilon)\delta_{k,y}+\frac \epsilon K,\quad k=1,2,\cdots,K$

    其中$ \epsilon $是一个很小的正数（如 0.1)，其物理意义为：样本标签不可信的比例。

    该类别分布的物理意义为：

    - 样本 $\mathbf{\vec x} $的类别为 $y $的概率为 $1-\frac {(K-1)\epsilon}K ​$。
    - 样本 $\mathbf{\vec x}$ 的类别为$ 1,2,\cdots,y-1,y+1,\dots,K $的概率均$ \frac \epsilon K $。

2. 论文指出：标签平滑正则化对`top-1` 错误率和`top-5` 错误率提升了大约 0.2% 。

### 4.4 Inception v4 & Inception - ResNet

1. `Inception v4` 和 `Inception-ResNet` 在同一篇论文中给出。论文通过实验证明了：结合残差连接可以显著加速`Inception` 的训练。

2. 性能比较：（综合采用了 `144 crops/dense` 评估的结果，数据集：`ILSVRC 2012` 的验证集 ）

    | 网络                | crops | Top-1 Error | Top-5 Error |
    | ------------------- | ----- | ----------- | ----------- |
    | ResNet-151          | dense | 19.4%       | 4.5%        |
    | Inception-v3        | 144   | 18.9%       | 4.3%        |
    | Inception-ResNet-v1 | 144   | 18.8%       | 4.3%        |
    | Inception-v4        | 144   | 17.7%       | 3.8%        |
    | Inception-ResNet-v2 | 144   | 17.8%       | 3.7%        |

3. `Inception-ResNet-v2` 参数数量约为 5500 万，`Inception-ResNet-v1/Inception-v4` 的参数数量也在该量级。

#### 4.4.1 Inception v4

1. 在`Inception v4` 结构的主要改动：

    - 修改了 `stem` 部分。

    - 引入了`Inception-A`、`Inception-B`、`Inception-C` 三个模块。这些模块看起来和`Inception v3` 变体非常相似。

      `Inception-A/B/C` 模块中，输入`feature map` 和输出`feature map` 形状相同。而`Reduction-A/B` 模块中，输出`feature map` 的宽/高减半、通道数增加。

    - 引入了专用的“缩减块”(`reduction block`)，它被用于缩减`feature map` 的宽、高。

      早期的版本并没有明确使用缩减块，但是也实现了其功能。

2. `Inception v4` 结构如下：（没有标记`V` 的卷积使用`same`填充；标记`V` 的卷积使用`valid` 填充）

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v4.png)

    - `stem` 部分的结构：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v4_stem.png)

    - `Inception-A`模块（这样的模块有 4 个）：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v4_A.png)

    - `Inception-B`模块（这样的模块有 7 个）：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v4_B.png)

    - `Inception-C`模块（这样的模块有 3 个）：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v4_C.png)

    - `Reduction-A`模块：(其中 k,l,m,n 分别表示滤波器的数量)

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v4_reduce_A.png)

      | 网络                | k    | l    | m    | n    |
      | ------------------- | ---- | ---- | ---- | ---- |
      | Inception-v4        | 192  | 224  | 256  | 384  |
      | Inception-ResNet-v1 | 192  | 192  | 256  | 384  |
      | Inception-ResNet-v2 | 256  | 256  | 256  | 384  |

    - `Reduction-B`模块：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v4_reduce_B.png)

#### 4.4.2 Inception-ResNet

1. 在`Inception-ResNet` 中，使用了更廉价的`Inception` 块：`inception` 模块的池化运算由残差连接替代。

    > 在`Reduction` 模块中能够找到池化运算。

2)  `Inception ResNet` 有两个版本：`v1` 和 `v2` 。

- `v1` 的计算成本和`Inception v3` 的接近，`v2` 的计算成本和`Inception v4` 的接近。
- `v1` 和`v2` 具有不同的`stem` 。
- 两个版本都有相同的模块`A、B、C` 和缩减块结构，唯一不同在于超参数设置。

3)  `Inception-ResNet-v1` 结构如下：

![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v1.png)

- `stem` 部分的结构：

  ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v1_stem.png)

- `Inception-ResNet-A`模块（这样的模块有 5 个）：

  ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v1_A.png)

- `Inception-B`模块（这样的模块有 10 个）：

  ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v1_17x17.png)

- `Inception-C`模块（这样的模块有 5 个）：

  ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v1_C.png)

- `Reduction-A`模块：同`inception_v4` 的 `Reduction-A`模块

- `Reduction-B`模块：

  ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v1_reduce_B.png)

4)  `Inception-ResNet-v2` 结构与`Inception-ResNet-v1` 基本相同 ：

- `stem` 部分的结构：同`inception_v4` 的 `stem` 部分。

  `Inception-ResNet-v2` 使用了`inception v4` 的 `stem` 部分，因此后续的通道数量与`Inception-ResNet-v1` 不同。

- `Inception-ResNet-A`模块（这样的模块有 5 个）：它的结构与`Inception-ResNet-v1` 的`Inception-ResNet-A`相同，只是通道数发生了改变。

  ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v2_A.png)

- `Inception-B`模块（这样的模块有 10 个）：它的结构与`Inception-ResNet-v1` 的`Inception-ResNet-B`相同，只是通道数发生了改变。

  ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v2_17x17.png)

- `Inception-C`模块（这样的模块有 5 个）：它的结构与`Inception-ResNet-v1` 的`Inception-ResNet-C`相同，只是通道数发生了改变。 

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v4_reduce_A.png)

    

- `Reduction-A`模块：同`inception_v4` 的 `Reduction-A`模块。

- `Reduction-B`模块：它的结构与`Inception-ResNet-v1` 的`Reduction-B`相同，只是通道数发生了改变。

  ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_v2_reduce_B.png)

5)  如果滤波器数量超过 1000，则残差网络开始出现不稳定，同时网络会在训练过程早期出现“死亡”：经过成千上万次迭代之后，在平均池化之前的层开始只生成 0 。

解决方案：在残差模块添加到`activation` 激活层之前，对其进行缩放能够稳定训练。降低学习率或者增加额外的`BN`都无法避免这种状况。

这就是`Inception ResNet` 中的 `Inception-A,Inception-B,Inception-C` 为何如此设计的原因。

- 将`Inception-A,Inception-B,Inception-C` 放置在两个`Relu activation` 之间。
- 通过线性的`1x1 Conv`（不带激活函数）来执行对残差的线性缩放。

![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_scale.png)

### 4.5 Xception

1. 一个常规的卷积核尝试在三维空间中使用滤波器抽取特征，包括：两个空间维度（宽度和高度）、一个通道维度。因此单个卷积核的任务是：同时映射跨通道的相关性和空间相关性。

    `Inception` 将这个过程明确的分解为一系列独立的相关性的映射：要么考虑跨通道相关性，要么考虑空间相关性。`Inception` 的做法是：

    - 首先通过一组`1x1` 卷积来查看跨通道的相关性，将输入数据映射到比原始输入空间小的三个或者四个独立空间。
    - 然后通过常规的`3x3` 或者 `5x5` 卷积，将所有的相关性（包含了跨通道相关性和空间相关性）映射到这些较小的三维空间中。

    一个典型的`Inception` 模块（`Inception V3` )如下：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_v3_1.png)

    可以简化为：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/xception2.png)

2. `Xception` 将这一思想发挥到极致：首先使用`1x1` 卷积来映射跨通道相关性，然后分别映射每个输出通道的空间相关性，从而将跨通道相关性和空间相关性解耦。因此该网络被称作`Xception:Extreme Inception` ，其中的`Inception` 块被称作 `Xception` 块。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/xception.png)

3. `Xception` 块类似于深度可分离卷积，但是它与深度可分离卷积之间有两个细微的差异：

    - 操作顺序不同：

      - 深度可分离卷积通常首先执行`channel-wise` 空间卷积，然后再执行`1x1` 卷积。
      - `Xception` 块首先执行`1x1` 卷积，然后再进行`channel-wise` 空间卷积。

    - 第一次卷积操作之后是否存在非线性：

      - 深度可分离卷积只有第二个卷积(`1x1` )使用了`ReLU` 非线性激活函数，`channel-wise` 空间卷积不使用非线性激活函数。
      - `Xception` 块的两个卷积（`1x1` 和 `3x3` ）都使用了`ReLU` 非线性激活函数。

    其中第二个差异更为重要。

4. 对`Xception` 进行以下的修改，都可以加快网络收敛速度，并获取更高的准确率：

    - 引入类似`ResNet` 的残差连接机制。
    - 在`1x1` 卷积和`3x3` 卷积之间不加入任何非线性。

5. `Xception` 的参数数量与`Inception V3` 相同，但是性能表现显著优于`Inception V3` 。这表明`Xception` 更加高效的利用了模型参数。

    - 根据论文`Xception: Deep Learning with Depthwise Separable Convolutions`，`Inception V3` 参数数量为 23626728，`Xception` 参数数量为 22855952 。

    - 在`ImageNet` 上的`benchmark` 为（单个模型，单次`crop` ）：

      | 模型         | top-1 accuracy | top-5 accuracy |
      | ------------ | -------------- | -------------- |
      | VGG-16       | 71.5%          | 90.1%          |
      | ResNet-152   | 77.0%          | 93.3%          |
      | Inception V3 | 78.2%          | 94.1%          |
      | Xception     | 79.0%          | 94.5%          |

## 五、ResNet

1. `ResNet` 提出了一种残差学习框架来解决网络退化问题，从而训练更深的网络。这种框架可以结合已有的各种网络结构，充分发挥二者的优势。

2. `ResNet`以三种方式挑战了传统的神经网络架构：

    - `ResNet` 通过引入跳跃连接来绕过残差层，这允许数据直接流向任何后续层。

      这与传统的、顺序的`pipeline` 形成鲜明对比：传统的架构中，网络依次处理低级`feature` 到高级`feature` 。

    - `ResNet` 的层数非常深，高达 1202 层。而`ALexNet` 这样的架构，网络层数要小两个量级。

    - 通过实验发现，训练好的 `ResNet` 中去掉单个层并不会影响其预测性能。而训练好的`AlexNet` 等网络中，移除层会导致预测性能损失。

3. 在`ImageNet`分类数据集中，拥有 152 层的残差网络，以`3.75% top-5` 的错误率获得了`ILSVRC 2015` 分类比赛的冠军。

4. 很多证据表明：残差学习是通用的，不仅可以应用于视觉问题，也可应用于非视觉问题。

### 5.1 网络退化问题

1. 学习更深的网络的一个障碍是梯度消失/爆炸，该问题可以通过`Batch Normalization` 在很大程度上解决。

2. `ResNet` 论文作者发现：随着网络的深度的增加，准确率达到饱和之后迅速下降，而这种下降不是由过拟合引起的。这称作网络退化问题。

    如果更深的网络训练误差更大，则说明是由于优化算法引起的：越深的网络，求解优化问题越难。如下所示：更深的网络导致更高的训练误差和测试误差。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_1.png)

3. 理论上讲，较深的模型不应该比和它对应的、较浅的模型更差。因为较深的模型是较浅的模型的超空间。较深的模型可以这样得到：先构建较浅的模型，然后添加很多恒等映射的网络层。

    实际上我们的较深的模型后面添加的不是恒等映射，而是一些非线性层。因此，退化问题表明：通过多个非线性层来近似横等映射可能是困难的。

4. 解决网络退化问题的方案：学习残差。

### 5.2 残差块

1. 假设需要学习的是映射$ \mathbf{\vec y}=H(\mathbf{\vec x}) ，$残差块使用堆叠的非线性层拟合残差：$\mathbf{\vec y}=F(\mathbf{\vec x},\mathbf W_i)+\mathbf{\vec x} 。$

    其中：

    - $\mathbf{\vec x} $和 $\mathbf{\vec y} $是块的输入和输出向量。

    - $F(\mathbf{\vec x},\mathbf W_i) $是要学习的残差映射。因为 $F(\mathbf{\vec x},\mathbf W_i)=H(\mathbf{\vec x})-\mathbf{\vec x} ，$因此称 F 为残差。

    - `+` ：通过`快捷连接`逐个元素相加来执行。`快捷连接` 指的是那些跳过一层或者更多层的连接。

      - 快捷连接简单的执行恒等映射，并将其输出添加到堆叠层的输出。
      - 快捷连接既不增加额外的参数，也不增加计算复杂度。

    - 相加之后通过非线性激活函数，这可以视作对整个残差块添加非线性，即 $\text{relu}(\mathbf{\vec y}) $。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_block.png)

2. 前面给出的残差块隐含了一个假设：$F(\mathbf{\vec x},\mathbf W_i) $和 $\mathbf{\vec x} $的维度相等。如果它们的维度不等，则需要在快捷连接中对 $\mathbf{\vec x} $执行线性投影来匹配维度：$\mathbf{\vec y}=F(\mathbf{\vec x},\mathbf W_i)+\mathbf W_s\mathbf{\vec x} 。$

    事实上当它们维度相等时，也可以执行线性变换。但是实践表明：使用恒等映射足以解决退化问题，而使用线性投影会增加参数和计算复杂度。因此 $\mathbf W_s $仅在匹配维度时使用。

3. 残差函数 F 的形式是可变的。

    - 层数可变：论文中的实验包含有两层堆叠、三层堆叠，实际任务中也可以包含更多层的堆叠。

      如果 F 只有一层，则残差块退化线性层：$\mathbf{\vec y} = \mathbf W\mathbf{\vec x}+\mathbf{\vec x} 。​$此时对网络并没有什么提升。

    - 连接形式可变：不仅可用于全连接层，可也用于卷积层。此时 F 代表多个卷积层的堆叠，而最终的逐元素加法`+` 在两个`feature map` 上逐通道进行。

      > 此时 `x` 也是一个`feature map`，而不再是一个向量。

4. 残差学习成功的原因：学习残差$ F(\mathbf{\vec x},\mathbf W_i) $比学习原始映射$ H(\mathbf{\vec x}) $要更容易。

    - 当原始映射 H 就是一个恒等映射时，F 就是一个零映射。此时求解器只需要简单的将堆叠的非线性连接的权重推向零即可。

      实际任务中原始映射 H 可能不是一个恒等映射：

      - 如果 H 更偏向于恒等映射（而不是更偏向于非恒等映射），则 F 就是关于恒等映射的抖动，会更容易学习。
      - 如果原始映射 H 更偏向于零映射，那么学习 H 本身要更容易。但是在实际应用中，零映射非常少见，因为它会导致输出全为 0。

    - 如果原始映射 H 是一个非恒等映射，则可以考虑对残差模块使用缩放因子。如`Inception-Resnet` 中：在残差模块与快捷连接叠加之前，对残差进行缩放。

      注意：`ResNet` 作者在随后的论文中指出：不应该对恒等映射进行缩放。因此`Inception-Resnet`对残差模块进行缩放。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_resnet_scale.png)

    - 可以通过观察残差 F 的输出来判断：如果 F 的输出均为 0 附近的、较小的数，则说明原始映射 H 更偏向于恒等映射；否则，说明原始映射 H 更偏向于非横等映射。

### 5.3 ResNet 分析

1. `Veit et al.` 认为`ResNet` 工作较好的原因是：一个`ResNet` 网络可以看做是一组较浅的网络的集成模型。

    但是`ResNet` 的作者认为这个解释是不正确的。因为集成模型要求每个子模型是独立训练的，而这组较浅的网络是共同训练的。

2. 论文`《Residual Networks Bahave Like Ensemble of Relatively Shallow Networks》` 对`ResNet` 进行了深入的分析。

    - 通过分解视图表明：`ResNet` 可以被视作许多路径的集合。

    - 通过研究`ResNet` 的梯度流表明：网络训练期间只有短路径才会产生梯度流，深的路径不是必须的。

    - 通过破坏性实验，表明：

      - 即使这些路径是共同训练的，它们也不是相互依赖的。
      - 这些路径的行为类似集成模型，其预测准确率平滑地与有效路径的数量有关。

#### 5.3.1 分解视图

1. 考虑从输出$ \mathbf {\vec y}_0 $到$ \mathbf {\vec y}_3​$ 的三个`ResNet` 块构建的网络。根据：

    $$
    \mathbf {\vec y}_3=\mathbf {\vec y}_2+f_3(\mathbf {\vec y}_2) =[\mathbf {\vec y}_1+f_2(\mathbf {\vec y}_1)]+f_3(\mathbf {\vec y}_1+f_2(\mathbf {\vec y}_1))\\ =[\mathbf {\vec y}_0+f_1(\mathbf {\vec y}_0)+f_2(\mathbf {\vec y}_0+f_1(\mathbf {\vec y}_0))]+f_3(\mathbf {\vec y}_0+f_1(\mathbf {\vec y}_0)+f_2(\mathbf {\vec y}_0+f_1(\mathbf {\vec y}_0)))
    $$
    下图中：左图为原始形式，右图为分解视图。分解视图中展示了数据从输入到输出的多条路径。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_unravel.png)

    对于严格顺序的网络（如`VGG` ），这些网络中的输入总是在单个路径中从第一层直接流到最后一层。如下图所示。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_unravel2.png)

2. 分解视图中， 每条路径可以通过二进制编码向量 $\mathbf {\vec b} =(b_1,b_2,\cdots,b_n),\quad b_i\in \{0,1\} $来索引：如果流过残差块$ f_i ，$则 $b_i=1 ；$如果跳过残差块$ f_i ，$则$ b_i=0 。$

    因此`ResNet` 从输入到输出具有$ 2^n$ 条路径，第 $i$ 个残差块$ f_i $的输入汇聚了之前的$ i-1 $个残差块的$ 2^{i-1} ​$条路径。

3. 普通的前馈神经网络也可以在单个神经元（而不是网络层）这一粒度上运用分解视图，这也可以将网络分解为不同路径的集合。

    它与`ResNet` 分解的区别是：

    - 普通前馈神经网络的神经元分解视图中，所有路径都具有相同的长度。
    - `ResNet` 网络的残差块分解视图中，所有路径具有不同的路径长度。

#### 5.3.2 路径长度分析

1. `ResNet` 中，从输入到输出存在许多条不同长度的路径。这些路径长度的分布服从二项分布。对于 n 层深的`ResNet`，大多数路径的深度为 $\frac n2 。$

    下图为一个 54 个块的`ResNet` 网络的路径长度的分布 ，其中`95%` 的路径只包含 19 ～ 35 个块。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_path_len.png)

#### 5.3.3 路径梯度分析

1. `ResNet` 中，路径的梯度幅度随着它在反向传播中经过的残差块的数量呈指数减小。因此，训练期间大多数梯度来源于更短的路径。

2. 对于一个包含 54 个残差块的`ResNet` 网络：

    - 下图表示：单条长度为 k 的路径在反向传播到 `input` 处的梯度的幅度的均值，它刻画了长度为 k 的单条路径的对于更新的影响。

      因为长度为 k 的路径有多条，因此取其平均。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_path_grad.png)

    - 下图表示：长度为 k 的所有路径在反向传播到 `input` 处的梯度的幅度的和。它刻画了长度为 k 的所有路径对于更新的影响。

      它不仅取决于长度为 k 的单条路径的对于更新的影响，还取决于长度为 k 的单条路径的数量。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_shallow_path.png)

3. 有效路径：反向传播到 `input` 处的梯度幅度相对较大的路径。

    `ResNet` 中有效路径相对较浅，而且有效路径数量占比较少。在一个 54 个块的`ResNet` 网络中：

    - 几乎所有的梯度更新都来自于长度为 5~17 的路径。
    - 长度为 5~17 的路径占网络所有路径的 0.45% 。

4. 论文从头开始重新训练`ResNet`，同时在训练期间只保留有效路径，确保不使用长路径。实验结果表明：相比于完整模型的 6.10% 的错误率，这里实现了 5.96% 的错误率。二者没有明显的统计学上的差异，这表明确实只需要有效路径。

    因此，`ResNet` 不是让梯度流流通整个网络深度来解决梯度消失问题，而是引入能够在非常深的网络中传输梯度的短路径来避免梯度消失问题。

5. 和`ResNet` 原理类似，随机深度网络起作用有两个原因：

    - 训练期间，网络看到的路径分布会发生变化，主要是变得更短。
    - 训练期间，每个`mini-batch` 选择不同的短路径的子集，这会鼓励各路径独立地产生良好的结果。

#### 5.3.4 路径破坏性分析

1. 在`ResNet` 网络训练完成之后，如果随机丢弃单个残差块，则测试误差基本不变。因为移除一个残差块时，`ResNet` 中路径的数量从$ 2^n $减少到$ 2^{n-1}，$留下了一半的路径。

    在`VGG` 网络训练完成之后，如果随机丢弃单个块，则测试误差急剧上升，预测结果就跟随机猜测差不多。因为移除一个块时，`VGG` 中唯一可行的路径被破坏。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_drop_index.png)

2. 删除`ResNet` 残差块通常会删除长路径。

    当删除了 $k $个残差块时，长度为$ x $的路径的剩余比例由下式给定：$percent=\frac {C_{n-k}^x}{C_n^x} 。$

    下图中：

    - 删除 10 个残差模块，一部分有效路径（路径长度为`5~17`）仍然被保留，模型测试性能会部分下降。
    - 删除 20 个残差模块，绝大部分有效路径（路径长度为`5~17`）被删除，模型测试性能会大幅度下降。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_remain_block.png)

3. `ResNet` 网络中，路径的集合表现出一种类似集成模型的效果。一个关键证据是：它们的整体表现平稳地取决于路径的数量。随着网络删除越来越多的残差块，网络路径的数量降低，测试误差平滑地增加（而不是突变）。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_delete_layers.png)

4. 如果在测试时重新排序网络的残差块，这意味着交换了低层映射和高层映射。采用`Kendall Tau rank` 来衡量网络结构被破坏的程度，结果表明：随着 `Kendall Tau rank` 的增加，预测错误率也在增加。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_reorder.png)

### 5.4 网络性能

1. `plain` 网络：一些简单网络结构的叠加，如下图所示。图中给出了四种`plain` 网络，它们的区别主要是网络深度不同。其中，输入图片尺寸 224x224 。

    `ResNet` 简单的在`plain` 网络上添加快捷连接来实现。

    > `FLOPs`：`floating point operations` 的缩写，意思是浮点运算量，用于衡量算法/模型的复杂度。
    >
    > `FLOPS`：`floating point per second`的缩写，意思是每秒浮点运算次数，用于衡量计算速度。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_struct.png)

2. 相对于输入的`feature map`，残差块的输出`feature map` 尺寸可能会发生变化：

    - 输出 `feature map` 的通道数增加，此时需要扩充快捷连接的输出`feature map` 。否则快捷连接的输出 `feature map` 无法和残差块的`feature map` 累加。

      有两种扩充方式：

      - 直接通过 0 来填充需要扩充的维度，在图中以实线标识。
      - 通过`1x1` 卷积来扩充维度，在图中以虚线标识。

    - 输出 `feature map` 的尺寸减半。此时需要对快捷连接执行步长为 2 的池化/卷积：如果快捷连接已经采用 `1x1` 卷积，则该卷积步长为 2 ；否则采用步长为 2 的最大池化 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_reorder.png)

3. 计算复杂度：

    |                   | VGG-19       | 34层 plain 网络 | Resnet-34   |
    | ----------------- | ------------ | --------------- | ----------- |
    | 计算复杂度(FLOPs) | 19.6 billion | 3.5 billion     | 3.6 billion |

4. 模型预测能力：在`ImageNet` 验证集上执行`10-crop` 测试的结果。

    - `A` 类模型：快捷连接中，所有需要扩充的维度的填充 0 。
    - `B` 类模型：快捷连接中，所有需要扩充的维度通过`1x1` 卷积来扩充。
    - `C` 类模型：所有快捷连接都通过`1x1` 卷积来执行线性变换。

    可以看到`C` 优于`B`，`B` 优于`A`。但是 `C` 引入更多的参数，相对于这种微弱的提升，性价比较低。所以后续的`ResNet` 均采用 `B` 类模型。

    | 模型        | top-1 误差率 | top-5 误差率 |
    | ----------- | ------------ | ------------ |
    | VGG-16      | 28.07%       | 9.33%        |
    | GoogleNet   | -            | 9.15%        |
    | PReLU-net   | 24.27%       | 7.38%        |
    | plain-34    | 28.54%       | 10.02%       |
    | ResNet-34 A | 25.03%       | 7.76%        |
    | ResNet-34 B | 24.52%       | 7.46%        |
    | ResNet-34 C | 24.19%       | 7.40%        |
    | ResNet-50   | 22.85%       | 6.71%        |
    | ResNet-101  | 21.75%       | 6.05%        |
    | ResNet-152  | 21.43%       | 5.71%        |

## 六、ResNet 变种

### 6.1 恒等映射修正

1.  在论文`《Identity Mappings in Deep Residual Networks》`中，`ResNet` 的作者通过实验证明了恒等映射的重要性，并且提出了一个新的残差单元来简化恒等映射。

#### 6.1.1 新残差块

1. 新的残差单元中，恒等映射添加到`ReLU` 激活函数之后。它使得训练变得更简单，并且提高了网络的泛化能力。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/cnn_resnet_identity_modified.png)

2. 假设 $\mathbf{\vec x}_l $是第$ l $个残差单元的输入特征$；\mathcal W_l=\{\mathbf W_{l,k}\mid 1\le k \le K\} $为一组与第$ l $个残差单元相关的权重（包括偏置项）$，K$ 是残差单元中的层的数量；$ \mathcal F $代表残差函数。则第$ l$ 个残差单元的输出为（它也等价于第$ l+1​$ 个残差单元的输入）：

    $\mathbf{\vec x}_{l+1}=\mathbf{\vec x}_l+\mathcal F(\mathbf{\vec x}_l,\mathcal W_l)$

    考虑递归，对于任意深的残差单元 L ，则有：

    $$
    \mathbf{\vec x}_{L}=\mathbf{\vec x}_{L-1}+\mathcal F(\mathbf{\vec x}_{L-1},\mathcal W_{L-1})\\ =\mathbf{\vec x}_{L-2}+\mathcal F(\mathbf{\vec x}_{L-1},\mathcal W*{L-1})+\mathcal F(\mathbf{\vec x}*{L-2},\mathcal W*{L-2})\\ \vdots\\ =\mathbf{\vec x}*{l}+\sum_{i=l}^{L-1}\mathcal F(\mathbf{\vec x}_i,\mathcal W_i)
    $$
    因此，对任意深的单元$ L $，其输入特征 $\mathbf{\vec x}_L$ 可以表示为浅层单元$ l $的特征$ \mathbf{\vec x}_l $加上一个形如$ \sum_{i=l}^{L-1}\mathcal F(\mathbf{\vec x}_i,\mathcal W_i) ​$的残差函数。

    这意味着：任意单元$ L$ 和 $l $之间都具有残差性。

3. 对于任意深的单元$ L ，$其输入特征$ \mathbf{\vec x}_L$ 可以表示为$：\mathbf{\vec x}_{L}=\mathbf{\vec x}_{0}+\sum_{i=0}^{L-1}\mathcal F(\mathbf{\vec x}_i,\mathcal W_i) $。即：之前所有残差函数输出的总和，再加上 $\mathbf{\vec x}_0 。$

    与之形成鲜明对比的是常规网络中，输入特征$ \mathbf{\vec x}_L $是一系列矩阵向量的乘积。即为：$\mathbf{\vec x}_{L}=\prod_{i=0}^{L-1}\mathbf W_i\mathbf{\vec x_0} ​$（忽略了激活函数和 `BN` ）。

4. 新的残差单元也更具有良好的反向传播特性。对于损失函数$ \mathcal L ，$有：

    $$
    \frac{\partial \mathcal L}{\partial \mathbf{\vec x}_l}= \left(\frac{\partial \mathbf{\vec x}_L}{\partial \mathbf{\vec x}_l}\right)^T\frac{\partial \mathcal L}{\partial \mathbf{\vec x}_L}=\left(1+\frac{\partial \sum_{i=l}^{L-1}\mathcal F(\mathbf{\vec x}_i,\mathcal W_i)}{\partial \mathbf{\vec x}_l}\right)^T\frac{\partial \mathcal L}{\partial \mathbf{\vec x}_L}
    $$
    可以看到：

    - 梯度$ \frac{\partial \mathcal L}{\partial \mathbf{\vec x}_l} $可以分解为两个部分：

      - $\frac{\partial \mathcal L}{\partial \mathbf{\vec x}_L} $：直接传递信息而不涉及任何权重。它保证了信息能够直接传回给任意浅层 l 。
      - $\left(\frac{\partial \sum_{i=l}^{L-1}\mathcal F(\mathbf{\vec x}_i,\mathcal W_i)}{\partial \mathbf{\vec x}_l}\right)^T\frac{\partial \mathcal L}{\partial \mathbf{\vec x}_L} $：通过各权重层来传递。

    - 在一个`mini-batch` 中，不可能出现梯度消失的情况。

      可能对于某个样本，存在$ \frac{\partial \sum*{i=l}^{L-1}\mathcal F(\mathbf{\vec x}_i,\mathcal W_i)}{\partial \mathbf{\vec x}_l}=-1 $的情况，但是不可能出现`mini-batch` 中所有的样本满足 $\frac{\partial \sum*{i=l}^{L-1}\mathcal F(\mathbf{\vec x}_i,\mathcal W_i)}{\partial \mathbf{\vec x}_l}=-1 。$

      这意味着：哪怕权重是任意小的，也不可能出现梯度消失的情况。

    > 对于旧的残差单元，由于恒等映射还需要经过`ReLU` 激活函数，因此当 $\mathbf{\vec x}+\mathcal F(\mathbf{\vec x},\mathcal W) <0$ 时饱和，其梯度为 0 。

5. 根据`3.` 和`4.` 的讨论表明：在前向和反向阶段，信号都能够直接传递到任意单元。

#### 6.1.2 快捷连接验证

1. 假设可以对快捷连接执行缩放（如线性的`1x1` 卷积），第 l 个残差单元的缩放因子为$ \lambda*l ，$其中$ \lambda_l$ 也是一个可以学习的参数。此时有：$\mathbf{\vec x}*{l+1}=\lambda*l\mathbf{\vec x}_l+\mathcal F(\mathbf{\vec x}_l,\mathcal W_l) ，以及：\mathbf{\vec x}*{L}=(\prod_{i=l}^{L-1}\lambda_i)\mathbf{\vec x}_l+\sum*{i=l}^{L-1}(\prod*{j=i+1}^{L-1}\lambda_j)\mathcal F(\mathbf{\vec x}_i,\mathcal W_i) 。$

    令：$\hat { \mathcal F}(\mathbf{\vec x}_i,\mathcal W_i) =(\prod_{j=i+1}^{L-1}\lambda_j)\mathcal F(\mathbf{\vec x}_i,\mathcal W_i)，$则有：

    $$
    \mathbf{\vec x}_{L}=\left(\prod_{i=l}^{L-1}\lambda_i\right)\mathbf{\vec x}_l+\sum*{i=l}^{L-1}\hat { \mathcal F}(\mathbf{\vec x}_i,\mathcal W_i)\\ \frac{\partial \mathcal L}{\partial \mathbf{\vec x}_l}= \left(\frac{\partial \mathbf{\vec x}_L}{\partial \mathbf{\vec x}_l}\right)^T\frac{\partial \mathcal L}{\partial \mathbf{\vec x}_L}=\left( \prod*{i=l}^{L-1}\lambda_i +\frac{\partial \sum_{i=l}^{L-1}\hat {\mathcal F}(\mathbf{\vec x}_i,\mathcal W_i)}{\partial \mathbf{\vec x}_l}\right)^T\frac{\partial \mathcal L}{\partial \mathbf{\vec x}_L}
    $$
    对于特别深的网络：如果$ \lambda*i \gt 1,\text{for all} \;i ，$ 则$ \prod*{i=l}^{L-1}\lambda*i $发生梯度爆炸；如果 $\lambda_i \lt 1,\text{for all} \;i ， $则 $\prod*{i=l}^{L-1}\lambda_i ​$发生梯度消失。这会丧失快捷连接的好处。

2. 如果对快捷连接执行的不是线性缩放，而是一个复杂的函数$ h*l ，$则上式括号中第一部分变成$：\prod*{i=l}^{L-1} h_i^{\prime} 。$其中 $h_i^\prime 为 h_i $的导数。

    这也会丧失快捷连接的好处，阻碍梯度的传播。

3. 下图所示为对快捷连接进行的各种修改：

    > 为了简化，这里没有画出`BN` 层。每个权重层的后面实际上都有一个`BN` 层。

    - `(a)`：原始的、旧的残差块。

    - `(b)`：对所有的快捷连接设置缩放。其中缩放因子$ \lambda=0.5 。$

      残差有两种配置：缩放（缩放因子 0.5）、不缩放。

    - `(c)`：对快捷连接执行门控机制。残差由$ g$ 来缩放，快捷连接由 $1-g $来缩放。

      其中$ g(\mathbf{\vec x}) = \sigma(\mathbf W_g\mathbf{\vec x})+\mathbf{\vec b}_g ， \sigma(z) = \frac{1}{1+e^{-z}} 。$

    - `(d)`：对快捷连接执行门控机制，但是残差并不进行缩放。

    - `(e)`： 对快捷连接执行`1x1` 卷积。

    - `(f)`：对快捷连接执行`dropout`，其中遗忘比例为 0.5 。

      在统计学上，它等效于一个缩放比例为 0.5 的缩放操作。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_modified1.png)

    在 `CIFAR-10` 上利用`ResNet-110` 的测试误差如下：（`fail` 表示测试误差超过 20% ）

    `on shortcut` 和 `on F` 列分别给出了快捷连接、残差块上的缩放比例。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_shortcut_test.png)

4. 最终结果表明：快捷连接是信息传递最直接的路径，快捷连接中的各种操作都会阻碍信息的传递，以致于对优化造成困难。

5. 理论上，对快捷连接执行`1x1` 卷积，会引入更多的参数。它应该比恒等连接具备更强大的表达能力。

    事实上，其训练误差要比恒等连接的训练误差高的多。这意味着模型退化是因为优化问题，而不是网络表达能力的问题。

#### 6.1.3 激活函数验证

1. 设残差块之间的函数为 f ，即：`+` 之后引入 f ：

    $\mathbf{\vec y}_l=\mathbf{\vec x}_l+\mathcal F(\mathbf{\vec x}_l,\mathcal W_l),\quad \mathbf{\vec x}_{l+1} = f(\mathbf{\vec y}_l)​$

    前面的理论推导均假设$ f $为恒等映射 $f(z) = z ，$而上面的实验中 $f(z) = ReLU(z) $。因此接下来考察 f 的影响。

2. 如下图所示，组件都相同，但是不同的组合导致不同的残差块或 $f 。$

    - `(a)`：原始的、旧的残差块，$ f(z)=ReLU(z) 。$

    - `(b)`：将`BN` 移动到`addition` 之后$， f(z)=ReLU(BN(z)) 。$

    - `(c)`：将`ReLU` 移动到`addition` 之前$，f(z) = z。$

      这种结构问题较大，因为理想的残差块的输出范围是 $(-\infty,+\infty) 。$这里的残差块经过个`ReLU` 之后的输出为非负，从而使得残差的输出为$ [0,+\infty)，$从而使得前向信号会逐级递增。这会影响网络的表达能力。

    - `(d)`：将`ReLU` 移动到残差块之前，$f(z) = z。​$

    - `(e)`： 将`BN` 和`ReLU` 移动到残差块之前$，f(z) = z。$

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_modified2.png)

3. 最终结果表明：`full pre-activation` 效果最好。有两个原因：

    - 快捷连接通路是顺畅的，这使得优化更加简单。

    - 对两个权重层的输入都执行了`BN`。

      所有其它四组结构中，只有第二个权重层的输入的到了标准化，第一个权重层的输入并未的到标准化。

#### 6.1.4 网络性能

1. 在 `ILSVRC 2012` 验证集上的评估结果：

    | 方法                            | 数据集增强        | train crop | test crop | top-1 误差 | top-5 误差 |
    | ------------------------------- | ----------------- | ---------- | --------- | ---------- | ---------- |
    | ResNet-152，原始残差块          | scale             | 224x224    | 224x224   | 23.0%      | 6.7%       |
    | ResNet-152，原始残差块          | scale             | 224x224    | 320x320   | 21.3%      | 5.5%       |
    | ResNet-152，full pre-activation | scale             | 224x224    | 320x320   | 21.1%      | 5.5%       |
    | ResNet-200，原始残差块          | scale             | 224x224    | 320x320   | 21.8%      | 6.0%       |
    | ResNet-200，full pre-activation | scale             | 224x224    | 320x320   | 20.7%      | 5.3%       |
    | ResNet-200，full pre-activation | scale + asp ratio | 224x224    | 320x320   | 20.1%      | 4.8%       |
    | Inception v3                    | scale + asp ratio | 299x299    | 299x299   | 21.2%      | 5.6%       |

### 6.2 ResNeXt

1. 通常提高模型准确率的方法是加深网络深度或者加宽网络宽度，但这些方法会增加超参数的数量、参数数量和计算量。

    `ResNeXt` 网络可以在不增加网络参数复杂度的前提下提高准确率，同时还减少了超参数的数量。

2. `ResNeXt` 的设计参考了`VGG` 和`Inception` 的设计哲学。

    - `VGG`：网络通过简单地层叠相同结构的层来实现，因此网络结构简单。其缺点是网络参数太多，计算量太大。

    - `Inception`：通过执行`分裂-变换-合并`策略来精心设计拓扑结构，使得网络参数较少，计算复杂度较低。这种`分裂-变换-合并`行为预期能够达到一个大的`dense` 层的表达能力，但是计算复杂度要低的多。

      其缺点是：

      - 每个“变换”中，滤波器的数量和尺寸等超参数都需要精细的设计。
      - 一旦需要训练新的任务（如新任务是一个`NLP` 任务），可能需要重新设计网络结构。因此可扩展性不高。

    - `ResNeXt` 结合了二者的优点：

      - 网络结构也是通过简单地层叠相同结构的层来实现。
      - 网络的每一层都执行了`分裂-变换-合并`策略。

3. 在相同的参数数量和计算复杂度的情况下，`ResNeXt` 的预测性能要优于`ResNet` 。

    - 它在`ILSVRC 2016` 分类任务中取得了第二名的成绩。
    - `101` 层的`ResNeXt` 就能够获得超过`200` 层`ResNet` 的准确率，并且计算量只有后者的一半。

4. `ResNeXt` 改进了`ResNet` 网络结构，并提出了一个新的维度，称作“基数”`cardinality`。基数是网络的深度和网络的宽度之外的另一个重要因素。

    作者通过实验表明：增加基数比增加网络的深度或者网络的宽度更有效。

#### 6.2.1 分裂-变换-合并

1. 考虑全连接网络中的一个神经元。假设输入为$ \mathbf{\vec x}=(x_1,x_2,\cdots,x*D) ，$为一个一度的输入向量（长度为 D ）。假设对应的权重为 $\mathbf{\vec w}=(w_1,w_2,\cdots,w_D)。$不考虑偏置和激活函数，则神经元的输出为：$\sum*{i=1}^Dw_ix_i $。

    它可以视作一个最简单的“分裂-变换-合并”：

    - 分裂：输入被分割成$ D $个低维（维度为零）嵌入。
    - 变换：每个低维嵌入通过对应的权重$ w_i$ 执行线性变换。
    - 合并：变换之后的结果通过直接相加来合并。

2. `Inception` 的“分裂-变换-合并”策略：

    - 分裂：输入通过`1x1` 卷积被分割成几个低维嵌入。
    - 变换：每个低维嵌入分别使用一组专用滤波器（`3x3`、`5x5` 等） 执行变换。
    - 合并：变换之后的结果进行合并（沿深度方向拼接）。

3. 对一个`ResNeXt` 块，其“分裂-变换-合并”策略用公式表述为：

    $\mathcal F(\mathbf{\vec x}) = \sum_{i=1}^C\mathcal T_i(\mathbf{\vec x}),\quad \mathbf{\vec y}=\mathbf{\vec x}+\mathcal F(\mathbf{\vec x})$

    其中：

    - $\mathcal T_i$ 为任意函数，它将 $\mathbf{\vec x} $映射为$ \mathbf{\vec x} $的一个低维嵌入，并对该低维嵌入执行转换。
    - $C$ 为转换的数量，也就是基数`cardinality`。

4. 在`ResNeXt` 中，为了设计方便$ \mathcal T_i $采取以下设计原则：

    - 所有的 $\mathcal T_i$ 具有相同的结构。这是参考了`VGG` 的层叠相同结构的层的思想。

    - $\mathcal T_i $的结构通常是：

      - 第一层：执行`1x1` 的卷积来产生 $\mathbf{\vec x}$ 的一个低维嵌入。
      - 第二层 ~ 倒数第二层：执行卷积、池化等等变换。
      - 最后一层：执行`1x1` 的卷积来将结果提升到合适的维度。

#### 6.2.2 ResNeXt 块

1. 一个`ResNeXt` 模块执行了一组相同的“变换”，每一个“变换”都是输入的一个低维嵌入，变换的数量就是基数 `C` 。

    如下所示：左图为`ResNet` 块；右图为`ResNeXt` 块。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/ResNeXt1.png)

2. `ResNeXt` 模块有两种等效的形式：图`(a)` 为标准形式，图`(b)`类似`Inception-ResNet` 模块。其中图`(b)` 的拼接是沿着深度方向拼接。

    - 等效的原因是：输入通道数为`128` 的`1x1` 卷积可以如下拆分：( 设输入张量为$ \mathbf I ，$输出张量为$ \mathbf O，$核张量为$ \mathbf K )​$

      $$
      \mathbf O*{i,j,k}=\sum*{s=1}^{128} \mathbf I*{s,j,k}\times \mathbf K*{i,s}\\ =\left(\sum*{s=1}^{4} \mathbf I*{s,j,k}\times \mathbf K*{i,s}\right)+\left(\sum*{s=5}^{8} \mathbf I*{s,j,k}\times \mathbf K*{i,s}\right)\\+\cdots+\left(\sum*{s=124}^{128} \mathbf I*{s,j,k}\times \mathbf K_{i,s}\right)\\ i = 1,2,\cdots,256
      $$
      经过这种拆分，图`(b)` 就等效于图`(a)`。其中：$i $表示输出单元位于 $i$ 通道，$s$ 表示输入单元位于$ s$ 通道，$j,k$ 表示通道中的坐标。

      > 本质原因是`1x1` 卷积是简单的对通道进行线性相加。它可以拆分为：先将输入通道分组，然后计算各组的子通道的线性和（`1x1` 卷积）；然后将所有组的和相加。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnetx_merge.png)

    - 图`(b)` 与`Inception-ResNet` 模块的区别在于：这里每一条路径都是相同的。

    - 图`(c)` 是一个分组卷积的形式，它就是用分组卷积来实现图`(b)`。它也是图`(b)` 在代码中的实现方式。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/ResNeXt2.png)

3. 通常`ResNeXt` 模块至少有三层。事实上它也可以有两层，此时它等效于一个宽的、密集模块。

    - 此时并没有通过 `1x1` 卷积进行降维与升维，而是在降维的过程中同时进行变换，在升维的过程中也进行变换。
    - 如下图所示，它等价于图`(c)` 中，去掉中间的变换层(`128,3x3,128` 层)，同时将第一层、第三层的 `1x1` 替换为`3x3` 卷积层。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/ResNeXt3.png)

1. `ResNeXt` 的两种重要超参数是：基数`C` 和颈宽`d` 。

    - 基数 `C`：决定了每个`ResNeXt` 模块有多少条路径。
    - 颈宽（`bottleneck width`）`d`：决定了`ResNeXt` 模块中第一层`1x1` 卷积降维的维度。

    > 这二者也决定了`ResNeXt` 模块等价形式中，通道分组卷积的通道数量为 `Cxd` 。

2. `ResNeXt` 的网络参数和计算量与同等结构的`ResNet` 几乎相同。以`ResNet-50` 为例（输入图片尺寸`224x224` ）：

    > `ResNeXt-50(32x4d)` 意思是：基数`C=32`，颈宽`d=4` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnetx_struct.png)

3. 在`ImageNet` 上进行的对比实验（验证集误差，`single crop` ）：

    - 基数 vs 颈宽：基数越大越好。

      | 模型        | 配置     | top-1 error(%) |
      | ----------- | -------- | -------------- |
      | ResNet-50   | C=1,d=64 | 23.9           |
      | ResNeXt-50  | C=2,d=40 | 23.0           |
      | ResNeXt-50  | C=4,d=24 | 22.6           |
      | ResNeXt-50  | C=8,d=14 | 22.3           |
      | ResNeXt-50  | C=32,d=4 | 22.2           |
      | ResNet-101  | C=1,d=64 | 22.0           |
      | ResNeXt-101 | C=2,d=40 | 21.7           |
      | ResNeXt-101 | C=4,d=24 | 21.4           |
      | ResNeXt-101 | C=8,d=14 | 21.3           |
      | ResNeXt-101 | C=32,d=4 | 21.2           |

    - 基数 vs 深度/宽度：基数越大越好。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnetx_width_deep.png)

4. 与其它模型的预测能力比较（验证集误差，`single crop`）：

    > `ResNet/ResNeXt` 的图片尺寸为`224x224` 和 `320x320`；`Inception` 的图片尺寸为`299x299` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnetx_performance.png)

### 6.3 随机深度网络

1. 随机深度网络提出了训练时随机丢弃网络层的思想，从而能够让网络深度增加到超过 1000 层，并仍然可以减少测试误差。

    如图所示：在`CIFAR-10` 上，`1202` 层的`ResNet` 测试误差要高于 `110` 层的`ResNet` ，表现出明显的过拟合。而 `1202` 层的随机深度网络（结合了`ResNet` ）的测试误差要低于 `110` 层的`ResNet` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/stochastic_depth_net_2.png)

2. 神经网络的表达能力主要由网络深度来决定，但是过深的网络会带来三个问题：反向传播过程中的梯度消失、前向传播过程中的`feature` 消失、训练时间过长。

    - 虽然较浅的网络能够缓解这几个问题，但是较浅的网络表达能力不足，容易陷入欠拟合。

    - 随机深度网络解决这一矛盾的策略是：构建具有足够表达能力的深度神经网络（具有数百层甚至数千层），然后：

      - 在网络训练期间，对每个`mini batch` 随机地移除部分层来显著的减小网络的深度。

        移除操作：删除对应的层，并用跳跃连接来代替。

      - 在网络测试期间，使用全部的网络层。

3. 随机深度的思想可以和`ResNet` 结合。因为`ResNet` 已经包含了跳跃连接，因此可以直接修改。

#### 6.3.1 随机深度

1. 假设`ResNet` 有 L 个残差块，则有：$\mathbf{\vec x}_{l+1}=\mathbf{\vec x}_l+\mathcal F(\mathbf{\vec x}_l,\mathcal W_l),\quad l=0,1,\cdots,L-1 。$其中：

    - $\mathbf{\vec x}_{l+1} $表示第$ l $个残差块的输出$，\mathbf{\vec x}_l $为第 $l$ 个残差块的输入（它也是第 $l-1 $个残差块的输出）。
    - $\mathcal W_l=\{\mathbf W_{l,k}\mid 1\le k \le K\}$ 为一组与第 $l $个残差单元相关的权重（包括偏置项）$，K $是残差单元中的层的数量。
    - $\mathcal F$ 代表残差函数。

2. 假设第$ l $个残差块是否随机丢弃由伯努利随机变量$ b_l\in \{0,1\} $来指示：当$ b_l=0 $时，第$ l $个残差块被丢弃；当 $b_l=1 $时，第$ l $个残差块被保留。

    因此有：$\mathbf{\vec x}_{l+1}=\mathbf{\vec x}_l+b_l\times \mathcal F(\mathbf{\vec x}_l,\mathcal W_l),\quad l=0,1,\cdots,L-1 。$

    对随机变量$ b_l ，​$令：

    $P(b_l)=\begin{cases} p_l,&b_l=1\\ 1-p_l ,&b_l=0 \end{cases}$

    其中$ p_l$ 称做保留概率或者存活概率，它是一个非常重要的超参数。

3. $p_l $的选择有两个策略：

    - 所有残差块的存活概率都相同：$p_0=p_1=\cdots=p*{L-2}=p*{L-1} 。$

    - 所有残差块的存活概率都不同，且根据残差块的深度进行线性衰减：

      $p_l=1-\frac {l+1}L(1-p_L),\quad l=0,1,\cdots,L-1​$

      其背后的思想是：靠近输入的层提取的是被后续层使用的低级特征，因此更应该被保留下来。

4. 给定第 $l $个残差块的保留概率$ p*l ，$则网络的深度$ \tilde L$ 的期望为（以残差块数量为单位）$：\mathbb E(\tilde L) = \sum*{l=1}^L p_l 。$

    - 对于均匀存活：$\mathbb E(\tilde L) = p_L\times L$

    - 对于线性衰减存活：

      $\mathbb E(\tilde L)=\frac{(1+p_L)\times L-(1-p_L)}{2}$

      当 p_L \lt 1 时，无论是均匀存活还是线性衰减存活，都满足$ \mathbb E(\tilde L) \lt L 。$因此随机深度网络在训练时具有更短的期望深度，从而节省了训练时间。

5. $p_l $的选择策略，以及$ p_L $的大小的选取需要根据实验仔细选择。

    - 根据作者的实验结果，作者推荐使用线性衰减存活概率，并选择 $p_L=0.5 。$此时有：$ \mathbb E(\tilde L)=\frac {(3L-1)}{4}$
    - 如果选择更小的 $p_L ​$将会带来更大的测试误差，但是会更大的加速训练过程。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/stochastic_depth_net_1.png)

6. 测试时，需要调整残差块的输出：$\mathbf{\vec x}_{l+1}^{test}=\mathbf{\vec x}_l^{test}+p_l\times \mathcal F(\mathbf{\vec x}_l^{test},\mathcal W_l),\quad l=0,1,\cdots,L-1 。$

7. 随机深度网络能够大大减少训练时间和测试误差。

    - 训练时间减小是因为：网络训练时，期望深度减小。

    - 测试误差减小是因为：

      - 网络训练时期望深度的减少，使得梯度链变短，从而加强了反向传播期间靠近输入层的梯度。
      - 随机深度网络可以被理解为一系列不同深度的网络的隐式集成的集成模型。

8. 随机深度网络可以视作 2^L 个隐式的神经网络的集成，这些被集成的网络都是权重共享的，类似于`Dropout` 。

    - 在训练时，对每个`mini batch`，只有其中之一得到了权重更新。
    - 在测试时，取所有被集成的网络的平均。

9. 随机深度网络和 `Dropout` 都可以被理解为一系列网络的隐式集成。

    - 随机深度集网络成了一系列具有不同深度的神经网络，而 `Dropout`集成了一系列具有不同宽度的神经网络。
    - `Dropout` 与`BN`配合使用时会失效，而随机深度可以和`BN` 配合使用。

10. 在随机深度网络中，由于训练时的随机深度，模型的测试误差的抖动相对于`ResNet` 会偏大。

#### 6.3.2 网络性能

1. 对`ResNet` （固定深度和随机深度）在三个数据集上进行比较，测试误差的结果如下：

    - `+` 表示执行了数据集增强。

    - 随机深度网络采用 p_L=0.5 。

    - `CIFAR10/100` 采用 110 层 `ResNet`，`ImageNet` 采用 152 层 `ResNet` 。

    - 虽然在 `ImageNet` 上随机深度模型没有提升，但是作者表示这是因为网络本身比较简单（相对 `ImageNet` 数据集）。如果使用更深的`ResNet`，则容易看到随机深度模型的提升。

      > 虽然这里模型的测试误差没有提升，但是训练速度大幅提升。

    | 网络               | CIFAR10+ | CIFAR100+ | ImageNet |
    | ------------------ | -------- | --------- | -------- |
    | ResNet（固定深度） | 6.41     | 27.76     | 21.78    |
    | ResNet（随机深度） | 5.25     | 24.98     | 21.98    |

2. 在 `CIFAR-10` 上，`ResNet` 随机深度网络的深度 L 、概率 p_L 与测试集误差的关系：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/stochastic_net.png)

## 七、SENet

1.  `SENet` 提出了一种新的架构单元来解决通道之间相互依赖的问题。它通过显式地对通道之间的相互依赖关系建模，自适应的重新校准通道维的特征响应，从而提高了网络的表达能力。
2.  `SENet` 以`2.251% top-5` 的错误率获得了`ILSVRC 2017` 分类比赛的冠军。
3.  `SENet` 是和`ResNet` 一样，都是一种网络框架。它可以直接与其他网络架构一起融合使用，只需要付出微小的计算成本就可以产生显著的性能提升。

### 7.1 SE 块

1. `SE` 块（`Squeeze-and-Excitation`）是`SENet` 中提出的新的架构单元，它主要由`squeeze` 操作和`excitation` 操作组成。

2. 对于给定的任何变换 $\mathbf F_{tr}: \mathbf X \rightarrow \mathbf U，$其中：$\mathbf X\in \mathbb R^{W^\prime\times H^\prime\times C^\prime} $为输入`feature map`，其尺寸为$ W^\prime\times H^\prime，$通道数为$ C ^\prime；\mathbf U\in \mathbb R^{W\times H\times C} $为输出`feature map`，其尺寸为 $W\times H，$通道数为$ C。$

    可以构建一个相应的`SE`块来对输出`feature map` $\mathbf U $执行特征重新校准：

    - 首先对输出`feature map` $\mathbf U $`squeeze`操作，它对每个通道的全局信息建模，生成一组通道描述符。
    - 然后是一个`excitation` 操作，它对通道之间的依赖关系建模，生成一组权重信息（对应于每个通道的权重）。
    - 最后输出`feature map` $\mathbf U ​$被重新加权以生成`SE` 块的输出。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet_seblock.png)

3. `SE` 块可以理解为注意力机制的一个应用。它是一个轻量级的门机制，用于对通道关系进行建模。

    通过该机制，网络学习全局信息（全通道、全空间）来选择性的强调部分特征，并抑制其它特征。

4. 设$ \mathbf U=[\mathbf u_1,\mathbf u_2,\cdots,\mathbf u*C] ， \mathbf u_c $为第 $c $个通道，是一个$ W\times H $的矩阵；设 $\mathbf X=[\mathbf x_1,\mathbf x_2,\cdots,\mathbf x*{C^\prime}] ， \mathbf x_j $为第 $j $个通道，是一个$ W^\prime\times H^\prime $的矩阵。

    - 需要学习的变换$ \mathbf F_{tr}=[\mathbf V_1,\mathbf V_2,\cdots,\mathbf V_C] $就是一组卷积核。 \mathbf V_c 为第 c 个卷积核，记做： $\mathbf V_c=[\mathbf v_c^{(1)},\mathbf v_c^{(2)},\cdots,\mathbf v_c^{(C^\prime)}] ，\mathbf v_c^{(j)} $为第 $c$ 个卷积核的第 $j $通道，是一个二维矩阵。则：

      $\mathbf u*c = \mathbf V_c *\mathbf X = \sum_{j=1}^{C^\prime} \mathbf v*c^{(j)}*\mathbf x_j$

      这里 `*` 表示卷积。同时为了描述简洁，这里忽略了偏置项。

    - 输出$ \mathbf u_c $考虑了输入 $\mathbf X$ 的所有通道，因此通道依赖性被隐式的嵌入到 $\mathbf V_c $中。

#### 7.1.1 squeeze 操作

1. `squeeze` 操作的作用是：跨空间$ W\times H $聚合特征来产生通道描述符。

    该描述符嵌入了通道维度特征响应的全局分布，包含了全局感受野的信息。

2. 每个学到的滤波器都是对局部感受野进行操作，因此每个输出单元都无法利用局部感受野之外的上下文信息。

    在网络的低层，其感受野尺寸很小，这个问题更严重。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet_squeeze.png)

    为减轻这个问题，可以将全局空间信息压缩成一组通道描述符，每个通道对应一个通道描述符。然后利用该通道描述符。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet_squeeze2.png)

3. 通常基于通道的全局平均池化来生成通道描述符（也可以考虑使用更复杂的聚合策略）。

    设所有通道的通道描述符组成一个向量$ \mathbf{\vec z} \in \mathbb R^C 。$则有：

    $\mathbf {\vec z}=[z_1,z_2,\cdots,z_C]\\ z_c=\frac{1}{W\times H}\sum*{i=1}^W\sum*{j=1}^Hu_c(i,j);\quad c=1,2,\cdots,C$

    .

#### 7.1.2 excitation 操作

1. `excitation` 操作的作用是：通过自门机制来学习每个通道的激活值，从而控制每个通道的权重。

2. `excitation` 操作利用了`squeeze` 操作输出的通道描述符 $\mathbf{\vec z} 。$

    - 首先，通道描述符$ \mathbf{\vec z} $经过线性降维之后，通过一个`ReLU` 激活函数。

      降维通过一个输出单元的数量为 $\frac Cr$ 的全连接层来实现，其中 $r $为降维比例。

    - 然后，`ReLU` 激活函数的输出经过线性升维之后，通过一个`sigmoid` 激活函数。

      升维通过一个输出单元的数量为$ C​$ 的全连接层来实现。

    通过对通道描述符$ \mathbf{\vec z} $进行降维，显式的对通道之间的相互依赖关系建模。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet_excitation.png)

3. 设`excitation` 操作的输出为向量$ \mathbf{\vec s}，$则有：$\mathbf{\vec s} = \sigma(\mathbf W_2 \text{ReLU}(\mathbf W_1\mathbf {\vec z})) 。$

    其中：$\sigma $为`sigmoid`激活函数，$\mathbf W_1 \in \mathbb R^{C/r\times C} $为降维层的权重$，\mathbf W_2 \in \mathbb R^{C\times C/r} $为升维层的权重，$r $为降维比例。

4. 在经过`excitation` 操作之后，通过重新调节 $\mathbf U $得到`SE` 块的输出。

    设`SE` 块的最终输出为$ \widetilde {\mathbf X}=[\widetilde{\mathbf x}_1,\widetilde{\mathbf x}_2,\cdots,\widetilde{\mathbf x}_C]，$则有：$\widetilde{\mathbf x}_c=s_c\times \mathbf{ u}_c,\quad c=1,2,\cdots,C $。这里$ s_c$ 为`excitaion` 操作的输出结果，它作为通道 $c $的权重。

    $s_c $不仅考虑了本通道的全局信息（由 $z_c$ 引入），还考虑了其它通道的全局信息（由$ \mathbf W_1,\mathbf W_2 引入）$。

#### 7.1.3 SE 块使用

1. 有两种使用`SE` 块来构建`SENet` 的方式：

    - 简单的堆叠`SE`块来构建一个新的网络。

    - 在现有网络架构中，用`SE` 块来替代原始块。

      下图中，左侧为原始`Inception` 模块，右侧为`SE-Inception` 模块。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/inception_senet.png)

      下图中，左侧为原始残差模块，右侧为`SE-ResNet` 模块。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/resnet_senet.png)

2. 如果使用`SE` 块来替代现有网络架构中的原始块，则所有额外的参数都包含在门机制的两个全连接层中。

    引入的额外参数的数量为：$\frac 2r\sum_{s=1}^S C_s^2 。$其中：$r $表示降维比例（论文中设定为 16），$S $指的是`SE` 块的数量，$C_s $表示第 $s​$ 个`SE` 块的输出通道的维度。

    如：`SE-ResNet-50` 在`ResNet-50` 所要求的大约 2500 万参数之外，额外引入了约 250 万参数，相对增加了 10%。

3. 超参数 $r $称作减少比率，它刻画了需要将通道描述符组成的向量压缩的比例。它是一个重要的超参数，需要在精度和复杂度之间平衡。

    网络的精度并不会随着 $r $的增加而单调上升，因此需要多次实验并选取其中最好的那个值。

    如下所示为`SE-ResNet-50` 采用不同的 r 在 `ImageNet` 验证集上的预测误差（`single-crop`）。`original` 表示原始的 `ResNet-50` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet5.png)

4. 虽然`SE`块可以应用在网络的任何地方，但是它在不同深度中承担了不同的作用。

    - 在网络较低的层中：对于不同类别的样本，特征通道的权重分布几乎相同。

      这说明在网络的最初阶段，特征通道的重要性很可能由不同的类别共享。即：低层特征通常更具有普遍性。

    - 在网络较高的层中：对于不同类别的样本，特征通道的权重分布开始分化。

      这说明在网络的高层，每个通道的值变得更具有类别特异性。即：高层特征通常更具有特异性。

    在网络的最后阶段的`SE` 块为网络提供重新校准所起到的作用，相对于网络的前面阶段的`SE` 块来说，更加不重要。

    这意味着可以删除最后一个阶段的`SE` 块，从而显著减少总体参数数量，仅仅带来一点点的损失。如：在`SE-ResNet-50`中，删除最后一个阶段的`SE` 块可以使得参数增加量下降到 4%，而在`ImageNet`上的`top-1` 错误率的损失小于 `0.1%` 。

    因此：`Se` 块执行特征通道重新校准的好处可以通过整个网络进行累积。

### 7.2 网络性能

1. 网络结构：其中 `fc,[16,256]` 表示 `SE` 块中的两个全连接层的输出维度。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet1.png)

2. 在`ImageNet` 验证集上的计算复杂度和预测误差比较(`single-crop`)。

    - `original` 列：从各自原始论文中给出的结果报告。

    - `re-implementation` 列：重新训练得到的结果报告。

    - `SENet` 列：通过引入`SE`块之后的结果报告。

    - `GFLOPs/MFLOPs`：计算复杂度，单位为 `G/M FLOPs` 。

    - `MobileNet` 采用的是 `1.0 MobileNet-224`，`ShuffleNet` 采用的是 `1.0 ShuffleNet 1x(g=3)` 。

    - 数据集增强和归一化：

      - 随机裁剪成 `224x224` 大小（`Inception` 系列裁剪成 `299x299` ）。
      - 随机水平翻转。
      - 输入图片沿着通道归一化：每个像素减去本通道的像素均值。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet2.png)

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet3.png)

3. 在`ImageNet` 验证集上的预测误差比较（`single-crop`）：

    其中 `SENet-154(post-challenge)` 是采用 `320x320` 大小的图片来训练的。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/senet4.png)

## 八、 DenseNet

1. `DenseNet`不是通过更深或者更宽的结构，而是通过特征重用来提升网络的学习能力。

2. `ResNet` 的思想是：创建从“靠近输入的层” 到 “靠近输出的层” 的直连。而`DenseNet` 做得更为彻底：将所有层以前馈的形式相连，这种网络因此称作`DenseNet` 。

3. `DenseNet` 具有以下的优点：

    - 缓解梯度消失的问题。因为每层都可以直接从损失函数中获取梯度、从原始输入中获取信息，从而易于训练。
    - 密集连接还具有正则化的效应，缓解了小训练集任务的过拟合。
    - 鼓励特征重用。网络将不同层学到的 `feature map` 进行组合。
    - 大幅度减少参数数量。因为每层的卷积核尺寸都比较小，输出通道数较少 (由增长率 k 决定)。

4. `DenseNet` 具有比传统卷积网络更少的参数，因为它不需要重新学习多余的`feature map` 。

    - 传统的前馈神经网络可以视作在层与层之间传递`状态`的算法，每一层接收前一层的`状态`，然后将新的`状态`传递给下一层。

      这会改变`状态`，但是也传递了需要保留的信息。

    - `ResNet` 通过恒等映射来直接传递需要保留的信息，因此层之间只需要传递`状态的变化` 。

    - `DenseNet` 会将所有层的`状态` 全部保存到`集体知识`中，同时每一层增加很少数量的`feture map` 到网络的`集体知识中`。

5. `DenseNet` 的层很窄（即：`feature map` 的通道数很小），如：每一层的输出只有 12 个通道。

### 8.1 DenseNet 块

1. 具有 L 层的传统卷积网络有 L 个连接，每层仅仅与后继层相连。

    具有 L 个残差块的 `ResNet` 在每个残差块增加了跨层连接，第 l 个残差块的输出为：$\mathbf{\vec x}_{l+1}=\mathbf{\vec x}_l+\mathcal F(\mathbf{\vec x}_l,\mathcal W_l) 。$其中$ \mathbf{\vec x}_l $是第 $l $个残差块的输入特征$；\mathcal W_l=\{\mathbf W_{l,k}\mid 1\le k \le K\} $为一组与第$ l $个残差块相关的权重（包括偏置项）$，K$ 是残差块中的层的数量$； \mathcal F $代表残差函数。

    具有 $L$ 个层块的`DenseNet` 块有$ \frac {L(L+1)}{2} ​$个连接，每层以前馈的方式将该层与它后面的所有层相连。对于第 l 层：所有先前层的`feature map` 都作为本层的输入，第 l 层具有 l 个输入`feature map` ；本层输出的`feature map` 都将作为后面 L-l 层的输入。

2. 假设`DenseNet`块包含 $L$ 层，每一层都实现了一个非线性变换$ H_l(\cdot) ，$其中$ l $表示层的索引。

    假设`DenseNet`块的输入为 $\mathbf x_0 ，$`DenseNet`块的第$ l $层的输出为 $\mathbf x_l $，则有：

    $\mathbf x_{l}=H_l([\mathbf x_0,\mathbf x_1,\cdots,\mathbf x_{l-1}])$

    其中 $[\mathbf x_0,\mathbf x_1,\cdots,\mathbf x_{l-1}] $表示 $0,\cdots,l-1 $层输出的`feature map` 沿着通道方向的拼接。

    `ResNet` 块与它不同。在`ResNet` 中，不同`feature map` 是通过直接相加来作为块的输出。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/denset_1.png)

3. 当 `feature map` 的尺寸改变时，无法沿着通道方向进行拼接。此时将网络划分为多个`DenseNet` 块，每块内部的 `feature map`尺寸相同，块之间的`feature map` 尺寸不同。

#### 8.1.1 增长率

1. `DenseNet` 块中，每层的 $H_l(\cdot) $输出的`feature map` 通道数都相同，都是 $k$ 个。$k$ 是一个重要的超参数，称作网络的增长率。

    第 l 层的输入`feature map` 的通道数为$： k_0+k\times(l-1) 。$其中 k_0 为输入层的通道数。

2. `DenseNet` 不同于现有网络的一个重要地方是：`DenseNet` 的网络很窄，即输出的 `feature map` 通道数较小，如：$ k=12。$

    一个很小的增长率就能够获得不错的效果。一种解释是：`DenseNet` 块的每层都可以访问块内的所有早前层输出的`feature map`，这些`feature map` 可以视作`DenseNet` 块的全局状态。每层输出的`feature map` 都将被添加到块的这个全局状态中，该全局状态可以理解为网络块的“集体知识”，由块内所有层共享。增长率 k 决定了新增特征占全局状态的比例。

    因此`feature map` 无需逐层复制（因为它是全局共享），这也是`DenseNet` 与传统网络结构不同的地方。这有助于整个网络的特征重用，并产生更紧凑的模型。

#### 8.1.2 非线性变换$ H_l$

1.  $H_l(\cdot) $可以是包含了 `Batch Normalization(BN)` 、`ReLU` 单元、池化或者卷积等操作的复合函数。
2.  论文中$ H_l(\cdot)$ 的结构为：先执行`BN` ，再执行`ReLU`，最后接一个`3x3` 的卷积，即：`BN-ReLU-Conv(3x3)` 。

#### 8.1.3 bottleneck

1. 尽管`DenseNet` 块中每层只产生 k 个输出`feature map`，但是它具有很多输入。当在$ H_l(\cdot)$ 之前采用 `1x1` 卷积实现降维时，可以减小计算量。

    $H_l(\cdot) $的输入是由第 $0,1,2,\cdots,l-1 ​$层的输出 `feature map` 组成，其中第 `0` 层的输出`feature map`就是整个`DensNet` 块的输入`feature map` 。

事实上第 $1,2,\cdots,l-1 $层从`DensNet` 块的输入`feature map` 中抽取各种特征。即 H_l(\cdot) 包含了`DensNet` 块的输入`feature map` 的冗余信息，这可以通过 `1x1` 卷积降维来去掉这种冗余性。

因此这种`1x1` 卷积降维对于`DenseNet` 块极其有效。

2. 如果在$ H_l(\cdot) $中引入 `1x1` 卷积降维，则该版本的`DenseNet` 称作`DenseNet-B` 。其 $H_l(\cdot) ​$结构为：先执行`BN` ，再执行`ReLU`，再接一个`1x1` 的卷积，再执行`BN` ，再执行`ReLU`，最后接一个`3x3` 的卷积。即：`BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)` 。

    其中`1x1` 卷积的输出通道数是个超参数，论文中选取为$ 4\times k 。$

### 8.2 过渡层

1. 一个`DenseNet` 网络具有多个`DenseNet`块，`DenseNet` 块之间由过渡层连接。`DenseNet` 块之间的层称为过渡层，其主要作用是连接不同的`DenseNet`块。

2. 过渡层可以包含卷积或池化操作，从而改变前一个`DenseNet` 块的输出`feature map` 的大小（包括尺寸大小、通道数量）。

    论文中的过渡层由一个`BN`层、一个`1x1` 卷积层、一个`2x2` 平均池化层组成。其中 `1x1` 卷积层用于减少`DenseNet` 块的输出通道数，提高模型的紧凑性。

    如果不减少`DenseNet` 块的输出通道数，则经过了 N 个`DenseNet` 块之后，网络的`feature map` 的通道数为：$k_0+N\times k\times (L-1) ，$其中 $k_0$ 为输入图片的通道数， $L $为每个`DenseNet` 块的层数。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/denset_2.png)

3. 如果`Dense` 块输出`feature map`的通道数为 $ m$ ，则可以使得过渡层输出`feature map` 的通道数为$  \theta m $ ，其中$  0 \lt\theta \le 1 $ 为压缩因子。

    - 当 $ \theta =1$  时，经过过渡层的`feature map` 通道数不变。

    - 当$  \theta \lt 1 $ 时，经过过渡层的`feature map` 通道数减小。此时的`DenseNet` 称做 `DenseNet-C` 。

      结合了`DenseNet-C` 和 `DenseNet-B` 的改进的网络称作 `DenseNet-BC` 。

### 8.3 网络性能

1. 网络结构：`ImageNet` 训练的`DenseNet` 网络结构，其中增长率 k=32 。

    - 表中的 `conv` 代表的是 `BN-ReLU-Conv` 的组合。如 `1x1 conv` 表示：先执行`BN`，再执行`ReLU`，最后执行`1x1` 的卷积。
    - `DenseNet-xx` 表示`DenseNet` 块有`xx` 层。如：`DenseNet-169` 表示 `DenseNet` 块有 L=169 层 。
    - 所有的 `DenseNet` 使用的是 `DenseNet-BC` 结构，输入图片尺寸为`224x224`，初始卷积尺寸为`7x7`、输出通道 `2k`、步长为`2` ，压缩因子 \theta=0.5 。
    - 在所有`DenseNet` 块的最后接一个全局平均池化层，该池化层的结果作为`softmax` 输出层的输入。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/densenet_imgnet.png)

2. 在`ImageNet` 验证集的错误率（`single-crop/10-crop`）：

    | 模型         | top-1 error(%) | top-5 error(%) |
    | ------------ | -------------- | -------------- |
    | DenseNet-121 | 25.02/23.61    | 7.71/6.66      |
    | DenseNet-169 | 23.80/22.08    | 6.85/5.92      |
    | DenseNet-201 | 22.58/21.46    | 6.34/5.54      |
    | DenseNet-264 | 22.15/20.80    | 6.12/5.29      |

    下图是`DenseNet` 和`ResNet` 在`ImageNet` 验证集的错误率的比较（`single-crop`）。左图为参数数量，右图为计算量。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/densenet_imagenet2.png)

    从实验可见：`DenseNet` 的参数数量和计算量相对`ResNet` 明显减少。

    - 具有 `20M` 个参数的`DenseNet-201` 与具有 `40M` 个参数的`ResNet-101` 验证误差接近。
    - 和`ResNet-101` 验证误差接近的`DenseNet-201` 的计算量接近于`ResNet-50`，几乎是`ResNet-101` 的一半。

3. `DenseNet` 在`CIFAR` 和 `SVHN` 验证集的表现：

    - `C10+` 和 `C100+` ：表示对`CIFAR10/CIFAR100` 执行数据集增强，包括平移和翻转。
    - 在`C10/C100/SVHN` 三列上的`DenseNet` 采用了 `Dropout` 。
    - `DenseNet` 的`Depth` 列给出的是 L 参数。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/densenet_compare.png)

    从实验可见：

    - 不考虑压缩因子和`bottleneck`，L 和 k 越大`DenseNet`表现更好。这主要是因为模型容量相应地增长。

      网络可以利用更大、更深的模型提高其表达学习能力，这也表明了`DenseNet` 不会受到优化困难的影响。

    - `DenseNet` 的参数效率更高，使用了压缩因子和`bottleneck` 的 `DenseNet-BC` 的参数利用率极高。

      这带来的一个效果是：`DenseNet-BC` 不容易发生过拟合。

      事实上在`CIFAR10` 上，`DenseNet` 从$  k=12\rightarrow k=24 ​$ 中，参数数量提升了 4 倍但是验证误差反而 `5.77` 下降到 `5.83`，明显发生了过拟合。而`DenseNet-BC` 未观察到过拟合。

4. `DenseNet` 提高准确率的一个可能的解释是：各层通过较短的连接（最多需要经过两个或者三个过渡层）直接从损失函数中接收额外的监督信息。

### 8.4 内存优化

#### 8.4.1 内存消耗

1. 虽然 `DenseNet` 的计算效率较高、参数相对较少，但是`DenseNet` 对内存不友好。考虑到`GPU` 显存大小的限制，因此无法训练较深的 `DenseNet` 。

2. 假设`DenseNet`块包含 L 层，对于第 l 层有：$ \mathbf x_{l}=H_l([\mathbf x_0,\mathbf x_1,\cdots,\mathbf x_{l-1}]) $ 。

    假设每层的输出`feature map` 尺寸均为 $ W \times H 、$ 通道数为 $ k ，$ $ H_l $ 由`BN-ReLU-Conv(3x3)` 组成，则：

    - 拼接`Concat`操作$  [\cdots] ：​$ 需要生成临时`feature map` 作为第 l 层的输入，内存消耗为$  WHk\times l ​$ 。
    - `BN` 操作：需要生成临时`feature map` 作为`ReLU` 的输入，内存消耗为$  WHk\times l 。$ 
    - `ReLU` 操作：可以执行原地修改，因此不需要额外的`feature map` 存放`ReLU` 的输出。
    - `Conv` 操作：需要生成输出`feature map` 作为第 l 层的输出，它是必须的开销。

    因此除了第$  1,2,\cdots,L $ 层的输出`feature map` 需要内存开销之外，第 $ l $ 层还需要$  2WHkl​$  的内存开销来存放中间生成的临时`feature map` 。

    整个 `DenseNet Block` 需要 $ WHk(1+L)L$  的内存开销来存放中间生成的临时`feature map` 。即`DenseNet Block` 的内存消耗为$  O(L^2)，​$ 是网络深度的平方关系。

3. 拼接`Concat`操作是必须的，因为当卷积的输入存放在连续的内存区域时，卷积操作的计算效率较高。而`DenseNet Block` 中，第 l 层的输入`feature map` 由前面各层的输出`feature map` 沿通道方向拼接而成。而这些输出`feature map` 并不在连续的内存区域。

    另外，拼接`feature map` 并不是简单的将它们拷贝在一起。由于`feature map` 在`Tensorflow/Pytorch` 等等实现中的表示为$  \mathbb R^{n\times d\times w\times h} $ （`channel first`)，或者 $ \mathbb R^{n\times w\times h\times d}$ (`channel last`），如果简单的将它们拷贝在一起则是沿着`mini batch` 维度的拼接，而不是沿着通道方向的拼接。

4. `DenseNet Block` 的这种内存消耗并不是`DenseNet Block` 的结构引起的，而是由深度学习库引起的。因为`Tensorflow/PyTorch` 等库在实现神经网络时，会存放中间生成的临时节点（如`BN` 的输出节点），这是为了在反向传播阶段可以直接获取临时节点的值。

    这是在时间代价和空间代价之间的折中：通过开辟更多的空间来存储临时值，从而在反向传播阶段节省计算。

5. 除了临时`feature map` 的内存消耗之外，网络的参数也会消耗内存。设 H_l 由`BN-ReLU-Conv(3x3)` 组成，则第 l 层的网络参数数量为：$ 9lk^2 $ （不考虑 `BN` ）。

    整个 `DenseNet Block` 的参数数量为 $ \frac{9k^2(L+1)L}{2} ，$ 即 $ O(L^2)。$ 因此网络参数的数量也是网络深度的平方关系。

    - 由于`DenseNet` 参数数量与网络的深度呈平方关系，因此`DenseNet` 网络的参数更多、网络容量更大。这也是`DenseNet` 优于其它网络的一个重要因素。
    - 通常情况下都有 $ WH\gt \frac{9k}{2} ，$ 其中 $ W,H $ 为网络`feature map` 的宽、高，$ k$  为网络的增长率。所以网络参数消耗的内存要远小于临时`feature map` 消耗的内存。

#### 8.4.2 内存优化

1. 论文`《Memory-Efficient Implementation of DenseNets》`通过分配共享内存来降低内存需求，从而使得训练更深的`DenseNet` 成为可能。

    其思想是利用时间代价和空间代价之间的折中，但是侧重于牺牲时间代价来换取空间代价。其背后支撑的因素是：`Concat`操作和`BN` 操作的计算代价很低，但是空间代价很高。因此这种做法在`DenseNet` 中非常有效。

2. 传统的`DenseNet Block` 实现与内存优化的`DenseNet Block` 对比如下（第 l 层，该层的输入`feature map` 来自于同一个块中早前的层的输出）：

    - 左图为传统的`DenseNet Block` 的第 l 层。首先将 `feature map` 拷贝到连续的内存块，拷贝时完成拼接的操作。然后依次执行`BN`、`ReLU`、`Conv` 操作。

      该层的临时`feature map` 需要消耗内存 $ 2WHkl，$ 该层的输出`feature map` 需要消耗内存 $ WHk $ 。

      - 另外某些实现（如`LuaTorch`）还需要为反向传播过程的梯度分配内存，如左图下半部分所示。如：计算 `BN` 层输出的梯度时，需要用到第 l 层输出层的梯度和`BN` 层的输出。存储这些梯度需要额外的 $ O(l k) $ 的内存。
      - 另外一些实现（如`PyTorch,MxNet`）会对梯度使用共享的内存区域来存放这些梯度，因此只需要 $ O(k) ​$ 的内存。

    - 右图为内存优化的`DenseNet Block` 的第 l 层。采用两组预分配的共享内存区`Shared memory Storage location` 来存`Concate` 操作和`BN` 操作输出的临时`feature map` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/densenet_mem_block.png)

3. 第一组预分配的共享内存区：`Concat` 操作共享区。第 $ 1,2,\cdots,L $ 层的 `Concat` 操作的输出都写入到该共享区，第$  l+1$  层的写入会覆盖第 l 层的结果。

    - 对于整个`Dense Block`，这个共享区只需要分配 $ WHkL $ （最大的`feature map` ）的内存，即内存消耗为 $ O(kL)$  (对比传统`DenseNet` 的 $ O(kL^2))。$ 

    - 后续的`BN` 操作直接从这个共享区读取数据。

    - 由于第$  l+1$  层的写入会覆盖第$  l $ 层的结果，因此这里存放的数据是临时的、易丢失的。因此在反向传播阶段还需要重新计算第$  l ​$ 层的`Concat` 操作的结果。

      因为`Concat` 操作的计算效率非常高，因此这种额外的计算代价很低。

4. 第二组预分配的共享内存区：`BN` 操作共享区。第 $ 1,2,\cdots,L$  层的 `BN` 操作的输出都写入到该共享区，第 l+1 层的写入会覆盖第 l 层的结果。

    - 对于整个`Dense Block`，这个共享区也只需要分配 $ WHkL $ （最大的`feature map` ）的内存，即内存消耗为 $ O(kL) $ (对比传统`DenseNet` 的$  O(kL^2))$ 。

    - 后续的卷积操作直接从这个共享区读取数据。

    - 与`Concat` 操作共享区同样的原因，在反向传播阶段还需要重新计算第 l 层的`BN` 操作的结果。

      `BN` 的计算效率也很高，只需要额外付出大约 5% 的计算代价。

5. 由于`BN` 操作和`Concat` 操作在神经网络中大量使用，因此这种预分配共享内存区的方法可以广泛应用。它们可以在增加少量的计算时间的情况下节省大量的内存消耗。

#### 8.4.3 优化结果

1. 如下图所示，`DenseNet` 不同实现方式的实验结果：

    - `Naive Implementation(LuaTorch)`：采用`LuaTorch` 实现的，不采用任何的内存共享区。
    - `Shared Gradient Strorage(LuaTorch)`：采用`LuaTorch` 实现的，采用梯度内存共享区。
    - `Shared Gradient Storage(PyTorch)`：采用`PyTorch` 实现的，采用梯度内存共享区。
    - `Shared Gradient+BN+Concat Strorate(LuaTorch)`：采用`LuaTorch` 实现的，采用梯度内存共享区、`Concat`内存共享区、`BN`内存共享区。
    - `Shared Gradient+BN+Concat Strorate(PyTorch)`：采用`LuaTorch` 实现的，采用梯度内存共享区、`Concat`内存共享区、`BN`内存共享区。

    注意：

    - `PyTorch` 自动实现了梯度的内存共享区。

    - 内存消耗是参数数量的线性函数。因为参数数量本质上是网络深度的二次函数，而内存消耗也是网络深度的二次函数。

      如前面的推导过程中，`DenseNet Block` 参数数量 $ P=\frac{9k^2(L+1)L}{2}，$ 内存消耗 M=WHk(1+L)L。因此 $ M=\frac{2WH}{9k} P，$ 即 $ M=O(P) 。​$ 

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/densenet_mem_compare.png)

2. 如下图所示，`DenseNet` 不同实现方式的训练时间差异（`NVIDIA Maxwell Titan-X` ）：

    - 梯度共享存储区不会带来额外时间的开销。
    - `Concat`内存共享区、`BN`内存共享区需要额外消耗 15%(`LuaTorch`) 或者 20% (`PyTorch`) 的时间。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/densenet_mem_time.png)

3. 如下图所示，不同`DenseNet`的不同实现方式在`ImageNet` 上的表现（`single-crop test`）：

    - `DenseNet cosine` 使用 $ \cos(\cdot) $ 学习率。

    - 经过内存优化的`DenseNet` 可以在单个工作站（`8 NVIDIA Tesla M40 GPU` ）上训练 264 层的网络，并取得了`top-1 error=20.26%` 的好成绩。

      > 网络参数数量：232 层`DenseNet`：`k=48,55M` 参数。 264 层`DenseNet`：`k=32,33M` 参数；`k=48,73M` 参数。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/densenet_mem_result.png)

## 九、小型网络

1. 目前神经网络领域的研究基本可以概括为两个方向：探索模型更好的预测能力，关注模型在实际应用中的难点。

    事实上卷积神经网络在图像识别领域超越了人类的表现，但是这些先进的网络需要较高的计算资源。这些资源需求超出了很多移动设备和嵌入式设备的能力（如：无人驾驶），导致实际应用中难以在这些设备上应用。

    小型网络就是为解决这个难点来设计的。

2. 小型网络的设计和优化目标并不是模型的准确率，而是在满足一定准确率的条件下，尽可能的使得模型小，从而降低对计算资源的需求，降低计算延迟。

3. 小型高效的神经网络的构建方法大致可以归为两类：对已经训练好的网络进行压缩，直接训练小网络模型。

4. 模型参数的数量决定了模型的大小，所谓的`小型网络` 指的是网络的参数数量较少。

    小型网络具有至少以下三个优势：

    - 较小的模型具有更高效的分布式训练效率。

      `Worker` 与`PS` 以及 `Worker` 与`Worker` 之间的通信是神经网络分布式训练的重要限制因素。在分布式数据并行训练中，通信开销与模型的参数数量成正比。较小的模型需要更少的通信，从而可以更快的训练。

    - 较小的模型在模型导出时开销更低。

      当在`tensorflow` 等框架中训练好模型并准备部署时，需要将模型导出。如：将训练好的模型导出到自动驾驶汽车上。模型越小，则数据导出需要传输的数据量越少，这样可以支持更频繁的模型更新。

    - 较小的模型可以在`FPGA` 和嵌入式硬件上部署。

      `FPGA` 通常只有小于`10MB` 的片上内存，并且没有外部存储。因此如果希望在`FPGA` 上部署模型，则要求模型足够小从而满足内存限制。

### 9.1 SqueezeNet 系列

#### 9.1.1 SqueezeNet

1. `squeezenet` 提出了`Fire` 模块，并通过该模型构成一种小型`CNN` 网络，在满足`AlexNet` 级别准确率的条件下大幅度降低参数数量。

2. `CNN` 结构设计三个主要策略：

    - 策略 1：部分的使用`1x1` 卷积替换`3x3` 卷积。因为`1x1` 卷积的参数数量比`3x3` 卷积的参数数量少了 `9` 倍。

    - 策略 2：减少`3x3` 卷积输入通道的数量。这会进一步降低网络的参数数量。

    - 策略 3：将网络中下采样的时机推迟到网络的后面。这会使得网络整体具有尺寸较大的`feature map` 。

      其直觉是：在其它不变的情况下，尺寸大的`feature map` 具有更高的分类准确率。

    策略`1、2` 是关于在尽可能保持模型准确率的条件下减少模型的参数数量，策略`3` 是关于在有限的参数数量下最大化准确率。

##### 9.1.1.1 Fire 模块

1. 一个`Fire` 模块由一个`squeeze` 层和一个`expand` 层组成。

    - `squeeze` 层：一个`1x1` 卷积层，输出通道数为超参数$  s_{1\times 1} $ 。
      - 通常选择超参数 $ s_{1\times 1} $ 满足：$ s_{1\times 1}\lt e_{1\times 1}+e_{3\times 3} 。$ 
      - 它用于减少`expand` 层的输入通道数，即：应用策略 2 。

    - `expand` 层：一个`1x1` 卷积层和一个`3x3` 卷积层，它们卷积的结果沿着深度进行拼接。

      - `1x1` 卷积输出通道数为超参数 $ e_{1\times 1} ，$ `3x3` 卷积输出通道数为超参数 $ e_{3\times 3} 。$ 
      - 选择`1x1` 卷积是应用了策略 1 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeeze_v1_fire.png)

##### 9.1.1.2 网络性能

1. 网络设计：

    - `SqueezeNet` 从一个独立的卷积层（`conv1`）开始，后跟 8 个`Fire` 模块（`fire2~9` ），最后连接卷积层`conv10`、全局平均池化层、`softmax` 输出层。
    - 从网络开始到末尾，每个`Fire` 模块的输出通道数逐渐增加。
    - 在`conv1、fire4、fire8` 之后执行最大池化，步长为 2。这种相对较晚的执行池化操作是采用了策略 3。
    - 在`expand` 层中的`3x3` 执行的是`same` 卷积，即：在原始输入上下左右各添加一个像素，使得输出的`feature map` 尺寸不变。
    - 在`fire9` 之后使用`Dropout`，遗忘比例为 0.5 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenet_v1_struct.png)

2. 网络参数：

    - $ s_{1\times1}$ `(#1x1 squeeze)` 列给出了超参数$  s_{1\times1} ，e_{1\times1}$ `(#1x1 expand)` 列给出了超参数 e*{1\times1} ，e*{3\times 3}`(#3x3 expand)` 列给出了超参数 e*{3\times 3} 。
    - $ s_{1\times 1}\; \text{sparsity} ，e_{1\times 1}\; \text{sparsity}，e_{3\times 3}\; \text{sparsity}，\text{# bits} ​$ 这四列给出了模型裁剪的参数。
    - `# parameter before pruning` 列给出了模型裁剪之前的参数数量。
    - `# parameter after pruning` 列给出了模型裁剪之后的参数数量。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenet_v1_params.png)

3. 模型性能：

    - 网络压缩方法列：给出了网络裁剪方法，包括`SVD(Denton et al. 2014)`、`Network pruning (Han et al. 2015b)`、`Deep compression (Han et al. 2015a)` 。
    - 数据类型列：给出了计算精度。
    - 模型压缩比例列：给出了模型相对于原始`AlexNet` 的模型大小压缩的比例。
    - `top-1 Accuracy/top-5 Accuracy` 列：给出了模型在`ImageNet` 测试集上的评估结果。

    可以看到，`SqueezeNet` 在满足同样准确率的情况下，模型大小比`AlexNet` 压缩了 50 倍。如果使用 `Deep Compression` 以及 `6 bit` 精度的情况下，模型大小比`AlexNet` 压缩了 510 倍。

    | CNN 网络结构 | 网络压缩方法     | 数据类型 | 模型大小 | 模型压缩比例 | top-1  Accuracy | top-5 Accuracy |
    | ------------ | ---------------- | -------- | -------- | ------------ | --------------- | -------------- |
    | AlexNet      | None             | 32 bit   | 240 MB   | 1x           | 57.2%           | 80.3%          |
    | AlexNet      | SVD              | 32 bit   | 48 MB    | 5x           | 56.0%           | 79.4%          |
    | AlexNet      | Network Pruning  | 32 bit   | 27 MB    | 9x           | 57.2%           | 80.3%          |
    | AlexNet      | Deep Compression | 5-8 bit  | 6.9 MB   | 35x          | 57.2%           | 80.3%          |
    | SqueezeNet   | None             | 32 bit   | 4.8 MB   | 50x          | 57.5%           | 80.3%          |
    | SqueezeNet   | Deep Compression | 8 bit    | 0.66 MB  | 363x         | 57.5%           | 80.3%          |
    | SqueezeNet   | Deep Compression | 6 bit    | 0.47 MB  | 510x         | 57.5%           | 80.3%          |

###### a. 超参数设计

1. 定义$  \text{base}_e $ 为地一个`fire` 模块的输出通道数，假设每 $ \text{freq}$  个 `fire` 模块将`fire` 模块的输出通道数增加 $ \text{incr}_e 。$ 如：$ \text{freq}=2，$ 则网络中所有`fire` 模块的输出通道数依次为：

    $ \text{base}_e,\text{base}_e,\text{base}_e+\text{incr}_e,\text{base}_e+\text{incr}_e,\text{base}_e+2*\text{incr}_e,\text{base}_e+2*\text{incr}_e,\cdots​$ 

    因此第 i 个 `fire` 模块的输出通道数为： $ e_i = \text{base}_e +\left(\text{incr}_e \times \lfloor\frac{i-1}{\text{freq}}\rfloor\right) ,i=1,2,\cdots,9 。$ 由于`fire` 模块的输出就是`fire` 模块中`expand` 层的输出，因此 e_i 也就是第 i 个`fire` 模块中`expand` 层的输出通道数。

    - 定义$  \text{pct}_{3\times 3} $ 为所有`fire` 模块的`expand` 层中，`3x3` 卷积数量的比例。这意味着：不同`fire` 模块的`expand` 层中`3x3` 卷积数量的占比都是同一个比例。则有：

      - 第 i 个`fire` 模块的`expand` 层中的`1x1` 卷积数量 $ e_{i,1\times 1}=e_i\times(1-\text{pct}_{3\times 3}) $ 。
      - 第 i 个`fire` 模块的`expand` 层中的`3x3` 卷积数量 $ e_{i,3\times 3}=e_i\times \text{pct}_{3\times 3} 。$ 

    - 定义 SR (`squeeze ratio`)为所有 `fire` 模块中，`squeeze` 层的输出通道数与`expand` 层的输出通道数的比例，称作压缩比。这意味着：不同`fire` 模块的压缩比都相同。

      则有：第 i 个 `fire` 模块的 `squeeze` 层的输出通道数为$  s_{i,1\times 1} = e_i\times SR 。$ 

2. 对于前面给到的`SqueezeNet` 有：$ \text{base}_e=128,\text{freq}=2,\text{incr}_e=128,\text{pct}_{3\times 3}=0.5,SR=0.125 。$ 

3. 评估超参数 SR：$ \text{base}_e=128,\text{freq}=2,\text{incr}_e=128,\text{pct}_{3\times 3}=0.5 ，$ SR 从 `0.15~1.0`。

    在 SR=0.75 时，`ImageNet top-5` 准确率达到峰值 `86.0%` ，此时模型大小为 `19MB` 。此后进一步增加 `SR` 只会增加模型大小而不会提高模型准确率。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenet_v1_SR.png)

4. 评估超参数$  \text{pct}_{3\times 3}：\text{base}_e=128,\text{freq}=2,\text{incr}_e=128,SR=0.5 ，\text{pct}_{3\times 3} $ 从 `1%~99%` 。

    在`3x3` 卷积占比（相对于`expand` 层的卷积数量的占比）为 `50%` 时，`ImageNet top-5` 准确率达到 `85.3%` 。此后进一步增加`3x3` 卷积的占比只会增加模型大小而几乎不会提高模型准确率。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenet_v1_percent.png)

###### b. 旁路连接

1. 采用`ResNet` 的思想，可以在`SqueezeNet` 中添加旁路连接。下图中，左图为标准的`SqueezeNet` ，中图为引入简单旁路连接的`SqueezeNet`，右图为引入复杂旁路连接的`SqueezeNet`。

    - 简单旁路连接：就是一个恒等映射。此时要求输入`feature map` 和残差`feature map` 的通道数相等。
    - 复杂旁路连接：针对输入`feature map` 和残差`feature map` 的通道数不等的情况，旁路采用一个`1x1` 卷积来调整旁路的输出通道数。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenet_v1_resnet.png)

    它们在`ImageNet` 测试集上的表现：

    | 模型结构                  | top-1 准确率 | top-5 准确率 | model size |
    | ------------------------- | ------------ | ------------ | ---------- |
    | SqueezeNet                | 57.5%        | 80.3%        | 4.8MB      |
    | SqueezeNet + 简单旁路连接 | 60.4%        | 82.5%        | 4.8MB      |
    | SqueezeNet + 复杂旁路连接 | 58.8%        | 82.0%        | 7.7MB      |

    因此添加简单旁路连接能够提升模型的准确率，还能保持模型的大小不变。

#### 9.1.2 SqueezeNext

1.  现有的神经网络在嵌入式系统上部署的主要挑战之一是内存和功耗，`SqueezeNext` 针对内存和功耗进行优化，是为功耗和内存有限的嵌入式设备设计的神经网络。

##### 9.1.2.1 SqueezeNext Block

1. `SqueezeNext` 块是在`Fire` 块的基础进行修改：

    - 将 `expand` 层的`3x3` 卷积替换为`1x3 + 3x1` 卷积，同时移除了 `expand` 层的拼接 `1x1` 卷积、添加了`1x1` 卷积来恢复通道数。
    - 通过两阶段的 `squeeze` 得到更激进的通道缩减，每个阶段的`squeeze` 都将通道数减半。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenext_block2.png)

2. `SqueezeNext` 块也采用类似`ResNet` 的旁路连接，从而可以训练更深的网络。

    下图中，左图为 `ResNet` 块，中图为 `SqueezeNet` 的 `Fire` 块，右图为`SqueezeNext` 块。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenext_block3.png)

##### 9.1.2.2 网络性能

1. 网络结构：如下为一个 23 层的`SqueezeNext` 网络（记做`SqueezeNext-23`）。

    - 相同颜色的`SqueezeNext` 块具有相同的输入`feature map` 尺寸和通道数。
    - 该网络结构描述为`[6,6,8,1]`，意思是：在第一个`conv/pooling` 层之后有四组`SqueezeNext` 块，每组`SqueezeNext` 分别有 6 个、6 个、8 个、1 个 `SqueezeNext` 块。
    - 在全连接层之前插入一个`1x1` 卷积层来降低全连接层的输入通道数，从而大幅降低全连接层的参数数量。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenext_23.png)

    23 层`SqueezeNext` 网络的另一种结构（记做`SqueezeNext-23v5`）：结构描述为`[2,4,14,1]` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenext_23v5.png)

2. `SqueezeNext`网络在`ImageNet` 上的预测准确率：

    - 参数降低倍数是相对于`AlexNet` 网络的参数而言。
    - `G-SqueezeNext-23` 是`SqueezeNext-23` 采用分组卷积的版本。

    | 模型             | top-1 准确率 | top-5 准确率 | 参数数量(百万) | 参数降低倍数 |
    | ---------------- | ------------ | ------------ | -------------- | ------------ |
    | AlexNet          | 57.10%       | 80.30%       | 60.9           | 1x           |
    | SqueezeNet       | 57.50%       | 80.30%       | 1.2            | 51x          |
    | SqueezeNext-23   | 59.05%       | 82.60%       | 0.72           | 84x          |
    | G-SqueezeNext-23 | 57.16%       | 80.23%       | 0.54           | 112x         |
    | SqueezeNext-34   | 61.39%       | 84.31%       | 1.0            | 61x          |
    | SqueezeNext-44   | 62.64%       | 85.15%       | 1.2            | 51x          |

3. 更宽和更深版本的`SqueezeNext`网络在`ImageNet` 上的预测准确率：

    - `1.5/2.0` 分别表示将网络拓宽`1.5/2` 倍。拓宽指的增加网络的`feature map` 的通道数，做法是增加第一个`conv` 的输出通道数。
    - 括号中的准确率是采用了数据集增强和超参数优化之后的最佳结果。

    | 模型                 | top-1 准确率  | top-5 准确率  | 参数数量(百万) |
    | -------------------- | ------------- | ------------- | -------------- |
    | 1.5-SqueezeNext-23   | 63.52%        | 85.66%        | 1.4            |
    | 1.5-SqueezeNext-34   | 66.00%        | 87.40%        | 2.1            |
    | 1.5-SqueezeNext-44   | 67.28%        | 88.15%        | 2.6            |
    | VGG-19               | 68.50%        | 88.50%        | 138            |
    | 2.0-SqueezeNext-23   | 67.18%        | 88.17%        | 2.4            |
    | 2.0-SqueezeNext-34   | 68.46%        | 88.78%        | 3.8            |
    | 2.0-SqueezeNext-44   | 69.59%        | 89.53%        | 4.4            |
    | MobileNet            | 67.50%(70.9%) | 86.59%(89.9%) | 4.2            |
    | 2.0-SqueezeNext-23v5 | 67.44%(69.8%) | 88.20%(89.5%) | 3.2            |

4. 硬件仿真结果：

    - 括号中的准确率是采用了数据集增强和超参数优化之后的最佳结果。

    - `Time` 表示模型的推断时间（相对耗时），`Energy` 表示模型的推断功耗。

    - `8x8,32KB` 和`16x16,128KB` 表示仿真硬件的配置：

      - `NxN` 表示硬件具有`NxN` 个`PE` 阵列。`processing element:PE` 是单个计算单元。
      - `32KB/128KB` 表示全局缓存。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenext_result.png)

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/squeezenext_result2.png)

5. 深度可分离卷积的计算密集性较差，因为其`计算/带宽` 比例较低，所以在某些移动设备上表现较差。

    > 一个可分离卷积的计算需要多次 `IO` 和计算才能完成，相比而言普通卷积只需要一次`IO` 和计算。

### 9.2 MobileNet 系列

#### 9.2.1 MobileNet

1. `MobileNet` 应用了`Depthwise` 深度可分离卷积来代替常规卷积，从而降低计算量，减少模型参数。

2. `MobileNet` 不仅产生了小型网络，还重点优化了预测延迟。

    与之相比，有一些小型网络虽然网络参数较少，但是预测延迟较大。

##### 9.2.1.1 深度可分离卷积

1. 对于传统的卷积层，单个输出`feature` 这样产生：

    - 首先由一组滤波器对输入的各通道执行滤波，生成滤波`feature` 。这一步仅仅考虑空间相关性。

    - 然后计算各通道的滤波`feature` 的加权和，得到单个`feature` 。

      这里不同位置处的通道加权和的权重不同，这意味着在空间相关性的基础上，叠加了通道相关性。

    `Depthwise` 深度可分离卷积打破了空间相关性和通道相关性的混合：

    - 首先由一组滤波器对输入的各通道执行滤波，生成滤波`feature` 。这一步仅仅考虑空间相关性。

    - 然后执行`1x1` 卷积来组合不同滤波`feature` 。

      这里不同位置处的通道加权和的权重都相同，这意味着这一步仅仅考虑通道相关性。

2. 假设：输入 `feature map` 是一个尺寸为 $ W_I\times H_I 、$ 输入通道数为 $ C_I $ 的张量 $ \mathbf X ，$ 输出`feature map` 是一个尺寸为 $ W_O\times H_O 、$ 输出通道数为$  C_O $ 的张量$  \mathbf Y $ 。假设核张量为$  \mathbf K$ ，其形状为$  C_O\times C_I\times W_K\times H_K 。$ 

    则对于标准卷积过程，有：$  Y*{j,m,n}=\sum*{i}\sum*{m^\prime}\sum*{n^\prime}K*{j,i,m^\prime,n^\prime}\times X*{i,m+m^\prime,n+n^\prime} 。$ 其中：$ i $ 为输入通道的索引；$ j $ 为输出通道的索引$ ；m, n $ 为空间索引， $ m^\prime,n^\prime $ 分别为对应空间方向上的遍历变量。

    其参数数量为：$ C_O\times C_I\times W_K\times H_K ；$ 其计算代价为$ ： C_I\times W_K\times H_K\times C_O\times W_O\times H_O。$ 

3. `depthwise` 深度可分离卷积中，假设核张量为$  \mathbf K $ ，其形状为：$  C_I\times W_K\times H_K 。$ 则对于 `depthwise` 深度可分离卷积过程：

    - 首先对输入的各通道执行滤波，有：$ \hat Y_{i,m,n}=\sum_{m^\prime}\sum_{n\prime}K_{i,m^\prime,n^\prime}\times X_{i,m+m^\prime,n+n^\prime} $ 。输出尺寸为 $ W_O\times H_O、$ 输出通道数为$  C_I 。$ 

      其参数数量为：$ C_I\times W_K\times H_K$ ；其计算代价为：$  C_I\times W_K\times H_K\times W_O\times H_O 。$ 

    - 然后对每个通道得到的滤波`feature` 执行`1x1`卷积，有：$ Y_{j,m,n}=\sum_{i}\tilde K_{i,j}\times \hat Y_{i,m,n} $ 。其中 $ \tilde {\mathbf K} ​$ 为`1x1` 卷积的核张量。

      其参数数量为：$ C_I\times C_O；$ 其计算代价为：$ C_I\times W_O\times H_O\times C_O 。$ 

    - 总的参数数量为：$ C_I\times W_K\times H_K+C_I\times C_O$ ，约为标准卷积的 $ \frac {1}{C_O}+\frac{1}{W_K\times H_K} 。$ 

      总的计算代价为：$ C_I\times W_K\times H_K\times W_O\times H_O+C_I\times W_O\times H_O\times C_O ，$ 约为标准卷积的 $ \frac{1}{C_O}+\frac{1}{W_K\times H_K} 。$ 

      通常卷积核采用`3x3` 卷积，而 $ C_O\gg 9 $ ，因此`depthwise` 卷积的参数数量和计算代价都是常规卷积的 $ \frac 18 $ 到 $ \frac 19 。$ 

4. 常规卷积和`Depthwise`可分离卷积的结构区别（带`BN` 和 `ReLU` ）：（左图为常规卷积，右图为`Depthwise` 可分离卷积）

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenet_1.png)

##### 9.2.1.2 网络结构

1. `MobileNeet` 网络结构如下表所示。其中：

    - `Conv` 表示标准卷积，`Conv dw` 表示深度可分离卷积。
    - 所有层之后都跟随`BN` 和 `ReLU` （除了最后的全连接层，该层的输出直接送入到`softmax` 层进行分类）。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenet_struct.png)

2. `MobileNet` 大量的参数和计算量都被消耗在 `1x1` 卷积上：

    - `Conv 1x1` 包含了所有的`1x1` 卷积层，包括可分离卷积中的`1x1` 卷积。
    - `Conv DW 3x3` 仅包括可分离卷积中的 `3x3` 卷积。

    | 层类型      | 乘-加运算 | 参数数量 |
    | ----------- | --------- | -------- |
    | Conv 1x1    | 94.86%    | 74.59%   |
    | Conv DW 3x3 | 3.06%     | 1.06%    |
    | Conv 3x3    | 1.19%     | 0.02%    |
    | 全连接层    | 0.18%     | 24.33%   |

3. 与训练大模型相反，训练`MobileNet` 时较少的采用正则化和数据集增强技术，因为`MobileNet` 是小模型，而小模型不容易过拟合。

    论文特别提到：在`depthwise` 滤波器上使用很少或者不使用 `L2` 正则化，因为它们的参数很少。

##### 9.2.1.3 宽度乘子 & 分辨率乘子

1. 尽管基本的`MobileNet` 架构已经很小，延迟很低，但特定应用需要更快的模型。为此`MobileNet` 引入了两个超参数：宽度乘子、分辨率乘子

2. 宽度乘子`width multiplier` ，记做 \alpha 。其作用是：在每层均匀的缩减网络（实际上是减小每层网络的输入、输出 `feature map` 的通道数量）。

    - 宽度乘子应用于第一层（是一个全卷积层）的输出通道数上。这也影响了后续所有`Depthwise`可分离卷积层的输入`feature map`通道数、输出`feature map`通道数。

      > 这可以通过直接调整第一层的输出通道数来实现。

    - 它大概以 $ \alpha^2$  的比例减少了参数数量，降低了计算量。

    - 通常将其设置为：0.25、0.5、0.75、1.0 四档。

3. 分辨率乘子`resolution multiplier`，记做 \rho 。其作用是：降低输出的`feature map` 的尺寸。

    - 分辨率乘子应用于输入图片上，改变了输入图片的尺寸。这也影响了后续所有`Depthwise`可分离卷积层的输入`feature map` 尺寸、输出`feature map` 尺寸。

      > 这可以通过直接调整网络的输入尺寸来实现。

    - 它不会改变模型的参数数量，但是大概以 $ \rho^2$  的比例降低计算量。

4. 如果模型同时实施了宽度乘子和分辨率乘子，则模型大概以$  \alpha^2 $ 的比例减少了参数数量，大概以$ \alpha^2\rho^2 $ 的比例降低了计算量。

    假设输入`feature map` 尺寸为`14x14`，通道数为 `512` ；卷积尺寸为`3x3`；输出`feature map` 尺寸为`14x14`，通道数为`512` 。

    | 层类型                                    | 乘-加操作（百万） | 参数数量（百万） |
    | ----------------------------------------- | ----------------- | ---------------- |
    | 常规卷积                                  | 462               | 2.36             |
    | 深度可分离卷积                            | 52.3              | 0.27             |
    | $\alpha=0.75$ 的深度可分离卷积            | 29.6              | 0.15             |
    | $\alpha=0.75,\rho=0.714 $的深度可分离卷积 | 15.1              | 0.15             |

##### 9.2.1.4 网络性能

1. 常规卷积和深度可分离卷积的比较：使用深度可分离卷积在`ImageNet` 上只会降低 1% 的准确率，但是计算量和参数数量大幅度降低。

    | 模型                | ImageNet Accuracy | 乘-加操作（百万） | 参数数量（百万） |
    | ------------------- | ----------------- | ----------------- | ---------------- |
    | 常规卷积的MobileNet | 71.7%             | 4866              | 29.3             |
    | MobileNet           | 70.6%             | 569               | 4.2              |

2. 更瘦的模型和更浅的模型的比较：在计算量和参数数量相差无几的情况下，采用更瘦的`MobileNet` 比采用更浅的`MobileNet` 更好。

    - 更瘦的模型：采用$  \alpha=0.75 $ 宽度乘子（`瘦` 表示模型的通道数更小）。
    - 更浅的模型：删除了`MobileNet` 中`5x Conv dw/s` 部分（即：5 层 `feature size=14x14@512` 的深度可分离卷积）。

    | 模型            | ImageNet Accuracy | 乘-加操作（百万） | 参数数量（百万） |
    | --------------- | ----------------- | ----------------- | ---------------- |
    | 更瘦的MobileNet | 68.4%             | 325               | 2.6              |
    | 更浅的MobileNet | 65.3%             | 307               | 2.9              |

3. 不同宽度乘子的比较：随着$  \alpha $ 降低，模型的准确率一直下降（$ \alpha=1$  表示基准 `MobileNet`）。

    - 输入分辨率：224x224。

    | with multiplier | ImageNet Accuracy | 乘-加 操作（百万） | 参数数量（百万） |
    | --------------- | ----------------- | ------------------ | ---------------- |
    | 1.0             | 70.6%             | 569                | 4.2              |
    | 0.75            | 68.4%             | 325                | 2.6              |
    | 0.5             | 63.7%             | 149                | 1.3              |
    | 0.25            | 50.6%             | 41                 | 0.5              |

4. 不同分辨率乘子的比较：随着 $ \rho $ 的降低，模型的准确率一直下降（$ \rho=1 $ 表示基准`MobileNet`）。

    - 宽度乘子：1.0 。
    - $ 224 \times 224$  对应$  \rho = 1，192 \times 192 对应 \rho =0.857 ，160\times 160 对应 \rho = 0.714，128 \times 128 对应 \rho = 0.571 。​$ 

    | resolution | ImageNet Accuracy | 乘-加 操作（百万） | 参数数量（百万） |
    | ---------- | ----------------- | ------------------ | ---------------- |
    | 224x224    | 70.6%             | 569                | 4.2              |
    | 192x192    | 69.1%             | 418                | 4.2              |
    | 160x160    | 67.2%             | 290                | 4.2              |
    | 128x128    | 64.4%             | 186                | 4.2              |

5. 根据$  \alpha \in \{1.0,0.75,0.5,0.25\}$  和 分辨率为 $ \{224,192,160,128\} $ 两两组合得到 16 个模型。

    - 绘制这 16 个模型的`accuracy` 和计算量的关系：近似于 `log` 关系，但是在$  \alpha=0.25$  有一个显著降低。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenetv1_accuracy.png)

    - 绘制这 16 个模型的`accuracy` 和参数数量的关系：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenetv1_accuracy2.png)

6. `MobileNet` 和其它模型的比较：

    - `MobileNet` 和 `VGG 16` 准确率相近，但是模型大小小了 32 倍，计算量小了 27 倍。
    - 瘦身的`MobileNet`（宽度乘子$  \alpha=0.5，$ 分辨率乘子$  \rho = 0.714$  ）和 `Squeezenet` 模型大小差不多，但是准确率更高，计算量小了 22 倍。

    | 模型               | ImageNet Accuracy | 乘-加 操作（百万） | 参数数量（百万） |
    | ------------------ | ----------------- | ------------------ | ---------------- |
    | 1.0 MobileNet-224  | 70.6%             | 569                | 4.2              |
    | GoogleNet          | 69.8%             | 1550               | 6.8              |
    | VGG 16             | 71.5%             | 15300              | 138              |
    | 0.50 MobileNet-160 | 60.2%             | 76                 | 1.32             |
    | Squeezenet         | 57.5%             | 1700               | 1.25             |
    | AlexNet            | 57.2%             | 720                | 60               |

#### 9.2.2 MobileNet V2

1. `MobileNet V2` 创新性的提出了具有线性`bottleneck` 的`Inverted` 残差块。

    这种块特别适用于移动设备和嵌入式设备，因为它用到的张量都较小，因此减少了推断期间的内存需求。

##### 9.2.2.1 线性 bottleneck

1. 一直以来，人们认为与任务相关的信息是嵌入到`feature map` 中的一个低维子空间。因此`feature map` 事实上有一定的信息冗余。如果缩减`feature map` 的通道数量（相当于降维），则可以降低计算量。

    `MobileNet V1` 就是采用宽度乘子，从而在计算复杂度和准确率之间平衡。

2. 由于`ReLU` 非线性激活的存在，缩减输入 `feature map` 的通道数量可能会产生不良的影响。

    输入`feature map` 中每个元素值代表某个特征，将所有图片在该`feature map` 上的取值扩成为矩阵：

    $ \mathbf P = \begin{bmatrix} P*{1,1}&P*{1,2}&\cdots&P*{1,n}\\ P*{2,1}&P*{2,2}&\cdots&P*{2,n}\\ \vdots&\vdots&\ddots&\vdots\\ P*{N,1}&P*{N,2}&\cdots&P_{N,n}\\ \end{bmatrix}$ 

    其中 N 为样本的数量， $ n=W\times H\times C $ 。即：行索引代表样本，列索引代表特征。所有特征由`feature map` 展平成一维得到。

    通常 $ N \gg n $ ，则输入矩阵 $ \mathbf P $ 的秩$  \text{rank}(\mathbf P) \lt n 。$ 

    - 对于`1x1` 卷积（不考虑`ReLU`），则`1x1` 卷积是输入特征的线性组合。输出 `featuremap` 以矩阵描述为：

      $ \mathbf P^* = \begin{bmatrix} P^*_{1,1}&P^*_{1,2}&\cdots&P^*_{1,n^*}\\ P^*_{2,1}&P^*_{2,2}&\cdots&P^*_{2,n^*}\\ \vdots&\vdots&\ddots&\vdots\\ P^*_{N,1}&P^*_{N,2}&\cdots&P^*_{N,n^*}\\ \end{bmatrix}$ 

      其中$  n^*=W\times H\times C^* ， C^* ​$ 为输出通道数。

    - 如果考虑`ReLU`，则输出 `featuremap` 为：

      $\mathbf P^{**} = ReLU(\mathbf P^*) = \begin{bmatrix} ReLU(P^*_{1,1})&ReLU(P^*_{1,2})&\cdots&ReLU(P^*_{1,n^*})\\ ReLU(P^*_{2,1})&ReLU(P^*_{2,2})&\cdots&ReLU(P^*_{2,n^*})\\ \vdots&\vdots&\ddots&\vdots\\ ReLU(P^*_{N,1})&ReLU(P^*_{N,2})&\cdots&ReLU(P^*_{N,n^*})\\ \end{bmatrix}​$

    - 对于输出的维度$ 1,2,\cdots,j,\cdots,n^*，$如果$ \mathbf P^*$在维度 $j $上的取值均小于 0 ，则由于 ReLU 的作用$，\mathbf P^{**} $在维度$ j $上取值均为 0 ，此时可能发生信息不可逆的丢失。

      - 如果 `1x1` 卷积的输出通道数很小（即 $C^* $较小），使得$ n\gt n^*=\text{rank}(\mathbf P^*)=\text{rank}(\mathbf P) ，$则$ \mathbf P^*​$的每个维度都非常重要。

        一旦经过 `ReLU` 之后$ \mathbf P^{**} $的信息有效维度降低（即$ \text{rank} (\mathbf P^{**}) \lt n^* ​$），则发生信息的丢失。且这种信息丢失是不可逆的。

        这使得输出`feature map` 的有效容量降低，从而降低了模型的表征能力。

      - 如果 `1x1` 卷积的输出通道数较大（即$ C^*$较大），使得 $n\gt n^*\gt \text{rank}(\mathbf P^*)=\text{rank}(\mathbf P) ，$则$ \mathbf P^*$ 的维度出现冗余。即使 $\mathbf P^*​$ 的某个维度被丢失，该维度的信息仍然可以从其它维度得到。

    - 如果 `1x1` 卷积的输出通道数非常小，使得$ \text{rank}(\mathbf P^*)\le \text{rank}(\mathbf P)，$则信息压缩的过程中就发生信息不可逆的丢失。

      上面的讨论的是 `ReLU` 的非线性导致信息的不可逆丢失。

3. 实验表明：`bootleneck` 中使用线性是非常重要的。

    虽然引入非线性会提升模型的表达能力，但是引入非线性会破坏太多信息，会引起准确率的下降。在`Imagenet` 测试集上的表现如下图：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenet_v2_linear.png)

##### 9.2.2.2 bottleneck block

1. `bottleneck block`：输入`feature map` 首先经过线性 `bottleneck` 来扩张通道数，然后经过深度可分离卷积，最后通过线性`bottleneck` 来缩小通道数。

    输入`bootleneck` 输出通道数与输入通道数的比例称作膨胀比。

    - 通常较小的网络使用略小的膨胀比效果更好，较大的网络使用略大的膨胀比效果更好。
    - 如果膨胀比小于 1 ，这就是一个典型的 `resnet` 残差块。

2. 可以在 `bottleneck block` 中引入旁路连接，这种`bottleneck block` 称作`Inverted` 残差块，其结构类似`ResNet` 残差块。

    - 在`ResNeXt` 残差块中，首先对输入`feature map` 执行`1x1` 卷积来压缩通道数，最后通过`1x1` 卷积来恢复通道数。

      这对应了一个输入 `feature map` 通道数先压缩、后扩张的过程。

    - 在`Inverted` 残差块中，首先对输入`feature map` 执行`1x1` 卷积来扩张通道数，最后通过`1x1` 卷积来恢复通道数。

      这对应了一个输入 `feature map` 通道数先扩张、后压缩的过程。这也是`Inverted` 残差块取名为`Inverted` 的原因。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenet_v2_inverted_block.png)

3. 当深度可分离卷积的步长为 `1` 时，`bottleneck block`包含了旁路连接。

    当深度可分离卷积的步长不为 `1` 时，`bottleneck block`不包含旁路连接。这是因为：输入`feature map` 的尺寸与块输出`feature map` 的尺寸不相同，二者无法简单拼接。

    虽然可以将旁路连接通过一个同样步长的池化层来解决，但是根据`ResNet`的研究，破坏旁路连接会引起网络性能的下降。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenet_v2_structure2.png)

4. 事实上旁路连接有两个插入的位置：在两个`1x1` 卷积的前后，或者在两个`Dwise` 卷积的前后。

    通过实验证明：在两个`1x1` 卷积的前后使用旁路连接的效果最好。在`Imagenet` 测试集上的表现如下图：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenet_v2_shortcut.png)

5. `bottleneck block` 可以看作是对信息的两阶段处理过程：

    - 阶段一：对输入`feature map` 进行降维，这一部分代表了信息的容量。
    - 阶段二：对信息进行非线性处理，这一部分代表了信息的表达。

    在`MobileNet v2` 中这二者是独立的，而传统网络中这二者是相互纠缠的。

##### 9.2.2.3 网络性能

1. `MobileNet V2` 的设计基于 `MobileNet v1` ，其结构如下：

    - 每一行代表一个或者一组相同结构的层，层的数量由 n 给定。

    - 相同结构指的是：

      - 同一行内的层的类型相同，由`Operator` 指定。其中`bottleneck` 指的是`bottleneck block` 。
      - 同一行内的层的膨胀比相同，由 `t` 指定。
      - 同一行内的层的输出通道数相同，由`c` 指定。
      - 同一行内的层：第一层采用步幅`s`，其它层采用步幅`1` 。

    - 采用`ReLU6` 激活函数，因为它在低精度浮点运算的环境下表现较好。

    - 训练过程中采用`dropout` 和`BN`。

        ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenet_v2_structure.png)

2. 与`MobileNet V1` 类似，`MobileNet V2` 也可以引入宽度乘子、分辨率乘子这两个超参数。

3. 网络推断期间最大内存需求（`通道数/内存消耗(Kb)`）：采用 16 bit 的浮点运算。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/mobilenet_v2_mem.png)

4. 网络在`ImageNet` 测试集上的表现：

    - 最后一列给出了预测单张图片的推断时间。

    | 网络                | Top 1 | Params（百万） | 乘-加 数量（百万） | CPU   |
    | ------------------- | ----- | -------------- | ------------------ | ----- |
    | MobileNet V1        | 70.6  | 4.2            | 575                | 113ms |
    | ShuffleNet (1.5)    | 71.5  | 3.4            | 292                | -     |
    | ShuffleNet (x2)     | 73.7  | 5.4            | 524                | -     |
    | NasNet-A            | 74.0  | 5.3            | 564                | 183ms |
    | MobileNet V2        | 72.0  | 3.4            | 300                | 75ms  |
    | MobileNet V2（1.4） | 74.7  | 6.9            | 585                | 143ms |

### 9.3 ShuffleNet 系列

#### 9.3.1 ShuffleNet

1. `ShuffleNet` 提出了 `1x1分组卷积+通道混洗` 的策略，在保证准确率的同时大幅降低计算成本。

    `ShuffleNet` 专为计算能力有限的设备（如：`10~150MFLOPs`）设计。在基于`ARM` 的移动设备上，`ShuffleNet` 与`AlexNet` 相比，在保持相当的准确率的同时，大约 13 倍的加速。

##### 9.3.1.1 ShuffleNet block

1. 在`Xception` 和`ResNeXt` 中，有大量的`1x1` 卷积，所以整体而言`1x1` 卷积的计算开销较大。如`ResNeXt` 的每个残差块中，`1x1` 卷积占据了`乘-加`运算的 93.4% （基数为 32 时）。

    在小型网络中，为了满足计算性能的约束（因为计算资源不够）需要控制计算量。虽然限制通道数量可以降低计算量，但这也可能会严重降低准确率。

    解决办法是：对`1x1` 卷积应用分组卷积，将每个 `1x1` 卷积仅仅在相应的通道分组上操作，这样就可以降低每个`1x1` 卷积的计算代价。

2. `1x1` 卷积仅在相应的通道分组上操作会带来一个副作用：每个通道的输出仅仅与该通道所在分组的输入（一般占总输入的比例较小）有关，与其它分组的输入（一般占总输入的比例较大）无关。这会阻止通道之间的信息流动，降低网络的表达能力。

    解决办法是：采用通道混洗，允许分组卷积从不同分组中获取输入。

    - 如下图所示：`(a)` 表示没有通道混洗的分组卷积；`(b)` 表示进行通道混洗的分组卷积；`(c)` 为`(b)` 的等效表示。
    - 由于通道混洗是可微的，因此它可以嵌入到网络中以进行端到端的训练。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shuffle_channel.png)

3. `ShuffleNet` 块的结构从`ResNeXt` 块改进而来：下图中`(a)` 是一个`ResNeXt` 块，`(b)` 是一个 `ShuffleNet` 块，`(c)` 是一个步长为`2` 的 `ShuffleNet` 块。

    在 `ShuffleNet` 块中：

    - 第一个`1x1` 卷积替换为`1x1` 分组卷积+通道随机混洗。

    - 第二个`1x1` 卷积替换为`1x1` 分组卷积，但是并没有附加通道随机混洗。这是为了简单起见，因为不附加通道随机混洗已经有了很好的结果。

    - 在`3x3 depthwise` 卷积之后只有`BN` 而没有`ReLU` 。

    - 当步长为 2 时：

      - 恒等映射直连替换为一个尺寸为 `3x3` 、步长为`2` 的平均池化。

      - `3x3 depthwise` 卷积的步长为`2` 。

      - 将残差部分与直连部分的`feature map` 拼接，而不是相加。

        因为当`feature map` 减半时，为了缓解信息丢失需要将输出通道数加倍从而保持模型的有效容量。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_block.png)

##### 9.3.1.2 网络性能

1. 在`Shufflenet` 中，`depthwise` 卷积仅仅在`1x1` 卷积的输出 `feature map` 上执行。这是因为 `depthwise` 很难在移动设备上高效的实现，因为移动设备的 `计算/内存访问` 比率较低，所以仅仅在`1x1` 卷积的输出 `feature map` 上执行从而降低开销。

2. `ShuffleNet` 网络由`ShuffleNet` 块组成。

    - 网络主要由三个`Stage` 组成。

      - 每个`Stage` 的第一个块的步长为 2 ，`stage` 内的块在其它参数上相同。
      - 每个`Stage` 的输出通道数翻倍。

    - 在 `Stage2` 的第一个`1x1` 卷积并不执行分组卷积，因此此时的输入通道数相对较小。

    - 每个`ShuffleNet` 块中的第一个`1x1` 分组卷积的输出通道数为：该块的输出通道数的 `1/4` 。

    - 使用较少的数据集增强，因为这一类小模型更多的是遇到欠拟合而不是过拟合。

    - 复杂度给出了计算量（`乘-加运算`），`KSize` 给出了卷积核的尺寸，`Stride` 给出了`ShuffleNet block` 的步长，`Repeat` 给出了 `ShuffleNet block` 重复的次数，`g` 控制了`ShuffleNet block` 分组的数量。

      > `g=1` 时，`1x1` 的通道分组卷积退化回原始的`1x1` 卷积。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_struct.png)

3. 超参数 `g` 会影响模型的准确率和计算量。在 `ImageNet` 测试集上的表现如下：

    - `ShuffleNet sx` 表示对`ShuffleNet` 的通道数增加到 `s` 倍。这通过控制 `Conv1` 卷积的输出通道数来实现。

    - `g` 越大，则计算量越小，模型越准确。

      其背后的原理是：小模型的表达能力不足，通道数量越大，则小模型的表达能力越强。

      - `g` 越大，则准确率越高。这是因为对于`ShuffleNet`，分组越大则生成的`feature map` 通道数量越多，模型表达能力越强。
      - 网络的通道数越小（如`ShuffleNet 0.25x` ），则这种增益越大。

    - 随着分组越来越大，准确率饱和甚至下降。

      这是因为随着分组数量增大，每个组内的通道数量变少。虽然总体通道数增加，但是每个分组提取有效信息的能力变弱，降低了模型整体的表征能力。

    - 虽然较大`g` 的`ShuffleNet` 通常具有更好的准确率。但是由于它的实现效率较低，会带来较大的推断时间。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_error.png)

4. 通道随机混洗的效果要比不混洗好。在 `ImageNet` 测试集上的表现如下：

    - 通道混洗使得分组卷积中，信息能够跨分组流动。
    - 分组数`g` 越大，这种混洗带来的好处越大。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_error2.png)

5. 多种模型在`ImageNet` 测试集上的表现：

    比较分为三组，每一组都有相差无几的计算量。$\Delta \text{err} $给出了在该组中，模型相对于 `MobileNet` 的预测能力的提升。

    - `MFLOPs` 表示`乘-加`运算量（百万），错误率表示`top-1 error` 。
    - `ShuffleNet 0.5x(shallow,g=3)` 是一个更浅的`ShuffleNet` 。考虑到`MobileNet` 只有 28 层，而`ShuffleNet` 有 50 层，因此去掉了`Stage 2-4` 中一半的块，得到一个教浅的、只有 26 层的 `ShuffleNet` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_compare.png)

6. 在移动设备上的推断时间（`Qualcomm Snapdragon 820 processor`，单线程）：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_time.png)

#### 9.3.2 ShuffleNet V2

1.  `ShuffleNet V2` 基于一系列对照实验提出了高效网络设计的几个通用准则，并提出了`ShuffleNet V2` 的网络结构。

##### 9.3.2.1 小型网络通用设计准则

1. 目前衡量模型推断速度的一个通用指标是`FLOPs`（即`乘-加` 计算量）。事实上这是一个间接指标，因为它不完全等同于推断速度。如：`MobileNet V2` 和 `NASNET-A` 的`FLOPs` 相差无几，但是`MobileNet V2` 的推断速度要快得多。

    如下所示为几种模型在`GPU` 和`ARM` 上的准确率（在`ImageNet` 验证集上的测试结果）、模型推断速度（通过`Batch/秒`来衡量）、计算复杂度（通过`FLOPs` 衡量）的关系。

    - 在`ARM` 平台上`batchsize=1` ， 在`GPU` 平台上`batchsize=8` 。

    - 准确率与模型容量成正比，而模型容量与模型计算复杂度成成比、计算复杂度与推断速度成反比。

      因此：模型准确率越高，则推断速度越低；模型计算复杂度越高，则推断速度越低。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_v2_speed.png)

2. `FLOPs` 和推断速度之间的差异有两个原因：

    - 除了`FLOPs` 之外，还有几个因素对推断速度有重要的影响。

      - 内存访问量（`memory access cost:MAC`）：在某些操作（如分组卷积）中，其时间开销占相当大比例。因此它可能是`GPU` 这种具有强大计算能力设备的瓶颈。
      - 模型并行度：相同`FLOPs` 的情况下，高并行度的模型比低并行度的模型快得多。

    - 即使相同的操作、相同的 `FLOPs`，在不同的硬件设备上、不同的库上，其推断速度也可能不同。

3. `MobileNet V2` 和 `ShuffleNet V1` 这两个网络非常具有代表性，它们分别采用了`group` 卷积和 `depth-wise` 卷积。这两个操作广泛应用于其它的先进网络。

    利用实验分析它们的推断速度（以推断时间开销来衡量）。其中：宽度乘子均为 1，`ShuffleNet V1` 的分组数`g=3`。

    从实验结果可知：`FLOPs` 指标仅仅考虑了卷积部分的计算量。虽然这部分消耗时间最多，但是其它操作包括数据`IO`、数据混洗、逐元素操作(`ReLU`、逐元素相加)等等时间开销也较大。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_v2_speed2.png)

4. 小型网络通用设计准则：

    - 准则一：输入通道数和输出通道数相等时，`MAC` 指标最小。

      假设输入`feature map` 尺寸为$ W\times H、$通道数为$ C_I，$输出通道数为 $C_O。$假设为`1x1` 卷积，则`FLOPs` 为：$B=C_I\times 1\times 1\times W\times H\times C_O = WHC_IC_O 。$

      其内存访问量为：输入`featuremap` 内存访问量+输出`featuremap` 内存访问量+卷积核内存访问量。因此有：$MAC=WHC_I+WHC_O+C_I\times 1\times 1\times C_O = WH(C_I+C_O)+C_IC_O 。$

      根据不等式 $C_I+C_O\ge 2\sqrt{C_IC_O}，$以及$ C_IC_O=\frac{B}{WH}，$则有：$MAC\ge 2\sqrt{WHB}+\frac{B}{WH} $。当 $C_I=C_O$ 时等式成立。

    - 准则二：大量使用分组卷积会增加`MAC` 。

      分组卷积可以降低`FLOPs` 。换言之，它可以在`FLOPs` 固定的情况下，增大`featuremap` 的通道数从而提高模型的容量。但是采用分组卷积可能增加`MAC` 。

      对于`1x1` 卷积，设分组数为 `g` ，则`FLOPs` 数为$ B=WHC_IC_O/g ，$内存访问量为

      $MAC=WH(C_I+C_O )+C_IC_O/g 。$

      当 $C_I=C_O$ 时，$MAC=2\sqrt{WHBg} +\frac{B}{WH} $最小。因此 MAC 随着 $g$ 的增加而增加。

    - 准则三：网络分支会降低并行度。

      虽然网络中采用分支（如`Inception`系列、`ResNet`系列）有利于提高模型的准确率，但是它对`GPU` 等具有高并行计算能力的设备不友好，因此会降低计算效率。

      另外它还带来了卷积核的`lauching`以及计算的同步等问题，这也是推断时间的开销。

    - 准则四：不能忽视元素级操作的影响。

      元素级操作包括`ReLU、AddTensor、AddBias` 等，它们的`FLOPs` 很小但是 `MAC` 很大。在`ResNet` 中，实验发现如果去掉`ReLU` 和旁路连接，则在`GPU` 和 `ARM` 上大约有 20% 的推断速度的提升。

##### 9.3.2.2 ShuffleNet V2 block

1. `ShuffleNet V1 block` 的分组卷积违反了准则二，`1x1` 卷积违反了准则一，旁路连接的元素级加法违反了准则四。而`ShuffleNet V2 block` 修正了这些违背的地方。

2. `ShuffleNet V2 block` 在 `ShuffleNet V1 block` 的基础上修改。`(a),(b)` 表示`ShuffleNet V1 block` （步长分别为 1、2），`(c),(d)` 表示`ShuffleNet V2 block` （步长分别为 1、2）。其中`GConv` 表示分组卷积，`DWConv` 表示`depthwise` 卷积。

    - 当步长为 1 时，`ShuffleNet V2 block` 首先将输入`feature map` 沿着通道进行拆分。设输入通道数为 $C_I ，$则拆分为 $C_I^\prime$ 和 $C_I-C_I^\prime 。$

      - 根据准则三，左路分支保持不变，右路分支执行各种卷积操作。

      - 根据准则一，右路的三个卷积操作都保持通道数不变。

      - 根据准则二，右路的两个`1x1` 卷积不再是分组卷积，而是标准的卷积操作。因为分组已经由通道拆分操作执行了。

      - 根据准则四，左右两路的`featuremap` 不再执行相加，而是执行特征拼接。

        可以将`Concat、Channel Shuffle、Channel Split` 融合到一个`element-wise` 操作中，这可以进一步降低`element-wise` 的操作数量。

    - 当步长为 2 时，`ShuffleNet V2 block` 不再拆分通道，因为通道数量需要翻倍从而保证模型的有效容量。

    - 在执行通道`Concat` 之后再进行通道混洗，这一点也与`ShuffleNet V1 block` 不同。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_v2_struct.png)

##### 9.3.2.3 网络性能

1. `ShuffleNet V2` 的网络结构类似`ShuffleNet V1`，主要有两个不同：

    - 用`ShuffleNet v2 block` 代替 `ShuffleNet v1 block` 。
    - 在`Global Pool` 之前加入一个 `1x1` 卷积。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_v2_struct2.png)

2. `ShuffleNet V2` 可以结合`SENet` 的思想，也可以增加层数从而由小网络变身为大网络。

    下表为几个模型在`ImageNet` 验证集上的表现（`single-crop`）。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_v2_result1.png)

3. `ShuffleNet V2` 和其它模型的比较：

    - 根据计算复杂度分成`40,140,300,500+` 等四组，单位：`MFLOPs` 。
    - 准确率指标为模型在`ImageNet` 验证集上的表现（`single-crop`）。
    - `GPU` 上的 `batchsize=8`，`ARM` 上的`batchsize=1` 。
    - 默认图片尺寸为`224x224`，标记为`*` 的图片尺寸为`160x160`，标记为`**` 的图片尺寸为`192x192` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/shufflenet_v2_result2.png)

4. `ShuffleNet V2` 准确率更高。有两个原因：

    - `ShuffleNet V2 block` 的构建效率更高，因此可以使用更多的通道从而提升模型的容量。

    - 在每个`ShuffleNet V2 block` ，有一半通道的数据（$C_I^\prime = C_I/2 ​$）直接进入下一层。这可以视作某种特征重用，类似于`DenseNet` 。

      但是`ShuffleNet v2 block` 在特征`Concat` 之后又执行通道混洗，这又不同于`DenseNet` 。

### 9.4 IGCV 系列

1.  设计小型化网络目前有两个代表性的方向，并且都取得了成功：

    - 通过一系列低秩卷积核（其中间输出采用非线性激活函数）去组合成一个线性卷积核或者非线性卷积核。如用`1x3 +3x1` 卷积去替代`3x3` 卷积 。
    - 使用一系列稀疏卷积核去组成一个密集卷积核。如：交错卷积中采用一系列分组卷积去替代一个密集卷积。

#### 9.4.1 IGCV

1. 简化神经网络结构的一个主要方法是消除结构里的冗余。目前大家都认为现在深度神经网络结构里有很强的冗余性。

    冗余可能来自与两个地方：

    - 卷积核空间方向的冗余。通过小卷积核替代（如，采用`3x3`、`1x3`、`3x1` 卷积核）可以降低这种冗余。
    - 卷积核通道方向的冗余。通过分组卷积、`depth-wise` 卷积可以降低这种冗余。

    `IGCV` 通过研究卷积核通道方向的冗余，从而减少网络的冗余性。

2. 事实上解决冗余性的方法有多种：

    - 二元化：将卷积核变成二值的`-1` 和`+1`，此时卷积运算的`乘加`操作变成了`加减` 操作。这样计算量就下降很多，存储需求也降低很多（模型变小）。

    - 浮点数转整数：将卷积核的 `32` 位浮点数转换成`16` 位整数。此时存储需求会下降（模型变小）。

      > 除了将卷积核进行二元化或者整数化之外，也可以将 `feature map` 进行二元化/整数化。

    - 卷积核低秩分解：将大卷积核分解为两个小卷积核 ，如：将`100x100` 分解为 `100x10` 和`10x100`、将`5x5` 分解为两个`3x3` 。

    - 稀疏卷积分解：将一个密集的卷积核分解为多个稀疏卷积核。如分组卷积、`depth-wise` 卷积。

##### 9.4.1.1 IGCV block

1. `IGCV` 提出了一种交错卷积模块，每个模块由相连的两层分组卷积组成，分别称作第一层分组卷积`primary group conv` 、第二层分组卷积`secondary group conv` 。

    - 第一层分组卷积采用`3x3` 分组卷积，主要用于处理空间相关性；第二层分组卷积采用`1x1` 分组卷积，主要用于处理通道相关性。

    - 每层分组卷积的每个组内执行的都是标准的卷积，而不是 `depth-wise` 卷积。

    - 这两层分组卷积的分组是交错进行的。假设第一层分组卷积有 `L` 个分组、每个分组 `M` 个通道，则第二层分组卷积有 `M` 个分组、每个分组`L` 个通道。

      - 第一层分组卷积中，同一个分组的不同通道会进入到第二层分组卷积的不同分组中。

        第二层分组卷积中，同一个分组的不同通道来自于第一层分组卷积的不同分组。

      - 这两层分组卷积是互补的、交错的，因此称作交错卷积`Interleaved Group Convolution:IGCV` 。

    - 这种严格意义上的互补需要对每层分组卷积的输出 `feature map` 根据通道数重新排列`Permutation`。这不等于混洗，因为混洗不能保证严格意义上的通道互补。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_block.png)

2. 由于分组卷积等价于稀疏卷积核的常规卷积，该稀疏卷积核是`block-diagonal` （即：只有对角线的块上有非零，其它位置均为 0 ）。

    因此`IGCV` 块等价于一个密集卷积，该密集卷积的卷积核由两个稀疏的、互补的卷积核相乘得到。

3. 假设第一层分组卷积的卷积核尺寸为 W\times H，则 `IGCV` 块的参数数量为：

    $T_{igc} = L\times M\times W\times H\times M +M\times L\times 1\times 1\times L=LM(WHM+L)$

    令 $G=LM $为`IGCV` 块 的输入`feature map` 通道数量，则$ T_{igc } = G^2(\frac{WH}{L}+\frac 1M) 。$

    对于常规卷积，假设卷积核的尺寸为 $W\times H，$输入输出通道数均为$ C，$则参数数量为：$T*{rc} = WHC^2 。$当参数数量相等，即$ T*{igc} = T_{rc} = T $时，则有：$C^2=G^2(\frac 1L +\frac{1}{MWH}) $。

    当 $\frac 1L +\frac{1}{MWH}\lt 1$ 时，有 $C\lt G 。$通常选择 $W=H=3 ，$考虑到$ L\gt1,M\gt1 ​$，因此该不等式几乎总是成立。于是 `IGCV` 块总是比同样参数数量的常规卷积块更宽（即：通道数更多）。

4. 在相同参数数量/计算量的条件下，`IGCV` 块（除了`L=1` 的极端情况）比常规卷积更宽，因此`IGCV` 块更高效。

    - 采用`IGCV` 块堆叠而成的`IGCV` 网络在同样参数数量/计算量的条件下，预测能力更强。

    - `Xception` 块、带有加法融合的分组卷积块都是`IGCV` 块的特殊情况。

      - 当 `M=1`，`L` 等于`IGCV` 块输入`feature map` 的输入通道数时，`IGCV` 块退化为`Xception` 块。

        ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_Xception.png)

      - 当 `L=1`，`M` 等于`IGCV` 块输入`feature map` 的输入通道数时，`IGCV` 块退化为采用`Summation` 融合的`Deeply-Fused Nets block` 。

        ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_Summation.png)

##### 9.4.1.2 网络性能

1. `IGCV`网络结构如下表所示，其中：`RegConv-Wc` 表示常规卷积， `W` 表示卷积核的尺寸为`WxW`，`c` 表示通道数量；`Summation` 表示加法融合层（即常规的`1x1` 卷积，而不是`1x1` 分组卷积）。

    - 在`IGCV` 块之后紧跟 `BN` 和 `ReLU`，即结构为：`IGC block+BN+ReLU` 。
    - `4x(3x3,8)` 表示分成 4 组，每组的卷积核尺寸为`3x3` 输出通道数为 8 。
    - 网络主要包含 3 个`stage`，`B` 表示每个`stage` 的块的数量。
    - 某些`stage` 的输出通道数翻倍（对应于`Output size` 减半），此时该`stage` 的最后一个 `block` 的步长为 2（即该`block` 中的`3x3` 卷积的步长为 2） 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_struct.png)

2. `IGCV`网络的模型复杂度如下所示。

    `SumFusion` 的参数和计算复杂度类似 `RegConv-W16`，因此没有给出。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_params.png)

3. `IGCV`模型在`CIFAR-10` 和 `CIFAR-100` 上的表现：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_result.png)

    - 如果增加恒等映射，则结果为：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_result3.png)

    - 所有的模型比较：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_result5.png)

4. `IGCV`模型在`ImageNet` 上的表现：（输入`224x224`，验证集，`single-crop` ）

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_result4.png)

5. 超参数`L` 和 `M` 的选取：

    通过`IGCV`在`CIFAR100` 上的表现可以发现：`L` 占据统治地位，随着 L 的增加模型准确率先上升到最大值然后缓缓下降。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV_LM.png)

#### 9.4.2 IGCV2

1.  `IGCV2 block` 的主要设计思想是：将 `IGCV block` 两个结构化的稀疏卷积核组成的交错卷积推广到更多个结构化的稀疏卷积核组成的交错卷积，从而进一步消除冗余性。

##### 9.4.2.1 IGCV2 block

1. 假设输入`feature map` 的通道数为 $C_I，$卷积核尺寸为$ W\times H，$输出通道数为 $C_O。$考虑输入 `feature map` 的一个 `patch`（对应卷积核大小）$\mathbf{\vec x}$，该 `patch` 经过卷积之后的结果为：$\mathbf{\vec y} = \mathbf W \mathbf{\vec x} 。$其中 $\mathbf{\vec x} $是一个长度为$ WH C_I $的向量，$\mathbf {\vec y}$ 是一个长度为$ C_O $的向量， $\mathbf W$ 是一个 $C_O$ 行、$WHC_I $列的矩阵。

    - `Xception、deep roots、IGCV1` 的思想是：将卷积核 $\mathbf W$ 分解为多个稀疏卷积的乘积：

      $\mathbf{\vec y} =\mathbf P^2\mathbf W^2\mathbf P^1\mathbf W^1 \mathbf{\vec x},\quad \mathbf W=\mathbf P^2\mathbf W^2\mathbf P^1\mathbf W^1$

      其中 $\mathbf W^1,\mathbf W^2 $至少有一个为块稀疏`block-wise sparse` 矩阵；$\mathbf P^1,\mathbf P^2$ 为置换矩阵，用于重新调整通道顺序； $\mathbf W=\mathbf P^2\mathbf W^2\mathbf P^1\mathbf W^1 $为一个密集矩阵。

    - 对于`IGCV1` ， 令 $G_i,i=1,2$ 为第$ i$ 层分组卷积的分组数，则：

      $\mathbf W^i=\begin{bmatrix} \mathbf W_1^i&\mathbf 0&\cdots&\mathbf 0\\ \mathbf 0&\mathbf W_2^i&\cdots&\mathbf 0\\ \vdots&\vdots&\ddots&\vdots\\ \mathbf 0&\mathbf 0&\cdots&\mathbf W_{G_i}^i\\ \end{bmatrix}$

      其中 $\mathbf W^i $表示第 i 层分组卷积的卷积核，它是一个`block-wise sparse` 的矩阵，其中第 j 组的卷积核矩阵为 $\mathbf W^i_j,1\le j\le G_i 。$

2. `IGCV2` 不是将$ \mathbf W$ 分解为 2 个 `block-wise sparse` 矩阵的乘积（不考虑置换矩阵），而是进一步分解为 L 个块稀疏矩阵的乘积：

    $\mathbf{\vec y} =\mathbf P_L\mathbf W_L\mathbf P*{L-1}\mathbf W*{L-1}\cdots\mathbf P_1\mathbf W_1 \mathbf{\vec x}=\left(\prod*{l=L}^1\mathbf P_l\mathbf W_l\right)\mathbf{\vec x}\\ \mathbf W=\mathbf P_L\mathbf W_L\mathbf P*{L-1}\mathbf W*{L-1}\cdots\mathbf P_1\mathbf W_1=\prod*{l=L}^1\mathbf P_l\mathbf W_l$

    其中 $\mathbf W_l $为块稀疏矩阵，它对应于第$ l $层分组卷积，即：`IGCV2` 包含了$ L $层分组卷积；$\mathbf P_l $为置换矩阵。

    - 为了设计简单，第$ l $层分组卷积中，每个分组的通道数都相等，设为$ K_l 。$
    - 如下所示为一个 `IGCV2 block` 。实线表示权重$ \mathbf W_l $，虚线表示置换 $\mathbf P_l ，​$加粗的线表示产生某个输出通道（灰色的那个通道）的路径。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV2_block.png)

##### 9.4.2.2 互补条件 & 平衡条件

1. 事实上并不是所有的 L 个块稀疏矩阵的乘积一定就是密集矩阵，这其中要满足一定的条件，即互补条件`complementary condition` 。互补条件用于保证：每个输出通道，有且仅有一条路径，该路径将所有输入通道与该输出通道相连。

2. 要满足互补条件，则有以下准则：

    对于任意 $m ，\mathbf W_L\left(\prod_{l=L-1}^m\mathbf P_l\mathbf W_l\right)$ 等价于一层分组卷积，且 $\mathbf W*{m-1}\left(\prod*{l=m-2}^1\mathbf P_l\mathbf W_l\right) ​$也等价于一层分组卷积。且这两层分组卷积是互补的：第一层分组卷积中，同一个分组的不同通道会进入到第二层分组卷积的不同分组中；第二层分组卷积中，同一个分组的不同通道来自于第一层分组卷积的不同分组。

    - 其物理意义为：从任意位置 m 截断，则前面一半等价于一层分组卷积，后面一半也等价于一层分组卷积。
    - 可以证明 `IGCV2 block` 满足这种互补条件。

3. 假设 `IGCV2 block` 的输入通道数为$ C $，输出通道数也为 $C ，$其 `L` 层分组卷积中第 l 层分组卷积每组卷积的输入通道数为$ K_l $。现在考虑 $\{K_1,\cdots,K_L\}$ 需要满足的约束条件。

    对于`IGCV2 block` 的某个输入通道，考虑它能影响多少个`IGCV2 block`的输出通道。

    - 设该输入通道影响了第 $l $层分组卷积的 $C_l $个输出通道。因为互补条件的限制，这$ C_l $个通道在第$ l+1 $层进入不同的分组，因此分别归属于$ C_l $组。而每个分组影响了$ K*{l+1} $个输出通道，因此有递归公式：

      $C_{l+1} = K_l\times C_l$

    - 考虑初始条件，在第`1` 层分组卷积有：$C_1= K_1\times 1 $。则最终该输入通道能够影响`IGCV2 block`的输出通道的数量为：$C_L=\prod_{l=1}^L K_l $。

    - 由于互补条件的限制：每个输出通道，有且仅有一条路径，该路径将所有输入通道与该输出通道相连。因此每个输入通道必须能够影响所有的输出通道。因此有：$C= C_L =\prod_{l=1}^L K_l 。$

4. 考虑仅仅使用尺寸为 `1x1` 和`WxH` 的卷积核。由于`1x1` 卷积核的参数数量更少，因此 `L` 层分组卷积中使用 1 层尺寸为 `WxH` 的卷积核、`L-1` 层尺寸为`1x1` 卷积核。

    - 假设第 1 层为`WxH` 卷积核，后续层都是`1x1` 卷积核，则总的参数数量为：$Q=C\sum_{l=2}^L K_l+CWHK_1 。$

      - 第一项来自于`1x1` 分组卷积。对于第$ l>=2$ 层的分组卷积，输入通道数为$ C$、分组数为$ C/K_l、$输出通道数为$ C $。则参数数量为：$K_l\times 1\times 1\times K_l\times \frac{C}{K_l}=C\times K_l $。
      - 第二项来自于`WxH` 分组卷积。参数数量为：$K_1\times W\times H\times K_1\times \frac{C}{K_1}=CWHK_1 。$

    - 根据互补条件 $C=\prod_{l=1}^L K_l，$以及`Jensen` 不等式有：

      $Q\ge CL(WHK_1\prod_{l=2}^LK_l)^{1/L}=CL(WHC)^{1/L}$

      等式成立时$ WHK_1=K_2=\cdots=K_L=(WHC)^{1/L} $。这称作平衡条件`balance condition` 。

5. 考虑选择使得 Q 最小的 L。根据

    $\frac{d \log Q}{ dL} = \frac 1L-\frac 1{L^2}\log(WHC) = 0$

    则有：$L_{min}= \log(WHC)$ 。

6. 当`Block` 的输出`feature map` 尺寸相对于输入`feature map` 尺寸减半（同时也意味着通道数翻倍）时，`Block` 的输出通道数不等于输入通道数。

    设输入通道数为 $C_I，$输出通道数为 $C_O，$则有：

    - 互补条件：$C_O=\prod_{l=1}^LK_l 。$

    - 平衡条件：

      设`feature map` 的尺寸缩减发生在`3x3` 卷积，则后续所有的`1x1` 卷积的输入通道数为$ C_O、$输出通道数也为 $C_O。$

      - 对于第 $l>=2 $层的分组卷积，参数数量为：$C_O\times K_l 。$
      - 对于第 $1$ 层分组卷积，参数数量为：$K_1\times W\times H\times \frac{C_O}{g}\times g ，$其中$ g=\frac{C_I}{K_l}$ 为分组数量。

      则总的参数数量为：$Q=C_O\sum_{l=2}^L K_l+C_OWHK_1\ge C_OL(WHC_O)^{1/L} $。等式成立时有：

      $WHK_1=K_2=\cdots=(WHC_O)^{1/L}$

    - 选择使得 $Q$ 最小的 $L$ 有：$L_{min} = \log(WHC_O) 。$

    因此对于`feature map` 尺寸发生缩减的`IGCV2 Block`，互补条件&平衡条件仍然成立，只需要将 C 替换为输出通道数$ C_O 。$

##### 9.4.2.3 网络性能

1. `IGCV2` 网络结构如下表所示：其中 x 为网络第一层的输出通道数量，也就是后面 `block` 的输入通道数 C。

    - 网络主要包含 3 个`stage`，`B` 表示每个`stage` 的块的数量。

    - 某些`stage` 的输出通道数翻倍，此时该`stage` 的最后一个 `block` 的步长为 2（即该`block` 中的`3x3` 卷积的步长为 2） 。

    - 每 2 个`block` 添加一个旁路连接，类似`ResNet`。 即在$ [x\times(3\times3,1),(1\times 1,x)] $这种结构添加旁路连接。

    - $x\times(3\times3,1) $表示输入通道数为 x 的 `3x3` 卷积。$(1\times1,x) $表示输出通道数为 x 的 `1x1` 卷积。

    - L 和 K 为`IGCV2` 的超参数。$ [L-1,x,(1\times 1,K)] $表示有 $L-1 $层分组卷积，每个组都是 K 个输入通道的`1x1` 卷积。

      > 事实上 `IGCV2` 每个 `stage` 都是 `L` 层，只是第一层的分组数为 1 。

    - 对于 `IGCV2(Cx)`，L=3 ；对于 `IGCV2*(Cx)`，$L^*=\lceil \log_K(x)\rceil+1 ​$。

      > `IGCV2*` 的 `(3x3,64)` 应该修改为 $(3\times 3,x) 。​$

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV2_struct.png)

2. 通过一个简化版的、没有`feature map` 尺寸缩减（这意味着`IGCV2 Block` 的输入、输出通道数保持不变 ）的`IGCV2` 网络来研究一些超参数的限制：

    - `IGCV2 block` 最接近互补条件限制条件 时，表现最好。

      如下所示，红色表示互补条件限制满足条件最好的情况。其中匹配程度通过 \frac{C}{K^L} 来衡量。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV2_complement.png)

    - `IGCV2 block`的 L 最佳的情况是：网络的宽度（通过 C 刻画）和稀疏性达到平衡，如下所示。

      - 下图中，每一种 L 都选择了对应的`Width` （即 C ）使得其满足互补条件限制，以及平衡条件，从而使得参数保持相同。

      - `Non-sparsity` 指的是`IGCV2 block` 等效的密集矩阵中非零元素的比例。

          ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV2_L.png)

    - `IGCV2` 网络越深越好、越宽越好，但是加宽网络比加深网络性价比更高。如下图所示：采用 `IGCV2*` ，深度用 `D` 表示、宽度用 `C` 表示。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV2_deep_wide.png)

3. `IGCV2` 网络与不同网络比较：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV2_result.png)

4. `IGCV2` 网络与`MobileNet` 的比较：（`ImageNet`）

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV2_result2.png)

#### 9.4.3 IGCV3

1.  `IGCV3` 结合了块稀疏的卷积核（如：交错卷积） 和低秩的卷积核（如：`bottleneck` 模块），并放松了互补条件。

##### 9.4.3.1 IGCV3 block

1. `IGCV3` 的`block` 修改自`IGCV2 block` ，它将`IGCV2 block` 的分组卷积替换为低秩的分组卷积。

    - 首先对输入`feature map` 进行分组，然后在每个分组中执行 `1x1` 卷积来扩张通道数。
    - 然后执行常规分组卷积，并对结果进行通道重排列。
    - 最后执行分组卷积，在每个分组中执行`1x1` 卷积来恢复通道数，并对结果进行通道重排。

    按照`IGCV2` 的描述方式，输入 `feature map` 的一个 `patch`$\mathbf{\vec x}，$经过`IGCV3 block` 之后的结果为：

    $\mathbf{\vec y} =\mathbf P^2\mathbf W^2\mathbf P^1\mathbf W^1 \mathbf W^0\mathbf{\vec x}$

    其中 $\mathbf W^0,\mathbf W^2 $为低秩的块稀疏矩阵；$\mathbf P^1,\mathbf P^2 $为置换矩阵，用于重新调整通道顺序 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV3_block.png)

2. 下图分别展示了`IGCV1 block`、`Inverted` 残差块、`IGCV3 block` 的区别。每个圆圈代表一个通道，箭头代表通道的数据流动。

    `IGCV1` 块中，两层分组卷积的卷积核都是稀疏的；在`Inverted` 残差块中，两层`1x1` 卷积的卷积核都是低秩的；在`IGCV3` 块中，两层`1x1` 卷积的卷积核是稀疏的，也都是低秩的。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV3_arch2.png)

3. 设`IGCV3 block` 第一层分组卷积(`1x1` 分组扩张卷积) 的分组数量为 $G_1；$第二层分组卷积（普通分组卷积）的分组数不变，也是 $G_1 ；$第三层分组卷积（`1x1` 分组压缩卷积）的分组数量为$ G_2 。$则$ G_1,G_2$ 是非常重要的超参数。

##### 9.4.3.2 loose 互补条件

1. 实验中发现：如果`IGCV3 block`如果严格遵守互补条件会导致卷积核过于稀疏，从而导致更大的内存消耗而没有带来准确率的提升。

    因此`IGCV3 block`提出了宽松互补条件`loose complementary condition`，从而允许输入通道到输出通道存在多条路径。

2. 在`IGCV3 block` 中定义了超级通道 `Super-Channel`，设为$ C_s 。​$在一个`IGCV3 block` 中，每个`feature map` 都具有同等数量的`Super-Channel` 。

    - 输入`feature map` 的通道数假设为 C ，因此每个输入超级通道包含 $\frac{C}{C_s} $条普通的输入通道。
    - 经过`1x1` 分组卷积扩张的输出通道数假设为$ C_{mid}$，则这里的每个超级通道包含 $\frac{C_{mid}}{C_s} ​$条普通的通道。
    - 经过常规分组卷积的输出通道数也是 $C_{mid}$，则这里的每个超级通道也包含$ \frac{C_{mid}}{C_s} $条普通的通道。
    - 最后经过`1x1` 分组卷积压缩的输出通道数也是$ C，$因此每个输出超级通道包含 $\frac{C}{C_s} $条普通的输出通道。

3. `loose complementary condition`：第一层分组卷积中，同一个分组的不同超级通道会进入到第三层分组卷积的不同分组中；第三层分组卷积中，同一个分组的不同超级通道来自于第一层分组卷积的不同分组。

    > 由于超级通道中可能包含多个普通通道，因此输入通道到输出通道存在多条路径。

4. 通常设置 $C_s=C，$即：超级通道数量等于输入通道数量。则每条超级通道包含的普通通道数量依次为：\${1,expand,expand,1\}，$其中 $expand$ 表示`1x1` 分组卷积的通道扩张比例。

    也可以将 C_s 设置为一个超参数，其取值越小，则互补条件放松的越厉害（即：输入通道到输出通道存在路径的数量越多）。

##### 9.4.3.3 网络性能

1. `IGCV3` 网络通过叠加`IGCV3 block` 而成。因为论文中的对照实验是在网络参数数量差不多相同的条件下进行。为了保证`IGCV3` 网络与`IGCV1/2,MobileNet,ShuffleNet` 等网络的参数数量相同，需要加深或者加宽`IGCV3`网络。

    论文中作者展示了两个版本：

    - 网络更宽的版本，深度与对比的网络相等的同时，加宽网络的宽度，记做`IGCV3-W`。参数为：$C_s=4,G_1=4,G_2=4 。$
    - 网络更深的版本，宽度与对比的网络相等的同时，加深网络的深度，记做`IGCV3-D`。参数为：$C_2=2,G_1=2,G_2=2 。$

2. `IGCV3` 与`IGCV1/2` 的对比：

    `IGCV3` 降低了参数冗余度，因此在相同的参数的条件下，其网络可以更深或者更宽，因此拥有比`IGCV1/2` 更好的准确率。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV3_result1.png)

3. `IGCV3` 和其它网络的在`ImageNet` 上的对比：

    - `MAdds` 用于刻画网络的计算量，单位是`百万乘-加运算` 。
    - $\text{xxNet}(\alpha)$ 中的 $\alpha$ 表示宽度乘子，它大概以 $\alpha^2 $的比例减少了参数数量，降低了计算量。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV3_result2.png)

4. 更深的`IGCV` 网络和更宽的`IGCV` 网络的比较：

    - 更深的网络带来的效果更好，这与人们一直以来追求网络的`深度` 的理念相一致。
    - 在宽度方向有冗余性，因此进一步增加网络宽度并不会带来额外的收益。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV3_result4.png)

5. `ReLU`的位置比较：

    第一个`IGCV3 block` 的`ReLU` 位置在两个`1x1` 卷积之后；第二个`IGCV3 block` 的`ReLU` 位置在`3x3` 卷积之后；第三个`IGCV3 block` 的 `ReLU` 在整个`block` 之后（这等价于一个常规卷积+`ReLU` ）。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV3_result3.png)

6. 超参数$ G_1,G_2$ 的比较：

    - 只要 $G_1,G_2$ 都是输入、输出通道数的公约数时，`loose` 互补条件就能满足，因此存在$ G_1,G_2 $的很多选择。

    - 实验发现，$G_1 $比较小、$G_2$ 比较大时网络预测能力较好。但是这里的`IGCV3` 网络采用的是同样的网络深度、宽度。

      如果采用同样的网络参数数量，则$ G_1=2,G_2=2$ 可以产生一个更深的、准确率更高的网络。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/IGCV3_result5.png)

### 9.5 CondenseNet

1. `CondenseNet` 基于`DenseNet`网络，它在训练的过程中自动学习一个稀疏的网络从而减少了`Densenet` 的冗余性。

    这种稀疏性类似于分组卷积，但是`CondenseNet` 从数据中自动学到分组，而不是由人工硬性规定分组。

2. `CondenseNet` 网络在`GPU` 上具有高效的训练速度，在移动设备上具有高效的推断速度。

3. `MobileNet、ShuffleNet、NASNet` 都采用了深度可分离卷积，这种卷积在大多数深度学习库中没有实现。而`CondenseNet` 采用分组卷积，这种卷积被大多数深度学习库很好的支持。

    > 准确的说，`CondenseNet` 在测试阶段采用分组卷积，在训练阶段采用的是常规卷积。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_group.png)

#### 9.5.1 网络剪枝

1. 有一些工作（`Compressing neural networks with the hashing trick`、`Deep networks with stochastic depth`）表明：`CNN`网络中存在大量的冗余。

    `DenseNet` 通过直接连接每个层的输出特征与之后的所有层来特征重用。但是如果该层的输出特征是冗余的或者不需要的，则这些连接会带来冗余。

2. `CNN` 网络可以通过权重剪枝从而在不牺牲模型准确率的条件下实现小型化。有不同粒度的剪枝技术：

    - 独立的权重剪枝：一种细粒度的剪枝，它可以带来高度稀疏的网络。但是它需要存储大量的索引来记录哪些连接被剪掉，并且依赖特定的硬件/软件设施。
    - `filter`级别剪枝：一种粗粒度的剪枝，它得到更规则的网络，且易于实现。但是它带来的网络稀疏性较低。

3. `CondenseNet` 也可以认为是一种网络剪枝技术，它与以上的剪枝方法都不同：

    - `CondenseNet` 的网络剪枝发生、且仅仅发生在训练的早期。这比在网络整个训练过程中都采用`L1` 正则化来获取稀疏权重更高效。
    - `CondenseNet` 的网络剪枝能产生比`filter`级别剪枝更高的网络稀疏性，而且网络结构也是规则的、易于实现的。

#### 9.5.2 LGC

1. 分组卷积在很多`CNN` 网络中大量使用。在`DenseNet` 中，可以使用`3x3` 分组卷积来代替常规的`3x3` 卷积从而减小模型大小，同时保持模型的准确率下降不大。

    但是实验表明：在`DenseNet` 中，`1x1` 分组卷积代替`1x1` 常规卷积时，会带来模型准确率的剧烈下降。

    对于一个具有 L 层的 `DenseNet Block`，`1x1` 卷积是第$ l=1,2,\cdots,L $层的第一个操作。该卷积的输入是由当前 `DenseNet Block` 内第 $1,2,\cdots,l-1 $层输出`feature map` 组成。因此采用`1x1` 分组卷积带来准确率剧烈下降的原因可能有两个：

    - 第 $1,2,\cdots,l-1 ​$层输出`feature` 具有内在的顺序，难以决策哪些`feature` 应该位于同一组、哪些`feature` 应该位于不同组。
    - 这些输出`feature` 多样性较强，缺少任何一些`feature` 都会减小模型的表达能力。

    因此将这些输出`feature` 强制性的、固定的分配到不相交的分组里会影响特征重用。

2. 一种解决方案是：在`1x1` 卷积之前，先对`1x1` 卷积操作的输入特征进行随机排列。

    这个策略会缓解模型准确率的下降，但是并不推荐。因为可以采用模型更小的`DenseNet` 网络达到同样的准确率，而且二者计算代价相同。与之相比，`特征混洗+1x1分组卷积` 的方案并没有任何优势。

3. 另一种解决方案是：通过训练来自动学习出`feature` 的分组。

    考虑到第 $1,2,\cdots,l-1​$ 层输出`feature`中，很难预测哪些`feature` 对于第 l 层有用，也很难预先对这些`feature` 执行合适的分组。因此通过训练数据来学习这种分组是合理的方案。这就是学习的分组卷积`Learned Group Convolution:LGC` 。

4. 在`LGC` 中：

    - 卷积操作被分为多组`filter`，每个组都会自主选择最相关的一批`feature` 作为输入。

      由于这组`filter` 都采用同样的一批`feature` 作为输入，因此就构成了分组卷积。

    - 允许某个`feature` 被多个分组共享（如下图中的`Input Feature 5、12`），也允许某个`feature` 被所有的分组忽略（如下图中的`Input Feature 2、4`）。

    - 即使某个`feature` 被第 l 层`1x1` 卷积操作的所有分组忽略，它也可能被第 l+1 层`1x1` 卷积操作的分组用到。

#### 9.5.3 训练和测试

1. `CondenseNet` 的训练、测试与常规网络不同。

    - `CondenseNet` 的训练是一个多阶段的过程，分为浓缩阶段和优化阶段。

      - 浓缩阶段`condensing stage`：训练的前半部分为浓缩阶段，可以有多个`condensing stage` （如下图所示有 2 个浓缩阶段）。

        每个`condensing stage` 重复训练网络一个固定的`iteration`，并在训练过程中：引入可以产生稀疏性的正则化、裁剪掉权重幅度较小的`filter` 。

      - 优化阶段`optimization stage`：训练的后半部分为优化阶段，只有一个`optimization stage`。

        在这个阶段，网络的`1x1` 卷积的分组已经固定，此阶段在给定分组的条件下继续学习`1x1` 分组卷积。

    - `CondenseNet` 的测试需要重新排列裁剪后的`filter`，使其重新组织成分组卷积的形式。因为分组卷积可以更高效的实现，并且节省大量计算。

    - 通常所有的浓缩阶段和所有的优化阶段采取`1:1` 的训练`epoch` 分配。假设需要训练 M 个`epoch`，有 N 个浓缩阶段，则：

      - 所有的浓缩阶段消耗 M/2 个`epoch`，每个浓缩阶段消耗 \frac {M}{2N} 个`epoch` 。
      - 所有的优化阶段也消耗 M/2 个`epoch` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_stages.png)

2. 假设卷积核的尺寸为$ W\times H，$输入通道数为 $C_I，$输出通道数为 $C_O 。$当采用`1x1` 卷积核时`4D` 的核张量退化为一个矩阵，假设该矩阵表示为$ \mathbf F\in \mathbb R^{C_O\times C_I} $。

    将该矩阵按行（即输出通道数）划分为同样大小的$ G $个组，对应的权重为：$\mathbf F^1,\cdots,\mathbf F^g,\cdots,\mathbf F^G 。$其中 $\mathbf F^g\in\mathbb R^{C_O/G\times C_I} ， F^g_{i,j} $对应第$ g $个分组的第$ i $个输出特征（对应于整体的第$ g\times G+i $个输出特征）和整体的第 j 个输入特征的权重。

    - `CondenseNet` 在训练阶段对每个分组筛选出对其不太重要的特征子集。

      对于第$ g$ 个分组，第$ j $个输入特征的重要性由该特征在 g 分组所有输出上的权重的绝对值均值来刻画：$\text{impt}^g(j)=\sum_{i=1}^{C_O/G}|F^g_{i,j}| 。$如果 $\text{impt}^g(j) $相对较小，则删除输入特征 $j$ ，这就产生了稀疏性。

    - 为了缓解权重裁剪带来的准确率损失，这里引入 `L1` 正则化。因为`L1` 正则化可以得到更稀疏的解，从而使得删除$ \text{impt}^g(j) $相对较小的连接带来的准确率损失较小（因为这些权重大多数为 0 或者很小的值）。

      `CondenseNet` 中引入分组`L1` 正则化`group-lasso`：$\sum*{g=1}^G\sum*{j=1}^{C_I}\sqrt{\sum_{i=1}^{C_O/G}(F^g_{i,j})^2} 。$这倾向于将$ \mathbf F^g $的某一列 $(F^g*{1,j},F^g*{2,j},\cdots,F^g_{C_o/G,j})$ 整体拉向 0 ，使得产生`filter` 级别的稀疏性（而不是权重级别的稀疏性）。

3. 考虑到`LGC` 中，某个`feature` 可以被多个分组分享，也可以不被任何分组使用，因此一个`feature` 被某个分组选取的概率不再是 $\frac{1}{G}。$ 定义浓缩因子 C，它表示每个分组包含$ \lfloor\frac{C_I}{C}\rfloor $的输入。

    - 如果不经过裁剪，则每个分组包含 $C_I $个输入；经过裁剪之后，最终保留$ \frac 1 C $比例的输入。

    - `filter` 裁剪融合于训练过程中。给定浓缩因子 C，则`CondenseNet` 包含$ C-1 ​$个`condensing stage` (如上图中，C=3 )。

      在每个`condensing stage` 结束时，每个分组裁剪掉整体输入$ \frac 1C$ 比例的输入。经过$ C-1 $个`condensing stage`之后，网络的每个分组仅仅保留 $\lfloor\frac{C_I}{C}\rfloor $的输入。

4. 在每个`condensing stage` 结束前后，`training loss` 都会突然上升然后下降。这是因为权重裁剪带来的突变。

    最后一个浓缩阶段突变的最厉害，因为此时每个分组会损失 `50%` 的权重。但是随后的优化阶段会逐渐降低`training loss` 。

    下图中，学习率采用`cosine` 学习率。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_loss.png)

5. 训练完成之后，`CondenseNet`引入`index layer` 来重新排列权重裁剪之后剩下的连接，使其成为一个`1x1`分组卷积。

    - 训练时，`1x1` 卷积是一个`LGC` 。
    - 测试时，`1x1` 卷积是一个`index layer` 加一个`1x1`分组卷积。

    下图中：左图为标准的 `DenseNet Block`，中间为训练期间的`CondenseNet Block`，右图为测试期间的`CondenseNet Block` 。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_train_test.png)

6. 在训练早期就裁剪了一些权重，而裁剪的原则是根据权重的大小。因此存在一个问题：裁剪的权重是否仅仅因为它们是用较小的值初始化？

    论文通过实验证明网络的权重裁剪与权重初始化无关。

#### 9.5.4 IGR 和 FDC

1. `CondenseNet` 对 `DenseNet` 做了两个修改：

    - 递增的学习率`increasing growth rate:IGR` ：

      原始的`DenseNet` 对所有的`DenseNet Block` 使用相同的增长率。考虑到`DenseNet` 更深的层更多的依赖`high-level` 特征而不是`low-level` 特征，因此可以考虑使用递增的增长率，如指数增长的增长率：$k=2^{m-1}k_0 $，其中 m 为 `Block` 编号。

      该策略会增加模型的参数，降低网络参数的效率，但是会大大提高计算效率。

    - 全面的连接`fully dense connectivity:FDC` ：

      原始的`DenseNet` 中只有`DenseNet Block` 内部的层之间才存在连接。在`CondenseNet` 中，将每层的输出连接到后续所有层的输入，无论是否在同一个`CondenseNet Block` 中。

      如果有不同的`feature map` 尺寸，则使用池化操作来降低分辨率。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_change.png)

2. 对`LGC、LGR、FDC` 的实验分析如下：（`CIFAR-10` 数据集）

    - 配置：默认 k=12，`3x3` 卷积为分组数量为 4 的分组卷积。

      - `LGC:learned group convolution`：`C=4` ，`1x1` 卷积为`LGC` 。
      - `IGR：exponentially increasing learning rate`：$k_0\in \{8,16,32\} 。$
      - `FDC:fully dense conectivity` 。

    - 相邻两条线之间的`gap` 表示对应策略的增益。

      - 第二条线和第一条线之间的`gap` 表示`IGC` 的增益。

      - 第三条线和第二条线之间的`gap` 表示`IGR` 的增益。

      - 第四条线和第三条线之间的`gap` 表示`FDC` 的增益。

        `FDC` 看起来增益不大，但是如果模型更大，根据现有曲线的趋势`FDC` 会起到效果。

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_compare1.png)

#### 9.5.5 网络性能

1. `CondenseNet` 与其它网络的比较：（`*` 表示使用`cosine` 学习率训练 600 个 epoch）

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_compare2.png)

2. `CondenseNet` 与其它裁剪技术的比较：

    ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_compare3.png)

3. `CondenseNet` 的超参数的实验：(`CIFAR-10`，`DenseNet-50` 为基准)

    - 裁剪策略：（`G=4` ）

      - `Full Model` ：不进行任何裁剪。
      - `Traditional Pruning` ：在训练阶段（300 个 epoch）完成时执行裁剪（因此只裁剪一次），裁剪方法与`LGC` 一样。然后使用额外的 300 个 epoch 进行微调。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_pruning_compare.png)

    - 分组数量：（`C=8`）

      - 这里的分组值得是`3x3` 分组卷积的分组数量。
      - 随着分组数量的增加，测试误差逐渐降低。这表明`LGC` 可以降低网络的冗余性。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_group_compare.png)

    - 浓缩因子：(`G=4`)

      可以看到：C\gt1 可以带来更好的效益；但是 C>1 时，网络的准确率和网络的`FLOPs` 呈现同一个函数关系。这表明裁剪网络权重会带来更小的模型，但是也会导致一定的准确率损失。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_c_compare.png)

4. 在`ImageNet` 上的比较：

    - 网络结构：

      为了降低参数，在`epoch 60` 时（一共`120`个`epoch` ）裁剪了全连接层`FC layer` 50% 的权重。思想与`1x1` 卷积的`LGC` 相同，只是 G=1,C=2 。

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condesenet_struct.png)

    - `CondenseNet` 与其它网络的比较：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_compare4.png)

    - 网络在`ARM` 处理器上的推断时间（输入尺寸为`224x224` ）：

      ![img](http://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/imgs/dl_cnn_classification/condensenet_infer_time.png)