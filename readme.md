# 文件结构

+ `consts.py` 定义了部分超参数和一些常量
+ `DataSet.py` 继承`torch.utils.data.Dataset`定义了数据集
+ `requirements.txt` python依赖的包
+ `ResNet.py` 实现图像分类模型
+ `TSAIE.py` 融合模型
+ `main.py` 程序入口



# 运行方式

## 安装依赖

```shell
pip install -r requirements.txt
```

## 运行

```shell
python main.py predict
```

`main.py`有一些运行参数:

```
train				训练模型
	--epoch			训练伦次
	--discard		丢弃上次的训练结果
	--txt_only		只训练文本模型
	--img_only		只训练图片模型
validate			使用次选项仅验证模型
predict				生成预测结果

--cpu				使用cpu训练模型
```

# 参考的库

模型的实现主要使用了pytorch。

文本特征的提取使用transforms获取了Bert预训练模型。

