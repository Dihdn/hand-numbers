# 🧠 手写数字识别（LeNet-MNIST）

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Stars](https://img.shields.io/github/stars/yourname/lenet-mnist?style=social)](https://github.com/yourname/lenet-mnist)

📦 本项目基于 **PyTorch** 实现 LeNet 卷积神经网络，用于 **MNIST 手写数字识别** 任务。

---

## 📁 项目结构
```
├── LeNet.py           # 主模型与训练脚本
├── LeNet_test.py      # 测试脚本
├── data/              # MNIST 数据集目录
├── img/               # 训练过程图片输出
├── lenet_data/        # 训练过程数据保存
├── module/            # 训练得到的模型参数
├── .vscode/           # VSCode 配置
├── README.md          # 项目说明
```

---

## 🧰 环境依赖

- 🐍 Python 3.10  
- 🔥 PyTorch 2.5  
- 📈 Matplotlib  
- ➗ Numpy  

## 📥 数据集说明
首次运行项目会自动下载 MNIST 数据集并保存至 data/ 目录中。

## 🚀 快速开始
## 🔧 训练模型
运行主训练脚本：

```sh
python LeNet.py
```
训练过程中会自动保存模型参数到 `module/`，损失曲线数据到 `lenet_data/`，并输出训练/测试损失与准确率。

## 📊 可视化结果

训练结束后会在 `img/lenet.jpg` 生成损失曲线图。

## 🧪 模型测试

可使用 [`LeNet_test.py`](LeNet_test.py) 对模型进行测试或推理。

## 📄 主要文件说明

- [`LeNet.py`](LeNet.py)：包含模型定义、训练流程、准确率计算与绘图函数。
- [`LeNet_test.py`](LeNet_test.py)：模型加载与测试脚本。

## 🔗 参考资料

- [📄 LeNet 原始论文](http://yann.lecun.com/exdb/lenet/)
- [📚 PyTorch 官方文档](https://pytorch.org/)
