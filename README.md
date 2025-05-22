
# Door 项目

## 项目简介

这是一个基于 Python 的机器学习项目，主要实现了随机森林分类器。项目使用了 pandas，scikit-learn，matplotlib 等库来进行数据处理和模型训练。

## 环境配置

1. 安装 Docker（https://www.docker.com/）。
2. 构建 Docker 镜像（镜像名示例：door-app）：
   ```
   docker build -t door-app .
   ```
3. 运行容器，进入容器环境进行调试或运行代码：
   ```
   docker run -it door-app /bin/bash
   ```
4. 在容器内确保已安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 依赖库

- pandas
- scikit-learn
- matplotlib

依赖库已写入 requirements.txt，可以通过以下命令安装：
```
pip install -r requirements.txt
```

## 使用说明

- 代码入口：main.py
- 核心逻辑封装在app文件夹中，包括生成模拟数据、训练随机森林模型、预测与评估以及生成前后数据对比图。

运行示例：
```
python main.py
```

## 代码结构

- main.py：程序入口，调用训练和评估函数。
- data_preocessing.py：包含数据的预处理代码
- model.py：包含机器学习模型的实现代码。
- visualization.py：包含数据的可视化对比图代码
- test.py：用于作者自我调试
- requirements.txt：项目依赖列表。
- sources.list：中科大镜像源，用于网络卡顿状况的非常规问题解决

## 备注

- 当前使用的是模拟数据，后续可以替换成实际数据集进行训练和测试。
- 欢迎大家提出问题和建议，感谢关注！

---

作者: Kuoopa  
邮箱: 1836194138@qq.com
