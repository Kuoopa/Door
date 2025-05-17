# 选择基础镜像，这里我们使用 Python 3.11
FROM python:3.11-slim

# 设置容器内的工作目录
WORKDIR /app


# 拷贝 requirements.txt 到镜像中
COPY requirements.txt .

# 安装 requirements.txt 中列出的所有依赖库
RUN pip install -r requirements.txt


# 将当前目录中的所有文件复制到容器内的 /app 目录
COPY . .

# 设置容器启动时执行的命令，这里是运行 main.py 文件
CMD ["python", "main.py"]