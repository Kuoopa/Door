from app.data_processing import load_sample_data
from app.model import train_and_evaluate
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)


def main():
    X, y = load_sample_data()
    accuracy = train_and_evaluate(X, y)
    print(f"模型准确率：{accuracy:.2f}")

    # 假设你的波形数据是一个列表或数组
    time = [0, 1, 2, 3, 4, 5]  # 时间或者采样点索引
    signal = [0, 0.5, 0.8, 0.4, 0.2, 0]  # 对应的波形幅值

    plt.plot(time, signal)
    plt.title("波形图示例")
    plt.xlabel("时间 / 采样点")
    plt.ylabel("幅值")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()




