from app.data_processing import load_sample_data
from app.model import train_and_evaluate
from app.visualization import plot_comparison
import logging

logging.basicConfig(level=logging.INFO)


def main():
    X, y = load_sample_data()
    y_test, y_pred, score = train_and_evaluate(X, y)
    print(f"模型R²得分：{score:.2f}")
    plot_comparison(y_test, y_pred)

if __name__ == "__main__":
    main()




