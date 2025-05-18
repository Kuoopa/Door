from app.data_processing import load_sample_data
from app.model import train_and_evaluate

import logging

logging.basicConfig(level=logging.INFO)

def main():
    X, y = load_sample_data()
    accuracy = train_and_evaluate(X, y)
    print(f"模型准确率：{accuracy:.2f}")

if __name__ == "__main__":
    main()
