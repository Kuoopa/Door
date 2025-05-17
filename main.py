import pandas as pd
import sklearn

print(pd.__version__)
print(sklearn.__version__)

from script import train_and_evaluate_model

if __name__ == "__main__":
    train_and_evaluate_model()
