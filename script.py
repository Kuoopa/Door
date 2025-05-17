import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate_model():
    # 生成模拟数据，100个样本，4个特征，2个类别
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

    # 分割训练集和测试集，比例70%训练，30%测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 初始化随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测测试集
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率：{accuracy:.2f}")