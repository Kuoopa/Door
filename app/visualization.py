import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
import os

# 设置中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
assert os.path.exists(font_path), "字体路径不存在，请确认路径正确"
chinese_font = FontProperties(fname=font_path)

# 避免负号被误显示成方块
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_comparison(y_true, y_pred, title="真实值 vs 预测值", save_path="comparison_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="真实值", marker='o')
    plt.plot(y_pred, label="预测值", marker='x')
    plt.title(title, fontproperties=chinese_font)
    plt.xlabel("样本索引", fontproperties=chinese_font)
    plt.ylabel("目标值", fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
