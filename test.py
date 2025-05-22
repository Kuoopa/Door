import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib

# 指定字体路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

# 创建字体对象
my_font = fm.FontProperties(fname=font_path)

# 设置字体
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 测试绘图
plt.text(0.5, 0.5, '测试中文显示', fontsize=20, ha='center')
plt.show()
