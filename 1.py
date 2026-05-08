import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns 

# ==========================================
# 0. 画图显示中文字体设置 (防止图表中文乱码)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # Windows用SimHei, Mac用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与预处理
# ==========================================
# 1. 数据加载与预处理
df = pd.read_excel('data1.xlsx')

# ★ 加上这一行，强制把前三列重命名为 F, I, z ★
df.columns = ['F', 'I', 'z']

# 提取特征变量(X)和目标变量(y)
I_data = df['I'].values
z_data = df['z'].values
F_data = df['F'].values
# ==========================================
# 2. 定义数学模型
# ==========================================
# 基本模型： F = K * sgn(I) * (I^2 / z^2)
def basic_model(X, K):
    I, z = X
    # np.sign(I) 处理电流方向，保证力的方向与电流方向一致
    return K * np.sign(I) * (I**2 / z**2)

# 进阶模型（非线性漏磁拓展，供精度不佳时备用）
def advanced_model(X, a1, a2, b1, b2):
    I, z = X
    return (a1 * I * np.abs(I) + a2 * I) / (z**2 + b1 * z + b2)

# ==========================================
# 3. 参数拟合 (使用最小二乘法)
# ==========================================
# 拟合基础模型
popt_basic, pcov_basic = curve_fit(basic_model, (I_data, z_data), F_data)
K_opt = popt_basic[0]

print(f"拟合成功！")
print(f"----> 求得电磁力系数 K = {K_opt:.6f}")
print(f"----> 基础模型公式为: F = {K_opt:.6f} * sgn(I) * (I^2 / z^2)")

# 如果你想尝试进阶模型，取消下面这行的注释：
popt_adv, pcov_adv = curve_fit(advanced_model, (I_data, z_data), F_data)

# ==========================================
# 4. 模型精度评估
# ==========================================
# 计算预测值
F_pred = basic_model((I_data, z_data), K_opt)

# 计算各项评价指标
r2 = r2_score(F_data, F_pred)
rmse = np.sqrt(mean_squared_error(F_data, F_pred))
mae = mean_absolute_error(F_data, F_pred)

print("\n--- 模型精度评估 ---")
print(f"决定系数 R^2 : {r2:.6f}  (越接近1越好，表明拟合程度高)")
print(f"均方根误差 RMSE: {rmse:.4f}")
print(f"平均绝对误差 MAE: {mae:.4f}")

# =========================================
# 优化后的论文级绘图代码
# =========================================
fig = plt.figure(figsize=(14, 6))

# ---- 图1：真实值与预测值的对比散点图  ----
ax1 = fig.add_subplot(121)
# 为了防止过度绘制，我们画散点，把点缩小，透明度调低
ax1.scatter(F_data, F_pred, s=2, alpha=0.3, color='#1f77b4', label='预测点')
# 画一条 y=x 的基准线
min_val = min(F_data.min(), F_pred.min())
max_val = max(F_data.max(), F_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线 (y=x)')

ax1.set_title('电磁力真实值与预测值对比', fontsize=14)
ax1.set_xlabel('实测电磁力 F_real (N)', fontsize=12)
ax1.set_ylabel('模型预测电磁力 F_pred (N)', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)


# ---- 图2：残差分布直方图 ----
ax2 = fig.add_subplot(122)
residuals = F_data - F_pred

# 使用直方图+KDE(核密度估计)展示误差的分布形态
sns.histplot(residuals, bins=100, kde=True, color='#ff7f0e', ax=ax2)
ax2.axvline(0, color='black', linestyle='--', lw=2)

ax2.set_title('预测误差(残差)分布规律', fontsize=14)
ax2.set_xlabel('残差 (F_real - F_pred) (N)', fontsize=12)
ax2.set_ylabel('频数 (样本点个数)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

