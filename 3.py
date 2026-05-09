import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 0. 绘图环境与中文字体设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 物理参数与常数输入
# ==========================================
mc = 33000.0                       # 车体质量 (kg)
mf = 3000.0 + 16 * 500.0           # 悬浮架总质量 (kg): 11000 kg
g = 9.8                            # 重力加速度 (m/s^2)
K = 0.079987                       # 第一问求得的电磁力系数

# ==========================================
# 2. 读取监测数据 (data3.xlsx)
# ==========================================
print("正在读取 data3.xlsx 监测数据...")
df = pd.read_excel('data3.xlsx')

# 使用 iloc 精确读取对应的列，避免表头空格导致的报错
t_data = df.iloc[:, 0].values          # 第0列：时间 t
h_data = df.iloc[:, 1].values          # 第1列：悬浮间隙 h
ddz_c_data = df.iloc[:, 2].values      # 第2列：车体加速度 ddz_c
I_matrix = df.iloc[:, 3:19].values     # 第3~18列：16个电磁铁的电流 I_1 ~ I_16

dt = t_data[1] - t_data[0]             # 计算采样步长 (通常为0.0001s)

# ==========================================
# 3. 核心计算：求解间隙加速度 ddh (带滤波降噪)
# ==========================================
# 提示写论文用：直接对位移求二次差分会产生极大的高频噪声导致模型失效。
# 这里采用 Savitzky-Golay 滤波器进行平滑求导，window_length=201, polyorder=3
print("正在运用 Savitzky-Golay 滤波器提取悬浮架加速度...")
ddh_data = savgol_filter(h_data, window_length=201, polyorder=3, deriv=2, delta=dt)

# ==========================================
# 4. 构建理想电磁力 F_ideal 与 客观需求力 F_req
# ==========================================
# (1) 计算理想状态下的理论总电磁力: F_ideal = Sum( K * sgn(I) * I^2 / h^2 )
# 采用矩阵向量化加速计算
I_squared_sgn = np.sign(I_matrix) * (I_matrix ** 2)
sum_I_term = np.sum(I_squared_sgn, axis=1)  # 每一行(时刻)对16个电磁铁求和
F_ideal = K * sum_I_term / (h_data ** 2)

# (2) 计算系统维持当前运动状态所需的实际客观受力 F_req
# 公式推导: F_req = mc*ddz_c - mf*ddh + (mc+mf)*g
F_req = mc * ddz_c_data - mf * ddh_data + (mc + mf) * g

# ==========================================
# 5. 最小二乘法参数辨识 
# ==========================================
eta_opt = np.sum(F_ideal * F_req) / np.sum(F_ideal ** 2)
F_fit = eta_opt * F_ideal

# 【新增验证指标计算】
r2 = r2_score(F_req, F_fit)
rmse = np.sqrt(mean_squared_error(F_req, F_fit))
residual = F_req - F_fit

if 0.8 <= eta_opt <= 1.2: fault_type = "正常运行"
elif eta_opt < 0.8: fault_type = "衰减故障"
else: fault_type = "激增故障"

# ==========================================
# 6. 故障诊断与结论输出
# ==========================================
print("\n" + "="*45)
print("★ 问题三：全车悬浮系统功率放大器故障诊断报告 ★")
print("="*45)
print(f"-> 测算得整车统一功率放大系数 η = {eta_opt:.6f}")

if 0.8 <= eta_opt <= 1.2:
    fault_type = "正常运行"
    conclusion = "功率放大系数在 [0.8, 1.2] 正常范围内，判定为【无故障】。"
elif eta_opt < 0.8:
    fault_type = "衰减故障"
    conclusion = f"功率放大系数低于 0.8，超出正常波动范围。\n-> 判定整车悬浮系统存在【功率衰减异常故障】。\n-> 这印证了问题二中列车因动力不足而向下掉落的物理现象！"
else:
    fault_type = "激增故障"
    conclusion = "功率放大系数高于 1.2，超出正常波动范围。\n-> 判定整车悬浮系统存在【功率激增异常故障】。"

print(conclusion)
print("="*45 + "\n")

# ==========================================
# 7. 【升级版】论文级可视化：双子图与残差分析
# ==========================================
# 降采样画图，让图表不至于因为点太密而发黑
step = 50
t_plot = t_data[::step]
F_req_plot = F_req[::step]
F_fit_plot = F_fit[::step]
residual_plot = residual[::step]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# 上半部分：受力拟合对比
ax1.plot(t_plot, F_req_plot, label='动力学客观需求合力 $F_{req}$', color='#d62728', linewidth=1.5, alpha=0.8)
ax1.plot(t_plot, F_fit_plot, label=r'辨识模型拟合电磁力 ($\eta$={:.4f})'.format(eta_opt), color='#2ca02c', linewidth=2, linestyle='--')
ax1.set_title(r'整车悬浮系统动力学辨识与残差分析 (诊断: {})'.format(fault_type), fontsize=16, fontweight='bold')
ax1.set_ylabel('总受力大小 (N)', fontsize=12)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

# 在ax1添加模型验证指标文本框
bbox_props = dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="gray", lw=1)
ax1.text(0.7, 0.5, 
         rf"最优辨识系数 $\eta$ = {eta_opt:.4f}" + "\n" +
         rf"拟合优度 $R^2$ = {r2:.4f}" + "\n" +
         rf"均方根误差 RMSE = {rmse:.2f} N", 
         transform=ax1.transAxes, fontsize=12, verticalalignment='bottom', bbox=bbox_props)

# 下半部分：残差图
ax2.plot(t_plot, residual_plot, color='#1f77b4', linewidth=1, alpha=0.8)
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('时间 t (s)', fontsize=12)
ax2.set_ylabel('拟合残差 (N)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)

# 调整布局
plt.tight_layout()
plt.show()