import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# 0. 绘图与中文字体设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 物理参数与常数
# ==========================================
mc = 33000.0                       # 车体质量 (kg)
mf = 3000.0 + 16 * 500.0           # 悬浮架总质量 (kg)
g = 9.8                            # 重力加速度 (m/s^2)
K = 0.079987                       # 第一问求得的电磁力系数

# ==========================================
# 2. 读取监测数据 (data4.xlsx)
# ==========================================
print("正在读取 data4.xlsx 监测数据...")
df = pd.read_excel('data4.xlsx')

t_data = df.iloc[:, 0].values          
h_data = df.iloc[:, 1].values          
ddz_c_data = df.iloc[:, 2].values      
I_matrix = df.iloc[:, 3:19].values     # shape: (N, 16)

dt = t_data[1] - t_data[0]
N = len(t_data)

# ==========================================
# 3. 动力学数据预处理
# ==========================================
print("正在提取悬浮架加速度与客观需求力...")
# SG滤波求加速度
ddh_data = savgol_filter(h_data, window_length=201, polyorder=3, deriv=2, delta=dt)

# 计算系统的客观需求总力 Y (形状: N x 1)
F_req = mc * ddz_c_data - mf * ddh_data + (mc + mf) * g

# 计算每个电磁铁的理想受力矩阵 X (形状: N x 16)
I_squared_sgn = np.sign(I_matrix) * (I_matrix ** 2)
# 利用广播机制：h_data[:, None] 将一维数组转为 N x 1 列向量
F_ideal_matrix = K * I_squared_sgn / (h_data[:, None] ** 2)

# ==========================================
# 4. 滑动窗口最小二乘法 (核心算法)
# ==========================================
print("正在执行滑动窗口最小二乘算法辨识16台电磁铁参数...")

# 定义时间窗大小与滑动步长 (假设采样率为0.0001s, 500个点=0.05秒)
window_size = 500 
step_size = 50

time_centers = []
eta_results = []  # 保存各个窗口的16个η值

for start_idx in range(0, N - window_size, step_size):
    end_idx = start_idx + window_size
    
    # 截取窗口内的数据
    X_window = F_ideal_matrix[start_idx:end_idx, :] # shape: (500, 16)
    Y_window = F_req[start_idx:end_idx]             # shape: (500,)
    
    # 使用最小二乘法求解: X * eta = Y
    # rcond=None 保证即使电流高度相关也能求出稳定伪逆解
    eta_window, residuals, rank, s = np.linalg.lstsq(X_window, Y_window, rcond=None)
    
    eta_results.append(eta_window)
    # 记录该窗口的中心时间
    time_centers.append(t_data[start_idx + window_size // 2])

time_centers = np.array(time_centers)
eta_results = np.array(eta_results) # shape: (M_windows, 16)

# ==========================================
# 4. 【新增】对辨识结果进行适度平滑，提升图表美观度
# ==========================================
# 使用 pandas 的 rolling 做简单滑动平均，消除高频辨识毛刺
eta_smoothed = pd.DataFrame(eta_results).rolling(window=10, min_periods=1, center=True).mean().values

# ==========================================
# 5. 生成 4x4 矩阵时序图表 (高颜值版)
# ==========================================
print("分析完成！正在生成各电磁铁诊断矩阵图...")

fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True, sharey=True)
fig.suptitle(r'各分体式电磁铁功率放大系数 $\eta_i(t)$ 动态监测图 (平滑降噪后)', fontsize=18, fontweight='bold')

fault_info = []

for i in range(16):
    row, col = divmod(i, 4)
    ax = axes[row, col]
    
    eta_line = eta_smoothed[:, i] # 使用平滑后的数据画图
    
    # 绘制正常区间绿带和边界
    ax.axhspan(0.8, 1.2, color='#2ca02c', alpha=0.15)
    ax.axhline(0.8, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axhline(1.2, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.8)
    
    # 画曲线
    ax.plot(time_centers, eta_line, color='#1f77b4', linewidth=1.5)
    ax.set_title(f'电磁铁 {i+1} 号', fontsize=12)
    ax.set_ylim(0.4, 1.6)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    avg_eta = np.mean(eta_line)
    if avg_eta < 0.8:
        ax.set_facecolor('#ffe6e6')
        ax.text(0.05, 0.85, "状态: 衰减故障", transform=ax.transAxes, color='darkred', fontweight='bold')
        fault_info.append(f"电磁铁 {i+1} : 总体均值 {avg_eta:.3f} (衰减)")
    elif avg_eta > 1.2:
        ax.set_facecolor('#fff2cc')
        ax.text(0.05, 0.85, "状态: 激增故障", transform=ax.transAxes, color='darkgoldenrod', fontweight='bold')
        fault_info.append(f"电磁铁 {i+1} : 总体均值 {avg_eta:.3f} (激增)")
    else:
        ax.text(0.05, 0.85, "状态: 正常", transform=ax.transAxes, color='green')

for ax in axes[-1, :]: ax.set_xlabel('时间 t (s)', fontsize=12)
for ax in axes[:, 0]: ax.set_ylabel(r'放大系数 $\eta$', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# ==========================================
# 6. 【重磅补充】生成全局统计分布检验图 (Boxplot 箱线图)
# ==========================================
print("正在生成统计学验证：箱线图分布检验...")

fig2 = plt.figure(figsize=(14, 6))
ax_box = fig2.add_subplot(111)

# 画正常状态的绿色安全带
ax_box.axhspan(0.8, 1.2, color='green', alpha=0.15, label='[0.8, 1.2] 正常容限区间')
ax_box.axhline(0.8, color='red', linestyle='--', alpha=0.7)
ax_box.axhline(1.2, color='red', linestyle='--', alpha=0.7)

# 绘制箱线图
box = ax_box.boxplot(eta_results, patch_artist=True, 
                     boxprops=dict(facecolor="lightblue", color="blue", alpha=0.7),
                     medianprops=dict(color="red", linewidth=2),
                     flierprops=dict(marker='o', color='red', alpha=0.3, markersize=3))

ax_box.set_title(r'16台电磁铁功率放大系数 $\eta$ 的全时段统计分布检验图', fontsize=16, fontweight='bold')
ax_box.set_xlabel('悬浮电磁铁编号', fontsize=14)
ax_box.set_ylabel(r'辨识得出的放大系数 $\eta$', fontsize=14)
ax_box.set_xticks(range(1, 17))
ax_box.grid(True, axis='y', linestyle='--', alpha=0.6)
ax_box.legend(loc='upper left', fontsize=12)

# 标记异常值（将出问题的电磁铁箱体标红）
for i in range(16):
    avg_val = np.mean(eta_results[:, i])
    if avg_val > 1.2 or avg_val < 0.8:
        box['boxes'][i].set_facecolor('#ff9999') # 异常的涂成红色
        ax_box.text(i+1, avg_val + 0.15, '异常故障!', color='darkred', 
                    horizontalalignment='center', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

plt.tight_layout()
plt.show()