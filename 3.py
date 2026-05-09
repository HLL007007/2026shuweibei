import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr  # 新增：用于计算相关系数

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

t_data = df.iloc[:, 0].values          
h_data = df.iloc[:, 1].values          
ddz_c_data = df.iloc[:, 2].values      
I_matrix = df.iloc[:, 3:19].values     

dt = t_data[1] - t_data[0]             

# ==========================================
# 3. 核心计算：求解间隙加速度 ddh (带滤波降噪)
# ==========================================
print("正在运用 Savitzky-Golay 滤波器提取悬浮架加速度...")
ddh_data = savgol_filter(h_data, window_length=201, polyorder=3, deriv=2, delta=dt)

# ==========================================
# 4. 构建理想电磁力 F_ideal 与 客观需求力 F_req
# ==========================================
I_squared_sgn = np.sign(I_matrix) * (I_matrix ** 2)
sum_I_term = np.sum(I_squared_sgn, axis=1)  
F_ideal = K * sum_I_term / (h_data ** 2)
F_req = mc * ddz_c_data - mf * ddh_data + (mc + mf) * g

# ==========================================
# 5. 【新增】统计学检验：线性相关性分析
# ==========================================
print("正在进行统计学检验：评估理论力与实际需求力的线性相关性...")
# 计算 Pearson 相关系数（评估线性关系强弱）
pearson_corr, p_value_p = pearsonr(F_ideal, F_req)
# 计算 Spearman 秩相关系数（评估单调关系强弱，对异常值更鲁棒）
spearman_corr, p_value_s = spearmanr(F_ideal, F_req)

print(f" -> 皮尔逊 (Pearson) 相关系数 : {pearson_corr:.5f} (p-value: {p_value_p:.2e})")
print(f" -> 斯皮尔曼 (Spearman) 秩相关系数: {spearman_corr:.5f} (p-value: {p_value_s:.2e})")

# ==========================================
# 6. 最小二乘法参数辨识 
# ==========================================
eta_opt = np.sum(F_ideal * F_req) / np.sum(F_ideal ** 2)
F_fit = eta_opt * F_ideal

r2 = r2_score(F_req, F_fit)
rmse = np.sqrt(mean_squared_error(F_req, F_fit))
residual = F_req - F_fit

# ==========================================
# 7. 故障诊断与结论输出
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
    conclusion = "功率放大系数低于 0.8，超出正常波动范围。\n-> 判定整车悬浮系统存在【功率衰减异常故障】。\n-> 这印证了问题二中列车因动力不足而向下掉落的物理现象！"
else:
    fault_type = "激增故障"
    conclusion = "功率放大系数高于 1.2，超出正常波动范围。\n-> 判定整车悬浮系统存在【功率激增异常故障】。"

print(conclusion)
print("="*45 + "\n")

# ==========================================
# 8. 论文级可视化图表生成 (双图输出)
# ==========================================
step = 50  # 降采样步长

# ----- 图1：相关性散点图 (证明模型合理性) -----
fig1 = plt.figure(figsize=(8, 6))

# 绘制散点图
plt.scatter(F_ideal[::step], F_req[::step], alpha=0.3, color='#1f77b4', edgecolors='none', label='系统受力观测数据点')

# 绘制最小二乘拟合直线
x_line = np.array([np.min(F_ideal), np.max(F_ideal)])
y_line = eta_opt * x_line
plt.plot(x_line, y_line, color='#d62728', linewidth=2.5, linestyle='--', label=rf'最优线性拟合线 (斜率 $\eta$={eta_opt:.4f})')

plt.title('理论电磁力与实际需求合力相关性散点检验', fontsize=15, fontweight='bold')
plt.xlabel('总理论电磁力 $F_{ideal}$ (N)', fontsize=12)
plt.ylabel('系统客观需求总力 $F_{req}$ (N)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 添加相关系数文本框
corr_text = (rf"Pearson 系数: {pearson_corr:.4f}" + "\n" +
             rf"Spearman 系数: {spearman_corr:.4f}" + "\n" +
             f"P-value 显著性: < 0.001")
plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='#f8f9fa', ec='gray', alpha=0.9))

plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.show()

# ----- 图2：时序动力学残差双子图 (证明拟合精准度) -----
t_plot = t_data[::step]
F_req_plot = F_req[::step]
F_fit_plot = F_fit[::step]
residual_plot = residual[::step]

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# 上半部分：受力拟合对比
ax1.plot(t_plot, F_req_plot, label='动力学客观需求合力 $F_{req}$', color='#d62728', linewidth=1.5, alpha=0.8)
ax1.plot(t_plot, F_fit_plot, label=r'辨识模型拟合电磁力 ($\eta$={:.4f})'.format(eta_opt), color='#2ca02c', linewidth=2, linestyle='--')
ax1.set_title(r'整车悬浮系统动力学辨识与残差分析 (诊断: {})'.format(fault_type), fontsize=16, fontweight='bold')
ax1.set_ylabel('总受力大小 (N)', fontsize=12)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

# 在ax1添加模型验证指标文本框
bbox_props = dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="gray", lw=1)
ax1.text(0.7, 0.45, 
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

plt.tight_layout()
plt.show()