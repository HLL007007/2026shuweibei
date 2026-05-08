import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ==========================================
# 0. 绘图环境与中文字体设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 物理模型参数定义
# ==========================================
mc = 33000.0                       # 车体质量 (kg)
mf = 3000.0 + 16 * 500.0           # 悬浮架总质量 (kg): 悬浮架 + 16个电磁铁
k = 2e7                            # 空气弹簧刚度 (N/m)
c = 8e4                            # 空气弹簧阻尼系数 (N·s/m)
g = 9.8                            # 重力加速度 (m/s^2)

# ==========================================
# 2. 实测数据读取与外力函数插值
# ==========================================
print("正在读取实测电磁力数据 (data2.xlsx)...")
df = pd.read_excel('data2.xlsx')

# 使用 iloc 防止表头含空格导致读取失败
t_data = df.iloc[:, 0].values       # 第0列为时间 t
F_1_to_16 = df.iloc[:, 1:17].values # 第1到16列为 16个电磁铁实际产生的电磁力

# 合成系统受到的总电磁力 Fm(t)
Fm_data = np.sum(F_1_to_16, axis=1)

# 构建连续时间的总力插值函数，供微分方程求解器调用
Fm_interp = interp1d(t_data, Fm_data, kind='linear', fill_value="extrapolate")

# ==========================================
# 3. 状态空间方程构建 (牛顿第二定律降阶)
# ==========================================
# Y[0]=z_c(车体位移), Y[1]=dz_c/dt, Y[2]=z_f(悬浮架位移), Y[3]=dz_f/dt
def system_odes(t, Y):
    y1, y2, y3, y4 = Y
    Fm_t = Fm_interp(t) # 获取当前时刻的外力
    
    dy1 = y2
    dy2 = (-k * (y1 - y3) - c * (y2 - y4) - mc * g) / mc
    dy3 = y4
    dy4 = (k * (y1 - y3) + c * (y2 - y4) - mf * g + Fm_t) / mf
    
    return [dy1, dy2, dy3, dy4]

# ==========================================
# 4. 设置初始条件并进行数值求解
# ==========================================
# 设定 t=0 时的系统状态
y1_0 = -(mc * g) / k   # 车体因重力产生的初始静压位移
y2_0 = 0.0             # 车体初速度为0
y3_0 = 0.0             # 悬浮架初位移为0 (停靠在轨道上)
y4_0 = 0.0             # 悬浮架初速度为0
Y0 = [y1_0, y2_0, y3_0, y4_0]

# 设置求解区间和评估点
t_span = (0, 10)
t_eval = np.linspace(0, 10, 5000) 

print("正在运用 RK45 算法求解动力学微分方程...")
# 调用求解器 (设定高精度 atol, rtol)
sol = solve_ivp(system_odes, t_span, Y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

# ==========================================
# 5. 提取指标与计算第9秒参数
# ==========================================
zc_t = sol.y[0]            # t时刻的车体垂向位移
zf_t = sol.y[2]            # t时刻的悬浮架垂向位移
h_t = 0.06 - zf_t          # t时刻的悬浮间隙

# 利用三次样条插值精准抓取 t=9.0s 时的结果
zf_func = interp1d(sol.t, zf_t, kind='cubic')
zf_9s = float(zf_func(9.0))
h_9s = 0.06 - zf_9s

print("\n--- 问题二求解结果 ---")
print(f"9秒时 悬浮架垂向位移 z_f(9) = {zf_9s:.6f} m")
print(f"9秒时 悬浮间隙 h(9)     = {h_9s:.6f} m")
print("------------------------\n")

# ==========================================
# 6. 生成结果分析图 (论文插图)
# ==========================================
fig = plt.figure(figsize=(14, 8))

# 子图1: 车体与悬浮架的位移历程曲线
ax1 = fig.add_subplot(211)
ax1.plot(sol.t, zc_t, label='车体位移 $z_c(t)$', color='#2ca02c', linewidth=2)
ax1.plot(sol.t, zf_t, label='悬浮架位移 $z_f(t)$', color='#d62728', linewidth=2)
ax1.set_title('车体与悬浮架 0~10秒 的垂向运动位移响应', fontsize=14)
ax1.set_xlabel('时间 t (s)', fontsize=12)
ax1.set_ylabel('垂向位移 z (m)', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# 子图2: 悬浮间隙变化规律
ax2 = fig.add_subplot(212)
ax2.plot(sol.t, h_t, label='悬浮间隙 $h(t)$', color='#9467bd', linewidth=2)
# 标记第9秒的状态点
ax2.axvline(9.0, color='grey', linestyle='-.', alpha=0.8)
ax2.scatter([9.0], [h_9s], color='red', s=60, zorder=5)
ax2.text(9.05, h_9s + 0.002, f"t=9s, h={h_9s:.5f}m", color='red', fontsize=12, fontweight='bold')
ax2.axhline(0.06, color='black', linestyle='--', label='初始轨道支撑面(h=0.06)')

ax2.set_title('电磁力作用下的悬浮间隙变化规律', fontsize=14)
ax2.set_xlabel('时间 t (s)', fontsize=12)
ax2.set_ylabel('悬浮间隙 h (m)', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()