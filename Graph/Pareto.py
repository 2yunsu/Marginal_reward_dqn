import matplotlib.pyplot as plt
import numpy as np

# x1과 x2의 범위를 정의합니다.
x1 = np.linspace(0.1, 14, 100)  # 0부터 14까지 100개의 점으로 나눈 범위
x2 = np.linspace(0.1, 14, 100)

# 한계효용 함수를 정의합니다.
def marginal_utility(x1, x2):
    return 10.0 * (0.8**(min(x1, x2) - 1))

# 무차별곡선을 나타내는 함수를 정의합니다.
def indifference_curve(x1, x2):
    return 10.0 * (0.8**(min(x1, x2) - 1))

# (14, 14)를 원점으로 하는 무차별곡선을 나타내는 함수를 정의합니다.
def new_indifference_curve(x1, x2):
    return 10.0 * (0.8**(min(x1 - 14, x2 - 14) - 1))

# 무차별곡선을 그립니다.
X1, X2 = np.meshgrid(x1, x2)
U = indifference_curve(X1, X2)

# (3,4)과 (4,3)의 효용을 표시합니다.
points1 = [(3, 4), (4, 3)]
U_points1 = [marginal_utility(x1, x2) for x1, x2 in points1]

# (2,5)과 (5,2)의 효용을 표시합니다.
points2 = [(2, 5), (5, 2)]
U_points2 = [marginal_utility(x1, x2) for x1, x2 in points2]

# (1,6)과 (6,1)의 효용을 표시합니다.
points3 = [(1, 6), (6, 1)]
U_points3 = [marginal_utility(x1, x2) for x1, x2 in points3]

plt.contour(x1, x2, U, levels=np.arange(0, np.max(U), 1), colors='gray', linestyles='dotted', alpha=0.5)

# (14, 14)를 원점으로 하는 무차별곡선을 그립니다.
U_new = new_indifference_curve(X1, X2)
plt.contour(x1, x2, U_new, levels=np.arange(0, np.max(U_new), 1), colors='purple', linestyles='dotted', alpha=0.5)

plt.xlabel(r'$\mathcal{X}_1$')
plt.ylabel(r'$\mathcal{X}_2$')
plt.title('Indifference Curve')

# 조합을 표시합니다.
plt.plot([p[0] for p in points1], [p[1] for p in points1], 'ro', label=f'U = {U_points1[0]:.2f}')
plt.plot([p[0] for p in points2], [p[1] for p in points2], 'bo', label=f'U = {U_points2[0]:.2f}')
plt.plot([p[0] for p in points3], [p[1] for p in points3], 'go', label=f'U = {U_points3[0]:.2f}')

# 그래프를 표시합니다.
plt.legend()
plt.show()
