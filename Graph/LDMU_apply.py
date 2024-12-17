import matplotlib.pyplot as plt
import numpy as np

marginal_rate = 0.8
reward_coefficient = 10

def f_i(x):
    return sum([reward_coefficient * (marginal_rate) ** i for i in range(x)])

def f_total(a, b):
    return f_i(a) + f_i(b)

# Q값 범위 설정
Q = [(6 - i, i) for i in range(7)]  # (Agent A, Agent B) 분배
Total_Utility = [f_total(a, b) for a, b in Q]

# 최고점을 기준으로 뒤집기
max_utility = max(Total_Utility)
Inverted_Utility = [max_utility - value for value in Total_Utility]

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(range(len(Q)), Inverted_Utility, label='Apply LDMU', color='blue')

# 상수함수 추가
plt.axhline(y=0, color='red', linestyle='-', label='Without LDMU')

# 그래프 꾸미기
plt.title('Convexity change due to application of the LDMU', fontsize=20)
plt.xlabel(r'$x_i$ (Agent A, Agent B)', fontsize=18)
plt.ylabel('Regret', fontsize=18)
plt.xticks(ticks=range(len(Q)), labels=[f'{a},{b}' for a, b in Q], rotation=45, fontsize=14)
plt.ylim(-2, max_utility-min(Total_Utility)+2)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('LDMU_apply.png')