import matplotlib.pyplot as plt

#Marginal Utility Graph
# n_goal_2의 범위를 정의합니다.
n_goal_2_values = range(1, 7)

# 각 n_goal_2 값에 대한 reward 값을 계산합니다.
rewards = [10.0*(0.8**(n-1)) for n in n_goal_2_values]

# 그래프를 그립니다.
plt.plot(n_goal_2_values, rewards, label='Reward', color='red', linestyle='solid', linewidth=2, marker='o', markersize=5)
plt.xlabel('Number of goods')
plt.ylabel('Utility')
plt.title('Margianl Utility Graph')

# 각 지점에 reward 값을 표시합니다.
for x, y in zip(n_goal_2_values, rewards):
    plt.text(x, y, f'{y:.2f}', ha='right', va='top')

# 그래프를 이미지 파일로 저장합니다.
plt.subplots_adjust(top=0.85)
plt.savefig('MU_Graph.png')  # 파일 이름은 원하는 대로 지정하십시오.
