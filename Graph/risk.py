import numpy as np

import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y1 = x
y2 = np.sqrt(10) * np.sqrt(x)

plt.plot(x, y1, label='Expected value of risky goods')
plt.plot(x, y2, label='Value of Money')
plt.xlabel('Price')
plt.ylabel('Utility')
plt.title('Utility function graph')
plt.legend()
plt.savefig('risk.png')