import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# df = pd.read_csv('result10.csv')

# ein = list(df["Ein"].values)
# epsilon = list(df["epsilon"].values)

# t = np.arange(0, 500)

# plt.figure(figsize=(10, 6))
# plt.plot(t, ein, label='Ein', color="#fCA311")
# plt.plot(t, epsilon, label='epsilon', color="#5448C8")

# plt.xlabel('t')
# plt.ylabel('Error')
# plt.title('problem 10')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig('hw7-p10.png')
# plt.close()

df = pd.read_csv('result11.csv')

ein = list(df["Ein"].values)
eout = list(df["Eout"].values)

t = np.arange(0, 500)

plt.figure(figsize=(10, 6))
plt.plot(t, ein, label='Ein', color="#fCA311")
plt.plot(t, eout, label='Eout', color="#5448C8")

plt.xlabel('t')
plt.ylabel('Error')
plt.title('problem 11')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('hw7-p11.png')
plt.close()
