import matplotlib.pyplot as plt
import pandas as pd

# read data
df = pd.read_csv('servoclipped.txt', header=None)
print(df)

plt.plot(df[0], '-o')
plt.show()
