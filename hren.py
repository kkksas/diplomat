import numpy as np
import matplotlib.pyplot as plt
 
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
 
x = [0,1,2,3,2,2,3,4]
ax.step(np.arange(len(x)), x)

 
plt.show()