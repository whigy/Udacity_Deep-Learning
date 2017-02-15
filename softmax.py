"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #pass  # TODO: Compute and return softmax(x)
    x = np.array(x)
    y = np.exp(x)
    y = y / np.sum(y, 0)
    return y
    

scores = [1.0, 2.0, 3.0]
print(softmax(scores))

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
print(softmax(scores))


# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

##### 10 times of scores: more close to 1 / 0 !!!!!!!!!!!!!!!!!!!!!!!!!
plt.plot(x, softmax(scores *10).T, linewidth=2)
plt.show()


### 1/10 times of scores: more uniform!!!!!!!!!!!!!!!!!
###since all the scores decrease in magnitude, the resulting softmax probabilities will be closer to each other.
plt.plot(x, softmax(scores/10).T, linewidth=2)
plt.show()