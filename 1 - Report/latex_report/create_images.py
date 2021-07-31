# Use matplotlib to create a plot of the sigmoid function and its derivative:
import matplotlib.pyplot as plt
import numpy as np

# Create plot of the sigmoid function and save it to a file ind eps color format
plt.figure()
# plot sigmoid function
plt.plot(np.arange(-5, 5, 0.1), [-1 + 2 / (1 + np.exp(-x)) for x in np.arange(-5, 5, 0.1)])
plt.xlim(-5, 5)
plt.ylim(-1.5, 1.5)
# Add legend:
plt.title("Sigmoid Function")
plt.grid(True)
plt.savefig("sigmoid.eps", format="eps", dpi=1000)
plt.draw()

# Plot derivative of the sigmoid function:
plt.figure()
plt.plot(
    np.arange(-5, 5, 0.1),
    [2 * np.exp(-x) / ((1 + np.exp(-x)) ** 2) for x in np.arange(-5, 5, 0.1)],
)
plt.xlim(-5, 5)
plt.ylim(-1, 2)
plt.grid(True)
plt.title("sigmoid derivative")
plt.savefig("sigmoid_diff.eps", format="eps", dpi=1000)
plt.draw()


# Plot inverse of the sigmoid function:
plt.figure()
plt.plot(
    np.arange(-0.99, 0.99, 0.01),
    [np.log((1 + x) / (1 - x)) for x in np.arange(-0.99, 0.99, 0.01)],
)
plt.autoscale()  # set axis limits automatically
plt.grid(True)
plt.title("sigmoid inverse")
plt.savefig("sigmoid_inv.eps", format="eps", dpi=1000)
plt.draw()

plt.figure()
# plot sigmoid function minus identity
plt.plot(
    np.arange(-5, 5, 0.1),
    [-x - 1 + 2 / (1 + np.exp(-x)) for x in np.arange(-5, 5, 0.1)],
)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
# Add legend:
plt.title("Sigmoid Function minus identity")
plt.grid(True)
plt.savefig("sigmoid_minus1.eps", format="eps", dpi=1000)
plt.draw()

# Create figure of ReLU function:
plt.figure()
plt.plot(np.arange(-5, 5, 0.1), [0 if x < 0 else x for x in np.arange(-5, 5, 0.1)])
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True)
plt.title("ReLU Function")
plt.savefig("ReLU.eps", format="eps", dpi=1000)
plt.draw()


plt.show()
