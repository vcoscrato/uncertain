from matplotlib import pyplot as plt
import numpy as np

bins = np.arange(1, 21)
good = np.linspace(1, 0.5, num=20) + np.random.normal(0.1, 0.01, 20)
bad = np.linspace(1, 1.5, num=20) + np.random.normal(0.1, 0.01, 20)
zero = np.linspace(1, 1, num=20) + np.random.normal(0.1, 0.01, 20)

f, ax = plt.subplots(figsize=(10, 5))
ax.plot(bins, zero, 'b-', label='Uncertainty measure 1')
ax.plot(bins, bad, 'r-x', label='Uncertainty measure 2')
ax.plot(bins, good, 'g--', label='Uncertainty measure 3')
ax.set_xlabel('Uncertainty bin', Fontsize=20)
ax.set_xticks(bins)
ax.set_ylabel('RMSE', usetex=True, Fontsize=30)
ax.legend()
f.tight_layout()
f.savefig('Results/example_intervals.pdf')