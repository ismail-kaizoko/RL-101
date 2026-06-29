
import matplotlib.pyplot as plt
import numpy as np

def value_function_vis(data):

    # Compute N
    N = int(np.sqrt(len(data)))
    print(N)
    
    matrix = data.reshape((N, N))

    # Plot
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar(label="Value")
    plt.title(f"{N}×{N} Matrix")
    plt.show()

    return matrix

