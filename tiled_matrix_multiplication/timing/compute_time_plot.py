import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('tiled_matrix_multiplication\\timing\compute_time.csv')


# Grouping the data by 'Matrix Size' and 'Tile Size', then calculating the mean and standard deviation
grouped = data.groupby(['Matrix Size', 'Tile Size'])['Matrix Multiplication Compute Time'].agg(['mean', 'std']).reset_index()

# Plotting
plt.figure(figsize=(10, 6))

# Getting unique matrix sizes for different lines
matrix_sizes = grouped['Matrix Size'].unique()

for matrix_size in matrix_sizes:
    subset = grouped[grouped['Matrix Size'] == matrix_size]
    plt.errorbar(subset['Tile Size'], subset['mean'], yerr=subset['std'], label=f'Matrix Size {matrix_size}', capsize=5, marker='o')

plt.xlabel('Tile Size')
plt.ylabel('Matrix Multiplication Compute Time')
plt.title('Matrix Multiplication Compute Time by Tile Size and Matrix Size')
plt.legend()
plt.grid(True)
plt.show()