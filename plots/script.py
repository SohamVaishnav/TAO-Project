import seaborn as sns
import matplotlib.pyplot as plt

# Sample data for BiPO
x = [2, 3, 5, 8 ,13]
y_case1 = [2.1, 1.3, 2.2, 0.4, 1.0]  # Replace with your actual BiPO data points

# Create a scatter plot
sns.lineplot(x=x, y=y_case1, marker='o')

# Add labels, title, and legend
plt.xlabel('Layer Number', fontsize=14)
plt.ylabel('Average Behavioral Score', fontsize=14)
plt.title('Coherence across layers with multiplier +1', fontsize=16)
plt.xticks(ticks=x)

# Save the plot in the specified folder
plt.savefig('layer_wise_plus_1.png')  # Replace with your desired folder path

# Show the plot (optional)
plt.show()