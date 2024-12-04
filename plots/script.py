import seaborn as sns
import matplotlib.pyplot as plt

# Sample data for BiPO
x = [2, 3, 5, 8 ,13]
y_case1 = [6.8, 6.1, 5.9, 6.7, 5.9]  # Replace with your actual BiPO data points

# Create a scatter plot
sns.lineplot(x=x, y=y_case1, marker='o')

# Add labels, title, and legend
plt.xlabel('Layer Number', fontsize=14)
plt.ylabel('Average Behavioral Score', fontsize=14)
plt.title('Refusal across layers with multiplier -1', fontsize=16)
plt.xticks(ticks=x)

# Save the plot in the specified folder
plt.savefig('layer_wise_refusal_minus_1.png')  # Replace with your desired folder path

# Show the plot (optional)
plt.show()