import seaborn as sns
import matplotlib.pyplot as plt

# Sample data for BiPO
x = [-2, -1, 0, 1, 2]
y_case1 = [5.5, 5.7, 6.2, 6.4, 6.8]  # Replace with your actual BiPO data points
y_case2 = [10, 6.7, 6.2, 6.0, 6.0]  # Replace with your actual BiPO data points
y_case3 = [10, 7.2, 6.5, 7.6, 8.4]

# Create a scatter plot
sns.lineplot(x=x, y=y_case1, marker='o', label='Case 1')
sns.lineplot(x=x, y=y_case2, marker='s', label='Case 2')  # Use a different marker for distinction
sns.lineplot(x=x, y=y_case3, marker='^', label='Case 3')  # Use another different marker

# Add labels, title, and legend
plt.xlabel('Multiplier', fontsize=14)
plt.ylabel('Average Behavioral Score', fontsize=14)
plt.title('Refusal Behavior Upon Steering', fontsize=16)
plt.xticks(ticks=x)
plt.legend(title='Cases', fontsize=12)  # Add a title to the legend (optional)

# Save the plot in the specified folder
plt.savefig('custom_refusal_comparison.png')  # Replace with your desired folder path

# Show the plot (optional)
plt.show()