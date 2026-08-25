import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. GENERATE DATA (NumPy & Pandas)
# ==========================================
# Line data
x = np.linspace(0, 10, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Scatter data
height = np.random.normal(170, 10, 50)
weight = height * 0.4 + np.random.normal(0, 5, 50)

# Histogram & Boxplot data (with outliers)
normal_scores = np.random.normal(75, 10, 200)
outliers = np.array([20, 30, 130]) # Extreme low and high scores
all_scores = np.concatenate([normal_scores, outliers])

# Pandas DataFrame for Bar Chart
df_sales = pd.DataFrame({
    "Product": ["Apples", "Bananas", "Cherries", "Dates"],
    "Revenue": [450, 800, 300, 650]
})


# ==========================================
# 2. CREATE THE CANVAS (2 rows, 3 columns)
# ==========================================
# figsize=(width, height) in inches
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))

# Give the whole dashboard a main title
fig.suptitle("Matplotlib Master Dashboard", fontsize=20, fontweight='bold')


# ==========================================
# 3. FILL THE SUBPLOTS
# ==========================================

# --- [Row 0, Col 0]: Line Plot (Customized limits & ticks) ---
axes[0, 0].plot(x, y_sin, color='blue', linestyle='-', label='Sine')
axes[0, 0].plot(x, y_cos, color='red', linestyle='--', label='Cosine')
axes[0, 0].set_title("Line Plot with Grid")
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle=':', alpha=0.7) # Add grid
axes[0, 0].set_xlim(0, 10) # Set strict X limits


# --- [Row 0, Col 1]: Scatter Plot (Transparency) ---
axes[0, 1].scatter(height, weight, color='green', marker='^', alpha=0.6)
axes[0, 1].set_title("Scatter Plot (Height vs Weight)")
axes[0, 1].set_xlabel("Height (cm)")
axes[0, 1].set_ylabel("Weight (kg)")


# --- [Row 0, Col 2]: Bar Chart (Direct from Pandas) ---
# Using df_sales DataFrame
axes[0, 2].bar("Product", "Revenue", data=df_sales, color=['purple', 'orange', 'cyan', 'pink'], edgecolor='black')
axes[0, 2].set_title("Bar Chart (From Pandas)")
axes[0, 2].set_ylabel("Revenue ($)")


# --- [Row 1, Col 0]: Histogram (Data Distribution) ---
axes[1, 0].hist(all_scores, bins=20, color='teal', edgecolor='white')
axes[1, 0].set_title("Histogram (Score Distribution)")
axes[1, 0].set_xlabel("Score")


# --- [Row 1, Col 1]: Boxplot (Outlier Detection) ---
# Using the same score data to show how boxplots handle outliers differently than histograms
axes[1, 1].boxplot(all_scores, vert=False) # vert=False makes it horizontal
axes[1, 1].set_title("Boxplot (Finding Outliers)")
axes[1, 1].set_xlabel("Score Range")


# --- [Row 1, Col 2]: Custom Text & Cleanup ---
# Sometimes you want a subplot just to display metrics or text!
axes[1, 2].axis('off') # Turn off the borders and axes
axes[1, 2].text(0.5, 0.5, "You have mastered\nMatplotlib Basics!", 
                fontsize=16, color='darkblue', 
                ha='center', va='center') # ha/va centers the text


# ==========================================
# 4. FINALIZE AND DISPLAY
# ==========================================
# tight_layout automatically fixes overlapping titles and labels
plt.tight_layout()

# Save the dashboard as a high-quality image file
fig.savefig("matplotlib_dashboard.png", dpi=300)
print("Dashboard saved to your computer as 'matplotlib_dashboard.png'!")

# Display the dashboard
plt.show()