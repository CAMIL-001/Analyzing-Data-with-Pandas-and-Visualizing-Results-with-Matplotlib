# Analyzing-Data-with-Pandas-and-Visualizing-Results-with-Matplotlib
Analyzing Data with Pandas and Visualizing Results with Matplotlib. Objective For this Assignment:  To load and analyze a dataset using the pandas library in Python. To create simple plots and charts with the matplotlib library for visualizing the data.


# Task 1: Load and Explore the Dataset

# Choose a dataset in CSV format (for example, you can use datasets like the Iris dataset, a sales dataset, or any dataset of your choice).
# Load the dataset using pandas.
# Display the first few rows of the dataset using .head() to inspect the data.
# Explore the structure of the dataset by checking the data types and any missing values.
# Clean the dataset by either filling or dropping any missing values.

    import pandas as pd

# 1. Load the dataset
# (Using the built-in Iris dataset from seaborn for demonstration)

import seaborn as sns
iris = sns.load_dataset('iris')
iris.to_csv('iris.csv', index=False)  # Save to CSV first to simulate real-world usage

# Now load from CSV
df = pd.read_csv('iris.csv')

# 2. Display the first few rows
print("=== First 5 Rows ===")
print(df.head())
print("\n")

# 3. Explore dataset structure
print("=== Dataset Structure ===")
print(f"Shape: {df.shape}")  # (rows, columns)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\n")

# 4. Data Cleaning
# Let's artificially introduce some missing values for demonstration
import numpy as np
df.iloc[10:15, 2:4] = np.nan  # Make some NaN values in columns 2-3, rows 10-14

print("=== After Introducing Missing Values ===")
print("Missing Values Count:")
print(df.isnull().sum())
print("\n")

# Cleaning approach 1: Fill with mean (for numerical columns)
df_filled = df.copy()
for col in df.select_dtypes(include=['float64']).columns:
    df_filled[col].fillna(df[col].mean(), inplace=True)

# Cleaning approach 2: Drop rows with missing values
df_dropped = df.dropna()

print("=== Cleaning Results ===")
print("Method 1: Filled with mean - Missing values:")
print(df_filled.isnull().sum())
print(f"\nShape after filling: {df_filled.shape}")

print("\nMethod 2: Dropped missing rows - Missing values:")
print(df_dropped.isnull().sum())
print(f"\nShape after dropping: {df_dropped.shape}")

# 5. Final cleaned dataset (using filled version)
clean_df = df_filled
print("\n=== Final Cleaned Dataset ===")
print(clean_df.head())



# Task 2: Basic Data Analysis

# Compute the basic statistics of the numerical columns (e.g., mean, median, standard deviation) using .describe().

# Perform groupings on a categorical column (for example, species, region, or department) and compute the mean of a numerical column for each group.

# Identify any patterns or interesting findings from your analysis.



import pandas as pd
import seaborn as sns

# Load and prepare the data
iris = sns.load_dataset('iris')
df = iris.copy()

# 1. Basic Statistics
print("=== Basic Statistics ===")
print(df.describe())
print("\n")

# 2. Grouping by Species
print("=== Mean Measurements by Species ===")
species_stats = df.groupby('species').mean()
print(species_stats)
print("\n")

# 3. Advanced Grouping (multiple statistics)
print("=== Detailed Statistics by Species ===")
detailed_stats = df.groupby('species').agg(['mean', 'median', 'std', 'min', 'max'])
print(detailed_stats)
print("\n")

# 4. Interesting Findings
print("=== Key Findings ===")

# Finding 1: Petal measurements show clear separation between species
print("\n1. Petal Dimensions Show Strong Species Differentiation:")
print("Setosa petals are significantly smaller than other species")
print(f"Setosa max petal length: {df[df['species']=='setosa']['petal_length'].max():.1f} cm")
print(f"Versicolor min petal length: {df[df['species']=='versicolor']['petal_length'].min():.1f} cm")

# Finding 2: Setosa has notably different sepal width
print("\n2. Sepal Width Distinguishes Setosa:")
print(f"Setosa mean sepal width: {species_stats.loc['setosa', 'sepal_width']:.2f} cm")
print(f"Other species average: {species_stats.loc[['versicolor','virginica'], 'sepal_width'].mean():.2f} cm")

# Finding 3: Virginica has the largest flowers
print("\n3. Virginica Has Largest Overall Dimensions:")
print("Largest mean petal length by species:")
print(species_stats['petal_length'].sort_values(ascending=False))

# 5. Visualization
import matplotlib.pyplot as plt

print("\n=== Generating Visualizations ===")
plt.figure(figsize=(12, 6))

# Boxplot of petal length by species
plt.subplot(1, 2, 1)
sns.boxplot(x='species', y='petal_length', data=df)
plt.title('Petal Length by Species')

# Scatter plot of sepal dimensions
plt.subplot(1, 2, 2)
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
plt.title('Sepal Dimensions')

plt.tight_layout()
plt.savefig('iris_analysis.png')
print("Visualizations saved to iris_analysis.png")

# Basic Statistics
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000

#Mean by Statistics
            sepal_length  sepal_width  petal_length  petal_width
species                                                        
setosa             5.006        3.428         1.462        0.246
versicolor         5.936        2.770         4.260        1.326
virginica          6.588        2.974         5.552        2.026

Petal Dimensions:

Setosa petals are distinctly smaller (max 1.9cm) compared to versicolor (minimum 3.0cm)

Virginica has the largest petals (mean 5.55cm)

Sepal Width:

Setosa has significantly wider sepals (mean 3.43cm) compared to others (~2.87cm)

Size Progression:

Clear size progression: setosa < versicolor < virginica across all measurements

Standard Deviations:

Petal measurements show higher variability than sepal measurements

Virginica shows the most variation in petal length (std=0.55)


# Task 3: Data Visualization

# Create at least four different types of visualizations:

# Line chart showing trends over time (for example, a time-series of sales data).

# Bar chart showing the comparison of a numerical value across categories (e.g., average petal length per species).

# Histogram of a numerical column to understand its distribution.

# Scatter plot to visualize the relationship between two numerical columns (e.g., sepal length vs. petal length).



     import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare data
iris = sns.load_dataset('iris')

# Create a time-series version by adding a dummy date column
dates = pd.date_range(start='2023-01-01', periods=len(iris), freq='D')
iris_with_time = iris.copy()
iris_with_time['date'] = dates
iris_with_time['cumulative_sepal_length'] = iris_with_time['sepal_length'].cumsum()

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(16, 10))

# 1. Line Chart (Trend over time)
plt.subplot(2, 2, 1)
sns.lineplot(x='date', y='cumulative_sepal_length', hue='species', 
             data=iris_with_time, palette="husl", linewidth=2.5)
plt.title('Cumulative Sepal Length Over Time', fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Sepal Length (cm)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Species')

# 2. Bar Chart (Comparison across categories)
plt.subplot(2, 2, 2)
mean_petal_length = iris.groupby('species')['petal_length'].mean().reset_index()
bar_plot = sns.barplot(x='species', y='petal_length', data=mean_petal_length, 
                       palette="coolwarm", edgecolor="black")
plt.title('Average Petal Length by Species', fontsize=14, pad=20)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Mean Petal Length (cm)', fontsize=12)

# Add value labels on bars
for p in bar_plot.patches:
    bar_plot.annotate(f"{p.get_height():.2f}", 
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 10), 
                     textcoords='offset points')

# 3. Histogram (Distribution of a numerical column)
plt.subplot(2, 2, 3)
sns.histplot(data=iris, x='sepal_width', bins=15, kde=True, 
             hue='species', palette="viridis", alpha=0.6)
plt.title('Distribution of Sepal Width by Species', fontsize=14, pad=20)
plt.xlabel('Sepal Width (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Species')

# 4. Scatter Plot (Relationship between two numerical columns)
plt.subplot(2, 2, 4)
scatter = sns.scatterplot(data=iris, x='sepal_length', y='petal_length', 
                         hue='species', palette="rocket", s=100, 
                         style='species', markers=["o", "s", "D"])
plt.title('Sepal Length vs Petal Length', fontsize=14, pad=20)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)

# Add regression lines
for species in iris['species'].unique():
    sns.regplot(data=iris[iris['species']==species], 
               x='sepal_length', y='petal_length', 
               scatter=False, ci=None, truncate=True)

plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')

# Final adjustments
plt.tight_layout(pad=3.0)
plt.suptitle('Iris Dataset Visual Analysis', y=1.02, fontsize=16)
plt.savefig('iris_visualizations.png', bbox_inches='tight', dpi=300)
plt.show()

               # Key Visualizations Produced:
# Line Chart:

Shows cumulative sepal length over time (artificial timeline)

Reveals growth patterns by species

Virginica accumulates length fastest due to larger flowers

# Bar Chart:

Compares mean petal length across species

Clear progression: setosa < versicolor < virginica

Exact values displayed on bars (setosa: 1.46cm, virginica: 5.55cm)

# Histogram:

Distribution of sepal width with KDE curves

Setosa shows distinct right-skewed distribution

Versicolor and virginica overlap more

# Scatter Plot:

Relationship between sepal and petal length

Strong positive correlation within each species

Different clusters clearly visible

Regression lines show species-specific trends

# Customization Features:
Consistent Styling: Professional whitegrid theme with uniform fonts

# Enhanced Readability:

Clear titles and axis labels

Value labels on bar chart

Legend placement optimization

# Visual Enhancements:

Different markers for species in scatter plot

Semi-transparent histogram bars

Color palettes optimized for color blindness

# Technical Precision:

Proper figure sizing and padding

High-resolution export (300dpi)

Tight layout to prevent overlap

# Insights Revealed:
Virginica consistently shows the largest measurements

Setosa is morphologically distinct in both size and proportions

Strong linear relationships within species suggest consistent growth patterns

Versicolor serves as an intermediate between the other two species



