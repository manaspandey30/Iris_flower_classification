import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the Iris dataset"""
    return pd.read_csv(file_path)

def create_pair_plot(df):
    """Create pair plot of all features"""
    plt.figure(figsize=(12, 8))
    sns.pairplot(df, hue='species', markers=["o", "s", "D"])
    plt.savefig('pair_plot.png')
    plt.close()

def create_box_plots(df):
    """Create box plots for each feature"""
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.boxplot(x='species', y='sepal_length', data=df)
    plt.subplot(2, 2, 2)
    sns.boxplot(x='species', y='sepal_width', data=df)
    plt.subplot(2, 2, 3)
    sns.boxplot(x='species', y='petal_length', data=df)
    plt.subplot(2, 2, 4)
    sns.boxplot(x='species', y='petal_width', data=df)
    plt.tight_layout()
    plt.savefig('box_plots.png')
    plt.close()

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

def main():
    # Load data
    df = load_data('IRIS.csv')
    
    # Create visualizations
    print("Creating data visualizations...")
    create_pair_plot(df)
    create_box_plots(df)
    create_correlation_heatmap(df)
    
    print("Visualizations saved as PNG files:")
    print("- pair_plot.png")
    print("- box_plots.png")
    print("- correlation_heatmap.png")

if __name__ == "__main__":
    main() 