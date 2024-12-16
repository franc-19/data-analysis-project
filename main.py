# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def load_and_clean_data():
    """
    Load the Iris dataset and perform initial data cleaning
    """
    try:
        # Load iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                         columns=iris['feature_names'] + ['target'])
        
        # Convert target numbers to species names
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        df['species'] = df['target'].map(species_map)
        
        # Drop the numeric target column as we now have species names
        df = df.drop('target', axis=1)
        
        print("Dataset loaded successfully!")
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def perform_basic_analysis(df):
    """
    Perform basic statistical analysis on the dataset
    """
    print("\nBasic Statistical Description:")
    print(df.describe())
    
    print("\nMean values by species:")
    print(df.groupby('species').mean())

def create_visualizations(df):
    """
    Create various visualizations of the data
    """
    # Set up the plotting style
    plt.style.use('seaborn')
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Line plot
    df.groupby('species').mean().plot(marker='o', ax=axes[0,0])
    axes[0,0].set_title('Average Measurements by Species')
    axes[0,0].set_ylabel('Measurement (cm)')
    axes[0,0].legend(loc='best')
    
    # 2. Bar plot
    sns.barplot(x='species', y='sepal length (cm)', data=df, ax=axes[0,1])
    axes[0,1].set_title('Average Sepal Length by Species')
    axes[0,1].set_ylabel('Sepal Length (cm)')
    
    # 3. Histogram
    sns.histplot(data=df, x='petal length (cm)', hue='species', 
                multiple="stack", ax=axes[1,0])
    axes[1,0].set_title('Distribution of Petal Length')
    axes[1,0].set_xlabel('Petal Length (cm)')
    
    # 4. Scatter plot
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                    hue='species', ax=axes[1,1])
    axes[1,1].set_title('Sepal Length vs Petal Length')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def main():
    # Load and clean data
    df = load_and_clean_data()
    
    if df is not None:
        # Perform basic analysis
        perform_basic_analysis(df)
        
        # Create visualizations
        create_visualizations(df)

if __name__ == "__main__":
    main()