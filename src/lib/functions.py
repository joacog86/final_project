
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import MinMaxScaler


def separate_data(df: pd.DataFrame) -> pd.DataFrame:
    
    '''Separates the input DataFrame into nunmerical and categorical dataframes'''
    
    numericals = df.select_dtypes(np.number)
    categoricals = df.select_dtypes(['object'])
    
    return numericals,  categoricals

def barplot(df, column_label, figsize=(4, 3), font_size=10):
    # Calculate the count and percentage of each unique value in the specified column
    value_counts = df[column_label].value_counts()
    percentage_counts = value_counts / len(df) * 100

    # Plot the bar graph
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=value_counts.index, y=value_counts.values)

    # Add labels and title
    ax.set_xlabel(column_label)
    ax.set_ylabel('Count / Percentage')
    ax.set_title(f'Count and Percentage of {column_label}')

    # Add text annotations inside the bars with both count and percentage
    for i, (p, percentage) in enumerate(zip(ax.patches, percentage_counts)):
        count = value_counts.iloc[i]
        ax.annotate(f'{count}\n{percentage:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', color='white', fontsize=font_size, weight='bold')

    plt.show()
    

# Defining function to create histograms of all the numerical columns

def make_histograms_cols(df: pd.DataFrame, figsize=(12, 15)):
    
    """
    Takes a dataframe and creates histograms for all the columns.
    Parameters:
    - df: DataFrame
    - figsize: Modifies the size of the plotting figure (default (12, 15))
    """
    
    num_cols = 2
    total_cols = len(df.columns)
    num_rows = (total_cols + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)

    for i, col in enumerate(df.columns):
        row_idx = i // num_cols
        col_idx = i % num_cols
        sns.histplot(x=df[col], data=df, ax=axes[row_idx, col_idx], color=sns.color_palette("muted")[0]) 
        axes[row_idx, col_idx].set_title(col)

    plt.show()
    

    def make_boxplot(df, churn_column='churn_value', palette='viridis'):
    cols = df.select_dtypes(include='number').columns
    num_cols = len(cols)
    num_rows = (num_cols + 2) // 3  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    for i, col in enumerate(cols):
        row_index = i // 3
        col_index = i % 3

        ax = axes[row_index, col_index]
        sns.boxplot(data=df, x=churn_column, y=col, palette=palette, ax=ax)
        ax.set_xlabel('Churn')
        ax.set_ylabel(None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'{col}', loc='left', weight='bold')

    # Remove empty subplots if any
    for i in range(num_cols, num_rows * 3):
        fig.delaxes(axes.flatten()[i])

    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical space between rows
    plt.tight_layout()
    plt.show()
    

    # Defining function to plot countplots

def make_countplots(df: pd.DataFrame, figsize=(12, 25)):
    
    """
    Takes a dataframe and creates countplots for all the columns.
    If the column has more than 5 categories, the data goes in the y axis.
    Bars are arranged in descending order based on count.
    Parameters:
    - df: DataFrame
    - figsize: Modifies the size of the plotting figure (default (12, 15))
    """
    
    num_cols = 2
    total_cols = len(df.columns)
    num_rows = (total_cols + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)

    for i, col in enumerate(df.columns):
        row_idx = i // num_cols
        col_idx = i % num_cols

        if df[col].nunique() > 5:
            order = df[col].value_counts().index
            sns.countplot(y=df[col], data=df, ax=axes[row_idx, col_idx], hue=df[col], palette='Set2', order=order)
        else:
            order = df[col].value_counts().index
            sns.countplot(x=df[col], data=df, ax=axes[row_idx, col_idx], hue=df[col], palette='Set2', order=order)

        axes[row_idx, col_idx].set_title(col)
        axes[row_idx, col_idx].set_xlabel('Count' if df[col].nunique() <= 5 else 'Frequency')
        axes[row_idx, col_idx].set_ylabel('Categories' if df[col].nunique() > 5 else 'Count')

    plt.show()
    
    
def calculate_percentage(df):
    percentage_dict = {}

    for column in df.columns:
        value_counts = df[column].value_counts()
        percentages = (value_counts / len(df) * 100).reset_index()
        percentages.columns = [column, 'Percentage']
        percentage_dict[column] = percentages

    return percentage_dict


def create_churn_pivot_table(df, columns):
    pivot_tables = []
    for column in columns:
        # Calculate Churn_Percentage
        churn_percentage = df.groupby(column)['churn_value'].mean() * 100
        churn_percentage = churn_percentage.rename('Churn_Percentage')

        # Calculate Count
        count = df.groupby(column)['churn_value'].count()
        count = count.rename('Count')

        # Merge the results
        pivot_table = pd.concat([churn_percentage, count], axis=1, sort=False)

        # Add a column for non-churn percentage
        pivot_table['Non_Churn_Percentage'] = 100 - pivot_table['Churn_Percentage']

        # Append the pivot table to the list
        pivot_tables.append((column, pivot_table))

    return pivot_tables

# Function to create a bar plot for each pivot table
def plot_churn_bar(pivot_tables, figsize=(15, 30), space_height=0.8, font_size=12):
    num_columns = len(pivot_tables)
    num_rows = math.ceil(num_columns / 3)  # 3 plots per row, adjust as needed

    fig, axes = plt.subplots(num_rows, 3, figsize=figsize, squeeze=False)
    
    # Adjust the height space between subplots
    fig.subplots_adjust(hspace=space_height)

    for i, (column, pivot_table) in enumerate(pivot_tables):
        # Calculate the position in the subplot grid
        row_idx = i // 3
        col_idx = i % 3

        # Create a bar plot
        ax = sns.barplot(x=pivot_table.index, y='Churn_Percentage', data=pivot_table, ax=axes[row_idx, col_idx])

        # Add labels and title
        ax.set_xlabel(column)
        ax.set_ylabel('Churn Percentage')
        ax.set_title(f'Churn Percentage by {column}')

        # Add text annotations inside the bars with adjusted font size
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha='center', va='center', color='white', fontsize=font_size, weight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
def plot_churn_custom_histogram(pivot_tables, figsize=(15, 30), space_height=0.8, font_size=12):
    num_columns = len(pivot_tables)
    num_rows = math.ceil(num_columns / 3)  # 3 plots per row, adjust as needed

    fig, axes = plt.subplots(num_rows, 3, figsize=figsize, squeeze=False)
    
    # Adjust the height space between subplots
    fig.subplots_adjust(hspace=space_height)

    for i, (column, pivot_table) in enumerate(pivot_tables):
        # Calculate the position in the subplot grid
        row_idx = i // 3
        col_idx = i % 3

        # Extract counts
        counts = pivot_table['Count']

        # Calculate bins based on the sum of counts
        bin_edges = [counts.iloc[:i+1].sum() for i in range(len(counts)+1)]

        # Create a histogram
        ax = sns.histplot(x=counts, bins=bin_edges, data=pivot_table, ax=axes[row_idx, col_idx], color=sns.color_palette("muted")[0])

        # Add labels and title
        ax.set_xlabel('Count')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Custom Histogram of Counts for {column}')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    
# Defining a function to compute the vif to identify possible multicolinearity issues

def compute_vif(df: pd.DataFrame, columns: list):

    X = df.loc[:, columns]
    # the calculation of variance inflation requires a constant
    X.loc[:,'intercept'] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.loc[vif['Variable']!='intercept'].sort_values('VIF', ascending=False).reset_index(drop=True)
    return vif



# Calculating skewness of numerical columns

def compute_skewness(df: pd.DataFrame, threshold: int=-2):
    
    '''
    Computes and prints the skewness of the columns in a dataframe.
    Inputs: pandas DataFrame
    '''
    
    print('Skewness of columns in the dataframe:\n')
    
    for col in df.columns:
        if st.skew(df[col]) > abs(threshold) or st.skew(df[col]) < threshold:
            print(f'{col}: {round(st.skew(df[col]), 2)} -> Out of threshold')
        else:
            print(f'{col}: {round(st.skew(df[col]), 2)}')


    
def get_unique_values(dataframe):
    unique_values_dict = {}
    
    for column in dataframe.columns:
        unique_values_dict[column] = pd.Series(dataframe[column].unique())

    unique_values_df = pd.DataFrame(unique_values_dict)
    return unique_values_df



def ordinal_cat_conversion_dict(df: pd.DataFrame, col_convert: str, churn_col: str) -> dict:
    '''
    Creates a mapping dictionary to convert an ordinal category based on the values of a numerical feature.
    Inputs:
    - df: pandas DataFrame
    - col_convert: categorical value to convert
    - churn_col: binary column indicating churn (1) or non-churn (0)
    Outputs:
    - mapping dictionary
    '''
    grouped_df = df.groupby(col_convert)[[churn_col]].mean().sort_values(by=churn_col, ascending=False)

    # Calculate the proportion of churn within each category
    churn_proportion = grouped_df[churn_col] / grouped_df.groupby(col_convert)[churn_col].transform('count')

    # Adjust the proportion based on the proportion of churn within each category using the maximum value
    adjusted_proportion = churn_proportion / churn_proportion.max()

    mapping_dict = adjusted_proportion.to_dict()

    return mapping_dict


def evaluate_models(model_pipeline, model_names, X_train_new, y_train, X_test_new, y_test):
    scores_means = {}
    scores_sds = {}

    for model, model_name in zip(model_pipeline, model_names):
        mean_score = np.mean(cross_val_score(model, X_train_new, y_train, cv=10)).round(2)
        sd_score = np.std(cross_val_score(model, X_train_new, y_train, cv=10)).round(2)
        
        model.fit(X_train_new, y_train)
        y_pred_test = model.predict(X_test_new)
        y_pred_train = model.predict(X_train_new)
        
        print("The accuracy of the model {} is {:.2f} in the TEST SET".format(model_name, model.score(X_test_new, y_test)))
        print("")
        print("The accuracy of the model {} in the TRAIN set is: {:.2f}".format(model_name, accuracy_score(y_train, y_pred_train)))
        print("The accuracy of the model {} in the TEST set is: {:.2f}".format(model_name, accuracy_score(y_test, y_pred_test)))
        print("The precision of the model {} in the TRAIN set is: {:.2f}".format(model_name, precision_score(y_train, y_pred_train)))
        print("The precision of the model {} in the TEST set is: {:.2f}".format(model_name, precision_score(y_test, y_pred_test)))
        print("The recall of the model {} in the TRAIN set is: {:.2f}".format(model_name, recall_score(y_train, y_pred_train)))
        print("The recall of the model {} in the TEST set is: {:.2f}".format(model_name, recall_score(y_test, y_pred_test)))
        print("The F1 of the model {} in the TRAIN set is: {:.2f}".format(model_name, f1_score(y_train, y_pred_train)))
        print("The F1 of the model {} in the TEST set is: {:.2f}".format(model_name, f1_score(y_test, y_pred_test)))
        print("The Kappa of the model {} in the TRAIN set is: {:.2f}".format(model_name, cohen_kappa_score(y_train, y_pred_train)))
        print("The Kappa of the model {} in the TEST set is: {:.2f}".format(model_name, cohen_kappa_score(y_test, y_pred_test)))
        print("")
        
        scores_means[model_name] = mean_score
        scores_sds[model_name] = sd_score
        
        cm_test = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(cm_test, display_labels=model.classes_)
        disp.plot()

    return scores_means, scores_sds


