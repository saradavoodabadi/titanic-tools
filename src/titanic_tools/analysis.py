import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TitanicAnalysis:
    """
    Titanic survival analysis and visualization tools.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed Titanic dataset (after cleaning/feature engineering).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def survival_rate(self, by: str | list[str] = None) -> pd.DataFrame:
        """
        Calculate survival rate overall or grouped by one or more columns.

        Parameters
        ----------
        by : str or list of str, optional
            Column name(s) to group by before calculating survival rate.

        Returns
        -------
        pandas.DataFrame
            Survival rate table.
        """
        df = self.df.copy()
        if by is None:
            rate = df['survived'].mean()
            return pd.DataFrame({'overall_survival_rate': [rate]})
        else:
            grouped = df.groupby(by)['survived'].mean().reset_index()
            grouped.rename(columns={'survived': 'survival_rate'}, inplace=True)
            return grouped

    def plot_survival_by(self, column: str):
        """
        Plot survival rate by a single categorical column.

        Parameters
        ----------
        column : str
            Column name to group by and plot.
        """
        df = self.df.copy()
        grouped = df.groupby(column)['survived'].mean().reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(x=column, y='survived', data=grouped, palette='muted')
        plt.title(f'Survival Rate by {column.capitalize()}')
        plt.ylabel('Survival Rate')
        plt.xlabel(column.capitalize())
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_survival_by_age_group(self):
        """
        Bin passengers into age groups and plot survival rates per group.

        Age Groups:
        - Child: 0–12
        - Teen: 13–17
        - Young Adult: 18–29
        - Adult: 30–49
        - Senior: 50+

        The function uses seaborn to create a barplot of survival rates by age group.
        """
        df = self.df.copy()

        # Define age bins and labels
        bins = [0, 12, 17, 29, 49, 120]
        labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

        # Calculate survival rate by age group
        survival_rates = df.groupby('age_group')['survived'].mean().reset_index()

        # Plotting
        plt.figure(figsize=(8, 5))
        sns.barplot(x='age_group', y='survived', data=survival_rates, palette='viridis')

        plt.title('Survival Rate by Age Group')
        plt.ylabel('Survival Rate')
        plt.xlabel('Age Group')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
