from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TitanicAnalysis:
    """
    Compute simple survival statistics and draw one bar plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned Titanic dataframe. Must include a 0/1 column for survival.
        Accepts either 'Survived' or 'survived' and normalizes to 'Survived'.
    """

    def __init__(self, df: pd.DataFrame):
        # Ensure a survival column exists, normalize its name.
        if "Survived" not in df.columns and "survived" not in df.columns:
            raise ValueError("DataFrame must contain a 'Survived' column (0/1).")
        self.df = df.rename(columns={"survived": "Survived"}).copy()

        # Convenience map so users can pass 'sex' or 'Sex'
        self._cols_lower = {c.lower(): c for c in self.df.columns}

    def survival_rate(self, by: str | list[str] | None = None):
        """
        Compute survival rate overall or grouped.

        Parameters
        ----------
        by : str or list of str or None, default None
            Column(s) to group by (e.g., 'sex', ['sex', 'pclass']).
            Case-insensitive for convenience.

        Returns
        -------
        float or pandas.Series/DataFrame
            Overall rate if ``by=None``, otherwise grouped rates.
        """
        if by is None:
            return float(self.df["Survived"].mean())

        # Allow case-insensitive column names for user friendliness
        col = (
            [self._cols_lower.get(c.lower(), c) for c in by]
            if isinstance(by, list)
            else self._cols_lower.get(by.lower(), by)
        )

        # observed=True silences a future pandas warning and is correct
        return (
            self.df.groupby(col, observed=True)["Survived"]
            .mean()
            .rename("SurvivalRate")
        )

    def plot_survival_by(self, by: str) -> plt.Axes:
        """
        Draw a bar chart of survival rate for a single categorical column.

        Parameters
        ----------
        by : str
            Categorical column to plot (e.g., 'sex', 'pclass', 'embarked').

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the bar plot.
        """
        col = self._cols_lower.get(by.lower(), by)
        rates = self.survival_rate(col).reset_index()
        ax = sns.barplot(data=rates, x=col, y="SurvivalRate")
        ax.set_title(f"Survival Rate by {col}")
        ax.set_ylabel("Rate (0–1)")
        return ax

    def describe_numeric(self) -> pd.DataFrame:
        """
        Describe numeric columns split by survival (0 vs 1).

        Returns
        -------
        pandas.DataFrame
            ``describe()`` summary for numeric columns grouped by 'Survived'.
        """
        num = self.df.select_dtypes("number")
        return num.groupby(self.df["Survived"]).describe()
