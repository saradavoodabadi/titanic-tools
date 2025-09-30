from __future__ import annotations
from pathlib import Path
import pandas as pd


class TitanicData:
    """
    Load and prepare the Titanic CSV.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the Titanic CSV file (e.g., 'data/Titanic-Dataset.csv').

    Attributes
    ----------
    csv_path : pathlib.Path
        Normalized path to the CSV file.
    df : pandas.DataFrame or None
        The most recently loaded/cleaned dataframe. Starts as None.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame | None = None

    def load(self, encoding: str = "utf-8") -> pd.DataFrame:
        """
        Read the CSV into a pandas DataFrame.

        Parameters
        ----------
        encoding : str, default "utf-8"
            Text encoding of the CSV file. Change if your file needs it.

        Returns
        -------
        pandas.DataFrame
            Raw dataframe with original columns from the CSV.
        """
        self.df = pd.read_csv(self.csv_path, encoding=encoding)
        return self.df

    def clean_basic(self) -> pd.DataFrame:
        """
        Perform simple, transparent cleaning focused on teaching.

        Operations
        ----------
        - Standardize column names to lower_snake_case.
        - Fill missing ``age`` with the column median.
        - Fill missing ``embarked`` with the most frequent value (mode).
        - Convert ``sex`` and ``embarked`` to 'category' dtype.

        Returns
        -------
        pandas.DataFrame
            Cleaned dataframe (also stored in ``self.df``).
        """
        if self.df is None:
            self.load()

        df = self.df.copy()

        # 1) Consistent column names -> lower snake_case
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # 2) Very basic imputations for teaching purposes
        if "age" in df.columns:
            df["age"] = df["age"].fillna(df["age"].median())
        if "embarked" in df.columns:
            df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])

        # 3) Mark obvious categoricals
        for col in ("sex", "embarked"):
            if col in df.columns:
                df[col] = df[col].astype("category")

        self.df = df
        return df

    def add_simple_features(self) -> pd.DataFrame:
        """
        Create small helper features for analysis.

        New Columns
        -----------
        family_size : int
            ``sibsp + parch + 1`` (includes the passenger).
        is_child : int
            1 if ``age < 16`` else 0.

        Returns
        -------
        pandas.DataFrame
            Dataframe with the added feature columns.
        """
        if self.df is None:
            raise ValueError("Call load()/clean_basic() before adding features.")

        df = self.df.copy()
        if {"sibsp", "parch"}.issubset(df.columns):
            df["family_size"] = df["sibsp"] + df["parch"] + 1
        if "age" in df.columns:
            df["is_child"] = (df["age"] < 16).astype(int)

        self.df = df
        return df

    def clean_advanced(self) -> pd.DataFrame:
        """
        Perform advanced data cleaning and feature engineering on the Titanic dataset.

        This includes:
        - Filling missing Age values with median grouped by Sex and Pclass
        - Filling missing Embarked values with the mode
        - Dropping Cabin (too sparse) and Ticket
        - Creating new features: FamilySize, IsAlone, Title (from Name)
        - Encoding categorical features (Sex, Embarked, Title)

        Returns
        -------
        pd.DataFrame
            Cleaned and feature-engineered DataFrame.
        """
        if self.df is None:
            self.load()

        df = self.df.copy()

        # 1) Consistent column names -> lower snake_case
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Fill Age: Median age by Sex and Pclass
        df["age"] = df.groupby(["sex", "pclass"])["age"].transform(lambda x: x.fillna(x.median()))

        # Fill Embarked: Use most frequent
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])

        # Drop Cabin (too many missing), Ticket (not useful), and Name after extracting Title
        df["has_cabin"] = df["cabin"].notnull().astype(int)
        df.drop(["cabin", "ticket"], axis=1, inplace=True)

        # Create FamilySize
        df["family_size"] = df["sibsp"] + df["parch"] + 1
        df["is_alone"] = (df["family_size"] == 1).astype(int)

        # Extract Title from Name
        df["title"] = df["name"].str.extract(" ([A-Za-z]+)\.", expand=False)
        rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
                       "Sir", "Jonkheer", "Dona"]
        df["title"] = df["title"].replace(rare_titles, "Rare")
        df["title"] = df["title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        df.drop("name", axis=1, inplace=True)

        # Encode categorical columns
        df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
        df = pd.get_dummies(df, columns=["embarked", "title"], drop_first=True)

        self.df = df
        return df

