# Titanic Tools

Beginner-friendly Python package for simple **Titanic Survival Analysis**.

## Installation
Clone this repo or download, then install in editable mode:

```bash
pip install -e .
```

## Usage Example

```python
from titanic_tools import TitanicData, TitanicAnalysis
import matplotlib.pyplot as plt

td = TitanicData("data/Titanic-Dataset.csv")
df = td.clean_basic()

ta = TitanicAnalysis(df)
print("Overall survival:", ta.survival_rate())
print("By sex:\n", ta.survival_rate(by="sex"))

ta.plot_survival_by("sex")
plt.show()
```

## Project Structure
```bash
src/titanic_tools/     # package (Python classes + functions)
  ├── __init__.py      # makes titanic_tools a package
  ├── data.py          # TitanicData class (loading, cleaning, feature engineering)
  └── analysis.py      # TitanicAnalysis class (analysis & visualizations)

notebooks/             # Jupyter Notebook tutorial
data/                  # Kaggle CSV (kept local; not in repo)
pyproject.toml         # build configuration
README.md              # project description
LICENSE                # MIT License
```


## Package Overview
### `titanic_tools.data`  
**Class: `TitanicData`**

Responsible for **loading, cleaning, and preparing the dataset**.

- `load()` → loads the CSV into a Pandas DataFrame.  
- `clean_basic()` → simple cleaning:
  - Standardizes column names to snake_case  
  - Fills missing age (median), embarked (mode)  
  - Converts categorical columns (`sex`, `embarked`)  
- `clean_advanced()` → more advanced cleaning:
  - One-hot encodes categorical features  
  - Extracts passenger title (Mr, Mrs, Miss, etc.) from names  
  - Creates helper features:
    - **family_size** = sibsp + parch + 1  
    - **is_alone** = indicator for solo travelers  
    - **has_cabin** = whether a passenger had a cabin entry  
  - Handles missing values more systematically  
- `add_simple_features()` → small helper features for teaching (family size, child flag).  



### `titanic_tools.analysis`  
**Class: `TitanicAnalysis`**

Provides **exploratory data analysis and visualizations**.

- `survival_rate(by=None)` → computes survival rate overall or grouped by a column (e.g., `sex`, `pclass`).  
- `plot_survival_by(column)` → bar plot of survival rates by a categorical column.  
- `plot_survival_by_age_group()` → creates age bins (child, teen, adult, senior) and visualizes survival by group.  
- (Optional advanced analysis can be added, such as family size survival, correlations, etc.)  

Visualizations are implemented using **Matplotlib** and **Seaborn**.

### Notebook (`notebooks/tutorial.ipynb`)

A step-by-step interactive tutorial that demonstrates:

1. **Loading the dataset**  
2. **Applying cleaning functions** (basic and advanced)  
3. **Inspecting missing values and data types**  
4. **Running survival analyses** (overall, by gender, by class, by embarkation, by age group)  
5. **Visualizations** (bar plots, correlation heatmap, survival distributions)  
6. **Interpretation of results** with markdown explanations  

This notebook is meant to be beginner-friendly and helps understand both the **Python code** and the **data science process**.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it with attribution.

---
## Future Work

Potential extensions that could make the project more advanced:

- Train a **logistic regression classifier** to predict survival.  
- Perform **feature importance analysis** to see which variables matter most.  
- Explore more advanced visualizations (stacked bar charts, survival curves).  
- Add a **CLI (command-line interface)** to run basic analyses without a notebook.  



