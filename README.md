# Titanic Tools

Beginner-friendly Python package for simple **Titanic Survival Analysis**.

## Installation
Clone this repo or download, then install in editable mode:

```bash
pip install -e .
```


##Usage Example

from titanic_tools import TitanicData, TitanicAnalysis
import matplotlib.pyplot as plt

td = TitanicData("data/Titanic-Dataset.csv")
df = td.clean_basic()

ta = TitanicAnalysis(df)
print("Overall survival:", ta.survival_rate())
print("By sex:\n", ta.survival_rate(by="sex"))

ta.plot_survival_by("sex")
plt.show()

##Project Structure

src/titanic_tools/     # package (classes + functions)
notebooks/             # tutorial notebook
data/                  # Kaggle CSV (kept local; not in repo)
pyproject.toml         # build config
README.md              # this file
LICENSE                # MIT



