```python
from titanic_tools import TitanicData, TitanicAnalysis

td = TitanicData("data/Titanic-Dataset.csv")  # keep CSV local (not in repo)
df = td.clean_basic()
ta = TitanicAnalysis(df)

print("Overall:", ta.survival_rate())
print("By sex:\n", ta.survival_rate(by="sex"))

import matplotlib.pyplot as plt
ta.plot_survival_by("sex")
plt.show()
