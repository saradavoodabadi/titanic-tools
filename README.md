from titanic_tools import TitanicData, TitanicAnalysis

td = TitanicData("data/Titanic-Dataset.csv")  # keep CSV local, not pushed
df = td.clean_basic()
ta = TitanicAnalysis(df)
print(ta.survival_rate(by="sex"))


