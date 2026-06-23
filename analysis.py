import pandas as pd
from pathlib import Path

private_excel = Path("private_data/my_real_smile_journal.xlsx")
private_csv = Path("private_data/my_real_smile_journal.csv")
sample_csv = Path("data/sample_smile_data.csv")

if private_excel.exists():
    print("Loading private Excel file...")
    df = pd.read_excel(private_excel)
elif private_csv.exists():
    print("Loading private CSV file...")
    df = pd.read_csv(private_csv)
else:
    print("Private data not found. Loading sample data...")
    df = pd.read_csv(sample_csv)

print(df.head())
print(df.info())