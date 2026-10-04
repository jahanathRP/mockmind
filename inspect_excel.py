import pandas as pd
import json

file_path = "d:/projects/2nd year 3rd sem ML project/ai interview simulator/mockmind/data/MockMind_complete_ml_dataset.xlsx"

try:
    xl = pd.ExcelFile(file_path)
    sheets = xl.sheet_names
    
    result = {"sheets": sheets, "columns": {}}
    for sheet in sheets:
        df = xl.parse(sheet)
        result["columns"][sheet] = list(df.columns)
        
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error: {e}")
