import pandas as pd
from typing import Dict, Any


class CleanerAgent:    
    def __init__(self):
        self.name = "DataCleaner"
    
    def clean_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        original_rows = len(df)
        
        df = df.drop_duplicates()
        
        df = df.dropna()
        
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        
        cleaned_rows = len(df)
        rows_removed = original_rows - cleaned_rows
        
        return {
            "cleaned_data": df,
            "metadata": {
                "original_rows": original_rows,
                "cleaned_rows": cleaned_rows,
                "rows_removed": rows_removed,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict()
            }
        }
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        validation_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns)
        }
        
        return validation_report