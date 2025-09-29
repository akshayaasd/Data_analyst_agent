import pandas as pd
import numpy as np
from typing import Dict, Any, List


class AnalystAgent:    
    def __init__(self):
        self.name = "DataAnalyst"
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        stats = {
            "summary": df.describe().to_dict(),
            "correlations": df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {},
            "numeric_columns_stats": {}
        }
        
        for col in numeric_cols:
            stats["numeric_columns_stats"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
        
        return stats
    
    def generate_insights(self, df: pd.DataFrame) -> List[str]:
        insights = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        insights.append(f"Dataset contains {len(df)} records with {len(df.columns)} features")
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                mean_val = df[col].mean()
                median_val = df[col].median()
                if mean_val > median_val * 1.2:
                    insights.append(f"'{col}' shows right-skewed distribution (mean > median)")
                elif mean_val < median_val * 0.8:
                    insights.append(f"'{col}' shows left-skewed distribution (mean < median)")
        
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                unique_count = df[col].nunique()
                if unique_count < 10:
                    insights.append(f"'{col}' has {unique_count} unique categories")
        
        return insights
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        outliers = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(df) * 100),
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
            }
        
        return outliers