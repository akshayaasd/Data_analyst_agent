import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os


class VisualizerAgent:    
    def __init__(self, output_dir: str = "outputs"):
        self.name = "DataVisualizer"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
    
    def create_distribution_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        saved_files = {}
        
        for col in numeric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram
            axes[0].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Distribution of {col}')
            
            # Boxplot
            axes[1].boxplot(df[col])
            axes[1].set_ylabel(col)
            axes[1].set_title(f'Boxplot of {col}')
            
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, f"distribution_{col}.png")
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            saved_files[col] = filepath
        
        return saved_files
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            return None
        
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_summary_report(self, stats: Dict[str, Any], insights: list) -> str:
        report_path = os.path.join(self.output_dir, "analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("KEY INSIGHTS:\n")
            f.write("-" * 60 + "\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("STATISTICAL SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            if "numeric_columns_stats" in stats:
                for col, col_stats in stats["numeric_columns_stats"].items():
                    f.write(f"\n{col}:\n")
                    for stat_name, value in col_stats.items():
                        f.write(f"  {stat_name}: {value:.2f}\n")
        
        return report_path