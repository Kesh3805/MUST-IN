"""
Results Summary Generator
Creates comprehensive summaries and visualizations of experiment results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class ResultsSummarizer:
    """
    Generates comprehensive summaries and visualizations of model results
    """
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def add_result(self, model_name, metrics):
        """
        Add a model's results for summary generation
        
        Args:
            model_name: Name of the model
            metrics: Dictionary containing accuracy, f1, precision, recall, roc_auc, etc.
        """
        result = {'Model': model_name}
        result.update(metrics)
        self.results.append(result)
    
    def generate_summary_table(self):
        """
        Generate a pandas DataFrame summarizing all results
        
        Returns:
            pd.DataFrame: Summary table of all model results
        """
        if not self.results:
            print("No results to summarize")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Round numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        return df
    
    def save_summary_csv(self, filename='model_comparison.csv'):
        """
        Save summary table to CSV file
        
        Args:
            filename: Name of the output CSV file
        """
        df = self.generate_summary_table()
        if df is not None:
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Summary saved to: {filepath}")
    
    def generate_comparison_plots(self):
        """
        Generate comparison plots for all models
        """
        df = self.generate_summary_table()
        if df is None:
            return
        
        # Metrics to plot (exclude non-numeric columns)
        metrics = [col for col in df.columns if col != 'Model' and df[col].dtype in ['float64', 'int64']]
        
        if not metrics:
            print("No numeric metrics found for plotting")
            return
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Sort by metric value for better visualization
            df_sorted = df.sort_values(by=metric, ascending=True)
            
            # Create horizontal bar chart
            bars = ax.barh(df_sorted['Model'], df_sorted[metric], color='steelblue')
            
            # Color the best performing model
            max_idx = df_sorted[metric].idxmax()
            bars[df_sorted.index.get_loc(max_idx)].set_color('darkgreen')
            
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}', 
                        fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (model, value) in enumerate(zip(df_sorted['Model'], df_sorted[metric])):
                ax.text(value, i, f' {value:.3f}', va='center', fontsize=9)
        
        # Hide extra subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'model_comparison_plots.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {filepath}")
        plt.close()
    
    def generate_heatmap(self):
        """
        Generate a heatmap showing all metrics for all models
        """
        df = self.generate_summary_table()
        if df is None:
            return
        
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        if numeric_df.empty:
            print("No numeric data for heatmap")
            return
        
        plt.figure(figsize=(10, len(df) * 0.5 + 2))
        
        # Create heatmap
        sns.heatmap(numeric_df.T, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=df['Model'], yticklabels=numeric_df.columns,
                   cbar_kws={'label': 'Score'})
        
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Model')
        plt.ylabel('Metric')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'performance_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to: {filepath}")
        plt.close()
    
    def generate_best_model_report(self, metric='accuracy'):
        """
        Generate a report highlighting the best performing model
        
        Args:
            metric: The metric to use for determining the best model
        """
        df = self.generate_summary_table()
        if df is None:
            return
        
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in results")
            return
        
        best_idx = df[metric].idxmax()
        best_model = df.iloc[best_idx]
        
        report = []
        report.append("=" * 70)
        report.append("BEST MODEL REPORT")
        report.append("=" * 70)
        report.append(f"\nBest Model: {best_model['Model']}")
        report.append(f"Based on: {metric.replace('_', ' ').title()}")
        report.append(f"\nPerformance Metrics:")
        report.append("-" * 70)
        
        for col in df.columns:
            if col != 'Model' and pd.api.types.is_numeric_dtype(df[col]):
                report.append(f"  {col.replace('_', ' ').title():<25}: {best_model[col]:.4f}")
        
        report.append("\n" + "=" * 70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save to file
        filepath = os.path.join(self.output_dir, 'best_model_report.txt')
        with open(filepath, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {filepath}")
    
    def generate_full_report(self):
        """
        Generate a complete report with all visualizations and summaries
        """
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE RESULTS REPORT")
        print("="*70 + "\n")
        
        # Generate summary table
        self.save_summary_csv()
        
        # Generate visualizations
        self.generate_comparison_plots()
        self.generate_heatmap()
        
        # Generate best model report
        self.generate_best_model_report()
        
        # Generate HTML summary
        self._generate_html_summary()
        
        print("\n" + "="*70)
        print("REPORT GENERATION COMPLETE")
        print("="*70)
    
    def _generate_html_summary(self):
        """
        Generate an HTML summary page with embedded visualizations
        """
        df = self.generate_summary_table()
        if df is None:
            return
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MUST-IN Results Summary</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        h1 {{
            margin: 0;
        }}
        .timestamp {{
            color: #ecf0f1;
            font-size: 14px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best-row {{
            background-color: #d5f4e6;
            font-weight: bold;
        }}
        .metric {{
            text-align: center;
        }}
        .visualization {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MUST-IN Framework - Results Summary</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Model Performance Comparison</h2>
    <table>
        <thead>
            <tr>
                {''.join(f'<th>{col}</th>' for col in df.columns)}
            </tr>
        </thead>
        <tbody>
"""
        
        # Find best model (by accuracy if available)
        best_col = 'accuracy' if 'accuracy' in df.columns else df.select_dtypes(include=['float64', 'int64']).columns[0]
        best_idx = df[best_col].idxmax()
        
        for idx, row in df.iterrows():
            row_class = 'best-row' if idx == best_idx else ''
            html += f'            <tr class="{row_class}">\n'
            for col in df.columns:
                value = row[col]
                cell_class = 'metric' if pd.api.types.is_numeric_dtype(df[col]) else ''
                display_value = f'{value:.4f}' if isinstance(value, (int, float)) else value
                html += f'                <td class="{cell_class}">{display_value}</td>\n'
            html += '            </tr>\n'
        
        html += """
        </tbody>
    </table>
    
    <div class="visualization">
        <h2>Performance Comparison Plots</h2>
        <img src="model_comparison_plots.png" alt="Model Comparison Plots">
    </div>
    
    <div class="visualization">
        <h2>Performance Heatmap</h2>
        <img src="performance_heatmap.png" alt="Performance Heatmap">
    </div>
    
    <div class="header" style="background-color: #27ae60; margin-top: 30px;">
        <h2 style="margin: 0;">Project Information</h2>
        <p>MUST-IN: Multilingual Explainable Hate Speech Detection Framework</p>
        <p>Framework for detecting hate speech in Hindi, Tamil, and English (including Romanized scripts)</p>
    </div>
</body>
</html>
"""
        
        filepath = os.path.join(self.output_dir, 'results_summary.html')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML summary saved to: {filepath}")
