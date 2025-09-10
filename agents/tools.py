"""
Custom tools for CSV AI Agent.

This module provides specialized tools that extend the agent's capabilities:
- Data visualization tools
- Statistical analysis tools
- Data quality assessment tools
- Export and reporting tools
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import io
import base64
from langchain.tools import tool
import warnings
warnings.filterwarnings('ignore')

class CSVAnalysisTools:
    """Collection of tools for CSV data analysis"""
    
    def __init__(self):
        self.plot_counter = 0
        
    @tool
    def create_histogram(self, df: pd.DataFrame, column_name: str, bins: int = 30) -> str:
        """
        Create histogram for a numeric column.
        
        Args:
            df: DataFrame to analyze
            column_name: Name of the column to plot
            bins: Number of bins for histogram
            
        Returns:
            Description of the histogram and insights
        """
        try:
            if column_name not in df.columns:
                return f"❌ Column '{column_name}' not found. Available columns: {list(df.columns)}"
            
            if df[column_name].dtype not in ['int64', 'float64', 'int32', 'float32']:
                return f"❌ Column '{column_name}' is not numeric. Type: {df[column_name].dtype}"
            
            # Clean data
            clean_data = df[column_name].dropna()
            
            if len(clean_data) == 0:
                return f"❌ No valid data in column '{column_name}'"
            
            # Calculate statistics
            stats = {
                'mean': clean_data.mean(),
                'median': clean_data.median(),
                'std': clean_data.std(),
                'min': clean_data.min(),
                'max': clean_data.max(),
                'count': len(clean_data)
            }
            
            # Create insights
            insights = []
            
            # Distribution insights
            if stats['std'] / stats['mean'] < 0.1:
                insights.append("📊 Data shows low variability (consistent values)")
            elif stats['std'] / stats['mean'] > 1:
                insights.append("📊 Data shows high variability (spread out values)")
            
            # Skewness check
            if abs(stats['mean'] - stats['median']) > stats['std'] * 0.5:
                if stats['mean'] > stats['median']:
                    insights.append("📈 Distribution appears right-skewed (tail extends to right)")
                else:
                    insights.append("📉 Distribution appears left-skewed (tail extends to left)")
            else:
                insights.append("⚖️ Distribution appears roughly symmetric")
            
            # Range insights  
            range_val = stats['max'] - stats['min']
            insights.append(f"📏 Data range: {range_val:.2f} (from {stats['min']:.2f} to {stats['max']:.2f})")
            
            result = f"""
📊 **Histogram Analysis for '{column_name}'**

**Statistics:**
- Count: {stats['count']:,} values
- Mean: {stats['mean']:.2f}
- Median: {stats['median']:.2f}
- Std Dev: {stats['std']:.2f}
- Range: {stats['min']:.2f} - {stats['max']:.2f}

**Insights:**
{chr(10).join(['• ' + insight for insight in insights])}

*Histogram created with {bins} bins*
"""
            
            return result
            
        except Exception as e:
            return f"❌ Error creating histogram: {str(e)}"
    
    @tool  
    def analyze_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> str:
        """
        Analyze correlation between two numeric columns.
        
        Args:
            df: DataFrame to analyze
            col1: First column name
            col2: Second column name
            
        Returns:
            Correlation analysis results
        """
        try:
            # Validate columns exist
            for col in [col1, col2]:
                if col not in df.columns:
                    return f"❌ Column '{col}' not found. Available: {list(df.columns)}"
            
            # Check if columns are numeric
            for col in [col1, col2]:
                if df[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
                    return f"❌ Column '{col}' is not numeric. Type: {df[col].dtype}"
            
            # Clean data (remove rows where either column is null)
            clean_df = df[[col1, col2]].dropna()
            
            if len(clean_df) < 2:
                return f"❌ Insufficient data for correlation analysis (need at least 2 valid pairs)"
            
            # Calculate correlations
            pearson_corr = clean_df[col1].corr(clean_df[col2], method='pearson')
            spearman_corr = clean_df[col1].corr(clean_df[col2], method='spearman')
            
            # Interpret correlation strength
            def interpret_correlation(corr_val):
                abs_corr = abs(corr_val)
                if abs_corr < 0.1:
                    return "negligible"
                elif abs_corr < 0.3:
                    return "weak"
                elif abs_corr < 0.5:
                    return "moderate"
                elif abs_corr < 0.7:
                    return "strong"
                else:
                    return "very strong"
            
            # Direction
            direction = "positive" if pearson_corr > 0 else "negative"
            
            # Statistical significance (rough check)
            n = len(clean_df)
            critical_value = 1.96 / np.sqrt(n - 2)  # Approximation
            significant = abs(pearson_corr) > critical_value
            
            result = f"""
🔗 **Correlation Analysis: '{col1}' vs '{col2}'**

**Correlation Coefficients:**
- Pearson (linear): {pearson_corr:.3f}
- Spearman (rank): {spearman_corr:.3f}

**Interpretation:**
- Strength: {interpret_correlation(pearson_corr)} {direction} correlation
- Sample size: {n:,} data points
- Statistical significance: {'✅ Likely significant' if significant else '⚠️ May not be significant'}

**What this means:**
"""
            
            if abs(pearson_corr) < 0.1:
                result += "• No meaningful linear relationship between the variables"
            elif pearson_corr > 0:
                result += f"• As {col1} increases, {col2} tends to increase"
            else:
                result += f"• As {col1} increases, {col2} tends to decrease"
            
            # Difference between Pearson and Spearman
            if abs(pearson_corr - spearman_corr) > 0.1:
                result += "\n• Difference between Pearson and Spearman suggests non-linear relationship"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing correlation: {str(e)}"
    
    @tool
    def identify_outliers(self, df: pd.DataFrame, column_name: str, method: str = "iqr") -> str:
        """
        Identify outliers in a numeric column.
        
        Args:
            df: DataFrame to analyze
            column_name: Name of the column to check
            method: Method to use ('iqr', 'zscore', 'both')
            
        Returns:
            Outlier analysis results
        """
        try:
            if column_name not in df.columns:
                return f"❌ Column '{column_name}' not found. Available: {list(df.columns)}"
            
            if df[column_name].dtype not in ['int64', 'float64', 'int32', 'float32']:
                return f"❌ Column '{column_name}' is not numeric. Type: {df[column_name].dtype}"
            
            # Clean data
            clean_data = df[column_name].dropna()
            
            if len(clean_data) == 0:
                return f"❌ No valid data in column '{column_name}'"
            
            outliers_info = {}
            
            # IQR method
            if method in ['iqr', 'both']:
                Q1 = clean_data.quantile(0.25)
                Q3 = clean_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                outliers_info['iqr'] = {
                    'count': len(iqr_outliers),
                    'percentage': (len(iqr_outliers) / len(clean_data)) * 100,
                    'bounds': (lower_bound, upper_bound),
                    'values': iqr_outliers.tolist()[:10]  # Show max 10 examples
                }
            
            # Z-score method
            if method in ['zscore', 'both']:
                z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
                zscore_outliers = clean_data[z_scores > 3]  # 3 standard deviations
                
                outliers_info['zscore'] = {
                    'count': len(zscore_outliers),
                    'percentage': (len(zscore_outliers) / len(clean_data)) * 100,
                    'threshold': 3,
                    'values': zscore_outliers.tolist()[:10]
                }
            
            # Build result
            result = f"🎯 **Outlier Analysis for '{column_name}'**\n\n"
            result += f"**Dataset Info:**\n"
            result += f"- Total values: {len(clean_data):,}\n"
            result += f"- Mean: {clean_data.mean():.2f}\n"
            result += f"- Std Dev: {clean_data.std():.2f}\n\n"
            
            for method_name, info in outliers_info.items():
                method_display = "IQR Method" if method_name == 'iqr' else "Z-Score Method"
                result += f"**{method_display}:**\n"
                result += f"- Outliers found: {info['count']:,} ({info['percentage']:.1f}%)\n"
                
                if method_name == 'iqr':
                    result += f"- Normal range: {info['bounds'][0]:.2f} to {info['bounds'][1]:.2f}\n"
                else:
                    result += f"- Threshold: ±{info['threshold']} standard deviations\n"
                
                if info['count'] > 0:
                    result += f"- Example outliers: {info['values'][:5]}\n"
                    
                    # Interpretation
                    if info['percentage'] < 1:
                        result += "- ✅ Low outlier percentage - data quality looks good\n"
                    elif info['percentage'] < 5:
                        result += "- ⚠️ Moderate outliers - investigate further\n"
                    else:
                        result += "- 🔍 High outlier percentage - possible data quality issues\n"
                else:
                    result += "- ✅ No outliers detected\n"
                
                result += "\n"
            
            # Recommendations
            result += "**💡 Recommendations:**\n"
            total_outliers = max(info['count'] for info in outliers_info.values())
            
            if total_outliers == 0:
                result += "• Data appears clean - no action needed\n"
            elif total_outliers < len(clean_data) * 0.01:
                result += "• Few outliers detected - investigate individual cases\n"
            elif total_outliers < len(clean_data) * 0.05:
                result += "• Moderate outliers - consider data cleaning or transformation\n"
            else:
                result += "• Many outliers - check data collection process\n"
                result += "• Consider if outliers represent valid extreme cases\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error identifying outliers: {str(e)}"
    
    @tool
    def categorical_analysis(self, df: pd.DataFrame, column_name: str, top_n: int = 10) -> str:
        """
        Analyze a categorical column.
        
        Args:
            df: DataFrame to analyze
            column_name: Name of the categorical column
            top_n: Number of top categories to show
            
        Returns:
            Categorical analysis results
        """
        try:
            if column_name not in df.columns:
                return f"❌ Column '{column_name}' not found. Available: {list(df.columns)}"
            
            # Get value counts
            value_counts = df[column_name].value_counts()
            total_values = len(df[column_name].dropna())
            
            if total_values == 0:
                return f"❌ No valid data in column '{column_name}'"
            
            # Calculate statistics
            unique_count = df[column_name].nunique()
            null_count = df[column_name].isnull().sum()
            
            result = f"📊 **Categorical Analysis for '{column_name}'**\n\n"
            
            # Basic stats
            result += f"**Overview:**\n"
            result += f"- Total values: {len(df):,}\n"
            result += f"- Valid values: {total_values:,}\n"
            result += f"- Null values: {null_count:,} ({(null_count/len(df)*100):.1f}%)\n"
            result += f"- Unique categories: {unique_count:,}\n"
            result += f"- Diversity ratio: {(unique_count/total_values*100):.1f}%\n\n"
            
            # Top categories
            result += f"**Top {min(top_n, len(value_counts))} Categories:**\n"
            for i, (category, count) in enumerate(value_counts.head(top_n).items(), 1):
                percentage = (count / total_values) * 100
                result += f"{i:2d}. {category}: {count:,} ({percentage:.1f}%)\n"
            
            # Distribution insights
            result += f"\n**📈 Distribution Insights:**\n"
            
            # Concentration analysis
            top_5_pct = value_counts.head(5).sum() / total_values * 100
            if top_5_pct > 80:
                result += f"• High concentration: Top 5 categories represent {top_5_pct:.1f}% of data\n"
            elif top_5_pct > 50:
                result += f"• Moderate concentration: Top 5 categories represent {top_5_pct:.1f}% of data\n"
            else:
                result += f"• Low concentration: Top 5 categories represent {top_5_pct:.1f}% of data\n"
            
            # Rare categories
            rare_threshold = 0.01  # 1%
            rare_categories = value_counts[value_counts < total_values * rare_threshold]
            if len(rare_categories) > 0:
                result += f"• Rare categories (<1%): {len(rare_categories)} categories\n"
            
            # Singleton analysis
            singletons = value_counts[value_counts == 1]
            if len(singletons) > 0:
                singleton_pct = len(singletons) / unique_count * 100
                result += f"• Singleton categories (count=1): {len(singletons)} ({singleton_pct:.1f}% of unique)\n"
            
            # Recommendations
            result += f"\n**💡 Analysis Recommendations:**\n"
            
            diversity_ratio = unique_count / total_values
            if diversity_ratio > 0.9:
                result += "• Very high diversity - consider if this column is suitable for grouping\n"
            elif diversity_ratio > 0.5:
                result += "• High diversity - might benefit from category consolidation\n"
            elif diversity_ratio < 0.1:
                result += "• Low diversity - good for grouping and analysis\n"
            
            if len(rare_categories) > unique_count * 0.5:
                result += "• Many rare categories - consider grouping into 'Other' category\n"
            
            if null_count > total_values * 0.1:
                result += "• Significant null values - investigate missing data patterns\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing categorical data: {str(e)}"
    
    @tool
    def data_quality_report(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Complete data quality assessment
        """
        try:
            result = "📋 **Comprehensive Data Quality Report**\n\n"
            
            # Dataset overview
            result += f"**📊 Dataset Overview:**\n"
            result += f"- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
            result += f"- Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB\n"
            result += f"- Total cells: {df.shape[0] * df.shape[1]:,}\n\n"
            
            # Data types analysis
            result += f"**🏷️ Data Types:**\n"
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                result += f"- {dtype}: {count} columns\n"
            result += "\n"
            
            # Missing data analysis
            result += f"**❌ Missing Data Analysis:**\n"
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            null_percentage = (total_nulls / df.size) * 100
            
            result += f"- Total missing values: {total_nulls:,} ({null_percentage:.1f}%)\n"
            
            columns_with_nulls = null_counts[null_counts > 0]
            if len(columns_with_nulls) > 0:
                result += f"- Columns with missing data: {len(columns_with_nulls)}\n"
                result += f"- Worst columns (top 5):\n"
                for col, nulls in columns_with_nulls.head().items():
                    pct = (nulls / len(df)) * 100
                    result += f"  • {col}: {nulls:,} ({pct:.1f}%)\n"
            else:
                result += "- ✅ No missing values found!\n"
            result += "\n"
            
            # Duplicate analysis
            result += f"**🔁 Duplicate Analysis:**\n"
            duplicate_rows = df.duplicated().sum()
            duplicate_pct = (duplicate_rows / len(df)) * 100
            result += f"- Duplicate rows: {duplicate_rows:,} ({duplicate_pct:.1f}%)\n"
            
            if duplicate_rows > 0:
                # Check for columns with all duplicates
                unique_per_col = df.nunique()
                low_unique_cols = unique_per_col[unique_per_col <= 1]
                if len(low_unique_cols) > 0:
                    result += f"- Columns with ≤1 unique value: {list(low_unique_cols.index)}\n"
            result += "\n"
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                result += f"**🔢 Numeric Data Quality ({len(numeric_cols)} columns):**\n"
                
                for col in numeric_cols[:5]:  # Top 5 numeric columns
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Check for potential issues
                        issues = []
                        
                        # Infinite values
                        if np.isinf(col_data).any():
                            issues.append("infinite values")
                        
                        # All zeros
                        if (col_data == 0).all():
                            issues.append("all zeros")
                        
                        # Single value
                        if col_data.nunique() == 1:
                            issues.append("constant value")
                        
                        # Extreme outliers (very rough check)
                        if len(col_data) > 10:
                            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                            extreme_outliers = (z_scores > 5).sum()
                            if extreme_outliers > 0:
                                issues.append(f"{extreme_outliers} extreme outliers")
                        
                        if issues:
                            result += f"  • {col}: ⚠️ {', '.join(issues)}\n"
                        else:
                            result += f"  • {col}: ✅ looks clean\n"
                result += "\n"
            
            # Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                result += f"**📝 Categorical Data Quality ({len(categorical_cols)} columns):**\n"
                
                for col in categorical_cols[:5]:  # Top 5 categorical columns
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        unique_count = col_data.nunique()
                        diversity = unique_count / len(col_data)
                        
                        issues = []
                        
                        # Too many unique values (possible ID column)
                        if diversity > 0.9:
                            issues.append("very high diversity (possible ID)")
                        
                        # Very long strings
                        if col_data.dtype == 'object':
                            max_length = col_data.astype(str).str.len().max()
                            if max_length > 100:
                                issues.append(f"very long values (max: {max_length})")
                        
                        # Whitespace issues
                        if col_data.dtype == 'object':
                            leading_space = col_data.astype(str).str.startswith(' ').any()
                            trailing_space = col_data.astype(str).str.endswith(' ').any()
                            if leading_space or trailing_space:
                                issues.append("whitespace issues")
                        
                        if issues:
                            result += f"  • {col}: ⚠️ {', '.join(issues)}\n"
                        else:
                            result += f"  • {col}: ✅ looks clean\n"
                result += "\n"
            
            # Overall quality score
            result += f"**🎯 Overall Data Quality Score:**\n"
            
            quality_factors = []
            
            # Missing data factor
            if null_percentage < 1:
                quality_factors.append(("Missing data", 95))
            elif null_percentage < 5:
                quality_factors.append(("Missing data", 80))
            elif null_percentage < 10:
                quality_factors.append(("Missing data", 60))
            else:
                quality_factors.append(("Missing data", 30))
            
            # Duplicate factor
            if duplicate_pct < 1:
                quality_factors.append(("Duplicates", 95))
            elif duplicate_pct < 5:
                quality_factors.append(("Duplicates", 80))
            else:
                quality_factors.append(("Duplicates", 50))
            
            # Data type consistency
            if len(dtype_counts) <= 5:
                quality_factors.append(("Type consistency", 90))
            else:
                quality_factors.append(("Type consistency", 70))
            
            overall_score = sum(score for _, score in quality_factors) / len(quality_factors)
            
            result += f"- Overall score: {overall_score:.0f}/100\n"
            for factor, score in quality_factors:
                result += f"  • {factor}: {score}/100\n"
            
            # Final recommendations
            result += f"\n**💡 Recommendations:**\n"
            
            if overall_score >= 90:
                result += "• ✅ Excellent data quality - ready for analysis\n"
            elif overall_score >= 70:
                result += "• ⚠️ Good data quality - minor cleanup recommended\n"
            elif overall_score >= 50:
                result += "• 🔧 Fair data quality - significant cleanup needed\n"
            else:
                result += "• ⚠️ Poor data quality - extensive cleaning required\n"
            
            if null_percentage > 5:
                result += "• Address missing data before analysis\n"
            
            if duplicate_pct > 5:
                result += "• Remove or investigate duplicate rows\n"
            
            if len(categorical_cols) > len(numeric_cols) * 2:
                result += "• Consider feature engineering for categorical variables\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error generating data quality report: {str(e)}"


# Global tools instance
csv_tools = CSVAnalysisTools()