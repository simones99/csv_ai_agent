"""
Prompt templates optimized for Qwen3 4B model.

This module contains all prompt templates used by the CSV AI Agent:
- Agent system prompts
- Task-specific prompts
- Error recovery prompts
- Output formatting templates
"""

from typing import Dict, Any, List
import pandas as pd


class PromptTemplates:
    """Manages all prompt templates for the CSV AI Agent"""
    
    def __init__(self):
        self.model_name = "Qwen3-4B"
        self.language = "Italian/English"  # Support both languages
        
    def get_agent_prefix(self, df: pd.DataFrame) -> str:
        """
        Main system prompt for the pandas agent.
        
        Optimized for Qwen3 4B:
        - Clear, structured instructions
        - Specific examples
        - Safety guidelines
        - Output format specifications
        """
        
        # Analyze dataset for context
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = list(df.select_dtypes(include=['datetime']).columns)
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        
        prompt = f"""You are an expert Data Analyst AI specialized in CSV data analysis using pandas.

DATASET CONTEXT:
- Dataset name: df
- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
- Memory: {memory_mb:.1f}MB
- Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}
- Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}
- Datetime columns ({len(datetime_cols)}): {datetime_cols[:3]}{'...' if len(datetime_cols) > 3 else ''}

YOUR CAPABILITIES:
🔍 EXPLORATORY DATA ANALYSIS
- Descriptive statistics and distributions
- Missing value analysis
- Outlier detection and analysis
- Data type assessment

📊 STATISTICAL ANALYSIS  
- Correlation analysis
- Hypothesis testing basics
- Trend identification
- Pattern recognition

📈 DATA INSIGHTS
- Automatic insight generation
- Business intelligence findings
- Data quality assessment
- Anomaly detection

🛠️ DATA OPERATIONS
- Filtering and grouping
- Aggregations and summaries
- Data cleaning suggestions
- Feature engineering ideas

RESPONSE GUIDELINES:

✅ DO:
- Always work with the DataFrame 'df'
- Provide clear, actionable insights
- Use specific numbers and percentages
- Explain statistical concepts simply
- Format numbers for readability (1,234.56)
- Include data quality observations
- Suggest next analysis steps

❌ DON'T:
- Modify df permanently without explicit request
- Make assumptions about business context
- Create external files without permission
- Use complex jargon without explanation
- Ignore null values or data quality issues

OUTPUT FORMAT:
Use this structure for comprehensive answers:

🔍 ANALYSIS:
[Your technical analysis here]

📊 KEY FINDINGS:
[Main results and numbers]

💡 INSIGHTS:
[What this means and why it matters]

⚠️ DATA NOTES:
[Any data quality issues or limitations]

🎯 RECOMMENDATIONS:
[Suggested next steps or actions]

Remember: Be precise, helpful, and always consider data quality in your analysis.
"""
        
        return prompt
    
    def get_agent_suffix(self) -> str:
        """
        Suffix prompt that guides response format.
        """
        return """
Begin your analysis now. Remember to:
1. Use the exact DataFrame variable name: df
2. Follow the structured output format above
3. Provide specific, quantified insights
4. Note any data quality issues you observe

Your analysis:
"""

    def get_quick_analysis_prompt(self, query: str, dataset_info: Dict[str, Any]) -> str:
        """
        Prompt for quick, focused analysis.
        
        Args:
            query: User's specific question
            dataset_info: Basic dataset information
        """
        return f"""
Dataset: {dataset_info.get('rows', 0):,} rows, {dataset_info.get('columns', 0)} columns

User Question: {query}

Instructions:
- Answer directly and concisely
- Focus only on what's asked
- Use df as the DataFrame name
- Include specific numbers when relevant
- If you can't answer, explain why clearly

Provide a focused response:
"""

    def get_insight_generation_prompt(self, focus_area: str = "general") -> str:
        """
        Prompt for generating automatic insights.
        
        Args:
            focus_area: Area to focus on ('general', 'quality', 'patterns', 'outliers')
        """
        
        focus_prompts = {
            'general': """
Analyze this dataset and provide 5 interesting insights that would be valuable for decision-making:

1. Data Overview - What type of data is this and what's its scope?
2. Key Patterns - What interesting patterns or trends do you see?
3. Data Quality - Are there any quality issues to be aware of?
4. Statistical Insights - What do the numbers tell us?
5. Business Implications - What actions might these insights suggest?

For each insight, provide:
- The finding
- Supporting evidence (specific numbers)
- Why it matters
""",
            'quality': """
Perform a comprehensive data quality analysis and provide insights about:

1. Completeness - Missing data patterns
2. Consistency - Data format and type issues  
3. Accuracy - Potential errors or outliers
4. Validity - Values that don't make sense
5. Timeliness - If applicable, data freshness

Focus on actionable quality insights that would help improve the data.
""",
            'patterns': """
Look for interesting patterns and relationships in the data:

1. Correlations - Strong relationships between variables
2. Distributions - Unusual or interesting data distributions
3. Segments - Natural groupings in the data
4. Trends - Changes over time (if applicable)
5. Anomalies - Unusual cases that stand out

Explain why each pattern is interesting and what it might indicate.
""",
            'outliers': """
Identify and analyze outliers and anomalies:

1. Statistical Outliers - Values far from normal ranges
2. Business Logic Outliers - Values that don't make business sense
3. Pattern Breaks - Data points that break normal patterns
4. Rare Combinations - Unusual combinations of values
5. Data Entry Errors - Likely mistakes in data collection

For each type, explain the potential impact and recommended actions.
"""
        }
        
        return focus_prompts.get(focus_area, focus_prompts['general'])
    
    def get_error_recovery_prompt(self, error: str, original_query: str, context: str = "") -> str:
        """
        Prompt for recovering from errors gracefully.
        
        Args:
            error: The error that occurred
            original_query: Original user query
            context: Additional context about the situation
        """
        return f"""
An error occurred while processing the query. Please help resolve this:

ORIGINAL QUERY: {original_query}

ERROR ENCOUNTERED: {error}

CONTEXT: {context}

Please:
1. Identify the likely cause of this error
2. Suggest a simpler alternative approach
3. If the query cannot be completed, explain why clearly
4. Recommend what the user should check or modify

Provide a helpful response that guides the user toward a solution.
"""

    def get_visualization_prompt(self, chart_type: str, data_info: Dict[str, Any]) -> str:
        """
        Prompt for generating visualization code.
        
        Args:
            chart_type: Type of chart requested
            data_info: Information about the data structure
        """
        return f"""
Create a {chart_type} visualization for this dataset.

DATASET INFO:
- Columns: {data_info.get('columns', [])}
- Numeric columns: {data_info.get('numeric_cols', [])}
- Categorical columns: {data_info.get('categorical_cols', [])}

REQUIREMENTS:
- Use matplotlib or plotly for visualization
- Include appropriate titles and axis labels
- Handle missing values appropriately
- Make it visually appealing and informative
- Add brief interpretation of what the chart shows

Provide the visualization code and explanation:
"""

    def get_comparison_prompt(self, item1: str, item2: str, comparison_type: str = "statistical") -> str:
        """
        Prompt for comparing two data elements.
        
        Args:
            item1: First item to compare (column, group, etc.)
            item2: Second item to compare
            comparison_type: Type of comparison ('statistical', 'categorical', 'temporal')
        """
        
        comparison_templates = {
            'statistical': f"""
Compare {item1} and {item2} statistically:

1. Descriptive Statistics - Mean, median, std dev, etc.
2. Distribution Shape - Skewness, outliers, normality
3. Variability - Which is more consistent?
4. Relationship - Correlation if applicable
5. Practical Significance - Which differences matter?

Provide specific numbers and clear conclusions.
""",
            'categorical': f"""
Compare the categorical distributions of {item1} and {item2}:

1. Category Counts - Frequency of each category
2. Proportions - Relative distributions
3. Diversity - How spread out are the values?
4. Rare Categories - Uncommon values in each
5. Overlap - Common categories between them

Highlight the most important differences.
""",
            'temporal': f"""
Compare {item1} and {item2} over time:

1. Trends - Overall direction of change
2. Seasonality - Recurring patterns
3. Volatility - Stability vs fluctuation  
4. Growth Rates - Speed of change
5. Turning Points - When did patterns change?

Focus on actionable temporal insights.
"""
        }
        
        return comparison_templates.get(comparison_type, comparison_templates['statistical'])
    
    def get_summary_prompt(self, analysis_type: str = "comprehensive") -> str:
        """
        Prompt for generating dataset summaries.
        
        Args:
            analysis_type: Type of summary ('quick', 'comprehensive', 'executive')
        """
        
        summary_templates = {
            'quick': """
Provide a quick 3-sentence summary of this dataset:
1. What type of data this is and its scope
2. The most important finding or pattern
3. One key insight for decision-making

Keep it concise and actionable.
""",
            'comprehensive': """
Create a comprehensive dataset summary covering:

📊 DATASET PROFILE:
- Data type, scope, and structure
- Size, completeness, and quality

🔍 KEY FINDINGS:
- Most important patterns and insights
- Statistical highlights
- Notable anomalies or outliers

💼 BUSINESS IMPLICATIONS:
- What this data tells us
- Recommended actions
- Areas for further investigation

Structure this as an executive briefing document.
""",
            'executive': """
Create an executive summary for leadership:

🎯 KEY TAKEAWAYS (3-5 bullet points)
- Most critical insights
- Business impact
- Recommended actions

📈 SUPPORTING EVIDENCE
- Key metrics and statistics
- Confidence level in findings

⚠️ CONSIDERATIONS
- Data limitations
- Areas needing attention
- Next steps

Keep it high-level, focused on business value.
"""
        }
        
        return summary_templates.get(analysis_type, summary_templates['comprehensive'])
    
    def get_data_cleaning_prompt(self, issues_found: List[str]) -> str:
        """
        Prompt for data cleaning recommendations.
        
        Args:
            issues_found: List of data quality issues detected
        """
        return f"""
Data quality issues detected: {', '.join(issues_found)}

Provide data cleaning recommendations:

🧹 IMMEDIATE ACTIONS:
- Critical issues that must be addressed
- Step-by-step cleaning procedures
- Code examples where helpful

⚠️ CONSIDERATIONS:
- Potential risks of each cleaning action
- Data that should be preserved
- Business logic to consider

✅ VALIDATION STEPS:
- How to verify cleaning was successful
- Metrics to monitor after cleaning
- Quality checks to implement

Prioritize recommendations by impact and ease of implementation.
"""

    def get_feature_engineering_prompt(self, analysis_goal: str = "general") -> str:
        """
        Prompt for feature engineering suggestions.
        
        Args:
            analysis_goal: Goal of the analysis ('prediction', 'segmentation', 'reporting')
        """
        
        goal_contexts = {
            'prediction': "for building predictive models",
            'segmentation': "for customer/data segmentation analysis", 
            'reporting': "for business reporting and dashboards",
            'general': "for general data analysis"
        }
        
        context = goal_contexts.get(analysis_goal, goal_contexts['general'])
        
        return f"""
Suggest feature engineering opportunities {context}:

🔧 NEW FEATURES TO CREATE:
- Derived metrics and ratios
- Date/time features (if applicable)
- Categorical combinations
- Statistical aggregations

📊 TRANSFORMATIONS:
- Normalization/scaling suggestions
- Encoding recommendations
- Distribution improvements

💡 BUSINESS LOGIC FEATURES:
- Domain-specific calculations
- Business rule implementations
- Risk/score calculations

Provide specific pandas code examples and explain the business value of each suggestion.
"""

    def get_multilingual_prompt(self, language: str = "italian") -> str:
        """
        Adjustment for multilingual responses.
        
        Args:
            language: Target language for responses
        """
        
        language_instructions = {
            'italian': """
ISTRUZIONI LINGUISTICHE:
- Rispondi principalmente in italiano
- Mantieni termini tecnici in inglese quando appropriato
- Usa formatting strutturato per chiarezza
- Spiega concetti statistici in modo semplice
""",
            'english': """
LANGUAGE INSTRUCTIONS:
- Respond in clear, professional English
- Use technical terms appropriately
- Structure responses for readability  
- Explain statistical concepts simply
""",
            'mixed': """
LANGUAGE INSTRUCTIONS:
- Use Italian for explanations and insights
- Keep technical terms and code in English
- Adapt language to user preference
- Be consistent within each response
"""
        }
        
        return language_instructions.get(language, language_instructions['mixed'])


# Global prompt templates instance
prompt_templates = PromptTemplates()