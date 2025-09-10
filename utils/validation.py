"""
Data validation utilities for CSV AI Agent.

This module provides comprehensive validation for:
- CSV file format and structure
- Data type detection and validation
- Memory safety checks
- Security validations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import chardet
import warnings

from config import config


class ValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class CSVValidator:
    """Comprehensive CSV validation for AI analysis"""
    
    def __init__(self):
        self.validation_report: Dict[str, Any] = {}
        
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Complete file validation pipeline.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dict with validation results and recommendations
        """
        file_path = Path(file_path)
        
        self.validation_report = {
            "file_path": str(file_path),
            "valid": False,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "file_info": {},
            "data_info": {}
        }
        
        try:
            # 1. Basic file checks
            self._validate_file_existence(file_path)
            self._validate_file_size(file_path)
            self._validate_file_extension(file_path)
            
            # 2. Content validation
            self._validate_file_encoding(file_path)
            self._validate_csv_structure(file_path)
            
            # 3. Data validation
            df = pd.read_csv(file_path, nrows=1000)  # Sample for validation
            self._validate_data_quality(df)
            self._validate_memory_requirements(file_path, df)
            
            # 4. Final assessment
            self.validation_report["valid"] = len(self.validation_report["errors"]) == 0
            
        except Exception as e:
            self.validation_report["errors"].append(f"Validation failed: {str(e)}")
            self.validation_report["valid"] = False
        
        return self.validation_report
    
    def _validate_file_existence(self, file_path: Path):
        """Check if file exists and is readable"""
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                f.read(1)  # Try to read first character
        except PermissionError:
            raise ValidationError(f"No permission to read file: {file_path}")
        except Exception as e:
            raise ValidationError(f"Cannot read file: {str(e)}")
        
        self.validation_report["file_info"]["readable"] = True
    
    def _validate_file_size(self, file_path: Path):
        """Validate file size against system limits"""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        self.validation_report["file_info"]["size_mb"] = round(file_size_mb, 2)
        
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            self.validation_report["errors"].append(
                f"File too large: {file_size_mb:.1f}MB (max: {config.MAX_FILE_SIZE_MB}MB)"
            )
        elif file_size_mb > config.MAX_FILE_SIZE_MB * 0.8:
            self.validation_report["warnings"].append(
                f"Large file: {file_size_mb:.1f}MB - may cause performance issues"
            )
        
        # Recommend optimal size
        if file_size_mb > 100:
            self.validation_report["recommendations"].append(
                "Consider using a smaller sample of your data for faster analysis"
            )
    
    def _validate_file_extension(self, file_path: Path):
        """Validate file extension"""
        extension = file_path.suffix.lower().lstrip('.')
        
        if extension not in config.ALLOWED_FILE_TYPES:
            self.validation_report["errors"].append(
                f"Unsupported file type: .{extension} (allowed: {', '.join(config.ALLOWED_FILE_TYPES)})"
            )
        
        self.validation_report["file_info"]["extension"] = extension
    
    def _validate_file_encoding(self, file_path: Path):
        """Detect and validate file encoding"""
        try:
            # Detect encoding using chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                encoding_info = chardet.detect(raw_data)
            
            detected_encoding = encoding_info.get('encoding', 'unknown')
            confidence = encoding_info.get('confidence', 0)
            
            self.validation_report["file_info"]["encoding"] = {
                "detected": detected_encoding,
                "confidence": round(confidence, 2)
            }
            
            # Warn about problematic encodings
            if confidence < 0.8:
                self.validation_report["warnings"].append(
                    f"Low encoding detection confidence ({confidence:.1%}). "
                    "File might have encoding issues."
                )
            
            # Test if file is readable as CSV
            try:
                pd.read_csv(file_path, nrows=5, encoding=detected_encoding)
            except UnicodeDecodeError:
                self.validation_report["errors"].append(
                    "File encoding not compatible with CSV parsing"
                )
                
        except Exception as e:
            self.validation_report["warnings"].append(f"Encoding detection failed: {str(e)}")
    
    def _validate_csv_structure(self, file_path: Path):
        """Validate CSV structure and format"""
        try:
            # Try reading first few rows
            sample_df = pd.read_csv(file_path, nrows=5)
            
            # Basic structure validation
            if len(sample_df.columns) == 0:
                self.validation_report["errors"].append("No columns found in CSV")
            
            if len(sample_df) == 0:
                self.validation_report["errors"].append("No data rows found in CSV")
            
            # Check for common CSV issues
            self._check_csv_formatting_issues(file_path, sample_df)
            
            self.validation_report["data_info"]["sample_columns"] = list(sample_df.columns[:10])
            self.validation_report["data_info"]["total_columns"] = len(sample_df.columns)
            
        except pd.errors.EmptyDataError:
            self.validation_report["errors"].append("CSV file is empty")
        except pd.errors.ParserError as e:
            self.validation_report["errors"].append(f"CSV parsing error: {str(e)}")
        except Exception as e:
            self.validation_report["errors"].append(f"CSV structure validation failed: {str(e)}")
    
    def _check_csv_formatting_issues(self, file_path: Path, sample_df: pd.DataFrame):
        """Check for common CSV formatting problems"""
        
        # Check for unnamed columns (often indicates header issues)
        unnamed_cols = [col for col in sample_df.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            self.validation_report["warnings"].append(
                f"Found {len(unnamed_cols)} unnamed columns - possible header issues"
            )
        
        # Check for columns with mostly null values
        high_null_cols = []
        for col in sample_df.columns:
            null_pct = sample_df[col].isnull().sum() / len(sample_df)
            if null_pct > 0.8:
                high_null_cols.append(col)
        
        if high_null_cols:
            self.validation_report["warnings"].append(
                f"Columns with >80% null values: {high_null_cols[:3]}..."
            )
        
        # Check for very long column names
        long_cols = [col for col in sample_df.columns if len(str(col)) > 50]
        if long_cols:
            self.validation_report["warnings"].append(
                f"Very long column names found (may affect display): {len(long_cols)} columns"
            )
    
    def _validate_data_quality(self, sample_df: pd.DataFrame):
        """Validate data quality in sample"""
        quality_info = {}
        
        # Data type distribution
        dtype_counts = sample_df.dtypes.value_counts()
        quality_info["dtypes"] = dict(dtype_counts)
        
        # Null value analysis
        null_counts = sample_df.isnull().sum()
        total_nulls = null_counts.sum()
        quality_info["null_values"] = {
            "total": int(total_nulls),
            "percentage": round((total_nulls / sample_df.size) * 100, 1)
        }
        
        # Duplicate analysis
        duplicate_rows = sample_df.duplicated().sum()
        quality_info["duplicates"] = {
            "count": int(duplicate_rows),
            "percentage": round((duplicate_rows / len(sample_df)) * 100, 1)
        }
        
        # Data type warnings
        if quality_info["null_values"]["percentage"] > 50:
            self.validation_report["warnings"].append(
                f"High percentage of null values: {quality_info['null_values']['percentage']}%"
            )
        
        if quality_info["duplicates"]["percentage"] > 20:
            self.validation_report["warnings"].append(
                f"High percentage of duplicate rows: {quality_info['duplicates']['percentage']}%"
            )
        
        # Check for problematic data types
        if 'object' in quality_info["dtypes"] and quality_info["dtypes"]['object'] > 20:
            self.validation_report["recommendations"].append(
                "Many text columns detected - consider data type optimization"
            )
        
        self.validation_report["data_info"]["quality"] = quality_info
    
    def _validate_memory_requirements(self, file_path: Path, sample_df: pd.DataFrame):
        """Estimate and validate memory requirements"""
        
        # Estimate full dataset size based on sample
        sample_size = len(sample_df)
        sample_memory = sample_df.memory_usage(deep=True).sum()
        
        # Estimate total rows (rough approximation)
        file_size_bytes = Path(file_path).stat().st_size
        estimated_total_rows = (file_size_bytes / (sample_memory / sample_size)) * 0.8  # Conservative
        
        # Estimate memory needed
        estimated_memory_mb = (sample_memory / sample_size) * estimated_total_rows / (1024**2)
        
        memory_info = {
            "estimated_rows": int(estimated_total_rows),
            "estimated_memory_mb": round(estimated_memory_mb, 1),
            "sample_memory_mb": round(sample_memory / (1024**2), 2)
        }
        
        # Memory validation
        if estimated_memory_mb > config.MAX_DATAFRAME_MEMORY_MB:
            self.validation_report["errors"].append(
                f"Dataset too large for memory: {estimated_memory_mb:.1f}MB "
                f"(limit: {config.MAX_DATAFRAME_MEMORY_MB}MB)"
            )
        elif estimated_memory_mb > config.MAX_DATAFRAME_MEMORY_MB * 0.8:
            self.validation_report["warnings"].append(
                f"Large dataset: {estimated_memory_mb:.1f}MB - may strain system resources"
            )
        
        self.validation_report["data_info"]["memory"] = memory_info
    
    def get_validation_summary(self) -> str:
        """Generate human-readable validation summary"""
        if not self.validation_report:
            return "No validation performed"
        
        summary = []
        
        # Status
        status = "✅ VALID" if self.validation_report["valid"] else "❌ INVALID"
        summary.append(f"**Validation Status**: {status}")
        
        # File info
        file_info = self.validation_report.get("file_info", {})
        if "size_mb" in file_info:
            summary.append(f"**File Size**: {file_info['size_mb']}MB")
        
        # Data info
        data_info = self.validation_report.get("data_info", {})
        if "total_columns" in data_info:
            summary.append(f"**Columns**: {data_info['total_columns']}")
        
        # Issues
        errors = self.validation_report.get("errors", [])
        if errors:
            summary.append("\n**❌ Errors:**")
            for error in errors:
                summary.append(f"  • {error}")
        
        warnings = self.validation_report.get("warnings", [])
        if warnings:
            summary.append("\n**⚠️ Warnings:**")
            for warning in warnings:
                summary.append(f"  • {warning}")
        
        recommendations = self.validation_report.get("recommendations", [])
        if recommendations:
            summary.append("\n**💡 Recommendations:**")
            for rec in recommendations:
                summary.append(f"  • {rec}")
        
        return "\n".join(summary)


# Global validator instance
csv_validator = CSVValidator()


def quick_validate_csv(file_path: str) -> Tuple[bool, str]:
    """
    Quick validation for basic checks.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Basic checks
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        if path.stat().st_size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File too large (max: {config.MAX_FILE_SIZE_MB}MB)"
        
        # Try reading first row
        pd.read_csv(file_path, nrows=1)
        
        return True, "File appears valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate an already loaded DataFrame.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Dict with validation results
    """
    validation = {
        "valid": True,
        "issues": [],
        "stats": {},
        "recommendations": []
    }
    
    try:
        # Basic stats
        validation["stats"] = {
            "shape": df.shape,
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
            "null_count": int(df.isnull().sum().sum()),
            "duplicate_count": int(df.duplicated().sum())
        }
        
        # Check for issues
        if validation["stats"]["null_count"] > 0:
            null_pct = (validation["stats"]["null_count"] / df.size) * 100
            if null_pct > 20:
                validation["issues"].append(f"High null percentage: {null_pct:.1f}%")
        
        if validation["stats"]["duplicate_count"] > 0:
            dup_pct = (validation["stats"]["duplicate_count"] / len(df)) * 100
            if dup_pct > 10:
                validation["issues"].append(f"High duplicate percentage: {dup_pct:.1f}%")
        
        # Memory check
        if validation["stats"]["memory_mb"] > config.MAX_DATAFRAME_MEMORY_MB:
            validation["valid"] = False
            validation["issues"].append("DataFrame exceeds memory limit")
        
        # Recommendations
        if len(df.columns) > 50:
            validation["recommendations"].append("Consider selecting only relevant columns for analysis")
        
        if validation["stats"]["memory_mb"] > 100:
            validation["recommendations"].append("Consider data type optimization to reduce memory usage")
        
    except Exception as e:
        validation["valid"] = False
        validation["issues"].append(f"DataFrame validation failed: {str(e)}")
    
    return validation