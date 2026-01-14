"""
Data loader for image selection evaluation with multiple CSV sources.

This module handles loading and merging data from:
1. Inference results CSV (generated summaries)
2. Annotations CSV (ground truth relevant images) 
3. Original dataset CSV (all available images)
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from src.config_utils import load_config


class ImageSelectionDataLoader:
    """
    Data loader for image selection evaluation that merges multiple CSV sources.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the data loader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.img_config = self.config["image_selection"]
        self.batch_config = self.img_config["batch_evaluation"]
        self.verbose = self.img_config.get("verbose", False)
        
        # Load and merge the datasets
        self._load_datasets()
        self._merge_datasets()
        
    def _load_datasets(self):
        """Load the three CSV datasets with robust error handling."""
        if self.verbose:
            print("Loading datasets...")
        
        # Load inference results
        inference_path = self.batch_config["inference_results_path"]
        if not os.path.exists(inference_path):
            raise FileNotFoundError(f"Inference results not found: {inference_path}")
        
        try:
            self.inference_df = pd.read_csv(inference_path)
            if self.verbose:
                print(f"Loaded {len(self.inference_df)} inference results from {inference_path}")
        except Exception as e:
            raise ValueError(f"Error loading inference results from {inference_path}: {e}")
        
        # Load annotations with robust CSV handling
        annotations_path = self.batch_config["annotations_path"]
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        
        try:
            # Try different CSV reading strategies
            self.annotations_df = self._load_csv_robust(annotations_path, "annotations")
            if self.verbose:
                print(f"Loaded {len(self.annotations_df)} annotations from {annotations_path}")
        except Exception as e:
            raise ValueError(f"Error loading annotations from {annotations_path}: {e}")
        
        # Load original dataset
        original_path = self.batch_config["original_dataset_path"]
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original dataset not found: {original_path}")
        
        try:
            self.original_df = pd.read_csv(original_path)
            if self.verbose:
                print(f"Loaded {len(self.original_df)} original dataset entries from {original_path}")
        except Exception as e:
            raise ValueError(f"Error loading original dataset from {original_path}: {e}")
    
    def _load_csv_robust(self, file_path: str, file_type: str) -> pd.DataFrame:
        """
        Load CSV with robust error handling for malformed data.
        
        Args:
            file_path: Path to CSV file
            file_type: Type of file for error messages
            
        Returns:
            Loaded DataFrame
        """
        if self.verbose:
            print(f"Loading {file_type} CSV: {file_path}")
        
        # Strategy 1: Try normal CSV reading
        try:
            df = pd.read_csv(file_path)
            if self.verbose:
                print(f"✅ Successfully loaded {file_type} using standard CSV reader")
            return df
        except pd.errors.ParserError as e:
            if self.verbose:
                print(f"⚠️  Standard CSV reader failed: {e}")
                print(f"Trying robust CSV reading strategies...")
        
                # Strategy 2: Try with different quote handling for JSON fields
        try:
            df = pd.read_csv(file_path, quotechar='"', quoting=1)  # QUOTE_ALL
            df = self._parse_json_columns(df, file_type)
            print(f"✅ Successfully loaded {file_type} with quote handling")
            return df
        except Exception as e:
            print(f"⚠️  Quote handling failed: {e}")

        # Strategy 3: Try JSON-aware parsing for CSV with JSON arrays
        try:
            df = self._load_csv_with_json_arrays(file_path, file_type)
            print(f"✅ Successfully loaded {file_type} with JSON-aware parsing")
            return df
        except Exception as e:
            print(f"⚠️  JSON-aware parsing failed: {e}")

        # Strategy 4: Try with error handling - skip bad lines
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            print(f"✅ Successfully loaded {file_type} by skipping bad lines")
            print(f"⚠️  Some lines may have been skipped due to formatting issues")
            return df
        except Exception as e:
            print(f"⚠️  Skip bad lines failed: {e}")
        
    def _parse_json_columns(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """
        Parse JSON strings in DataFrame columns to Python objects.
        
        Args:
            df: DataFrame to process
            file_type: Type of file for error messages
            
        Returns:
            DataFrame with JSON columns parsed
        """
        import json
        
        # For annotations file, parse relevant_images column if it exists
        if file_type == "annotations" and "relevant_images" in df.columns:
            try:
                df["relevant_images"] = df["relevant_images"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith('[') else x
                )
                print(f"✅ Parsed JSON arrays in relevant_images column")
            except Exception as e:
                print(f"⚠️  Could not parse JSON in relevant_images: {e}")
        
        # For original dataset, parse Image_Paths column if it exists
        if "Image_Paths" in df.columns:
            try:
                df["Image_Paths"] = df["Image_Paths"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith('[') else x
                )
                print(f"✅ Parsed JSON arrays in Image_Paths column")
            except Exception as e:
                print(f"⚠️  Could not parse JSON in Image_Paths: {e}")
        
        return df
    
    def _load_csv_with_json_arrays(self, file_path: str, file_type: str) -> pd.DataFrame:
        """
        Load CSV with special handling for JSON arrays in fields.
        
        Args:
            file_path: Path to CSV file
            file_type: Type of file for error messages
            
        Returns:
            Loaded DataFrame with JSON arrays parsed
        """
        import csv
        import json
        
        print(f"🔧 Using JSON-aware CSV parsing for {file_type}")
        
        # Read CSV manually to handle JSON arrays properly
        rows = []
        headers = None
        
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            # Use csv.reader with proper quoting
            reader = csv.reader(f, quotechar='"', quoting=csv.QUOTE_ALL)
            
            for i, row in enumerate(reader):
                if i == 0:
                    headers = row
                else:
                    rows.append(row)
        
        if not headers:
            raise ValueError("No headers found in CSV")
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        # Parse JSON columns
        df = self._parse_json_columns(df, file_type)
        
        return df

        # Strategy 4: Try reading line by line to identify problematic lines
        try:
            print(f"🔍 Analyzing CSV structure...")
            problematic_lines = self._analyze_csv_structure(file_path)
            
            if problematic_lines:
                print(f"Found {len(problematic_lines)} problematic lines:")
                for line_num in problematic_lines[:5]:  # Show first 5
                    print(f"  Line {line_num}")
                
                # Try reading with error handling
                df = pd.read_csv(file_path, on_bad_lines='skip')
                print(f"✅ Successfully loaded {file_type} by skipping problematic lines")
                return df
            else:
                raise ValueError("Could not identify specific problematic lines")
                
        except Exception as e:
            print(f"⚠️  Line-by-line analysis failed: {e}")
        
        # Strategy 5: Last resort - try with different separator
        try:
            # Check if it might be semicolon-separated
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if ';' in first_line and first_line.count(';') > first_line.count(','):
                    df = pd.read_csv(file_path, sep=';')
                    print(f"✅ Successfully loaded {file_type} using semicolon separator")
                    return df
        except Exception as e:
            print(f"⚠️  Alternative separator failed: {e}")
        
        # If all strategies fail, provide helpful error message
        error_msg = (
            f"Failed to load {file_type} CSV file: {file_path}\n"
            f"This is likely due to malformed CSV data such as:\n"
            f"- Unescaped commas in JSON fields\n"
            f"- Inconsistent number of columns\n"
            f"- Improper quoting\n\n"
            f"Please check the file structure and ensure JSON fields are properly quoted."
        )
        raise ValueError(error_msg)
    
    def _analyze_csv_structure(self, file_path: str) -> List[int]:
        """
        Analyze CSV structure to find problematic lines.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of problematic line numbers
        """
        problematic_lines = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                return problematic_lines
            
            # Get expected number of fields from header
            header = lines[0].strip()
            expected_fields = len(header.split(','))
            print(f"Expected {expected_fields} fields based on header")
            
            # Check each line
            for i, line in enumerate(lines[1:], start=2):  # Start from line 2 (after header)
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Simple field count (this won't handle quoted commas perfectly, but gives an idea)
                field_count = len(line.split(','))
                
                if field_count != expected_fields:
                    problematic_lines.append(i)
                    if len(problematic_lines) <= 3:  # Show details for first few
                        print(f"Line {i}: Expected {expected_fields} fields, found {field_count}")
                        print(f"  Content: {line[:100]}..." if len(line) > 100 else f"  Content: {line}")
        
        except Exception as e:
            print(f"Error analyzing CSV structure: {e}")
        
        return problematic_lines
    
    def _merge_datasets(self):
        """Merge the three datasets on Article_ID/id."""
        if self.verbose:
            print("Merging datasets...")
        
        # Get column names from config
        inf_cols = self.batch_config["columns"]["inference_results"]
        ann_cols = self.batch_config["columns"]["annotations"]  
        orig_cols = self.batch_config["columns"]["original_dataset"]
        
        # Rename ID columns to have a common name for merging
        inference_df = self.inference_df.rename(columns={inf_cols["id_column"]: "article_id"})
        annotations_df = self.annotations_df.rename(columns={ann_cols["id_column"]: "article_id"})
        original_df = self.original_df.rename(columns={orig_cols["id_column"]: "article_id"})
        
        # Start with inference results (these are the articles we have summaries for)
        merged_df = inference_df.copy()
        
        # Merge with annotations (left join - some articles might not have annotations)
        merged_df = merged_df.merge(
            annotations_df[["article_id", ann_cols["gold_images_column"]]], 
            on="article_id", 
            how="left"
        )
        
        # Merge with original dataset (left join - to get all available images)
        merged_df = merged_df.merge(
            original_df[["article_id", orig_cols["image_paths_column"]]], 
            on="article_id", 
            how="left"
        )
        
        self.merged_df = merged_df
        
        if self.verbose:
            print(f"Merged dataset contains {len(self.merged_df)} articles")
            print(f"Articles with gold annotations: {self.merged_df[ann_cols['gold_images_column']].notna().sum()}")
            print(f"Articles with original images: {self.merged_df[orig_cols['image_paths_column']].notna().sum()}")
        
    def parse_image_paths(self, paths_str: Any) -> List[str]:
        """Parse image paths from string (JSON format)."""
        if pd.isna(paths_str) or not paths_str:
            return []
        
        # Debug: Print what we're trying to parse
        if self.verbose:
            print(f"🔍 Parsing image paths: {repr(paths_str)} (type: {type(paths_str)})")
        
        try:
            if isinstance(paths_str, str):
                # Clean up the string first - remove any escaped quotes or malformed JSON
                cleaned_str = paths_str.strip()
                
                # Handle case where string represents a Python list literally
                if cleaned_str.startswith("['") and cleaned_str.endswith("']"):
                    # This is a string representation of a Python list
                    # Use ast.literal_eval for safe evaluation
                    import ast
                    try:
                        result = ast.literal_eval(cleaned_str)
                        if self.verbose:
                            print(f"✅ Parsed using ast.literal_eval: {len(result)} paths")
                        return result
                    except (ValueError, SyntaxError) as e:
                        if self.verbose:
                            print(f"⚠️  ast.literal_eval failed: {e}, trying JSON")
                
                # Try JSON parsing
                result = json.loads(cleaned_str)
                if self.verbose:
                    print(f"✅ Parsed using JSON: {len(result)} paths")
                return result
            else:
                result = list(paths_str)
                if self.verbose:
                    print(f"✅ Converted to list: {len(result)} paths")
                return result
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            if self.verbose:
                print(f"⚠️  Parsing failed ({e}), treating as single path")
            # Fallback: treat as single path
            return [str(paths_str)]
    
    def get_candidate_images(self, row: pd.Series) -> Tuple[List[str], str]:
        """
        Get candidate images for an article based on configuration.
        
        Args:
            row: Row from merged dataset
            
        Returns:
            List of candidate image paths
        """
        use_inference_images = self.batch_config["use_inference_images"]
        inf_cols = self.batch_config["columns"]["inference_results"]
        orig_cols = self.batch_config["columns"]["original_dataset"]
        
        if use_inference_images:
            # Use images from inference results (the K images used during summary generation)
            image_paths = self.parse_image_paths(row.get(inf_cols["image_paths_column"], []))
            source = "inference"
        else:
            # Use all available images from original dataset
            image_paths = self.parse_image_paths(row.get(orig_cols["image_paths_column"], []))
            source = "original"
            
        return image_paths, source
    
    def get_dataset_for_evaluation(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Get the processed dataset ready for evaluation.
        
        Args:
            max_samples: Maximum number of samples to return (None for all)
            
        Returns:
            Processed DataFrame ready for evaluation
        """
        df: pd.DataFrame = self.merged_df.copy()
        
        # Apply max_samples if specified
        if max_samples and len(df) > max_samples:
            random_seed = self.img_config.get("random_seed", 42)
            df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)
            print(f"Sampled {max_samples} articles for evaluation")
        
        # Filter out articles without summaries
        inf_cols = self.batch_config["columns"]["inference_results"]
        summary_col = inf_cols["summary_column"]
        
        before_filter = len(df)
        mask = df[summary_col].notna() & (df[summary_col] != "")
        filtered_df = df[mask].reset_index(drop=True)
        assert isinstance(filtered_df, pd.DataFrame)  # Type hint for linter
        after_filter = len(filtered_df)
        
        if before_filter != after_filter:
            print(f"Filtered out {before_filter - after_filter} articles without summaries")
        
        return filtered_df
    
    def get_evaluation_data(self, article_id: Any) -> Dict[str, Any]:
        """
        Get all evaluation data for a specific article.
        
        Args:
            article_id: Article ID to get data for
            
        Returns:
            Dictionary containing summary, candidate images, and gold images
        """
        row = self.merged_df[self.merged_df["article_id"] == article_id].iloc[0]
        
        # Get column names
        inf_cols = self.batch_config["columns"]["inference_results"]
        ann_cols = self.batch_config["columns"]["annotations"]
        
        # Get summary
        summary = str(row[inf_cols["summary_column"]])
        
        # Get reference summary if available
        reference_summary = None
        if "reference_summary" in row.index and pd.notna(row["reference_summary"]):
            reference_summary = str(row["reference_summary"])
        
        # Get candidate images
        candidate_images, image_source = self.get_candidate_images(row)
        
        # Get gold images
        gold_images = self.parse_image_paths(row.get(ann_cols["gold_images_column"], []))
        
        return {
            "article_id": article_id,
            "summary": summary,
            "reference_summary": reference_summary,
            "candidate_images": candidate_images,
            "gold_images": gold_images,
            "image_source": image_source,
            "num_candidates": len(candidate_images),
            "num_gold": len(gold_images)
        }
    
    def print_dataset_stats(self):
        """Print statistics about the loaded dataset."""
        df = self.merged_df
        inf_cols = self.batch_config["columns"]["inference_results"]
        ann_cols = self.batch_config["columns"]["annotations"]
        orig_cols = self.batch_config["columns"]["original_dataset"]
        
        print("\n" + "=" * 50)
        print("Dataset Statistics")
        print("=" * 50)
        print(f"Total articles: {len(df)}")
        print(f"Articles with summaries: {df[inf_cols['summary_column']].notna().sum()}")
        print(f"Articles with gold annotations: {df[ann_cols['gold_images_column']].notna().sum()}")
        print(f"Articles with inference images: {df[inf_cols['image_paths_column']].notna().sum()}")
        print(f"Articles with original images: {df[orig_cols['image_paths_column']].notna().sum()}")
        
        # Sample statistics
        sample_row = df.iloc[0]
        sample_data = self.get_evaluation_data(sample_row["article_id"])
        
        print(f"\nSample article (ID: {sample_data['article_id']}):")
        print(f"  Summary length: {len(sample_data['summary'])} chars")
        print(f"  Candidate images ({sample_data['image_source']}): {sample_data['num_candidates']}")
        print(f"  Gold images: {sample_data['num_gold']}")
        
        if self.batch_config["use_inference_images"]:
            print(f"\n📋 Configuration: Using inference images (K images used during summary generation)")
        else:
            print(f"\n📋 Configuration: Using all available images from original dataset")
        
        print("=" * 50)
    
    def validate_data_consistency(self) -> Dict[str, Any]:
        """
        Validate data consistency across the three datasets.
        
        Returns:
            Dictionary with validation results
        """
        inf_cols = self.batch_config["columns"]["inference_results"]
        ann_cols = self.batch_config["columns"]["annotations"]
        orig_cols = self.batch_config["columns"]["original_dataset"]
        
        # Get unique article IDs from each dataset
        inf_ids = set(self.inference_df[inf_cols["id_column"]].unique())
        ann_ids = set(self.annotations_df[ann_cols["id_column"]].unique())
        orig_ids = set(self.original_df[orig_cols["id_column"]].unique())
        
        # Find overlaps and missing IDs
        inf_ann_overlap = inf_ids.intersection(ann_ids)
        inf_orig_overlap = inf_ids.intersection(orig_ids)
        
        missing_annotations = inf_ids - ann_ids
        missing_original = inf_ids - orig_ids
        
        validation_results = {
            "total_inference_articles": len(inf_ids),
            "total_annotation_articles": len(ann_ids),
            "total_original_articles": len(orig_ids),
            "inference_with_annotations": len(inf_ann_overlap),
            "inference_with_original": len(inf_orig_overlap),
            "missing_annotations": len(missing_annotations),
            "missing_original": len(missing_original),
            "coverage_annotations": len(inf_ann_overlap) / len(inf_ids) if inf_ids else 0,
            "coverage_original": len(inf_orig_overlap) / len(inf_ids) if inf_ids else 0
        }
        
        return validation_results


def load_image_selection_data(config_path: str, max_samples: Optional[int] = None) -> Tuple[ImageSelectionDataLoader, pd.DataFrame]:
    """
    Convenience function to load and prepare image selection data.
    
    Args:
        config_path: Path to configuration file
        max_samples: Maximum number of samples to return
        
    Returns:
        Tuple of (data_loader, evaluation_dataframe)
    """
    loader = ImageSelectionDataLoader(config_path)
    df = loader.get_dataset_for_evaluation(max_samples)
    
    return loader, df 