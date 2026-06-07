import polars as pl
from pathlib import Path
import argparse

def convert_jsonl_to_parquet(input_folder: str, output_folder: str = None):
    input_path = Path(input_folder)
    
    if output_folder is None:
        output_path = input_path
    else:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_path}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL file(s)")
    
    for jsonl_file in jsonl_files:
        try:
            df = pl.read_ndjson(jsonl_file)
            
            core_columns = ["comment_id", "label"]
            optional_columns = ["source_file", "source_row", "post_id"]
            
            keep_columns = [col for col in core_columns + optional_columns if col in df.columns]
            
            if not keep_columns:
                print(f"✗ No core columns found in {jsonl_file.name}")
                continue
            
            df = df.select(keep_columns)
            
            parquet_file = output_path / f"{jsonl_file.stem}.parquet"
            df.write_parquet(parquet_file)
            print(f"✓ Converted: {jsonl_file.name} -> {parquet_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {jsonl_file.name}: {e}")

def convert_and_merge_to_single_parquet(input_folder: str, output_file: str = "combined_output.parquet"):
    input_path = Path(input_folder)
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_path}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL file(s)")
    
    all_dfs = []
    core_columns = ["comment_id", "label"]
    
    for jsonl_file in jsonl_files:
        try:
            df = pl.read_ndjson(jsonl_file)
            
            optional_columns = ["source_file", "source_row", "post_id"]
            keep_columns = [col for col in core_columns + optional_columns if col in df.columns]
            
            if not keep_columns:
                print(f"✗ No core columns found in {jsonl_file.name}")
                continue
            
            df = df.select(keep_columns)
            
            for col in core_columns:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
            
            df = df.select(core_columns + [col for col in optional_columns if col in df.columns])
            
            all_dfs.append(df)
            print(f"✓ Loaded: {jsonl_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {jsonl_file.name}: {e}")
    
    if all_dfs:
        combined_df = pl.concat(all_dfs, how="diagonal_relaxed")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.write_parquet(output_path)
        print(f"\n✓ Combined {len(all_dfs)} files into {output_path}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Columns: {combined_df.columns}")
    else:
        print("No data to combine")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL files to Parquet")
    parser.add_argument("input_folder", help="Input folder containing JSONL files")
    parser.add_argument("--output", "-o", help="Output folder for Parquet files (default: same as input)")
    parser.add_argument("--merge", "-m", action="store_true", help="Merge all files into a single Parquet")
    parser.add_argument("--output-file", "-f", default="combined_output.parquet", help="Output filename when merging (default: combined_output.parquet)")
    
    args = parser.parse_args()
    
    if args.merge:
        convert_and_merge_to_single_parquet(args.input_folder, args.output_file)
    else:
        convert_jsonl_to_parquet(args.input_folder, args.output)