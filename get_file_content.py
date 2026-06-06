from pathlib import Path
import os
import argparse

def files_to_markdown(folder_path, output_file="output.md", glob_pattern="*"):
    """
    Print the content of all files in a folder to a markdown file.
    Each file's path becomes a markdown heading (#).

    Args:
        folder_path: Path object or string path to the folder
        output_file: Name of the output markdown file
    """
    folder = Path(folder_path)
    excluded_path = ["svc", "png", "pyc"]

    with open(output_file, 'w', encoding='utf-8') as md_file:
        for file_path in sorted(folder.rglob(glob_pattern)):
            if file_path.is_file():
                try:
                    _, file_path_extension = os.path.splitext(file_path)
                    if file_path_extension in excluded_path:
                        continue
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content) < 10:
                        continue
                    
                    md_file.write(f"# {file_path}\n\n")
                    file_path_extension = file_path_extension[1:]
                    md_file.write(f"``` {file_path_extension}\n")
                    md_file.write(content)
                    md_file.write("\n\n```\n\n")

                except UnicodeDecodeError:
                    md_file.write(f"*[Binary file - could not read content]*\n\n```\n\n")
                except Exception as e:
                    md_file.write(f"*[Error reading file: {e}]*\n\n```\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, help="Input folder to be recursively crawled")
    parser.add_argument("--o", type=str, help="Output markdown")
    parser.add_argument("--gl", type=str, default="*", help="Glob pattern")
    args = parser.parse_args()
    folder_path = Path(args.i)
    files_to_markdown(folder_path, args.o, args.gl)
    print(f"Markdown file created with all file contents from {folder_path}")

if __name__ == "__main__":
    main()
