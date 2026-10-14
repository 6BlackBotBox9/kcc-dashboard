import os

# Project root directory
PROJECT_ROOT = r"C:\Users\Priyaranjan\Downloads\kcc_dashboard\kcc_dashboard"

def search_keyword_in_codebase():
    target = input("Enter the filename or keyword to search (e.g., schemes): ").strip()
    
    if not target:
        print("No search term entered. Exiting.")
        return

    print(f"\nSearching for '{target}' in all .py files...\n" + "=" * 60)
    found_matches = 0

    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith('.py') and file != 'check_file_usage.py':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if target.lower() in line.lower():
                                rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                                print(f"File: {rel_path}")
                                print(f"  └─ Line {line_num}: {line.strip()}\n")
                                found_matches += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print("=" * 60)
    if found_matches == 0:
        print(f"RESULT: '{target}' is NOT referenced anywhere in your Python codebase.")
    else:
        print(f"RESULT: Found {found_matches} reference(s) to '{target}'.")

if __name__ == "__main__":
    search_keyword_in_codebase()