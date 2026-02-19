import os

# The path to the source code we want to index
CODE_DIR = "./Tax-Calculator/taxcalc"

def search_tax_logic(keyword):
    print(f"--- Searching for Tax Logic: '{keyword}' ---")
    matches = 0
    for root, dirs, files in os.walk(CODE_DIR):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line_no, line in enumerate(lines):
                        if keyword.lower() in line.lower():
                            # Print the file name and the specific line of code
                            print(f"File: {file} (Line {line_no}) -> {line.strip()}")
                            matches += 1
    
    if matches == 0:
        print("No matching logic found.")
    else:
        print(f"\n✅ Found {matches} instances of '{keyword}' logic.")

# Example: Search for how the 'Standard Deduction' is calculated
if __name__ == "__main__":
    search_tax_logic("standard_deduction")
