# import os

# def map_keywords_in_code(keywords, directory):
#     """
#     Scans the directory for a list of keywords and returns their exact locations.
#     """
#     occurrences = {}
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith((".py", ".xml")):
#                 path = os.path.join(root, file)
#                 with open(path, "r", errors="ignore") as f:
#                     for line_num, line in enumerate(f, 1):
#                         for word in keywords:
#                             if word in line:
#                                 if word not in occurrences:
#                                     occurrences[word] = []

                                        
#                                             # Add the file and line number
#                                             occurrences[word].append(f"{file} (Line {line_num})")
#     return occurrences

import os
def map_keywords_in_code(keywords, project_path):
    # 1. Start with your empty dictionary
    occurrences = {} 

    # 2. Open your files and loops
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                with open(full_path, "r") as f:
                    for line_no, line in enumerate(f, 1):
                        
                        # --- PLACE THE CASE-INSENSITIVE LOGIC HERE ---
                        clean_line = line.lower()
                        
                        for word in keywords:
                            if word.lower() in clean_line:
                                # If this is the first time finding this word, 
                                # make a list for it in our bucket
                                if word not in occurrences:
                                    occurrences[word] = []
                                
                                # Add the file and line number
                                occurrences[word].append(f"{file}:{line_no}")
                                
    return occurrences

# Example usage for your IRS project
tax_keywords = ["StandardDeduction", "FilingStatus", "Blind", "Age65", "Dependents"]
project_path = "/Users/andreabennett/CS/CODESMITH_IRS_PROJECT/Tax-Calculator"
findings = map_keywords_in_code(tax_keywords, project_path)
