import xml.etree.ElementTree as ET
import os

FACTS_FILE = "./fact-graph/demo/all-facts.xml"

def search_legal_facts(keyword):
    if not os.path.exists(FACTS_FILE):
        print(f"❌ Error: Cannot find {FACTS_FILE}")
        return

    print(f"--- Searching Legal Fact Graph: '{keyword}' ---")
    tree = ET.parse(FACTS_FILE)
    root = tree.getroot()
    
    matches = 0
    # Search through the XML elements for the keyword
    for child in root.iter():
        # Check attributes and text for the keyword
        content = str(child.attrib) + str(child.text)
        if keyword.lower() in content.lower():
            print(f"Found Law Snippet: {ET.tostring(child, encoding='unicode').strip()[:200]}...")
            matches += 1
            
    if matches == 0:
        print("No legal facts found for that topic.")
    else:
        print(f"\n✅ Found {matches} legal definitions for '{keyword}'.")

if __name__ == "__main__":
    # Searching for the same topic to see the 'Law' side
    search_legal_facts("StandardDeduction")