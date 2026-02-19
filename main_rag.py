import code_rag
import fact_graph_rag
import sys

# The "Brain" - helps the RAG find related concepts automatically
RELATED_TOPICS = {
    "Standard Deduction": ["Filing Status", "Tax Year", "Dependency"],
    "Child Tax Credit": ["Dependent", "Adjusted Gross Income"],
    "Capital Gains": ["Asset Type", "Holding Period"]
}

def generate_tax_report(user_input):
    # Determine the full list of topics to search
    # If the user asks for a known topic, add its relatives!
    search_list = [user_input]
    if user_input in RELATED_TOPICS:
        search_list.extend(RELATED_TOPICS[user_input])
    
    with open("ai_context.txt", "w") as f:
        class Logger:
            def write(self, message):
                sys.__stdout__.write(message)
                f.write(message)
            def flush(self): pass

        sys.stdout = Logger()

        print("="*60)
        print(f"RECURSIVE RAG REPORT: {user_input.upper()}")
        print(f"Topics analyzed: {', '.join(search_list)}")
        print("="*60)
        
        for topic in search_list:
            law_key = topic.replace(" ", "")
            code_key = topic.lower().replace(" ", "_")
            
            print(f"\n>>> ANALYZING ENTITY: {topic}")
            print("-" * 30)
            fact_graph_rag.search_legal_facts(law_key)
            code_rag.search_tax_logic(code_key)
        
        sys.stdout = sys.__stdout__

    print(f"\n💾 Recursive context saved to 'ai_context.txt'.")

if __name__ == "__main__":
    # Test the recursion!
    generate_tax_report("Standard Deduction")