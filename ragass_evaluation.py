"""
Multi-LLM SQL Query Evaluation System with Excel Export
Compares different LLMs and generates comprehensive evaluation spreadsheet
"""

import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import ContextPrecision, Faithfulness, FactualCorrectness
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from nltk.translate.bleu_score import sentence_bleu
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': '3306',
    'username': 'root',
    'password': 123,
    'database_schema': 'text_to_sql'
}

# LLM configurations to test
LLM_CONFIGS = [
    {
        'name': 'Llama 3.1 8B',
        'model': 'llama3.1-8b',
        'api_key': 'csk-8yf4hrx6pdpyvem4ctthrm3rxhwtw93hv6pxkcm5e4nrtpwy',
        'temperature': 0.3,
        'provider': 'cerebras'
    },
    {
        'name': 'Llama 4 scout',
        'model': 'llama-4-scout-17b-16e-instruct',
        'api_key': 'csk-8yf4hrx6pdpyvem4ctthrm3rxhwtw93hv6pxkcm5e4nrtpwy',
        'temperature': 0.3,
        'provider': 'cerebras'
    },
    {
        'name': 'GPT-OSS 120B',
        'model': 'gpt-oss-120b',
        'api_key': 'csk-8yf4hrx6pdpyvem4ctthrm3rxhwtw93hv6pxkcm5e4nrtpwy',
        'temperature': 0.3,
        'provider': 'cerebras'
    },
    {
        'name': 'Qwen 3 32B',
        'model': 'qwen-3-32b',
        'api_key': 'csk-8yf4hrx6pdpyvem4ctthrm3rxhwtw93hv6pxkcm5e4nrtpwy',
        'temperature': 0.3,
        'provider': 'cerebras'
    },
    

]

# Test queries
TEST_QUERIES = [
    {
        'question': 'What was the budget of Product 12',
        'reference': "SELECT `2017_budget` FROM `2017_budgets` WHERE `product_name` = 'Product 12';"
    },
    {
        'question': 'What are the names of all products in the products table?',
        'reference': "SELECT `product_name` FROM products;"
    },
    {
        'question': 'List all customer names from the customers table.',
        'reference': "SELECT `customer_names` FROM customers;"
    },
    {
        'question': 'Find the name and state of all regions in the regions table.',
        'reference': "SELECT name, state FROM regions;"
    },
    {
        'question': 'What is the name of the customer with Customer Index = 1',
        'reference': "SELECT `customer_names` FROM customers WHERE `customer_index` = 1;"
    }
]

# ============================================
# DATABASE SETUP
# ============================================

def setup_database():
    """Initialize database connection"""
    mysql_uri = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database_schema']}"
    db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=1)
    return db

# ============================================
# LLM INITIALIZATION
# ============================================

def create_llm(config):
    """Create LLM based on configuration"""
    if config['provider'] == 'cerebras':
        return ChatCerebras(
            api_key=config['api_key'],
            model=config['model'],
            temperature=config['temperature'],
            max_tokens=2000
        )
    # Add more providers here (OpenAI, Anthropic, etc.)
    return None

# ============================================
# SQL QUERY PROCESSING
# ============================================

def get_schema(db):
    """Get database schema"""
    return db.get_table_info()

def extract_sql_query(response):
    """Extract SQL query from LLM response"""
    # Pattern to match SQL queries
    pattern = r"```sql\s*(.*?)\s*```|SELECT\s+.*?;"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        query = match.group(1) if match.group(1) else match.group(0)
        return query.strip() if query else None
    
    # Fallback: find SELECT statement
    select_pattern = r"SELECT\s+.*"
    select_match = re.search(select_pattern, response, re.IGNORECASE | re.DOTALL)
    if select_match:
        query = select_match.group(0).strip()
        if not query.endswith(';'):
            query += ';'
        return query
    
    return None

def create_sql_chain(llm, db):
    """Create SQL query generation chain"""
    template = """Based on the table schema below, write a SQL query that would answer the user's question:
Remember : Only provide me the sql query dont include anything else.
           Provide me sql query in a single line dont add line breaks.
Table Schema:
{schema}

Question: {question}
SQL Query:
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        RunnablePassthrough.assign(schema=lambda _: get_schema(db))
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    return chain

# ============================================
# EVALUATION METRICS
# ============================================

def calculate_bleu_score(reference, hypothesis):
    """Calculate BLEU score"""
    try:
        return sentence_bleu([reference.split()], hypothesis.split())
    except:
        return 0.0

def calculate_rouge_score(reference, hypothesis):
    """Calculate ROUGE score"""
    try:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)[0]
        return scores["rouge-1"]["f"]
    except:
        return 0.0

def calculate_exact_match(reference, response):
    """Check if query exactly matches reference"""
    # Normalize queries for comparison
    ref_normalized = ' '.join(reference.lower().split())
    resp_normalized = ' '.join(response.lower().split())
    return 1.0 if ref_normalized == resp_normalized else 0.0

def calculate_syntax_validity(query, db):
    """Check if SQL query is syntactically valid"""
    try:
        db.run(query)
        return 1.0
    except:
        return 0.0

# ============================================
# MAIN EVALUATION FUNCTION
# ============================================

def evaluate_llm(llm_config, db, test_queries):
    """Evaluate a single LLM on test queries"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {llm_config['name']}")
    print(f"{'='*60}")
    
    # Create LLM and chain
    llm = create_llm(llm_config)
    sql_chain = create_sql_chain(llm, db)
    
    results = []
    
    for idx, test in enumerate(test_queries, 1):
        print(f"\nQuery {idx}/{len(test_queries)}: {test['question']}")
        
        # Generate SQL query
        response = sql_chain.invoke({"question": test['question']})
        extracted_query = extract_sql_query(response)
        
        if extracted_query is None:
            extracted_query = response.strip()
            if not extracted_query.endswith(';'):
                extracted_query += ';'
        
        print(f"Generated: {extracted_query}")
        
        # Calculate metrics
        bleu = calculate_bleu_score(test['reference'], extracted_query)
        rouge = calculate_rouge_score(test['reference'], extracted_query)
        exact_match = calculate_exact_match(test['reference'], extracted_query)
        syntax_valid = calculate_syntax_validity(extracted_query, db)
        
        results.append({
            'Question': test['question'],
            'Reference': test['reference'],
            'Generated': extracted_query,
            'BLEU': bleu,
            'ROUGE-1': rouge,
            'Exact Match': exact_match,
            'Syntax Valid': syntax_valid
        })
    
    return results

def evaluate_with_ragas(llm_config, db, test_queries, results):
    """Run RAGAS evaluation"""
    print(f"\nRunning RAGAS evaluation for {llm_config['name']}...")
    
    # Create evaluator LLM
    llm = create_llm(llm_config)
    evaluator_llm = LangchainLLMWrapper(llm)
    
    # Create samples
    context = get_schema(db)
    samples = []
    
    for i, test in enumerate(test_queries):
        sample = SingleTurnSample(
            user_input=test['question'],
            retrieved_contexts=[context],
            response=results[i]['Generated'],
            reference=test['reference']
        )
        samples.append(sample)
    
    # Create dataset
    dataset = EvaluationDataset(samples=samples)
    
    # Evaluate
    try:
        ragas_result = evaluate(
            dataset=dataset,
            metrics=[ContextPrecision(), Faithfulness(), FactualCorrectness()],
            llm=evaluator_llm
        )
        
        print(f"RAGAS result type: {type(ragas_result)}")
        print(f"RAGAS result: {ragas_result}")
        
        # Convert EvaluationResult to dict
        metrics_dict = {}
        
        # Try different ways to extract data
        if hasattr(ragas_result, 'to_pandas'):
            # Method 1: If it has a to_pandas method, use it
            try:
                df = ragas_result.to_pandas()
                metrics_dict = df.mean().to_dict()
                print(f"Extracted from pandas: {metrics_dict}")
            except:
                pass
        
        if not metrics_dict and hasattr(ragas_result, '__dict__'):
            # Method 2: Get from __dict__
            for key, value in ragas_result.__dict__.items():
                if not key.startswith('_') and isinstance(value, (int, float)):
                    metrics_dict[key] = value
            print(f"Extracted from __dict__: {metrics_dict}")
        
        if not metrics_dict:
            # Method 3: Try to convert directly to dict
            try:
                if isinstance(ragas_result, dict):
                    metrics_dict = ragas_result
                else:
                    metrics_dict = dict(ragas_result)
                print(f"Converted to dict: {metrics_dict}")
            except:
                pass
        
        # Method 4: Access specific attributes
        if not metrics_dict:
            try:
                metrics_dict = {
                    'context_precision': getattr(ragas_result, 'context_precision', 'N/A'),
                    'faithfulness': getattr(ragas_result, 'faithfulness', 'N/A'),
                    'factual_correctness(mode=f1)': getattr(ragas_result, 'factual_correctness', 'N/A')
                }
                print(f"Extracted from attributes: {metrics_dict}")
            except:
                pass
        
        return metrics_dict if metrics_dict else {}
            
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

# ============================================
# EXCEL EXPORT
# ============================================

def create_excel_report(all_results, output_file='llm_evaluation_report.xlsx'):
    """Create comprehensive Excel report"""
    print(f"\nCreating Excel report: {output_file}")
    
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Sheet 1: Summary Comparison
            summary_data = []
            for llm_name, data in all_results.items():
                results = data['results']
                ragas = data.get('ragas', {})
                
                # Safely extract RAGAS scores
                def safe_get_score(ragas_dict, key, default='N/A'):
                    try:
                        if isinstance(ragas_dict, dict):
                            val = ragas_dict.get(key, default)
                            if isinstance(val, (int, float)):
                                return round(val, 4)
                            return val
                        return default
                    except:
                        return default
                
                summary_data.append({
                    'LLM': llm_name,
                    'Avg BLEU': round(sum(r['BLEU'] for r in results) / len(results), 4),
                    'Avg ROUGE-1': round(sum(r['ROUGE-1'] for r in results) / len(results), 4),
                    'Exact Match Rate': round(sum(r['Exact Match'] for r in results) / len(results), 4),
                    'Syntax Valid Rate': round(sum(r['Syntax Valid'] for r in results) / len(results), 4),
                    'Context Precision': 'Not Applicable (SQL Task)',
                    'Faithfulness': 'Not Applicable (SQL Task)',
                    'Factual Correctness': 'Not Applicable (SQL Task)'

                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Detailed Results for each LLM
            for llm_name, data in all_results.items():
                results_df = pd.DataFrame(data['results'])
                sheet_name = llm_name[:31]  # Excel sheet name limit
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Sheet 3: Side-by-side Query Comparison
            comparison_data = []
            for i, query in enumerate(TEST_QUERIES):
                row = {'Question': query['question'], 'Reference': query['reference']}
                for llm_name, data in all_results.items():
                    row[f"{llm_name} - Generated"] = data['results'][i]['Generated']
                    row[f"{llm_name} - BLEU"] = round(data['results'][i]['BLEU'], 4)
                    row[f"{llm_name} - ROUGE"] = round(data['results'][i]['ROUGE-1'], 4)
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel(writer, sheet_name='Query Comparison', index=False)
            
            # Sheet 4: Raw Data
            raw_data = []
            for llm_name, data in all_results.items():
                for result in data['results']:
                    raw_data.append({
                        'LLM': llm_name,
                        **result
                    })
            
            raw_df = pd.DataFrame(raw_data)
            raw_df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        print(f"✓ Excel report saved: {output_file}")
        
    except Exception as e:
        print(f"Error creating Excel report: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Save as CSV
        print("\nFalling back to CSV export...")
        try:
            summary_df.to_csv('summary.csv', index=False)
            print("✓ Saved summary.csv")
            
            for llm_name, data in all_results.items():
                results_df = pd.DataFrame(data['results'])
                filename = llm_name.replace(' ', '_').lower() + '.csv'
                results_df.to_csv(filename, index=False)
                print(f"✓ Saved {filename}")
        except Exception as csv_error:
            print(f"CSV export also failed: {csv_error}")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    print("="*60)
    print("Multi-LLM SQL Query Evaluation System")
    print("="*60)
    
    # Setup database
    print("\nConnecting to database...")
    db = setup_database()
    print("✓ Database connected")
    
    # Store all results
    all_results = {}
    
    # Evaluate each LLM
    for llm_config in LLM_CONFIGS:
        try:
            # Basic evaluation
            results = evaluate_llm(llm_config, db, TEST_QUERIES)
            
            # RAGAS evaluation
            ragas_result = evaluate_with_ragas(llm_config, db, TEST_QUERIES, results)
            
            all_results[llm_config['name']] = {
                'results': results,
                'ragas': ragas_result if ragas_result else {}
            }
            
            print(f"\n✓ {llm_config['name']} evaluation complete")
            
        except Exception as e:
            print(f"\n✗ Error evaluating {llm_config['name']}: {e}")
            continue
    
    # Create Excel report
    if all_results:
        create_excel_report(all_results)
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)
    else:
        print("\n✗ No results to export")

if __name__ == "__main__":
    main()