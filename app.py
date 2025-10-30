

from flask import Flask, render_template, request, jsonify, session
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import re
import os
import warnings
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, text, inspect
import uuid
warnings.filterwarnings('ignore')

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================
# CONFIGURATION
# ============================================

DB_CONFIG = {
    'host': 'localhost',
    'port': '3306',
    'username': 'root',
    'password': '123',
    'database_schema': 'dynamic_db'
}

LLM_CONFIGS = {
    'llama3.1-8b': {
        'name': 'Llama 3.1 8B',
        'model': 'llama3.1-8b',
        'api_key': CEREBRAS_API_KEY,
        'temperature': 0.3
    },
    'llama-4-scout': {
        'name': 'Llama 4 Scout',
        'model': 'llama-4-scout-17b-16e-instruct',
        'api_key': CEREBRAS_API_KEY,
        'temperature': 0.3
    },
    'gpt-oss-120b': {
        'name': 'GPT-OSS 120B',
        'model': 'gpt-oss-120b',
        'api_key': CEREBRAS_API_KEY,
        'temperature': 0.3
    },
    'qwen-3-32b': {
        'name': 'Qwen 3 32B',
        'model': 'qwen-3-32b',
        'api_key': CEREBRAS_API_KEY,
        'temperature': 0.3
    }
}

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# ============================================
# DATABASE FUNCTIONS
# ============================================

def get_engine():
    """Create SQLAlchemy engine"""
    mysql_uri = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database_schema']}"
    return create_engine(mysql_uri)

def create_database_if_not_exists():
    """Create database if it doesn't exist"""
    try:
        # Connect without database
        mysql_uri = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
        engine = create_engine(mysql_uri)
        
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database_schema']}"))
            conn.commit()
        
        print(f"✓ Database '{DB_CONFIG['database_schema']}' ready")
        return True
    except Exception as e:
        print(f"✗ Database creation failed: {e}")
        return False

def setup_database():
    """Initialize database connection"""
    mysql_uri = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database_schema']}"
    db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=3)
    return db

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_column_name(col_name):
    """Clean column names for SQL compatibility"""
    # Remove special characters and replace spaces with underscores
    col_name = str(col_name).strip()
    col_name = re.sub(r'[^\w\s]', '', col_name)
    col_name = re.sub(r'\s+', '_', col_name)
    col_name = col_name.lower()
    
    # Ensure it doesn't start with a number
    if col_name[0].isdigit():
        col_name = 'col_' + col_name
    
    return col_name

def process_uploaded_file(file, user_id):
    """Process uploaded CSV/Excel file and create database table"""
    try:
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        # Read file based on extension
        if file_ext == 'csv':
            df = pd.read_csv(file)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        else:
            return {'success': False, 'error': 'Unsupported file format'}
        
        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Generate table name from filename
        table_name = clean_column_name(filename.rsplit('.', 1)[0])
        table_name = f"{user_id}_{table_name}"
        
        # Create table in database
        engine = get_engine()
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        return {
            'success': True,
            'table_name': table_name,
            'columns': list(df.columns),
            'row_count': len(df),
            'original_filename': filename
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_user_tables(user_id):
    """Get all tables created by user"""
    try:
        engine = get_engine()
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        
        # Filter tables by user_id prefix
        user_tables = [t for t in all_tables if t.startswith(f"{user_id}_")]
        
        tables_info = []
        for table in user_tables:
            columns = [col['name'] for col in inspector.get_columns(table)]
            original_name = table.replace(f"{user_id}_", "")
            tables_info.append({
                'table_name': table,
                'original_name': original_name,
                'columns': columns
            })
        
        return tables_info
    except Exception as e:
        print(f"Error getting user tables: {e}")
        return []

def delete_user_table(user_id, table_name):
    """Delete a specific table"""
    try:
        # Verify table belongs to user
        if not table_name.startswith(f"{user_id}_"):
            return {'success': False, 'error': 'Unauthorized'}
        
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
            conn.commit()
        
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Initialize database on startup
create_database_if_not_exists()

# ============================================
# LLM FUNCTIONS
# ============================================

def create_llm(config):
    """Create LLM based on configuration"""
    return ChatCerebras(
        api_key=config['api_key'],
        model=config['model'],
        temperature=config['temperature'],
        max_tokens=2000
    )

def get_schema(user_id):
    """Get database schema for user's tables"""
    try:
        db = setup_database()
        inspector = inspect(get_engine())
        all_tables = inspector.get_table_names()
        
        # Filter user's tables
        user_tables = [t for t in all_tables if t.startswith(f"{user_id}_")]
        
        if not user_tables:
            return "No tables uploaded yet. Please upload CSV or Excel files first."
        
        schema_info = []
        for table in user_tables:
            columns = inspector.get_columns(table)
            col_info = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
            original_name = table.replace(f"{user_id}_", "")
            schema_info.append(f"Table: {table} (Original: {original_name})\nColumns: {col_info}")
        
        return "\n\n".join(schema_info)
    except Exception as e:
        return f"Error loading schema: {e}"

def extract_sql_query(response):
    """Extract SQL query from LLM response"""
    pattern = r"```sql\s*(.*?)\s*```|SELECT\s+.*?;"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        query = match.group(1) if match.group(1) else match.group(0)
        return query.strip() if query else None
    
    select_pattern = r"SELECT\s+.*"
    select_match = re.search(select_pattern, response, re.IGNORECASE | re.DOTALL)
    if select_match:
        query = select_match.group(0).strip()
        if not query.endswith(';'):
            query += ';'
        return query
    
    return None

def create_sql_chain(llm, user_id):
    """Create SQL query generation chain"""
    template = """Based on the table schema below, write a SQL query that would answer the user's question.
Remember: Only provide the SQL query, don't include anything else.
Provide the SQL query in a single line, don't add line breaks.

Table Schema:
{schema}

Question: {question}
SQL Query:
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        RunnablePassthrough.assign(schema=lambda _: get_schema(user_id))
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    return chain

def execute_query(query, user_id):
    """Execute SQL query and return results"""
    try:
        # Verify query only accesses user's tables
        query_lower = query.lower()
        
        # Get user's tables
        engine = get_engine()
        inspector = inspect(engine)
        user_tables = [t for t in inspector.get_table_names() if t.startswith(f"{user_id}_")]
        
        # Check if query references any tables
        has_user_table = any(table in query_lower for table in user_tables)
        
        if not has_user_table and user_tables:
            return {'success': False, 'error': 'Query must reference uploaded tables'}
        
        db = setup_database()
        result = db.run(query)
        return {'success': True, 'data': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Render main page"""
    # Generate or get user ID from session
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())[:8]
    
    return render_template('index.html', models=LLM_CONFIGS)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())[:8]
        
        user_id = session['user_id']
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                result = process_uploaded_file(file, user_id)
                results.append(result)
            else:
                results.append({
                    'success': False,
                    'error': f'Invalid file: {file.filename}'
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'tables': get_user_tables(user_id)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tables', methods=['GET'])
def get_tables():
    """Get user's uploaded tables"""
    try:
        if 'user_id' not in session:
            return jsonify({'tables': []})
        
        user_id = session['user_id']
        tables = get_user_tables(user_id)
        
        return jsonify({'success': True, 'tables': tables})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tables/<table_name>', methods=['DELETE'])
def delete_table(table_name):
    """Delete a table"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user_id = session['user_id']
        result = delete_user_table(user_id, table_name)
        
        if result['success']:
            return jsonify({
                'success': True,
                'tables': get_user_tables(user_id)
            })
        else:
            return jsonify(result), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process user query and generate SQL"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Please upload files first'}), 400
        
        user_id = session['user_id']
        data = request.json
        question = data.get('question', '').strip()
        model_key = data.get('model', 'llama3.1-8b')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Check if user has uploaded tables
        user_tables = get_user_tables(user_id)
        if not user_tables:
            return jsonify({'error': 'Please upload CSV or Excel files first'}), 400
        
        if model_key not in LLM_CONFIGS:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Get model config
        config = LLM_CONFIGS[model_key]
        
        # Create LLM and chain
        llm = create_llm(config)
        sql_chain = create_sql_chain(llm, user_id)
        
        # Generate SQL query
        response = sql_chain.invoke({"question": question})
        extracted_query = extract_sql_query(response)
        
        if extracted_query is None:
            extracted_query = response.strip()
            if not extracted_query.endswith(';'):
                extracted_query += ';'
        
        # Execute query
        execution_result = execute_query(extracted_query, user_id)
        
        return jsonify({
            'success': True,
            'model': config['name'],
            'question': question,
            'sql_query': extracted_query,
            'execution': execution_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/schema', methods=['GET'])
def get_db_schema():
    """Get database schema"""
    try:
        if 'user_id' not in session:
            return jsonify({
                'success': True,
                'schema': 'No files uploaded yet. Please upload CSV or Excel files to get started.'
            })
        
        user_id = session['user_id']
        schema = get_schema(user_id)
        
        return jsonify({'success': True, 'schema': schema})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    models = [{'key': key, 'name': config['name']} for key, config in LLM_CONFIGS.items()]
    return jsonify({'models': models})

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)