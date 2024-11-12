from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os
import json
from sqlalchemy import create_engine, inspect
from pymongo import MongoClient
from bson.json_util import dumps
from sqlalchemy import text
import logging
logging.basicConfig(level=logging.ERROR)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

PASSWORD = "Test01"

# In-memory storage for authenticated sessions
authenticated_sessions = {}

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if password == PASSWORD:
        session_id = str(uuid.uuid4())
        authenticated_sessions[session_id] = username
        return jsonify({"status": "success", "session_id": session_id, "username": username}), 200
    else:
        return jsonify({"status": "failed", "message": "Invalid password"}), 401

@app.route('/auth/logout', methods=['POST'])
def logout():
    session_id = request.json.get('session_id')
    if session_id in authenticated_sessions:
        del authenticated_sessions[session_id]
        return jsonify({"status": "success", "message": "Logged out successfully"}), 200
    return jsonify({"status": "failed", "message": "Invalid session"}), 401

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process-file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Create the uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            headers = df.columns.tolist()
            data_types = df.dtypes.apply(lambda x: x.name).tolist()
            
            # Correlation matrix for numerical columns
            numeric_df = df.select_dtypes(include=[np.number])
            correlation_matrix = numeric_df.corr().round(2).fillna(0).to_dict()
            
            # NEW CODE: Prepare data for charts
            chart_data = {}
            for header in headers:
                if df[header].dtype in ['int64', 'float64']:
                    data = df[header].tolist()
                    
                    # Scatter plot data
                    scatter_data = [{'x': i, 'y': val} for i, val in enumerate(data)]
                    
                    # Bar chart data
                    unique_values, counts = np.unique(data, return_counts=True)
                    bar_data = {
                        'labels': [str(val) for val in unique_values],
                        'datasets': [{
                            'label': header,
                            'data': counts.tolist(),
                        }]
                    }
                    
                    # Histogram data
                    hist, bin_edges = np.histogram(data, bins='auto')
                    histogram_data = {
                        'labels': [f'{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)],
                        'datasets': [{
                            'label': header,
                            'data': hist.tolist(),
                        }]
                    }
                    
                    chart_data[header] = {
                        'scatter': scatter_data,
                        'bar': bar_data,
                        'histogram': histogram_data
                    }
            
            # MODIFIED: Added chart_data to the result
            result = {
                "headers": [{"name": name, "type": dtype} for name, dtype in zip(headers, data_types)],
                "correlation_matrix": correlation_matrix,
                "chart_data": chart_data
            }
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Attempt to remove the file, but don't raise an error if it fails
            try:
                os.remove(file_path)
            except OSError:
                pass  # Ignore errors when trying to remove the file
    else:
        return jsonify({"error": "File type not allowed"}), 400

# NEW ROUTE: Added to fetch specific chart data if needed
@app.route('/get-chart-data', methods=['POST'])
def get_chart_data():
    data = request.json
    header = data.get('header')
    chart_data = data.get('chart_data')
    
    if not header or not chart_data or header not in chart_data:
        return jsonify({"error": "Invalid header or missing chart data"}), 400
    
    return jsonify(chart_data[header])

def create_db_engine(db_type, username, password, host, port, database):
    if db_type == 'postgresql':
        return create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')
    elif db_type == 'mysql':
        return create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    elif db_type == 'oracle':
        return create_engine(f'oracle+cx_oracle://{username}:{password}@{host}:{port}/{database}')
    elif db_type == 'sqlserver':
        return create_engine(f'mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server')
    else:
        raise ValueError('Unsupported database type')

@app.route('/api/connect', methods=['POST'])
def connect_db():
    data = request.json
    db_type = data['dbType']
    host = data['host']
    port = data['port']
    username = data['username']
    password = data['password']
    database = data['database']

    CLUSTER_NAME = "testCluster01"
    CLUSTER_PWD = "Discover42"

    try:
        engine = create_db_engine(db_type, username, password, host, port, database)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/table-details', methods=['POST'])
def get_table_details():
    data = request.json
    db_type = data['dbType']
    host = data['host']
    port = data['port']
    username = data['username']
    password = data['password']
    database = data['database']
    table_name = data['tableName']

    try:
        engine = create_db_engine(db_type, username, password, host, port, database)
        inspector = inspect(engine)

        # Get column details for the specified table
        columns = inspector.get_columns(table_name)
        column_details = []
        for col in columns:
            if isinstance(col, dict):
                column_details.append({
                    'name': col.get('name', 'Unknown'),
                    'type': str(col.get('type', 'Unknown'))
                })
            else:
                app.logger.warning(f"Unexpected column data type: {type(col)}")

        # Get primary key information
        pk = inspector.get_pk_constraint(table_name)
        primary_key = pk.get('constrained_columns', []) if isinstance(pk, dict) else []

        # Get foreign key information
        fks = inspector.get_foreign_keys(table_name)
        foreign_keys = []
        for fk in fks:
            if isinstance(fk, dict):
                foreign_keys.append({
                    'columns': fk.get('constrained_columns', []),
                    'referredTable': fk.get('referred_table', 'Unknown')
                })

        # Get index information
        indexes = inspector.get_indexes(table_name)
        index_details = []
        for idx in indexes:
            if isinstance(idx, dict):
                index_details.append({
                    'name': idx.get('name', 'Unknown'),
                    'columns': idx.get('column_names', [])
                })

        # Records Count
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            record_count = result.scalar()

        return jsonify({
            'tableName': table_name,
            'columns': column_details,
            'primaryKey': primary_key,
            'foreignKeys': foreign_keys,
            'indexes': index_details,
            'recordCount': record_count
        })

    except Exception as e:
        app.logger.error(f"Error in get_table_details: {str(e)}")
        app.logger.exception("Exception traceback:")
        return jsonify({'error': str(e)}), 500

@app.route('/api/table-records', methods=['POST'])
def get_table_records():
    data = request.json
    db_type = data['dbType']
    host = data['host']
    port = data['port']
    username = data['username']
    password = data['password']
    database = data['database']
    table_name = data['tableName']
    page = data.get('page', 1)
    per_page = data.get('perPage', 100)

    try:
        engine = create_db_engine(db_type, username, password, host, port, database)
        
        with engine.connect() as connection:
            # Get total count of records
            count_query = text(f"SELECT COUNT(*) FROM {table_name}")
            total_records = connection.execute(count_query).scalar()

            # Calculate offset
            offset = (page - 1) * per_page

            # Fetch paginated records
            query = text(f"SELECT * FROM {table_name} LIMIT :limit OFFSET :offset")
            result = connection.execute(query, {"limit": per_page, "offset": offset})
            
            # Convert to list of dictionaries
            records = [dict(row._mapping) for row in result]

        return jsonify({
            'records': records,
            'totalRecords': total_records,
            'page': page,
            'perPage': per_page
        })

    except Exception as e:
        logging.error(f"Error in get_table_details: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
# # MongoDB connection
# CLUSTER_NAME = "testCluster01"
# CLUSTER_PWD = "Discover42"
# uri = f""

# client = MongoClient(uri)
# db = client['user_feedback']  # database name

# # Ensure the collection exists
# if 'userqueries' not in db.list_collection_names():
#     db.create_collection('userqueries')

# collection = db['userqueries']  # get reference to existing collection

# @app.route('/api/submit-query', methods=['POST'])
# def submit_query():
#     try:
#         data = request.json
#         # Validate data
#         if not all(key in data for key in ['name', 'email', 'query']):
#             return jsonify({"error": "Missing required fields"}), 400

#         # Insert data into existing MongoDB collection
#         result = collection.insert_one(data)

#         if result.inserted_id:
#             return jsonify({"message": "Query submitted successfully"}), 200
#         else:
#             return jsonify({"error": "Failed to submit query"}), 500

#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/api/get-queries', methods=['GET'])
# def get_queries():
#     try:
#         queries = list(collection.find())
#         return dumps(queries), 200
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)