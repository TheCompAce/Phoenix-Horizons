import pyodbc
import json
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Export Access database structure to JSON.')
parser.add_argument('input_db', help='Path to the input Access database.')
parser.add_argument('output_json', help='Path to the output JSON file.')
args = parser.parse_args()

# Connect to the Access database
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=' + args.input_db + ';'
)
cnxn = pyodbc.connect(conn_str)
cursor = cnxn.cursor()

# Get the list of tables
tables = cursor.tables(tableType='TABLE').fetchall()

# Initialize an empty dictionary to hold the table structures
table_structures = {}

# Loop through the tables and get their structure
for table in tables:
    table_name = table.table_name
    columns = cursor.columns(table=table_name).fetchall()
    table_structures[table_name] = [{'name': column.column_name, 'type': column.type_name, 'size': column.column_size} for column in columns]

# Write the table structures to a JSON file
with open(args.output_json, 'w') as f:
    json.dump(table_structures, f, indent=4)
