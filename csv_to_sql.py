import pandas as pd
from sqlalchemy import create_engine

# Replace 'your_data.csv' with the path to your CSV file
csv_file = 'anime_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Drop rows with 'Unknown' values in 'rating' or 'genre' columns
df = df[(df['rating'] != 'Unknown') & (df['genre'] != 'Unknown')]

# MySQL database connection parameters
mysql_user = 'root'
mysql_password = ''
mysql_host = 'localhost'
mysql_db = 'anime'

# Create a SQLAlchemy engine for MySQL connection
engine = create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")

# Specify the table name
table_name = 'anime'

# Insert all rows into the MySQL table
df.to_sql(table_name, engine, index=False, if_exists='replace')

print(f"All rows have been inserted into the MySQL table '{table_name}'.")
