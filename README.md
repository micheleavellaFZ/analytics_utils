# Analytics Utils

## DatabaseConnection

```python
from analytics_utils import DatabaseConnection
import pandas as pd

# Best way to use it:
# The 'with' statement automatically handles connection, disconnection,
# commit (on success) and rollback (on exception).
with DatabaseConnection(DB_CONFIG_PATH) as conn:
    conn.run_query("DROP TABLE IF EXISTS employees") # run query
    conn.run_query("""
        CREATE TABLE employees (
          id SERIAL PRIMARY KEY,
          name VARCHAR(100) NOT NULL
        );
        INSERT INTO employees (name) VALUES ('John Doe');
    """)
    df_up = pd.DataFrame([{"id": 1, "name": "michele"}])
    conn.upload_df_into_db("employees", df_up, "ON CONFLICT (id) DO NOTHING") # upload to database, third arg is optional
    df = conn.get_df_from_query("SELECT * FROM employees") # fetch dataframe from db
    conn.run_query("DROP TABLE IF EXISTS employees")
```

## send_email 

```python
from analytics_utils import send_email

# send email with 'airflow.automation@fiscozen.it'
send_email(
    obj="prova",
    body="PROVAAAAA",
    receivers_emails=["michele.avella@fiscozen.it", "michele.avella.98@gmail.com"],
    smt_pass="asdj asdj jkfs asdj",
)
```

## compare_dataframe

```python
from analytics_utils import compare_dataframe

df1 = pd.DataFrame(...)
df2 = pd.DataFrame(...)

comparison = compare_dataframe(df1, df2)
print(comparison.summary_message()) # print summary 

# comparison data
comparison.len_left                 # len of df1
comparison.len_right                # len of df2 
comparison.rows_only_in_left        # number of rows in df1 not in df2
comparison.rows_only_in_right       # number of rows in df2 not in df1 
comparison.diff_dataframe           # dataframe with the rows that are different 
```
