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

## Prompt

Analyze the provided Python script and generate a comprehensive package-level docstring for it.

**Instructions:**

1.  **Read and Understand:** Carefully examine the Python code below to understand its overall purpose, functionality, and main components (functions, classes, etc.).
2.  **Infer Content:** Based *only* on the provided script's content (code structure, function names, logic, any existing comments), determine:
    * A concise summary of what the script does.
    * A slightly more detailed explanation of its process or goal.
    * The key steps or features it implements.
3.  **Generate Docstring:** Create a single Python docstring string with the following structure:
    * **Line 1:** A brief, one-sentence summary of the package/script.
    * **Line 2:** A blank line.
    * **Lines 3+:** A more detailed description (2-4 sentences) explaining what the script does.
    * **Following Lines:** A section titled "Key Features:" followed by a bulleted list (using '-' or '*') outlining the primary capabilities or actions performed by the script.
4.  **Format:** Ensure the entire output is formatted as a standard Python docstring, enclosed in triple quotes (`"""Docstring content goes here"""`). Do not include any other text before or after the docstring itself.

**Python Script Content:** ...

