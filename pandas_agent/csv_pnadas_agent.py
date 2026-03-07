import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# 2. Loading the data
df = pd.read_csv("../sample.csv", encoding="utf-16")

# 3. Initializing Gemini model
# Choose the appropriate model (gemini-1.5-flash for speed, gemini-1.5-pro for high accuracy)
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    max_retries=2
)

# 4. Creating the Agent
# I added 'prefix' here to make it an "expert" of your organization
custom_prefix = """
You are an expert data analyst working in a secure corporate environment.
You have access to a pandas dataframe (df). 
Your goal is to provide precise, data-driven answers.
If the user asks for a calculation, write the python code and execute it.
Always verify that the column names exist before running the code.
"""

agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True, 
    allow_dangerous_code=True,  # Critical for running Python code
    prefix=custom_prefix
)

# 5. Example query
query = "How many rows are there in the table and what is the average of the height column?"
try:
    response = agent.run(query)
    print(f"\n--- System Response ---\n{response}")
except Exception as e:
    print(f"Error running the model: {e}")