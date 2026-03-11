import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import matplotlib.pyplot as plt
import os


st.title("📊 Enterprise Data Agent")

# 2. Loading the data
#df = pd.read_csv("../sample.csv", encoding="utf-16")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
	df = pd.read_csv(uploaded_file, encoding="utf-16")

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
		prefix=custom_prefix,
		return_intermediate_steps=True
	)

	# 5. Example query
	default_query = "How many rows are there in the table and what is the average of the height column?"
	query = st.text_input("Ask something (e.g., 'Show a bar chart of CPU usage'):")
	if query=='a':
		query = default_query

	if query:
		with st.spinner("Analyzing..."):
			try:
				response = agent.invoke(query)
				print(f"\n--- System Response ---\n{response}")
			except Exception as e:
				print(f"Error running the model: {e}")
			
			st.subheader("Result:")
			st.write(response["output"])
			# 2. Extract the Python code from the intermediate steps and display it
			with st.expander("View Logic (Python Code)"):
				# Extracts the code that the agent ran
				steps = response.get("intermediate_steps", [])
				if steps:
					action_log = steps[0][0].tool_input # # 2. Extracting first action's code
					st.code(action_log, language="python")
			
			# 3. Display a plot if one was created in memory (Matplotlib)
			if plt.get_fignums():
				st.pyplot(plt.gcf())
				plt.clf() # clear the memory
