import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

st.title("📊 Enterprise Data Agent")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # הגדרת ה-Agent עם יכולת לראות את שלבי הביניים
    llm = ChatOpenAI(model="gpt-4o", temperature=0) 
    agent = create_pandas_dataframe_agent(llm, 
    df, 
    verbose=True, 
    allow_dangerous_code=True, 
    return_intermediate_steps=True)

    query = st.text_input("Ask something (e.g., 'Show a bar chart of CPU usage'):")

    if query:
        with st.spinner("Analyzing..."):
            # הרצה ששומרת את שלבי הביניים (כולל הקוד)
            response = agent.invoke(query)
            
            # 1. הצגת התשובה הטקסטואלית
            st.subheader("Result:")
            st.write(response["output"])

            # 2. שליפת קוד הפייתון מתוך שלבי הביניים והצגתו
            with st.expander("View Logic (Python Code)"):
                # מחלץ את הקוד שה-Agent הריץ
                steps = response.get("intermediate_steps", [])
                if steps:
                    action_log = steps[0][0].tool_input # שליפת הקוד מהצעד הראשון
                    st.code(action_log, language="python")

            # 3. הצגת גרף אם נוצר כזה בזיכרון (Matplotlib)
            if plt.get_fignums():
                st.pyplot(plt.gcf())
                plt.clf() # ניקוי הזיכרון