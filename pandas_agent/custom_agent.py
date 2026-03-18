import json
import re

# ---------- utils ----------
def normalize_text(text):
    """Convert LLM response to raw string"""
    
    # case: list of dicts כמו אצלך
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)

    # case: dict
    if isinstance(text, dict):
        if "text" in text:
            return text["text"]
        return json.dumps(text)

    return str(text)
    
    
def extract_json(text):
    text = normalize_text(text).strip()

    # remove code blocks
    if "```" in text:
        parts = text.split("```")
        text = parts[-1]

    # direct parse
    try:
        return json.loads(text)
    except:
        pass

    # regex fallback
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None

    return None


SAFE_GLOBALS = {
    "__builtins__": {
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "print": print,
        "abs": abs,
        "round": round,
    }
}

def run_code_safe(code, df):
    local_vars = {"df": df}

    try:
        exec(code, SAFE_GLOBALS, local_vars)
        return local_vars.get("result", "No result variable")
    except Exception as e:
        return f"ERROR: {e}"


# ---------- main agent ----------

def run_df_agent(llm, df, question, max_steps=5):
    prompt = """
You are a data analyst working with a pandas DataFrame called df.

Return ONLY valid JSON in this format:

{
  "thought": "...",
  "action": "python" | "final",
  "code": "...",
  "final_answer": "..."
}

Rules:
- If you need to compute משהו → action="python"
- If you are done → action="final"
- NEVER return both code and final answer
- Python code MUST store output in variable בשם result
"""

    history = []

    for step in range(1, max_steps + 1):
        full_prompt = f"{prompt}\nQuestion: {question}\nHistory: {history}"

        response = llm.invoke(full_prompt)
        text = response.content if hasattr(response, "content") else str(response)

        print("RAW TYPE:", type(text))
        print("RAW:", text)

        data = extract_json(text)

        print(f"\n=== STEP {step} ===")
        print("RAW:", text)

        if not data:
            print("⚠️ JSON parse failed")
            continue

        print("PARSED:", data)

        action = data.get("action")

        if action == "final":
            print("\n✅ FINAL ANSWER:")
            print(data.get("final_answer"))
            return data.get("final_answer")

        elif action == "python":
            code = data.get("code", "")
            print("\n▶ Running code:\n", code)

            result = run_code_safe(code, df)
            print("🧾 Observation:", result)

            history.append({
                "code": code,
                "observation": result
            })

        else:
            print("⚠️ Unknown action")

    return "❌ Failed to reach final answer"