import os
import argparse
import json

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chardet
from gemini_api import call_gemini_api

CONFIG_PATH = "config.json"

def load_config() -> dict:
    """Load the configuration from config.json."""
    enc = detect_encoding(CONFIG_PATH)
    with open(CONFIG_PATH, "r", encoding=enc) as f:
        return json.load(f)

def detect_encoding(path: str) -> str:
    """
    Look at the first few kB and guess an encoding.

    If a UTF‑16 BOM is present we return 'utf-16' immediately; otherwise
    defer to chardet and finally default to 'utf-8'.
    """
    with open(path, "rb") as f:
        raw = f.read(4096)

    # check for a UTF‑16 BOM
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"

    enc = chardet.detect(raw).get("encoding")
    return enc or "utf-8"

def read_csv(path: str, config: dict) -> pd.DataFrame:
    """Read a CSV, handling common encoding issues gracefully.

    The function tries, in order:
    1. utf‑8‑sig (strips BOMs such as the 0xff 0xfe sequence you saw).
    2. encoding detected by chardet.
    3. fallback encoding from config (default 'latin1') with errors='replace'.
    """
    # try to eat a BOM first
    try:
        return pd.read_csv(path, encoding="utf-16")
    except UnicodeDecodeError:
        pass

    enc = detect_encoding(path)
    try:
        return pd.read_csv(path, encoding=enc)
    except UnicodeDecodeError as e:
        fallback = config.get("csv_encoding_fallback", "latin1")
        print(f"warning: {e!r}, retrying with {fallback}")
        return pd.read_csv(path, encoding=fallback, errors="replace")

def make_prompt(df: pd.DataFrame, config: dict) -> str:
    """Create a prompt using the template from config, populated with CSV schema and sample rows."""
    template = config.get("system_prompt", "")
    # the config stores the prompt as a list of lines for readability
    if isinstance(template, list):
        template = "\n".join(template)

    # Build column schema string
    dtypes = df.dtypes.astype(str).to_dict()
    column_schema = "\n\n".join([f"{col}: {dtype}" for col, dtype in dtypes.items()])

    # Get first 5 rows (or less)
    sample_rows = df.head(5)
    example_rows = sample_rows.to_string(index=False)

    # Replace placeholders
    prompt = template.replace("{COLUMN_SCHEMA}", column_schema)
    prompt = prompt.replace("{EXAMPLE_ROWS}", example_rows)

    return prompt

def ask_llm(prompt: str, config: dict) -> str:
    """Call Gemini LLM with the prompt using model from config."""
    model = config.get("model", "gemini-3-flash-preview")
    result = call_gemini_api(prompt, model=model)
    return result

def save_schema(csv_path: str, schema_text: str) -> str:
    schema_path = os.path.splitext(csv_path)[0] + "_schema.txt"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(schema_text)
    return schema_path

def make_documents(
    df: pd.DataFrame,
    schema: str,
    dataset_type: str,
    source_file: str,
) -> list[Document]:
    """
    Turn each row of *df* into a Document using *schema* as a
    `str.format` template.

    The metadata for each document contains all row fields plus the
    supplied *dataset_type* and *source_file* strings.
    """
    documents: list[Document] = []

    for _, row in df.iterrows():
        # ensure missing keys don't raise
        row_dict = row.to_dict()
        row_dict.setdefault("date", "unknown")

        # 1. formatted text
        row_text = schema.format(**row_dict)

        # 2. metadata – use the passed‑in values instead of fixed literals
        metadata = {
            **row_dict,
            "dataset_type": dataset_type,
            "source_file": source_file,
        }

        # 3. assemble document
        doc = Document(page_content=row_text, metadata=metadata)
        documents.append(doc)

    return documents

def main():
    parser = argparse.ArgumentParser(
        description="read a CSV, ask an LLM to describe its schema, save result"
    )
    parser.add_argument(
        "csvfile",
        nargs="?",
        default="sample.csv",
        help="path to the CSV file (default: sample.csv)"
    )
    args = parser.parse_args()

    config = load_config()
    df = read_csv(args.csvfile, config)
    #prompt = make_prompt(df, config)
    #schema = ask_llm(prompt, config)
    #out = save_schema(args.csvfile, schema)
    #print("Schema written to", out)
    
    schema = "The person is named {name}."\
    "They are {age} years old."\
    "The measured height is {height}."\
    "The measured weight is {weight}."
    #docs = make_documents(df, schema, "drone_flights", os.path.basename(args.csvfile))
    # --- Create vector store ---
    # embeddings from gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    #vectorstore = FAISS.from_documents(docs, embeddings)
    # --- Save locally ---
    #vectorstore.save_local("index_folder")

    # --- Load back ---
    vectorstore = FAISS.load_local(
        "index_folder",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"dataset_type": "drone_flights"}
        }
    )

    all_docs = vectorstore.similarity_search(
    "What is Ari age?",
    k=5,
    filter={"dataset_type": "drone_flights"}
    )

    for doc in all_docs:
        print(doc.metadata)
        print(doc.page_content)
        print("---")
if __name__ == "__main__":
    main()