from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import streamlit as st
from fastembed import TextEmbedding


ROOT = Path(__file__).resolve().parent
QA_PATH = Path(os.getenv("QA_CSV_PATH", ROOT / "QA.csv"))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DEFAULT_MIN_SCORE = float(os.getenv("MIN_SIMILARITY", "0.35"))


def load_qa(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"question", "answer"}.issubset(reader.fieldnames):
            raise ValueError("QA.csv must contain question and answer columns.")
        rows = [
            {key: (value or "").strip() for key, value in row.items() if key}
            for row in reader
        ]
    rows = [row for row in rows if row.get("question") and row.get("answer")]
    if not rows:
        raise ValueError("QA.csv contains no usable question and answer rows.")
    return rows


@st.cache_resource(show_spinner="Loading Chinese semantic search model...")
def load_search_index(csv_path: str, model_name: str):
    rows = load_qa(Path(csv_path))
    model = TextEmbedding(model_name=model_name)
    documents = [f"\u554f\u984c\uff1a{row['question']}\n\u7b54\u6848\uff1a{row['answer']}" for row in rows]
    vectors = np.asarray(list(model.embed(documents)), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return rows, model, vectors


def find_answer(question: str, rows, model, vectors, min_score: float):
    query_vector = np.asarray(next(model.embed([question])), dtype=np.float32)
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
    scores = vectors @ query_vector
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    if best_score < min_score:
        return None, best_score
    return rows[best_index], best_score


st.set_page_config(page_title="\u8eca\u7f8e\u77e5\u8b58\u554f\u7b54", page_icon="\U0001f697", layout="centered")
st.title("\U0001f697 \u8eca\u7f8e\u77e5\u8b58\u554f\u7b54")
st.caption("\u8a62\u554f\u6d17\u8eca\u3001\u934d\u819c\u3001\u8eca\u6f06\u4fdd\u990a\u6216\u6c7d\u8eca\u7f8e\u5bb9\u554f\u984c\uff0c\u5f9e\u77e5\u8b58\u5eab\u5c0b\u627e\u6700\u76f8\u95dc\u7684\u56de\u7b54\u3002")

with st.sidebar:
    st.header("\u8a2d\u5b9a")
    min_score = st.slider(
        "\u76f8\u95dc\u5ea6\u9580\u6abb",
        min_value=0.0,
        max_value=1.0,
        value=min(1.0, max(0.0, DEFAULT_MIN_SCORE)),
        step=0.01,
        help="\u5206\u6578\u4f4e\u65bc\u9580\u6abb\u6642\uff0c\u7cfb\u7d71\u4e0d\u6703\u7d66\u51fa\u4e0d\u5920\u76f8\u95dc\u7684\u77e5\u8b58\u5eab\u56de\u7b54\u3002",
    )
    st.caption(f"Embedding model: `{MODEL_NAME}`")

try:
    qa_rows, embedder, document_vectors = load_search_index(str(QA_PATH), MODEL_NAME)
except Exception as exc:
    st.error(f"\u7121\u6cd5\u8f09\u5165\u554f\u7b54\u8cc7\u6599\u6216\u8a9e\u610f\u6a21\u578b\uff1a{exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("\u8a66\u8a66\uff1a\u934d\u819c\u5f8c\u53ef\u4ee5\u7528\u6d17\u8eca\u6a5f\u55ce\uff1f")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("source"):
            with st.expander("\u53c3\u8003\u7684\u77e5\u8b58\u5eab\u554f\u984c"):
                st.write(message["source"])
                st.caption(f"\u8a9e\u610f\u76f8\u95dc\u5206\u6578\uff1a{message['score']:.3f}")

if question := st.chat_input("\u8f38\u5165\u60f3\u8a62\u554f\u7684\u554f\u984c\u2026"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("\u6b63\u5728\u641c\u5c0b\u76f8\u95dc\u7b54\u6848\u2026"):
            result, score = find_answer(question, qa_rows, embedder, document_vectors, min_score)
        if result is None:
            response = "\u77e5\u8b58\u5eab\u4e2d\u6c92\u6709\u627e\u5230\u8db3\u5920\u76f8\u95dc\u7684\u7b54\u6848\uff0c\u8acb\u63db\u500b\u65b9\u5f0f\u63cf\u8ff0\uff0c\u6216\u806f\u7d61\u5ba2\u670d\u78ba\u8a8d\u3002"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response = result["answer"]
            st.markdown(response)
            with st.expander("\u53c3\u8003\u7684\u77e5\u8b58\u5eab\u554f\u984c"):
                st.write(result["question"])
                st.caption(f"\u8a9e\u610f\u76f8\u95dc\u5206\u6578\uff1a{score:.3f}")
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "source": result["question"], "score": score}
            )
