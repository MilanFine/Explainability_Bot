# 📦 SKU Optimization Explainability Agent

This project builds a conversational assistant to explain decisions in SKU-level and load-level optimization for supply chain operations. The agent intelligently routes questions to appropriate tools or a retrieval-augmented generation (RAG) chain, using pre- and post-optimization Excel files.

---

## 🚀 Features

- ✅ Smart Query Routing using LLM (Groq + DeepSeek)
- ✅ Structured DataFrame Querying
- ✅ Tools for:
  - `get_sku_info`
  - `get_skus_in_load`
  - `compare_skus`
  - `get_alternate_skus`
  - `structured_query_tool` (for count/aggregation)
  - `rag_chain_tool` (for general QA)
- ✅ LangGraph-based decision flow with LLM + ToolNode
- ✅ Streamlit session integration for persistent memory
- ✅ RAG backed by FAISS + HuggingFace embeddings on Excel files

---

## 📁 Project Structure
├── pre_opti_model.xlsx # Pre-optimization input file
├── post_opti_model.xlsx # Post-optimization output file
├── app.py # Frontend code
├── chatbot copy.py # Main logic file
├── README.md # This file
├── requirements.txt # All dependencies
├── htmltemplates.py # Templates for frontend styling

## 📁 Install Dependencies
pip install -r requirements.txt

## Create .env file
Make sure to create a .env file and add the API keys that is in use

## To run the app
streamlit run app.py