"""
Synthesis and RAG Chain Orchestration.
Constructs strict prompt templates, contextual grounding, and LLM invocation.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = """You are an enterprise AI assistant answering questions using only the provided context.
If the answer cannot be determined strictly from the context, state: 'The provided documentation does not contain sufficient information to answer this question.'

Context:
{context}

Question:
{question}

Answer:"""

def format_docs(docs: List[Any]) -> str:
    """Format retrieved documents with index tags and source metadata."""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else "unknown"
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        formatted.append(f"[Document {idx} | Source: {source}]\n{content}")
    return "\n\n".join(formatted)

def get_rag_chain(retriever: Optional[Any] = None, model_name: str = "gpt-4o", temperature: float = 0.0):
    """
    Constructs an end-to-end LCEL (LangChain Expression Language) RAG chain:
    Retriever (MMR) -> Document Formatting -> Prompt Template -> Chat Model -> Output Parser
    """
    try:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY", "mock-key-for-validation")
        llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)
    except Exception:
        llm = None

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    if retriever is not None and llm is not None:
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    return prompt
