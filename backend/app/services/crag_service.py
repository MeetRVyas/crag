import re
from typing import List, TypedDict, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.models.crag import Score, KeepOrDrop
from app.services.llm_factory import build_llm
from app.config import settings

# TODO : Logging

# Graph state
class State(TypedDict):
    # Input
    question: str
    
    # Retrieval
    hypothetical_answer: str
    docs: List[Document]

    # Evaluation
    good_docs: List[Document]
    verdict: str # "CORRECT" | "INCORRECT" | "AMBIGUOUS"
    reason: str

    # Web search
    web_query: str
    web_docs: List[Document]
    
    # Refinement
    strips: List[str]          # Decomposed sentences
    kept_strips: List[str]     # Filtered sentences
    refined_context: str       # Final context string
    
    # Output
    answer: str


# Corrective RAG pipeline using LangGraph
class CRAG_Service :
    # Relevance score thresholds
    UPPER_THRESHOLD = 0.7
    LOWER_THRESHOLD = 0.3

    # Max sentences to evaluate during refinement (keeps latency predictable)
    MAX_REFINE_SENTENCES = 20

    # Max characters for the hypothetical answer (used in HyDE prompt)
    HYDE_MAX_TOKENS = 5000

    def __init__(
        self,
        session_id: str,
        retriever,
        model_name: str,
        provider: str = "ollama",
        api_keys: dict = None,
    ):
        self.session_id = session_id
        self.retriever = retriever
        self.api_keys = api_keys or {}

        self.llm = build_llm(
            provider=provider,
            model=model_name,
            api_keys=self.api_keys,
        )

        self.tavily = self._initialize_tavily(settings.TOPIC)
        self.app = self._build_graph()

    def _initialize_tavily(self, topic: str):
        """
        Initialise the Tavily web-search tool.
        Returns None gracefully if no API key is provided,
        the pipeline will simply skip web search.
        """
        api_key = self.api_keys.get("tavily", "")
        if not api_key:
            # We return None here and handle it gracefully in the node
            # Don't raise a error on initialization
            # as the users have a choice to disable web search
            # TODO : Log a warning here
            print("No Tavily API key found — web search will be disabled for this session.")
            return None

        return TavilySearch(
            max_results = 5,
            topic = topic or "general",
            tavily_api_key = api_key
        )

    def _build_graph(self) :
        """Defines the nodes and edges of the LangGraph state machine"""
        workflow = StateGraph(State)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("evaluate", self.evaluate)
        workflow.add_node("rewrite", self.rewrite)
        workflow.add_node("research", self.research)
        workflow.add_node("refine", self.refine)
        workflow.add_node("generate", self.generate)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluate,
            {"refine": "refine", "rewrite": "rewrite"},
        )

        workflow.add_edge("rewrite", "research")
        workflow.add_edge("research", "refine")
        workflow.add_edge("refine", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _route_after_evaluate(self, state: State) -> str:
        """Route to refine (CORRECT) or rewrite (INCORRECT / AMBIGUOUS)."""
        return "refine" if state["verdict"] == "CORRECT" else "rewrite"

    def run(self, question: str) -> dict:
        return self.app.invoke({
            "question":           question,
            "hypothetical_answer": "",
            "docs":               [],
            "good_docs":          [],
            "verdict":            "",
            "reason":             "",
            "web_query":          "",
            "web_docs":           [],
            "strips":             [],
            "kept_strips":        [],
            "refined_context":    "",
            "answer":             "",
        })

    # Nodes

    def retrieve(self, state : State) -> Dict[str, Any] :
        """Node 1 : Retrieve documents using HyDE *Hypothetical Document Embeddings approach"""
        print("RETRIEVE")
        question = state["question"]

        # HyDE Approach
        # The user's query (question) might not be enough to answer the user's question
        # So we ask LLM to give a hypothetical answer without any context or facts
        hyde_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional who gives answers to questions with unknown context.\n"
                "Write a concise, technical passage that answers the question.\n"
                "Return the answer in under {max_tokens} characters.",
            ),
            ("human", "Question: {question}"),
        ])

        chain = hyde_prompt | self.llm
        response = chain.invoke({
            "question": question,
            "max_tokens": self.HYDE_MAX_TOKENS,
        })

        hypothetical_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        docs = self.retriever.invoke(hypothetical_text)

        return {
            "hypothetical_answer": hypothetical_text,
            "docs": docs
        }
    
    def evaluate(self, state: State) -> Dict[str, Any]:
        """Node 2 : Scores each document 0-1"""
        print("EVALUATE")
        question = state["question"]
        docs = state["docs"]

        # Setup scoring chain
        parser = PydanticOutputParser(pydantic_object = Score)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a strict retrieval evaluator for RAG.\n"
                "You will be given ONE retrieved chunk and a question.\n"
                "Return a relevance score in [0.0, 1.0].\n"
                "  1.0 → chunk alone is sufficient to answer fully\n"
                "  0.0 → chunk is irrelevant\n"
                "Be conservative with high scores.\n"
                "Also return a short reason.\n"
                "{format_instructions}\n"
                "Do NOT return schema. Do NOT explain.",
            ),
            ("human", "Question: {question}\n\nChunk:\n{context}"),
        ])
        
        chain = prompt | self.llm | parser

        good_docs: List[Document] = []
        scores: List[float] = []

        for doc in docs:
            try:
                res : Score = chain.invoke({
                    "question": question,
                    "context": doc.page_content,
                    "format_instructions": parser.get_format_instructions()
                })
                score = res.score
            except Exception as e :
                # TODO : Log a warning here
                score = 0.0

            scores.append(score)
            if score > self.LOWER_THRESHOLD:
                good_docs.append(doc)

        # Determine Verdict
        max_score = max(scores) if scores else 0.0
        
        if max_score > self.UPPER_THRESHOLD :
            verdict = "CORRECT"
        elif max_score > self.LOWER_THRESHOLD:
            verdict = "AMBIGUOUS"
        else:
            verdict = "INCORRECT"


        return {
            "good_docs": good_docs,
            "verdict": verdict,
            "reason": f"Max relevance score: {max_score:.2f}",
        }

    def rewrite(self, state: State) -> Dict[str, Any]:
        """Node 3 : Rewrite question for Web Search"""
        print("REWRITE")
        question = state["question"]

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Rewrite the user question into a web search query composed of keywords.\n"
                "Rules:\n"
                "  - Keep it short (6–14 words).\n"
                "  - If the question implies recency (recent / latest / last week / "
                "last month), add a constraint like (last 30 days).\n"
                "  - Do NOT answer the question.\n"
                "  - Do NOT explain.",
            ),
            ("human", "Question: {question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            query = chain.invoke({"question": question})
        except Exception :
            # TODO : Log a warning here
            query = question # Fallback

        return {"web_query": query}

    def research(self, state: State) -> Dict[str, Any]:
        """Node 4: Search the Web"""
        print("RESEARCH")
        query = state.get("web_query", state["question"])
        
        if not self.tavily:
            print("Tavily not configured, skipping web search.")
            return {"web_docs": []}

        try:
            # Tavily returns a list of dicts
            results = self.tavily.invoke({"query": query})
            web_docs: List[Document] = []

            for r in results.get("results", []):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "") or r.get("snippet", "")

                # Enrich content so the LLM has source attribution in context
                enriched = f"TITLE: {title}\nURL: {url}\nCONTENT: {content}"

                web_docs.append(Document(
                    page_content=enriched,
                    metadata={"source": url, "title": title},
                ))
            return {"web_docs": web_docs}
        except Exception as e:
            print("Tavily search failed: %s", e)
            return {"web_docs": []}

    def refine(self, state: State) -> Dict[str, Any]:
        """Node 5: Refine context"""
        print("REFINE")

        docs = self._select_docs_by_verdict(state)
        sentences = self._decompose_into_sentences(docs)

        parser = PydanticOutputParser(pydantic_object=KeepOrDrop)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Read the sentence and the user question.\n"
                "Return 'keep': true if the sentence contains information relevant to the question.\n"
                "Return 'keep': false if it is irrelevant, filler, or metadata.\n"
                "{format_instructions}",
            ),
            ("human", "Question: {question}\nSentence: {sentence}"),
        ])

        chain = prompt | self.llm | parser
        kept: List[str] = []

        for sent in sentences[: self.MAX_REFINE_SENTENCES]:
            try:
                res: KeepOrDrop = chain.invoke({
                    "question": state["question"],
                    "sentence": sent,
                    "format_instructions": parser.get_format_instructions()
                })
                if res.keep:
                    kept.append(sent)
            except:
                kept.append(sent)  # fail-open

        refined_context = "\n".join(kept).strip()
        return {
            "strips": sentences,
            "kept_strips": kept,
            "refined_context": refined_context
        }

    def generate(self, state: State) -> Dict[str, Any]:
        """Node 6: Generate Final Answer"""
        print("GENERATE")
        question = state["question"]
        context = state.get("refined_context", "").strip()

        if not context:
            return {"answer": "I could not find enough relevant information to answer your question."}
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Answer ONLY using the provided refined context.\n"
                "If the context is empty or insufficient, say: "
                "'I don't know based on the provided documents.'",
            ),
            ("human", "Question: {question}\n\nRefined context:\n{context}"),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "question": question,
            "context": context,
        })

        answer = response.content if hasattr(response, "content") else str(response)
        
        return {"answer": answer}

    def _select_docs_by_verdict(self, state: State) -> List[Document]:
        """Return the appropriate document list based on the pipeline verdict."""
        verdict = state["verdict"]
        if verdict == "CORRECT":
            return state.get("good_docs", [])
        elif verdict == "INCORRECT":
            return state.get("web_docs", [])
        else:  # AMBIGUOUS
            return state.get("good_docs", []) + state.get("web_docs", [])

    @staticmethod
    def _decompose_into_sentences(docs: List[Document]) -> List[str]:
        """
        Join all document content and split into individual sentences.
        Filters out very short fragments (< 20 chars).
        """
        full_text = " ".join(d.page_content for d in docs)
        sentences = re.split(r"(?<=[.!?])\s+", full_text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]