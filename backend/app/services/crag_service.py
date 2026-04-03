import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import List, TypedDict, Dict, Any, Optional

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
    HYDE_MAX_TOKENS = 100

    def __init__(
        self,
        session_id: str,
        retriever,
        model_name: str,
        provider: str = "ollama",
        api_keys: dict = None,
        redis=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.session_id = session_id
        self.retriever = retriever
        self.api_keys = api_keys or {}

        # Redis client and event loop reference — used for SSE status push.
        # Both are optional; if absent, status tracking is silently skipped.
        self.redis = redis
        self._loop = loop

        self.llm = build_llm(
            provider=provider,
            model=model_name,
            api_keys=self.api_keys,
        )

        self.tavily = self._initialize_tavily(settings.TOPIC)
        self.app = self._build_graph()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _push_status(self, step: str, message: str) -> None:
        """
        Push a pipeline status event to Redis from a synchronous context.

        The CRAG pipeline runs in a ThreadPoolExecutor thread (via
        run_in_executor), so we cannot simply `await` an async call.
        Instead we schedule the coroutine on the event loop that owns the
        Redis connection and wait for it to complete (2 s timeout).
        This is non-fatal: any failure is silently swallowed so the
        pipeline is never interrupted by a status-tracking error.
        """
        if not self.redis or not self._loop:
            return
        event = json.dumps({
            "step": step,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._push_status_coro(event),
                self._loop,
            )
            future.result(timeout=2)
        except Exception:
            pass  # Status tracking is always non-fatal

    async def _push_status_coro(self, event: str) -> None:
        """Async side of status push — runs on the main event loop."""
        key = f"pipeline:status:{self.session_id}"
        await self.redis.rpush(key, event)
        # 5-minute safety TTL so orphaned lists don't accumulate in Redis
        await self.redis.expire(key, 300)

    def _push_complete(self, verdict: str) -> None:
        """Push the terminal completion marker to the status list."""
        if not self.redis or not self._loop:
            return
        event = json.dumps({
            "complete": True,
            "verdict": verdict,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._push_status_coro(event),
                self._loop,
            )
            future.result(timeout=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

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
        try:
            result = self.app.invoke({
                "question":            question,
                "hypothetical_answer": "",
                "docs":                [],
                "good_docs":           [],
                "verdict":             "",
                "reason":              "",
                "web_query":           "",
                "web_docs":            [],
                "strips":              [],
                "kept_strips":         [],
                "refined_context":     "",
                "answer":              "",
            })
            # Push the completion marker so the SSE consumer can close
            self._push_complete(result.get("verdict", "UNKNOWN"))
            return result
        except Exception as e:
            # Push an error completion so the SSE consumer doesn't hang
            self._push_complete("ERROR")
            import traceback
            traceback.print_exc()
            print(f"CRAG pipeline crashed: {e}")
            raise

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def retrieve(self, state: State) -> Dict[str, Any]:
        """Node 1: Retrieve documents using HyDE (Hypothetical Document Embeddings)"""
        self._push_status("retrieve", "Retrieving relevant documents from your index…")
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
                "ONLY RETURN A STRING AS ANSWER.\n"
                "Return the answer in under {max_tokens} characters.",
            ),
            ("human", "Question: {question}"),
        ])

        chain = hyde_prompt | self.llm
        response = chain.invoke({
            "question": question,
            "max_tokens": self.HYDE_MAX_TOKENS,
        })

        hypothetical_text = self._extract_text(response)

        docs = self.retriever.invoke(hypothetical_text)

        return {
            "hypothetical_answer": hypothetical_text,
            "docs": docs
        }
    
    def evaluate(self, state: State) -> Dict[str, Any]:
        """Node 2 : Scores each document 0-1"""
        self._push_status("evaluate", "Evaluating document relevance…")
        print("EVALUATE")
        question = state["question"]
        docs = state["docs"]

        # Setup scoring chain
        parser = PydanticOutputParser(pydantic_object=Score)
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
            ("human", "Question: {question}\n\nChunk:\n{chunk}"),
        ])
        
        chain = prompt | self.llm | parser

        good_docs: List[Document] = []
        scores: List[float] = []

        for doc in docs:
            try:
                res : Score = chain.invoke({
                    "question": question,
                    "chunk": doc.page_content,
                    "format_instructions": parser.get_format_instructions(),
                })
                score = res.score
                time.sleep(1)
            except Exception as e :
                # TODO : Log a warning here
                score = 0.0
                time.sleep(1)

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

        print(f"Verdict -> {verdict} | good -> {len(good_docs)} bad -> {len(docs) - len(good_docs)}")
        return {
            "good_docs": good_docs,
            "verdict": verdict,
            "reason": f"Max relevance score: {max_score:.2f}",
        }

    def rewrite(self, state: State) -> Dict[str, Any]:
        """Node 3 : Rewrite question for Web Search"""
        self._push_status("rewrite", "Rewriting query for web search…")
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
        self._push_status("research", "Searching the web for additional context…")
        print("RESEARCH")
        query = state.get("web_query", state["question"])
        
        if not self.tavily:
            print("Tavily not configured, skipping web search.")
            return {"web_docs": []}

        try:
            # Tavily returns a list of dicts
            results = self.tavily.invoke({"query": query})
            web_docs: List[Document] = []

            if isinstance(results, list) :
                raw = results
            elif isinstance(results, dict) :
                raw = results.get("results", [])
            else :
                raw = []

            for r in raw :
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
        """Node 5: Decompose context into sentences and keep only relevant ones"""
        self._push_status("refine", "Filtering context to the most relevant sentences…")
        print("REFINE")

        docs = self._select_docs_by_verdict(state)
        sentences = self._decompose_into_sentences(docs)
        # sentences = sentences[:self.MAX_REFINE_SENTENCES]

        parser = PydanticOutputParser(pydantic_object=KeepOrDrop)
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Read the sentence and the user question.\n"
                "Return 'keep': true if the sentence contains information relevant to the question.\n"
                "Return 'keep': false if it is irrelevant, filler, or metadata.\n"
                "{format_instructions}",
            ),
            ("human", "Question: {question}\n\nSentence: {sentence}"),
        ])

        chain = prompt | self.llm | parser
        kept: List[str] = []

        print("Refining -> ", end = "| ")

        
        for sent in sentences :
            try:
                res: KeepOrDrop = chain.invoke({
                    "question": state["question"],
                    "sentence": sent,
                    "format_instructions": parser.get_format_instructions()
                })
                print(res.keep, end = " | ")
                if res.keep:
                    kept.append(sent)
                time.sleep(1)
            except:
                kept.append(sent)  # fail-open
                time.sleep(1)
        print()

        refined_context = "\n".join(kept).strip()
        return {
            "strips": sentences,
            "kept_strips": kept,
            "refined_context": refined_context
        }

    def generate(self, state: State) -> Dict[str, Any]:
        """Node 6: Generate Final Answer"""
        self._push_status("generate", "Generating your answer…")
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

        answer = self._extract_text(response)
        return {"answer": answer}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
        step = len(full_text.strip()) // 10
        sentences = re.split(r"(?<=[.!?])\s+", full_text)
        result = []
        curr = ""
        for s in sentences :
            if len(curr) < step :
                curr += s.strip()
            else :
                result.append(curr.strip())
                curr = s.strip()
        result.append(curr.strip())
        return result
    
    @staticmethod
    def _extract_text(response) -> str:
        """Safely extract string content from any LLM response object."""
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                # Extract text from list of content blocks
                text = ""
                for block in content :
                    if isinstance(block, dict) :
                        text += block.get("text", "").strip()
                    else :
                        text += str(block).strip()
            else:
                text = str(content)
        else:
            text = str(response)
        return text