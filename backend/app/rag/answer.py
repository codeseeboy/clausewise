"""Answer flow: retrieve clauses, screen them, ask the model, check the reply."""

from app.rag.guardrails import NOT_COVERED, CheckedAnswer, check_reply, screen_evidence
from app.rag.llm import FakeLLM, OllamaLLM
from app.rag.prompt import build_messages
from app.retrieval.base import Retriever
from app.schemas import Hit
from app.tracing import add_tokens, step


def answer_question(
    query: str, retriever: Retriever, llm: OllamaLLM | FakeLLM, context_chunks: int
) -> tuple[CheckedAnswer, list[Hit]]:
    """Returns the checked answer and the clauses that were given to the model."""
    hits = retriever.search(query, context_chunks)
    with step("guardrails"):
        hits, notes = screen_evidence(hits)
    if not hits:
        return CheckedAnswer("refused", NOT_COVERED, [], notes + ["nothing to answer from"]), hits

    with step("llm"):
        reply = llm.chat(build_messages(query, hits))
    add_tokens(prompt=reply.prompt_tokens, completion=reply.completion_tokens)

    with step("guardrails"):
        checked = check_reply(reply.text, {hit.chunk.chunk_id for hit in hits})
    checked.problems = notes + checked.problems
    return checked, hits
