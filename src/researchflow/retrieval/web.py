from duckduckgo_search import DDGS

from researchflow.config.settings import get_settings
from researchflow.state.models import Evidence, ResearchTask, SourceType


def _search_tavily(query: str, max_results: int) -> list[dict]:
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

    settings = get_settings()
    wrapper = TavilySearchAPIWrapper(tavily_api_key=settings.tavily_api_key)
    return wrapper.results(query, max_results=max_results)


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def retrieve_web(
    task: ResearchTask,
    *,
    sequence: int,
    query: str | None = None,
) -> list[Evidence]:
    settings = get_settings()
    search_query = query or task.question
    if settings.tavily_api_key:
        hits = _search_tavily(search_query, settings.retrieval_top_k)
    else:
        hits = _search_duckduckgo(search_query, settings.retrieval_top_k)

    items: list[Evidence] = []
    for index, hit in enumerate(hits):
        suffix = "" if index == 0 else f"_{index}"
        title = str(hit.get("title") or hit.get("url") or "Web result")
        url = hit.get("url")
        content = str(hit.get("content") or hit.get("body") or hit.get("snippet") or "")
        items.append(
            Evidence(
                evidence_id=f"E{sequence}{suffix}",
                research_id=task.id,
                source_type=SourceType.WEB,
                title=title,
                url=url,
                content=content[:4000],
                metadata={
                    "published_date": hit.get("published_date"),
                    "retrieval_score": round(0.05 * (settings.retrieval_top_k - index), 4),
                },
            )
        )
    return items
