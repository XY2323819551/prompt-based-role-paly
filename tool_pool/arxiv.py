import arxiv
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ArxivResult:
    """Arxiv搜索结果的数据类"""
    title: str
    authors: List[str]
    summary: str
    published: str
    pdf_url: str
    entry_id: str


def search_arxiv(query: str, max_results:int= 1) -> str:
    """
    在Arxiv上搜索学术论文。arxiv搜索工具查找相关论文。

    Args:
        query (str): 搜索查询字符串
        max_results (int): 整数类型，返回结果的最大数量。默认为1。

    Returns:
        str: 格式化的搜索结果字符串
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for result in client.results(search): 
            paper = ArxivResult(
                title=result.title,
                authors=[str(author) for author in result.authors],
                summary=result.summary,
                published=result.published.strftime("%Y-%m-%d"),
                pdf_url=result.pdf_url,
                entry_id=result.entry_id,
            )
            results.append(paper)

        if not results:
            return "No papers found for the given query."

        formatted_results = []
        for i, paper in enumerate(results, 1):
            formatted_paper = (
                f"{i}. Title: {paper.title}\n"
                f"   Authors: {', '.join(paper.authors)}\n"
                f"   Published: {paper.published}\n"
                f"   Summary: {paper.summary}\n"
                f"   PDF URL: {paper.pdf_url}\n"
                f"   ArXiv ID: {paper.entry_id}\n"
            )
            formatted_results.append(formatted_paper)

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching arxiv: {str(e)}"


if __name__ == "__main__":
    # query = "attention is all you need"
    query = "Transformer architecture"
    print(f"Searching arxiv for: {query}")
    print("-" * 80)
    results = search_arxiv(query)
    print(results)
    