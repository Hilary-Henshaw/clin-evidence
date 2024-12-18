"""PubMed search client using NCBI E-utilities REST API."""

from __future__ import annotations

import logging
from typing import Any
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_ESEARCH_ENDPOINT = f"{_BASE_URL}esearch.fcgi"
_EFETCH_ENDPOINT = f"{_BASE_URL}efetch.fcgi"
_REQUEST_TIMEOUT = 15.0


class PubMedSearchClient:
    """Queries PubMed for clinical literature via E-utilities."""

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Search PubMed for abstracts matching the query.

        Two-step: esearch (get PMIDs) then efetch (get abstracts).

        Args:
            query: Clinical search query.
            max_results: Maximum number of articles to return.

        Returns:
            List of dicts with keys:
            title, url, content (abstract), pmid.
        """
        try:
            pmids = self._esearch(query, max_results)
            if not pmids:
                logger.info(
                    "PubMed esearch returned no PMIDs",
                    extra={"query_len": len(query)},
                )
                return []
            articles = self._efetch(pmids)
            logger.info(
                "PubMed search complete",
                extra={
                    "pmids": len(pmids),
                    "articles": len(articles),
                },
            )
            return articles
        except Exception:
            logger.warning(
                "PubMed search failed",
                exc_info=True,
            )
            return []

    def _esearch(self, query: str, max_results: int) -> list[str]:
        """Run esearch to retrieve PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "usehistory": "n",
        }
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            resp = client.get(_ESEARCH_ENDPOINT, params=params)
            resp.raise_for_status()

        data: dict[str, Any] = resp.json()
        id_list: list[str] = data.get("esearchresult", {}).get("idlist", [])
        return id_list

    def _efetch(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Fetch article abstracts for the given PMIDs."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            resp = client.get(_EFETCH_ENDPOINT, params=params)
            resp.raise_for_status()

        return self._parse_efetch_xml(resp.text)

    def _parse_efetch_xml(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse PubMed efetch XML into article dicts."""
        articles: list[dict[str, Any]] = []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            logger.warning("Failed to parse PubMed XML", exc_info=True)
            return articles

        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            title_el = article.find(".//ArticleTitle")
            title = (
                "".join(title_el.itertext())
                if title_el is not None
                else "Unknown Title"
            )

            abstract_parts: list[str] = []
            for text_el in article.findall(".//AbstractText"):
                label = text_el.get("Label", "")
                text = "".join(text_el.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

            abstract = " ".join(abstract_parts).strip()
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            if abstract:
                articles.append(
                    {
                        "title": title,
                        "url": url,
                        "content": abstract,
                        "pmid": pmid,
                    }
                )

        return articles
