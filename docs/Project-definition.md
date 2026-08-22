# LangGraph Research Agent — Project Definition

## 1. Problem Statement

The goal of this project is to build a reliable, general-purpose research agent using LangGraph that can take a complex research question, break it into smaller sub-questions, gather relevant information from multiple sources, evaluate the quality and relevance of the retrieved evidence, and synthesize the findings into a structured research report. The system should combine information from an internal knowledge base through RAG with information retrieved from external sources when appropriate. Unlike a basic LLM chatbot that generates an answer directly from its internal knowledge, the system will follow an explicit agentic workflow involving research planning, information retrieval, evidence evaluation, synthesis, and validation. The final response should provide traceable, citation-backed findings, distinguish well-supported conclusions from uncertain information, identify conflicting evidence when present, and avoid unsupported claims. When sufficient reliable evidence cannot be found, the system should explicitly communicate the evidence gap rather than hallucinating an answer.

## 2. Supported Research Questions

The system will focus on multi-step research questions that require information gathering, evidence evaluation, and synthesis rather than simple question answering.

The initial system will support:

* **Factual questions** that require information from multiple sources.
* **Comparison questions** involving two or more technologies, products, companies, approaches, or concepts.
* **Analytical questions** that require synthesizing information from several sources.
* **"Why" and "How" questions** where the answer requires evidence-based explanation.
* **Multi-hop questions** where answering the final question requires finding and connecting information from multiple sources.
* **Questions involving conflicting information**, where the system must identify and explain differences between sources.

The system is intended to be domain-agnostic. It should be possible to research topics from areas such as technology, business, science, and other information-rich domains without changing the core agent workflow.

The initial version will not target high-risk domains such as medical diagnosis, legal advice, or financial investment recommendations.

## 3. Final Output

The system should produce a structured research report rather than a simple conversational response.

The report should contain:

1. **Research Question** — the original question being investigated.
2. **Executive Summary** — a concise answer to the research question.
3. **Key Findings** — the major conclusions identified during research.
4. **Detailed Analysis** — supporting reasoning and evidence for each important finding.
5. **Sources / Citations** — sources supporting the claims made in the report.
6. **Conflicting Evidence** — important disagreements between sources, when applicable.
7. **Evidence Gaps / Limitations** — information that could not be reliably established.
8. **Conclusion** — the final evidence-based assessment.

Important factual claims should be traceable to the evidence used to support them.

## 4. What Counts as Good Evidence?

Evidence should be evaluated based on multiple factors rather than simply whether a source was retrieved.

The system should consider:

* **Relevance** — Does the source directly support the claim?
* **Authority** — Is the source trustworthy and appropriate for the subject?
* **Recency** — Is the information sufficiently current for the research question?
* **Specificity** — Does the source provide concrete information or evidence?
* **Citations **— Is the claim supported by independent sources?

Sources should generally be prioritized in the following order:

### Strong Evidence

* Peer-reviewed research papers
* Official documentation
* Government or regulatory sources
* Official company reports and publications
* Primary datasets or first-party sources

### Moderate Evidence

* Reputable technical publications
* Established news organizations
* Industry reports
* Expert analysis

### Weak Evidence

* Personal blogs
* Forums
* Social media discussions
* Unverified websites
* Content that makes claims without supporting evidence

Lower-quality sources may still be useful for discovery, but important conclusions should preferably be supported by stronger evidence or multiple independent sources.

## 5. Insufficient or Conflicting Evidence

The system should not generate unsupported conclusions when reliable evidence is unavailable.

### No Evidence

If the system cannot find sufficient reliable evidence, it should explicitly state that the available evidence is insufficient to answer the question confidently.

### Partial Evidence

If only some parts of a research question can be answered, the system should provide the supported findings and clearly identify which parts remain unanswered.

### Conflicting Evidence

If credible sources disagree, the system should not arbitrarily choose one answer. It should:

1. Identify the disagreement.
2. Present the relevant evidence from each side.
3. Evaluate the relative quality and recency of the sources.
4. Explain which position currently has stronger support, if this can reasonably be determined.
5. Clearly communicate any remaining uncertainty.

### Low-Quality Evidence

If only weak sources are available, the system should communicate that the resulting conclusion has limited confidence rather than presenting it as an established fact.

## Guiding Principle

> **The system should prefer acknowledging insufficient evidence over generating an unsupported answer.**
