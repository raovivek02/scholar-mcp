Model: `gemini-3.5-flash-lite` - 8 cases

| metric | score |
|---|---|
| tool_selection | 1.0 |
| retrieval_recall | 0.889 |
| grounded | 1.0 |
| citation_support | 1.0 |
| graceful_failure | 1.0 |

| case | tools ok | recall | grounded | turns |
|---|---|---|---|---|
| lookup-details | True | 1.0 | True | 3 |
| recent-preprints | True | None | True | 2 |
| forward-citations | True | None | True | 3 |
| foundations-graph | True | None | True | 3 |
| bibtex | True | 1.0 | True | 2 |
| related-work | True | 0.6666666666666666 | True | 3 |
| similar-papers | True | None | True | 4 |
| missing-paper | True | None | True | 3 |
