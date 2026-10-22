# Troubleshooting

- **Groq is not available:** set `GROQ_API_KEY`, use mock mode for offline control-flow checks, and restart Streamlit. Live mode lists model IDs from the authenticated Groq endpoint.
- **Selected model fails:** model availability can change. Refresh Streamlit and choose a currently active chat model from the sidebar.
- **Mesh search returns unavailable:** run `python scripts/download_mesh.py`; confirm NLM URLs are reachable and no partial downloads are present.
- **PubMed rate limit/network error:** add `NCBI_EMAIL`, optionally an API key, and retry later. Requests use backoff; do not increase call frequency.
- **No scientific conclusion in mock mode:** expected. Mock mode shows retrieval/trace flow but refuses to assess evidence or invent findings. Configure Groq for actual reasoning.
- **No labeled evaluation metrics:** the selected official split may be unlabeled, or inference has not run. Inspect loader's `labeled` count.
- **Invalid structured output:** check provider compatibility and model output format; errors are surfaced instead of silently repaired into research decisions.

