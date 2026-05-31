# ChatSMITH Chatbot Evaluation

This directory contains reusable question sets and manual-review tooling for testing generated website chatbots. The focus is commercial and customer-service quality: contact details, store locations, delivery, returns, payments, products, policies, social links, and hallucination resistance.

## Question Sets

- `questions/customer_service_smoke.json` - 10 high-signal questions for quick checks after major changes.
- `questions/customer_service_standard.json` - broader customer-service question bank grouped by category.

Generated results are written to `evaluation/results/` and ignored by Git.

## Run Backend

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

The target website must already be scraped and indexed so `/api/chat` can use RAG with a `website_id`.

## Run Smoke Evaluation

```bash
python3 scripts/run_chatbot_eval.py \
  --website-id WEBSITE_ID \
  --question-file evaluation/questions/customer_service_smoke.json \
  --output-prefix evaluation/results/site_smoke
```

## Run Full Evaluation

```bash
python3 scripts/run_chatbot_eval.py \
  --website-id WEBSITE_ID \
  --question-file evaluation/questions/customer_service_standard.json \
  --output-prefix evaluation/results/site_full
```

Useful options:

```bash
--category contact_support
--limit 10
--include-debug
--delay-seconds 0.5
--timeout 60
```

## Summarize Results

```bash
python3 scripts/summarize_chatbot_eval.py evaluation/results/site_smoke.csv
```

## Review Results

Open the generated CSV and fill:

- `manual_score`
- `manual_notes`
- `issue_type`

Scoring:

- `2` = Correct answer with useful source.
- `1` = Partially correct, missing some detail, or weak source.
- `0` = Wrong, hallucinated, or says not enough information when website clearly has the data.
- `N/A` = Website likely does not contain this information.

Startup-quality target:

- Customer-service questions should be 85-90% correct.
- Hallucination-sensitive questions should be almost 100% safe.
- Important answers should include sources.

## Automatic Flags

The runner adds heuristic flags for review only:

- `missing_sources`
- `short_answer`
- `possible_hallucination`
- `missing_url`
- `missing_phone`
- `non_rag_mode`

These flags are not final judgments; use them to decide where to inspect first.

## Debugging Workflow

If an answer is wrong:

1. Check the scraped JSON knowledge file.
2. Check generated chunks.
3. Check retrieval results and reranking debug.
4. Check the RAG prompt and API response.
5. Check frontend display.
