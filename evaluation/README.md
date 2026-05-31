# ChatSMITH Chatbot Evaluation

This directory contains reusable question sets and manual-review tooling for testing generated website chatbots. The focus is commercial and customer-service quality: contact details, store locations, delivery, returns, payments, products, policies, social links, and hallucination resistance.

## Question Sets

- `questions/customer_service_smoke.json` - 10 high-signal questions for quick checks after major changes.
- `questions/customer_service_essential.json` - 30-40 important questions for real website validation.
- `questions/customer_service_standard.json` - full customer-service question bank grouped by category.

Generated results are written to `evaluation/results/` and ignored by Git.

## Run Backend

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

The target website must already be scraped and indexed so `/api/chat` can use RAG with a `website_id`.

## Evaluation Levels

1. Smoke test
   - 10 questions.
   - Run after every code change.
   - Fast confidence check.

2. Essential test
   - 30-40 questions.
   - Run after generating a chatbot for a real website.
   - Covers main customer-service use cases.

3. Standard/full test
   - 70-100 questions.
   - Run before demo, PR, or major release.
   - Helps find deeper weaknesses.

## Run Smoke Evaluation

```bash
python3 scripts/run_chatbot_eval.py \
  --website-id WEBSITE_ID \
  --question-file evaluation/questions/customer_service_smoke.json \
  --output-prefix evaluation/results/site_smoke
```

## Run Essential Evaluation

```bash
python3 scripts/run_chatbot_eval.py \
  --website-id WEBSITE_ID \
  --question-file evaluation/questions/customer_service_essential.json \
  --output-prefix evaluation/results/site_essential
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
python3 scripts/summarize_chatbot_eval.py evaluation/results/site_essential.csv
```

## Review Results

Open the generated CSV and fill:

- `manual_score`
- `manual_notes`
- `issue_type`

Manual scoring:

- `2` = Correct, complete, well-formatted, useful source.
- `1` = Partially correct or formatting/source issue.
- `0` = Wrong, hallucinated, or says not enough information when website clearly has the data.
- `N/A` = Website likely does not contain this information.

Issue types:

- `correct`
- `scraper_missing_data`
- `chunker_missing_chunk`
- `retrieval_wrong_chunk`
- `prompt_refusal`
- `answer_format_issue`
- `source_format_issue`
- `hallucination`
- `frontend_display_issue`
- `api_error`

Startup-quality target:

- Customer-service questions should be 85-90% correct.
- Hallucination-sensitive questions should be almost 100% safe.
- Important answers should include sources.

## Automatic Flags

The runner adds heuristic flags for review only:

- `missing_sources`
- `very_short_answer`
- `possible_hallucination`
- `missing_url`
- `missing_markdown_link`
- `missing_phone`
- `missing_email`
- `missing_hours`
- `missing_bullets`
- `missing_city_grouping`
- `raw_unformatted_long_answer`
- `non_rag_mode`

These flags are not final judgments; use them to decide where to inspect first.

## Debugging Workflow

If an answer is wrong:

1. Check the scraped JSON knowledge file.
2. Check generated chunks.
3. Check retrieval results and reranking debug.
4. Check the RAG prompt and API response.
5. Check frontend display.
