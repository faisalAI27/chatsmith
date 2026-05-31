# ChatSMITH Manual Evaluation Review

Use this template after running `scripts/run_chatbot_eval.py`.

## Scoring

- `2` = Correct answer with a useful source.
- `1` = Partially correct, missing some detail, or weak source.
- `0` = Wrong, hallucinated, or says not enough information when the website clearly has the data.
- `N/A` = Website likely does not contain this information.

## Review Columns

Add or fill these fields in the generated CSV:

- `manual_score`
- `manual_notes`
- `issue_type`

## Issue Types

- `scraper_missing_data`
- `chunker_missing_chunk`
- `retrieval_wrong_chunk`
- `prompt_refusal`
- `source_format_issue`
- `hallucination`
- `frontend_display_issue`
- `api_error`
- `correct`

## Review Process

1. Read the question, expected behavior, answer, source count, warnings, and flags.
2. Open source URLs or inspect retrieved chunks if the answer looks weak.
3. Assign `manual_score`.
4. Add short `manual_notes`.
5. Set `issue_type` if the answer is not fully correct.
