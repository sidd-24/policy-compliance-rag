# Evaluation report

33 questions on the sample policies: 25 answerable, 8 not covered by any policy.

## Retrieval

| Chunking | Chunks | Hit@1 | Hit@4 | MRR | Top-1 citation shows the right section |
|---|---|---|---|---|---|
| Fixed 800 characters | 22 | 96% | 100% | 0.98 | 40% |
| Section-aware | 42 | 96% | 100% | 0.98 | 96% |

## Refusal gate (best-match relevance threshold)

| Threshold | Answerable passed | Unanswerable refused |
|---|---|---|
| 0.20 | 100% | 0% |
| 0.25 | 100% | 12% |
| 0.30 | 100% | 50% |
| 0.35 (used) | 100% | 50% |
| 0.40 | 100% | 50% |
| 0.45 | 96% | 62% |
| 0.50 | 80% | 75% |

## End-to-end answers (openai/gpt-oss-120b)

| Metric | Result |
|---|---|
| Answer contains the expected facts | 100% |
| Answer cites the expected section | 100% |
| Unanswerable questions refused | 100% |
| Answerable questions wrongly refused | 0 |
| Answers without any citation | 0 |

### Misses

None.
