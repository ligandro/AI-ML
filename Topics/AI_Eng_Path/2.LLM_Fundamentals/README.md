# Step 2 (3–4 Weeks): LLM Fundamentals

## Outcome
Understand how LLMs behave and how to reliably control outputs for real applications.

## Topics to Cover
- How LLMs work (tokens, context windows, temperature)
- Prompt engineering patterns
- Function/tool calling
- Structured outputs (JSON mode / schema outputs)
- Evaluating LLM response quality

## Weekly Plan
### Week 1
- Learn tokenization and context limits.
- Experiment with prompt formats: role prompting, examples, constraints.
- Track how temperature and top-p affect output consistency.

### Week 2
- Implement function calling with at least 3 tools.
- Build schema-constrained outputs using JSON validation.
- Add fallback prompts for malformed outputs.

### Week 3
- Design evaluation rubrics (correctness, faithfulness, format compliance).
- Create a small benchmark set (30–50 prompts).
- Measure response quality using repeatable scoring.

## Hands-On Projects
1. **Prompt Lab**
   - Same task, 10 prompt variants.
   - Evaluate quality and consistency.
   - Document best prompt template.

2. **Structured LLM Assistant**
   - Input: user query.
   - Output: strict JSON schema (validated).
   - Include retry-and-repair logic for invalid JSON.

## Evaluation Template
Score each response from 1–5 on:
- Task correctness
- Hallucination risk
- Output format validity
- Latency
- Cost per call

## Completion Criteria
- You can design robust prompts for predictable behavior.
- You can wire model tool calls and parse outputs safely.
- You can evaluate output quality with repeatable metrics.

## Suggested Deliverable
Create a GitHub repo: **`llm-control-patterns`** with prompt tests, JSON schema validators, and evaluation scripts.
