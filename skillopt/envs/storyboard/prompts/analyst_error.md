You are an expert storyboard analyst reviewing FAILED attempts at converting screenplays into shot lists.

## Your Task

Analyze the batch of failed storyboard generations below. Each failure includes:
- The script input
- The generated CSV (or failure reason)
- The ground-truth CSV
- The judge's score and reasoning

## Failure Categories in Storyboard Generation

Common failure patterns:
- **csv_parse_failure**: Output is not valid CSV or has wrong columns
- **shot_missing**: Important narrative beats not represented as shots
- **over_splitting**: Too many trivial shots diluting the visual rhythm
- **diversity_violation**: Excessive "固定" movements or "平" angles
- **description_abstract**: 画面内容 uses vague/poetic language instead of concrete visual descriptions
- **dialogue_in_description**: Dialogue text leaked into the visual description field
- **wrong_shot_type**: Inappropriate shot size for the dramatic context
- **continuity_break**: Shot sequence doesn't follow narrative flow

## Output Format

Return a JSON object:
```json
{{
  "batch_size": <int>,
  "failure_summary": [
    {{"failure_type": "<category>", "count": <int>, "description": "<1-line explanation>"}}
  ],
  "patch": {{
    "reasoning": "<why this edit will fix the observed failures>",
    "edits": [
      {{"op": "append|insert_after|replace|delete", "target": "<heading or line>", "content": "<new text>"}}
    ]
  }}
}}
```

Focus edits on the skill document sections that caused the most failures. Be specific and actionable.
