You are an expert storyboard analyst reviewing SUCCESSFUL attempts at converting screenplays into shot lists.

## Your Task

Analyze the batch of successful storyboard generations below. Each success includes:
- The script input
- The generated CSV
- The ground-truth CSV
- The judge's score and reasoning

## What to Look For

Identify which parts of the current skill document contributed to success:
- Effective instructions that guided good shot composition
- Rules that ensured diversity in camera work
- Patterns that produced concrete, visual descriptions
- Structural guidance that maintained narrative flow

## Output Format

Return a JSON object:
```json
{{
  "batch_size": <int>,
  "success_patterns": [
    {{"pattern": "<what worked>", "count": <int>, "evidence": "<brief example>"}}
  ],
  "patch": {{
    "reasoning": "<why reinforcing these patterns will improve consistency>",
    "edits": [
      {{"op": "append|insert_after|replace|delete", "target": "<heading or line>", "content": "<new text>"}}
    ]
  }}
}}
```

Suggest edits that reinforce successful patterns or make implicit good behavior explicit in the skill document.
