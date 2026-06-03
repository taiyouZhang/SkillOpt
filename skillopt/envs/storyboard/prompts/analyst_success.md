You are an expert storyboard analyst reviewing SUCCESSFUL attempts at converting screenplays into shot lists.

## Skill Architecture

The skill you are optimizing consists of TWO sections (marked by `<!-- SKILLFILE:... -->` delimiters):

1. **Director section** (`director`): Reads the script, analyzes Scene Intent, divides Narrative Beats, makes editorial decisions about shot density, and sets camera movement tone + angle strategy.

2. **DP section** (`dp`): Takes the Director's notes and designs concrete shots — shot type, angle, camera movement, composition. Outputs the final CSV storyboard.

## Your Task

Analyze the batch of successful storyboard generations below. Each success includes:
- The 5-stage pipeline execution trace
- The generated CSV
- The ground-truth CSV
- The judge's score and reasoning

## What to Look For

Identify which parts of each section contributed to success:
- Director: effective beat decomposition, good cut point decisions, appropriate density choices
- DP: correct shot type progressions, diverse camera work, concrete visual descriptions

## Output Format

Return a JSON object:
```json
{{
  "batch_size": <int>,
  "success_patterns": [
    {{"pattern": "<what worked>", "count": <int>, "evidence": "<brief example>", "source_section": "director|dp"}}
  ],
  "patch": {{
    "reasoning": "<why reinforcing these patterns will improve consistency>",
    "edits": [
      {{"op": "append|insert_after|replace|delete", "target": "<heading or text within the target section>", "content": "<new text>"}}
    ]
  }}
}}
```

Suggest edits that reinforce successful patterns or make implicit good behavior explicit in the skill document.
