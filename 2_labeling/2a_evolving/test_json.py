import json
import re

def _extract_json_original(raw: str) -> dict:
    """Try direct parse, then find first { to last }."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {raw[:200]!r}")

def _extract_json_improved(raw: str) -> dict:
    """More robust extraction: try to find the longest valid JSON object."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find all { ... } blocks and pick the one that is valid and longest
    potential_starts = [m.start() for m in re.finditer(r'\{', raw)]
    potential_ends = [m.start() for m in re.finditer(r'\}', raw)]
    
    best_json = None
    best_len = -1
    
    # Iterate from outermost to innermost
    for start in potential_starts:
        for end in reversed(potential_ends):
            if end > start:
                candidate = raw[start:end+1]
                if len(candidate) > best_len:
                    try:
                        parsed = json.loads(candidate)
                        best_json = parsed
                        best_len = len(candidate)
                    except json.JSONDecodeError:
                        continue
    
    if best_json is not None:
        return best_json
        
    raise ValueError(f"Could not parse JSON from response: {raw[:200]!r}")

# Test cases
cases = [
    'Here is the JSON: {"a": 1}',
    'Thought: I will return {"a": 1}. JSON: {"b": 2}',
    '```json\n{"a": 1}\n```',
    'Some text before { "a": 1 } and after',
    'Malformed { "a": 1 } and then valid { "b": 2 }',
    'HTML <script>var x = { y: 1 };</script> { "valid": true }'
]

for c in cases:
    print(f"Input: {c!r}")
    try:
        print(f"  Original: {_extract_json_original(c)}")
    except Exception as e:
        print(f"  Original failed: {e}")
    try:
        print(f"  Improved: {_extract_json_improved(c)}")
    except Exception as e:
        print(f"  Improved failed: {e}")
    print()
