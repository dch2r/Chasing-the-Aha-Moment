"""Record reward inputs without changing the v2 reward values."""
import json, os
from collections import defaultdict
from pathlib import Path
from rewards import contents, correctness_reward, format_reward, extract_answer, number, score

def make_logged_reward(path, max_tokens=512):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Diagnostic log already exists. Start a new run instead of mixing runs.')
    path.touch()
    call_id = 0
    def logged_correctness_reward(prompts, completions, answer, completion_ids=None,
                                  source_id=None, trainer_state=None, **kwargs):
        nonlocal call_id
        texts = contents(completions)
        n = len(texts)
        if len(prompts) != n or len(answer) != n:
            raise ValueError('Prompt/completion/answer lengths differ')
        if source_id is not None and len(source_id) != n:
            raise ValueError('source_id length mismatch')
        if completion_ids is not None and len(completion_ids) != n:
            raise ValueError('completion_ids length mismatch')
        rewards = correctness_reward(completions, answer)
        formats = format_reward(completions)
        call_id += 1
        rows = []
        for i, (prompt, text, gold) in enumerate(zip(prompts, texts, answer)):
            extracted = extract_answer(text)
            numeric = number(extracted) if extracted is not None else None
            result = score(text, gold)
            if extracted is None:
                status = 'missing_or_ambiguous_answer_field'
            elif numeric is None:
                status = 'unsupported_numeric_answer'
            elif not result['correct']:
                status = 'parsed_answer_mismatch'
            else:
                status = 'correct'
            tokens = len(completion_ids[i]) if completion_ids is not None else None
            rows.append(dict(call_id=call_id, completion_index=i,
                optimizer_steps_completed=getattr(trainer_state, 'global_step', None),
                source_id=int(source_id[i]) if source_id is not None else None,
                prompt=prompt, gold=str(gold), response=text, extracted_answer=extracted,
                normalized_answer=str(numeric) if numeric is not None else None,
                status=status, generated_tokens=tokens,
                hit_token_cap=tokens >= max_tokens if tokens is not None else None,
                correctness_reward=rewards[i], format_reward=formats[i],
                total_reward=rewards[i]+formats[i], **result))
        with path.open('a') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
            f.flush(); os.fsync(f.fileno())
        return rewards
    return logged_correctness_reward

def summarize(path):
    rows=[json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    groups=defaultdict(list)
    counts=defaultdict(int)
    for r in rows:
        groups[(r['call_id'], json.dumps(r['prompt'], sort_keys=True))].append(r['total_reward'])
        counts[r['status']]+=1
    return dict(logged_completions=len(rows), reward_calls=len({r['call_id'] for r in rows}),
        prompt_groups=len(groups), groups_with_reward_variation=sum(len(set(v))>1 for v in groups.values()),
        status_counts=dict(counts), format_valid=sum(r['format_valid'] for r in rows),
        hit_token_cap=sum(r['hit_token_cap'] is True for r in rows),
        note='Counts describe sampled training responses, not held-out accuracy. Missing answer fields need human inspection.')
