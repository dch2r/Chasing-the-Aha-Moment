"""V2: numeric correctness independent of strict formatting. No GPU required."""
import re
from decimal import Decimal, InvalidOperation
SYSTEM_PROMPT = 'Solve the problem. Respond with <reasoning>your solution</reasoning> followed by <answer>the numeric answer</answer>.'
SCORER_VERSION = 'v2-answer-field'
PATTERN = re.compile(r'\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>\s*', re.S)
TAGS = ('<reasoning>', '</reasoning>', '<answer>', '</answer>')
def parse(text):
    """Strict full-response format, used only for format scoring."""
    if not isinstance(text, str) or any(text.count(t) != 1 for t in TAGS):
        return None
    m = PATTERN.fullmatch(text)
    if not m or not all(x.strip() for x in m.groups()):
        return None
    return m.group(1).strip(), m.group(2).strip()
def number(text):
    """Decimal number; allow a single leading currency symbol and valid commas."""
    text = str(text).strip()
    if text[:1] in ('$', '£', '€'):
        text = text[1:].strip()
    if not re.fullmatch(r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?', text):
        return None
    try:
        n = Decimal(text.replace(',', ''))
        return n if n.is_finite() else None
    except InvalidOperation:
        return None
def extract_answer(text):
    """Require one closed answer field; never infer from intermediate numbers."""
    if not isinstance(text, str):
        return None
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return None
    m = re.search(r'<answer>(.*?)</answer>', text, re.S)
    if not m:
        return None
    # Permit whitespace/Markdown emphasis after the final field, not extra claims.
    if m.string[m.end():].strip(' \t\r\n*_'):
        return None
    return m.group(1).strip()
def score(text, gold):
    answer = extract_answer(text)
    predicted = number(answer) if answer is not None else None
    target = number(gold)
    valid_format = parse(text) is not None
    correct = predicted is not None and target is not None and predicted == target
    return dict(format_valid=valid_format, answer_parseable=predicted is not None,
                correct=correct, structured_correct=correct and valid_format)
def contents(completions):
    return [x if isinstance(x, str) else x[0]['content'] for x in completions]
def correctness_reward(completions, answer, **kwargs):
    texts = contents(completions)
    if len(texts) != len(answer):
        raise ValueError('Completion/answer lengths differ')
    return [2.0 * score(t, a)['correct'] for t, a in zip(texts, answer)]
def format_reward(completions, **kwargs):
    return [0.5 if parse(t) else 0.0 for t in contents(completions)]
def gold_answer(solution):
    if '####' not in solution:
        raise ValueError('Missing GSM8K answer delimiter')
    gold = solution.rsplit('####', 1)[1].strip()
    if number(gold) is None:
        raise ValueError(f'Unsupported gold answer: {gold!r}')
    return gold
