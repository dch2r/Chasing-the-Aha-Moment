"""Frozen v3 evaluation policy: unique tagged numeric answer + bounded presentation normalization.
No searching for reference numbers or inferring answers from intermediate work.
Untagged responses remain unparseable. Units are labels; no unit conversion.
"""
import re
from decimal import Decimal
VERSION = 'v3-tagged-numeric-presentation'
NUM = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
UNITS = r'(?:%|percent|hours?|minutes?|seconds?|days?|weeks?|months?|years?|dollars?|cents?|pounds?|ounces?|miles?|feet|foot|inches|inch|square feet|square foot|pages?|books?|posters?|people|persons?|cows?|pigs?|fish|stamps?|pieces?|bottles?)'
VALUE = re.compile(rf'\s*[$£€]?\s*({NUM})\s*(?:{UNITS})?\s*[.!]?\s*', re.I)
TAIL = re.compile(rf'\s*(?:{UNITS})?\s*[.!]?\s*', re.I)
FORMAT = re.compile(r'\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>\s*', re.S)
def numeric(value):
    value = str(value).strip().strip('*_').strip()
    m = VALUE.fullmatch(value)
    return Decimal(m[1].replace(',', '')) if m else None

def extract(text):
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return None, 'missing_or_duplicate_answer_tags'
    m = re.search(r'<answer>(.*?)</answer>', text, re.S)
    if not m:
        return None, 'invalid_tag_order'
    tail = text[m.end():].strip().strip('*_').strip()
    # Remove emphasis around the field before terminal punctuation too.
    tail = tail.replace('**','').replace('__','').strip()
    if not TAIL.fullmatch(tail):
        return None, 'ambiguous_trailing_text'
    value = numeric(m[1])
    return (value, 'parsed') if value is not None else (None, 'unsupported_answer_value')

def score(text, gold):
    value, status = extract(text)
    target = numeric(gold)
    if target is None:
        raise ValueError('Unsupported reference answer')
    m = FORMAT.fullmatch(text)
    valid = bool(m and all(t.strip() for t in m.groups()) and all(text.count(t)==1 for t in ('<reasoning>','</reasoning>','<answer>','</answer>')))
    correct = value is not None and value == target
    return dict(correct=correct, answer_parseable=value is not None,
                extracted_answer=str(value) if value is not None else None,
                extraction_status=status, format_valid=valid, structured_correct=correct and valid)
