"""Minimal OpenAI chat-completions client over urllib (no extra dependency). Errors propagate."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_env():
    env = {}
    for line in (HERE / '.env').read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip().strip('"\'')
    return env


class LLMError(RuntimeError):
    pass


def chat(messages, tools, *, model, api_key, tool_choice='required', timeout=90, retries=2,
         reasoning_effort='none'):
    # gpt-6-luna rejects function tools unless reasoning_effort is 'none' on chat/completions.
    body = {'model': model, 'messages': messages, 'tools': tools, 'tool_choice': tool_choice,
            'reasoning_effort': reasoning_effort}
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode('utf-8', 'replace')[:500]
            if e.code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(2 * (attempt + 1)); last = f'HTTP {e.code}'; continue
            raise LLMError(f'HTTP {e.code}: {detail}') from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(2 * (attempt + 1)); last = str(e); continue
            raise LLMError(f'network: {e}') from None
    raise LLMError(last)
