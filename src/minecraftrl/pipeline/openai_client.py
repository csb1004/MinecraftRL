"""Responses API client for the demo slow path (same transport and rules as semantic.OpenAISemantic).

Paid calls need explicit permission; the local budget is spent before each attempt, failures included. A network failure
(connection error, timeout, HTTP 429/5xx) is retried once while budget remains (evaluation round 1, 2026-11-05: a single
transport failure ended a run); a client error (other 4xx) is never retried.
The key is sent only in the Authorization header and never written to logs or receipts.
"""
import json
import time
import urllib.error
import urllib.request

from ..semantic import _transport, response_text


class ResponsesClient:
    def __init__(self, model, key, *, allow_paid, transport=None, max_calls, timeout_s=90., retries=1, retry_wait_s=3., sleep=time.sleep):
        if not allow_paid: raise PermissionError('paid_api_not_authorized')
        if not model or not key: raise ValueError('model_and_key_required')
        if type(max_calls) is not int or not 1 <= max_calls <= 50: raise ValueError('invalid_call_budget')
        self.model = model; self._key = key; self.transport = transport or _transport; self.timeout_s = timeout_s
        self.remaining = max_calls; self.calls = []; self.retries = retries; self.retry_wait_s = retry_wait_s; self.sleep = sleep

    def create(self, payload):
        body = json.dumps(dict(payload, model=self.model, store=False), ensure_ascii=False).encode('utf-8')
        for attempt in range(self.retries + 1):
            if self.remaining <= 0: raise RuntimeError('api_call_budget_exhausted')
            self.remaining -= 1
            request = urllib.request.Request('https://api.openai.com/v1/responses', data=body,
                                             headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'})
            try: response = self.transport(request, timeout=self.timeout_s)
            except Exception as exc:
                status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                self.calls.append(dict(ok=False, error=type(exc).__name__, status=status))
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt == self.retries or self.remaining <= 0:
                    raise RuntimeError('responses_transport_failed') from None
                wait = self.retry_wait_s
                if status == 429:  # rate limit: the server's Retry-After (seconds), else 20 s; never more than 60 s
                    try: wait = float((exc.headers or {}).get('Retry-After') or 20.)
                    except (TypeError, ValueError): wait = 20.
                    wait = min(max(wait, 0.), 60.)
                self.sleep(wait); continue
            self.calls.append(dict(ok=True, usage=usage(response)))
            return response


def output_json(response): return json.loads(response_text(response))


def url_citations(response):
    """URLs the answer cited or the web_search tool actually returned (with a strict JSON format there are no annotations)."""
    urls = set()
    for item in response.get('output') or []:
        if item.get('type') == 'web_search_call':
            urls.update(s['url'] for s in (item.get('action') or {}).get('sources') or [] if isinstance(s, dict) and s.get('url'))
        for part in item.get('content') or [] if item.get('type') == 'message' else []:
            urls.update(a['url'] for a in part.get('annotations') or [] if a.get('type') == 'url_citation' and a.get('url'))
    return urls


def usage(response): return dict(response.get('usage') or {})
