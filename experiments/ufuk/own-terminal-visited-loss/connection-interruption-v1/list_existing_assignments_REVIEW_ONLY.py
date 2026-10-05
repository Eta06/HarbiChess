"""Review-only cached-token GET; never allocates, authenticates, refreshes or prunes."""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

URL = 'https://colab.research.google.com/tun/m/assignments?authuser=0'
CACHE = Path('/workspace/work/harbichess/colab-cli/config/home/token.json')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def inspect_existing(output):
    # Existing credential is read in memory only; no OAuth flow/refresh/token write.
    cached = json.loads(CACHE.read_text())
    expiry = datetime.fromisoformat(cached['expiry'].replace('Z', '+00:00'))
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    if expiry <= datetime.now(UTC) or not cached.get('token'):
        raise RuntimeError('cached-credential-expired-no-authentication-attempted')
    request = urllib.request.Request(URL, method='GET', headers={
        'Accept': 'application/json', 'X-Colab-Client-Agent': 'colab-cli',
        'Authorization': 'Bearer '+cached['token'],
    })
    result = {'schema':'existing-colab-assignment-readonly-metadata-v1',
              'observed_epoch':time.time(),'method':'GET','allocation_or_reset':False,
              'local_registry_written':False,'authentication_or_refresh_attempted':False}
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=20) as response:
            result['http_status'] = response.status
            body = response.read(1024**2+1)
            if len(body)>1024**2:
                raise RuntimeError('bounded-metadata-response-exceeded')
        text = body.decode()
        if text.startswith(")]}'\n"):
            text = text[5:]
        assignments = json.loads(text)['assignments']
        result['assignment_count'] = len(assignments)
        # Runtime proxy URL/token/expiry and raw endpoint deliberately omitted.
        result['assignments'] = [{
            'endpoint_sha256':hashlib.sha256(item['endpoint'].encode()).hexdigest(),
            'accelerator':item.get('accelerator'), 'variant':item.get('variant'),
        } for item in assignments]
        result['status'] = 'readonly-server-assignments-observed'
    except urllib.error.HTTPError as error:
        result.update({'http_status':error.code,'status':'metadata-request-failed-details-suppressed'})
    except Exception as error:
        result.update({'status':'metadata-request-failed-details-suppressed',
                       'error_type':type(error).__name__})
    with output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--execute-reviewed',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.execute_reviewed:
        raise SystemExit('REVIEW ONLY: explicit root approval required before this GET')
    print(json.dumps(inspect_existing(a.output)))
