"""End-to-end llama-server check: chat, tool-call round trip, and vision.
usage: e2e_test.py <server-url> <image.png> <transcript.md>"""
import base64, json, sys, time, urllib.request

url, image, out = sys.argv[1], sys.argv[2], sys.argv[3]
log = []

def chat(body):
    req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r['choices'][0]['message'], r.get('timings', {}), time.time() - t

def show(title, msg, timings, secs):
    log.append(f'### {title}\n')
    if msg.get('reasoning_content'):
        log.append('**thinking:** ' + msg['reasoning_content'].strip().replace('\n', ' ')[:600] + '\n')
    if msg.get('tool_calls'):
        for tc in msg['tool_calls']:
            log.append(f"**tool call:** `{tc['function']['name']}({tc['function']['arguments']})`\n")
    if msg.get('content'):
        log.append('**answer:** ' + msg['content'].strip() + '\n')
    tps = timings.get('predicted_per_second')
    log.append(f'*{secs:.2f}s' + (f', {tps:.0f} tok/s decode' if tps else '') + '*\n')

# 1. plain chat
m, t, s = chat({'messages': [{'role': 'user', 'content': 'In two sentences, why is the sky blue?'}], 'temperature': 0, 'max_tokens': 600})
show('Chat', m, t, s)

# 2. tool round trip
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
msgs = [{'role': 'user', 'content': 'Should I take an umbrella in Manchester today?'}]
m, t, s = chat({'messages': msgs, 'tools': tools, 'temperature': 0, 'max_tokens': 600})
show('Tool call (turn 1)', m, t, s)
if m.get('tool_calls'):
    tc = m['tool_calls'][0]
    msgs.append({'role': 'assistant', 'content': m.get('content') or '', 'tool_calls': m['tool_calls']})
    msgs.append({'role': 'tool', 'tool_call_id': tc.get('id', 'call_0'),
                 'content': json.dumps({'city': 'Manchester', 'condition': 'heavy showers', 'chance_of_rain': 0.9, 'temp_c': 11})})
    m, t, s = chat({'messages': msgs, 'tools': tools, 'temperature': 0, 'max_tokens': 600})
    show('Tool result → answer (turn 2)', m, t, s)

# 3. vision
b64 = base64.b64encode(open(image, 'rb').read()).decode()
m, t, s = chat({'messages': [{'role': 'user', 'content': [
    {'type': 'text', 'text': 'Describe this screenshot in two sentences. What application is it?'},
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}}]}], 'temperature': 0, 'max_tokens': 800})
show('Vision (mmproj)', m, t, s)

open(out, 'w', encoding='utf-8').write('\n'.join(log))
print('\n'.join(log))
