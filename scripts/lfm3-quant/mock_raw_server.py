"""Mock /v1/chat/completions that answers with raw <tool_call> text (no
structured tool_calls), to exercise tool-eval's fallback parser."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        last = body['messages'][-1]
        if last['role'] == 'tool':
            content = 'It is 18 degrees and raining.'
        elif 'json' in last['content']:
            content = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        else:
            content = ('<think>need weather</think><tool_call>\n<function=get_weather>\n'
                       '<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>')
        out = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': content}}]}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out))); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a):
        pass

HTTPServer(('127.0.0.1', 18099), H).serve_forever()
