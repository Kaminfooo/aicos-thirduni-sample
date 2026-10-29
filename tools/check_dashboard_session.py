"""Check dashboard session/reconnect behavior with Node's isolated VM and fake transport.

Uses no account, real WebSocket, camera, browser profile or third-party dependency.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
const vm=require('node:vm'),assert=require('node:assert/strict');
let input='';process.stdin.on('data',c=>input+=c);process.stdin.on('end',()=>{
const source=JSON.parse(input).source;
function page(token){
 const fields=new Map(),sockets=[],timers=[];
 function field(id){if(!fields.has(id))fields.set(id,{value:'',textContent:'',className:'',style:{}});return fields.get(id)}
 function Socket(){this.sent=[];this.send=text=>this.sent.push(JSON.parse(text));this.close=()=>{};sockets.push(this)}
 const auth={token:()=>token};
 const context={window:{location:{protocol:'http:',origin:'http://127.0.0.1:8081'},AicosAuth:auth},AicosAuth:auth,
  document:{getElementById:field,querySelector:()=>null},WebSocket:Socket,
  setTimeout:fn=>{timers.push(fn);return timers.length},clearTimeout:()=>{},console};
 vm.runInNewContext(source,context,{timeout:2000});return {field,sockets,timers};
}
let p=page('');p.field('btnConnect').onclick();assert.equal(p.sockets.length,0);assert.equal(p.field('statusTxt').textContent,'SIGN IN REQUIRED');
p=page('test-session-placeholder');assert.equal(p.sockets.length,1);p.sockets[0].onopen();assert.equal(p.field('statusTxt').textContent,'AUTHENTICATING…');
p.sockets[0].onmessage({data:JSON.stringify({kind:'snapshot',events:[]})});assert.equal(p.field('statusTxt').textContent,'CONNECTED');
p.sockets[0].onclose({code:4401});assert.equal(p.timers.length,0);assert.equal(p.field('statusTxt').textContent,'SESSION EXPIRED — SIGN IN');
p=page('test-session-placeholder');p.sockets[0].onclose({code:1006});assert.equal(p.timers.length,1);
console.log(JSON.stringify({checks:4,passed:4,failures:0,real_connections:0,checks_named:[
'no connection without session','connected only after authenticated snapshot','no retry loop after authentication rejection','network close schedules one reconnect']}));
});
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / '.runtime'):
        parser.error('Choose a new report file inside project .runtime')
    page = (ROOT / 'dashboard/live-ops.html').read_text(encoding='utf-8')
    scripts = re.findall(r'<script(?:\s[^>]*)?>([\s\S]*?)</script>', page)
    result = subprocess.run(['node', '-e', HARNESS], input=json.dumps({'source': '\n'.join(scripts)}),
                            capture_output=True, text=True, timeout=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(result.stdout if result.returncode == 0 else result.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
