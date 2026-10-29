"""Start/status/stop ONLY the isolated localhost judge workspace on port 8084."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime/judge-workspace-server-20261027'
RECEIPT = RUNTIME / 'server-process.json'
PORT = 8084
URL = f'http://127.0.0.1:{PORT}'
MODULE = 'deploy.judge_workspace:create_app'


def identity(pid):
    command = f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
        capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        # WMI can be unavailable to an ordinary Windows account. Query the same
        # PID through the existing local inference environment; do not infer
        # ownership from a port, a process name, or a stale receipt.
        probe = '''import json,psutil,subprocess,sys
try:
 p=psutil.Process(int(sys.argv[1]))
 print(json.dumps({'ProcessId':p.pid,'ExecutablePath':p.exe(),'CommandLine':subprocess.list2cmdline(p.cmdline())}))
except psutil.NoSuchProcess:
 print('null')
'''
        fallback = subprocess.run([str(ROOT / '.venv-inference/Scripts/python.exe'),
            '-I', '-B', '-c', probe, str(int(pid))], capture_output=True, text=True,
            timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        if fallback.returncode:
            raise RuntimeError('Both process identity lookups failed; no process started or stopped. ' + fallback.stderr.strip())
        return json.loads(fallback.stdout)
    value = json.loads(result.stdout) if result.stdout.strip() else None
    return value


def owns(value):
    return bool(value and str(ROOT) in value.get('CommandLine', '') and
                MODULE in value.get('CommandLine', '') and '--port 8084' in value.get('CommandLine', '') and
                Path(value.get('ExecutablePath') or '').resolve() == (ROOT / '.venv/Scripts/python.exe').resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop'])
    action = parser.parse_args().action
    prior = json.loads(RECEIPT.read_text()) if RECEIPT.is_file() else None
    process = identity(prior['pid']) if prior else None
    if action == 'stop':
        if not owns(process):
            raise SystemExit('No matching project server found; nothing terminated.')
        # Identity above verifies the recorded project-owned venv launcher. Its
        # tree includes the base Python server and the ML venv/base interpreters.
        # Stop that tree only; never use a process-name-wide kill.
        subprocess.run(['taskkill.exe', '/PID', str(prior['pid']), '/T', '/F'], check=True,
                       capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        print('Stopped only the verified project judge server process tree.')
        return
    if action == 'status':
        print(json.dumps({'running': owns(process), 'url': URL, 'pid': prior['pid'] if owns(process) else None}))
        return
    if owns(process):
        print('Already running: ' + URL)
        return
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', PORT)) == 0:
            raise SystemExit('Port 8084 is occupied. Nothing started or terminated. Identify it with Get-NetTCPConnection -LocalPort 8084.')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    tmp = RUNTIME / 'tmp'
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, TEMP=str(tmp), TMP=str(tmp))
    python = ROOT / '.venv/Scripts/python.exe'
    with (RUNTIME / 'server-stdout.txt').open('a', encoding='utf-8') as stdout, (RUNTIME / 'server-stderr.txt').open('a', encoding='utf-8') as stderr:
        proc = subprocess.Popen([str(python), '-B', '-m', 'uvicorn', MODULE, '--factory', '--app-dir', str(ROOT),
            '--host', '127.0.0.1', '--port', str(PORT), '--workers', '1', '--no-access-log'],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    RECEIPT.write_text(json.dumps({'pid': proc.pid, 'module': MODULE, 'url': URL,
                                 'started': time.time(), 'root': str(ROOT)}, indent=2), encoding='utf-8')
    # Bounded startup: four checks only, at most 48 seconds, no background polling.
    for attempt in range(4):
        time.sleep(3 if attempt == 0 else 10)
        if proc.poll() is not None:
            raise SystemExit('Startup exited ' + str(proc.returncode) + '; see ' + str(RUNTIME / 'server-stderr.txt'))
        try:
            with urlopen(URL + '/health', timeout=2) as response:
                health = json.load(response)
            if health.get('ok') and health.get('mode') == 'local-fresh-inference':
                print('Ready: ' + URL)
                return
        except OSError:
            continue
    raise SystemExit('Startup was not confirmed within the bounded check window. Inspect the saved process/log before attempting another start.')


if __name__ == '__main__':
    main()
