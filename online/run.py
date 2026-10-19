"""List or plan experiments without importing frameworks or contacting APIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from release_runtime import ONLINE, RUNS, checked_output

STUDIES = {
    'e10': 'Component controls: clean / template / records / combined',
    'e13': 'Three layouts under natural retrieval and full content supply',
    'e14': 'Matching versus mismatching condition records',
    'e16': 'DeepSeek: replay / restore / length-matched benign addition',
    'e16-kimi': 'Kimi: restore and benign addition only',
    'e17': 'Five methods at three competition levels',
    'e17mg': 'MemoryGraft adaptation and concurrent fully merged controls',
    'e19q': 'Retrieval-only whole-task / per-object query comparison',
    'e19x': 'ADK execution under query mode × layout',
    'pg': 'PG scans of paired reviewed Chinese/English input projections',
}


def plan(args):
    if args.study == 'e19x' and args.platform != 'adk':
        raise ValueError('E19x execution was ADK only; LangMem uses E19q retrieval.')
    platform = {'adk': 'ADK', 'langmem': 'LangMem'}[args.platform]
    tag = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if Path(tag).name != tag or tag in ('.', '..') or any(c in tag for c in '/\\:'):
        raise ValueError('run-id must be a simple directory name')
    out = checked_output(RUNS / platform / args.study / tag)
    study = 'e16' if args.study == 'e16-kimi' else args.study
    directory = ONLINE / 'experiments' / platform / ('_' + study)
    command = [sys.executable, '-B']
    model = args.model or ('Kimi-K3' if args.study == 'e16-kimi' else 'DeepSeek-V4-Flash-0731')
    commands = []
    if study in ('e10', 'e13', 'e14', 'e16'):
        base = command + [str(directory / 'run_all.py'), '--run-root', str(out), '--model', model]
        if args.scenario:
            base += ['--only-scenario', args.scenario]
        if args.study == 'e16-kimi':
            commands = [base + ['--only-arm', 'restore'], base + ['--only-arm', 'sham', '--resume']]
        else:
            commands = [base]
    elif study in ('e17', 'e17mg'):
        entry = 'run_all_mg.py' if study == 'e17mg' else 'run_all.py'
        commands = [command + [str(directory / entry), '--batch', args.competition, '--run-root', str(out), '--model', model]]
    elif study == 'e19x':
        commands = [command + [str(directory / 'run_all_x.py'), '--batch', 'x-main', '--run-root', str(out), '--model', model],
                    command + [str(directory / 'analyze_x.py'), '--root', str(out), '--batch', 'x-main']]
    elif study == 'e19q':
        base = command + [str(directory / 'retrieval_compare.py')]
        commands = [base + ['--levels', 'base']]
        if args.platform == 'langmem':
            commands.append(base + ['--m', '12', '--w-list', '6', '8', '10'])
    elif study == 'pg':
        commands = [command + [str(ONLINE / 'experiments/_stats/e15_detector_audit/run_audit.py'), '--langs', 'zh,en']]
    return {'study': args.study, 'platform': args.platform, 'output': str(out),
            'model': None if study in ('e19q', 'pg') else model,
            'commands': commands,
            'requires_chat_or_embedding_service': study != 'pg' and (study != 'e19q' or args.platform == 'langmem'),
            'note': 'PG uses locally cached/downloadable classifier weights, not a chat API.' if study == 'pg' else 'Exports are local simulated tool statuses only.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--study', choices=STUDIES)
    parser.add_argument('--platform', choices=['adk', 'langmem'], default='adk')
    parser.add_argument('--scenario', choices=['s1', 's2'])
    parser.add_argument('--model')
    parser.add_argument('--run-id')
    parser.add_argument('--competition', choices=['base', 'mid', 'stress', 'all'], default='all')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--execute', action='store_true', help='Run the displayed commands; this can incur API costs.')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(STUDIES, ensure_ascii=False, indent=2))
        return
    if not args.study:
        parser.error('--study or --list is required')
    try:
        info = plan(args)
    except ValueError as e:
        parser.error(str(e))
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    # Planning never makes directories or imports model code. Execution does.
    env = os.environ.copy()
    env.update(CMI_RUN_ROOT=info['output'], PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    for cmd in info['commands']:
        subprocess.run(cmd, cwd=ONLINE, env=env, check=True)


if __name__ == '__main__':
    main()
