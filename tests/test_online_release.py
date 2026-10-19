"""Release regression checks: standard library only, no model/API calls."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / 'online'
sys.path.insert(0, str(ONLINE))
import release_runtime as rt


def load(relative):
    path = ONLINE / relative
    spec = importlib.util.spec_from_file_location('release_test_' + hashlib.sha1(str(path).encode()).hexdigest(), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OnlineReleaseTests(unittest.TestCase):
    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            rt.checked_output(ROOT / 'formal-data')
        with self.assertRaises(ValueError):
            rt.checked_output(rt.RUNS / '..' / '..' / 'outside')

    def test_direct_e16_driver_rejects_external_root_before_writing(self):
        outside = ROOT / ('forbidden-' + uuid.uuid4().hex)
        for platform in ('ADK', 'LangMem'):
            driver = load(f'experiments/{platform}/_e16/run_all.py')
            with self.assertRaises(ValueError):
                driver.main(['--run-root', str(outside), '--model', 'unused'])
            self.assertFalse(outside.exists())

    def test_single_episode_checker_path_is_guarded(self):
        from argparse import Namespace
        with self.assertRaises(ValueError):
            rt.validate_output_args(Namespace(run_dir=ROOT / 'formal-data'))

    def test_fresh_episode_has_empty_log_and_rejects_reuse(self):
        base = ROOT / 'outputs' / 'online' / 'tests'
        base.mkdir(parents=True, exist_ok=True)
        directory = base / uuid.uuid4().hex
        directory.mkdir()
        try:
            with patch.object(rt, 'RUNS', directory.resolve()):
                p = rt.prepare_fresh_episode(directory / 'episode')
                self.assertEqual((p / 'tool_calls.jsonl').read_text(), '')
                (p / 'tool_calls.jsonl').write_text('old evidence')
                with self.assertRaises(FileExistsError):
                    rt.prepare_fresh_episode(p)
                self.assertEqual((p / 'tool_calls.jsonl').read_text(), 'old evidence')
        finally:
            (directory / 'episode' / 'tool_calls.jsonl').unlink(missing_ok=True)
            if (directory / 'episode').exists():
                (directory / 'episode').rmdir()
            directory.rmdir()

    def test_all_six_early_runners_use_fresh_episode_guard(self):
        for platform in ('ADK', 'LangMem'):
            for exp in ('e10', 'e13', 'e14'):
                text = (ONLINE / f'experiments/{platform}/_{exp}/run_episode.py').read_text(encoding='utf8')
                self.assertIn('prepare_fresh_episode(run_dir)', text)
                self.assertNotIn('if tool_calls:', text)

    def test_e13_exposure_checks_each_request_and_duplicate_blocks(self):
        for platform in ('ADK', 'LangMem'):
            m = load(f'experiments/{platform}/_e13/check_episode.py')
            inst = m._load_module('s1').INSTANCES[0]
            full = '[ID:T] ' + ' '.join(f'recon-sync {c} rc=0' for c in inst['codes'])
            tag = 'PAST_CONVERSATIONS' if platform == 'ADK' else 'memories'
            block = f'<{tag}>\n{full}\n</{tag}>'
            def request(text):
                return {'payload': {'contents': [{'parts': [{'text': text}]}]}} if platform == 'ADK' else {'payload': {'system_prompt': text}}
            def evaluate(requests):
                if platform == 'ADK':
                    return m.p2_and_exposure(requests, 's1', 1, 'B0')
                return m.p2_and_exposure([{'value': {'content': full}}], requests, 's1', 1, 'B0')
            self.assertTrue(evaluate([request(block), request(block)])['full_exposure_all_requests'])
            self.assertFalse(evaluate([request(block), request('')])['full_exposure_all_requests'])
            self.assertFalse(evaluate([request(block), request(block + '\n' + block)])['full_exposure_all_requests'])
            self.assertFalse(evaluate([])['full_exposure_all_requests'])

    def test_snapshots_are_complete_and_hash_valid(self):
        for platform in ('ADK', 'LangMem'):
            snapshots = list((ONLINE / f'experiments/{platform}/_e16/snapshots').glob('s*/epi-*.json'))
            self.assertEqual(len(snapshots), 60)
            identities = set()
            for p in snapshots:
                d = json.loads(p.read_text(encoding='utf8'))
                expected = d.pop('snapshot_sha256')
                digest = hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                self.assertEqual(digest, expected)
                identities.add((d['scenario'], d['episode_index']))
            self.assertEqual(len(identities), 60)

    def test_release_freezes_are_portable_and_current(self):
        for platform in ('ADK', 'LangMem'):
            for study, filename in (('e17', 'freeze.json'), ('e17mg', 'freeze_mg.json')):
                path = ONLINE / f'experiments/{platform}/_{study}/{filename}'
                receipt = json.loads(path.read_text(encoding='utf8'))
                self.assertIn('not the historical', receipt['receipt_kind'])
                base = ONLINE / 'experiments' if study == 'e17' else ONLINE
                sections = ('files',) if study == 'e17' else ('new_files', 'reused_from_e17_v1', 'shared_e17_materials', 'e11_scene_templates')
                for section in sections:
                    for relative, digest in receipt.get(section, {}).items():
                        self.assertNotIn('\\', relative)
                        self.assertEqual(hashlib.sha256((base / relative).read_bytes()).hexdigest()[:16], digest)

    def test_reviewed_detector_projections_are_complete(self):
        directory = ONLINE / 'experiments/_stats/e12_request_guard_audit'
        rows = [json.loads(x) for x in (directory / 'projections_full.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertEqual(len(rows), 240)
        self.assertEqual(len({(r['platform'], r['scenario'], r['case'], r['episode']) for r in rows}), 240)
        for r in rows:
            p = directory / r['proj_file']
            self.assertTrue(p.is_file())
            # Historical projection hashes were over normalized text, not CRLF bytes.
            self.assertEqual(hashlib.sha256(p.read_text(encoding='utf8').encode()).hexdigest(), r['projection_sha256'])
            self.assertNotIn('sources', r)

    def test_list_and_all_dry_plans_without_model_dependencies(self):
        environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        listed = subprocess.run([sys.executable, '-B', str(ONLINE / 'run.py'), '--list'], capture_output=True, text=True, check=True, env=environment)
        for study in json.loads(listed.stdout):
            result = subprocess.run([sys.executable, '-B', str(ONLINE / 'run.py'), '--study', study, '--platform', 'adk', '--run-id', 'unit-test-plan', '--dry-run'], capture_output=True, text=True, check=True, env=environment)
            info = json.loads(result.stdout)
            self.assertTrue(info['commands'])
            self.assertFalse(Path(info['output']).exists())
            for command in info['commands']:
                self.assertTrue(Path(command[2]).is_file())

    def test_no_automatic_dotenv_or_original_absolute_paths(self):
        for p in (ONLINE / 'experiments').rglob('*.py'):
            text = p.read_text(encoding='utf8')
            tree = ast.parse(text)
            self.assertNotIn('env_path.read_text', text, str(p))
            self.assertNotIn('load_dotenv(', text, str(p))
            self.assertNotIn('D:\\paper-4A', text, str(p))
            self.assertNotIn('D:/paper-4A', text, str(p))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == 'load_secrets_env':
                    self.assertFalse(any(isinstance(n, ast.Call) for n in ast.walk(node)), str(p))


if __name__ == '__main__':
    unittest.main()
