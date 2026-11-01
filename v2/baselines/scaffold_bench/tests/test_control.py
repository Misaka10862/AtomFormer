import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import validate_attempts
from stage2 import Supervisor, cpu_capacity_from_limits


class ControlTests(unittest.TestCase):
    def test_legacy_protocol_rejected_before_hardware_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            s = Supervisor({'run_root': d}, shutdown=lambda: self.fail('unexpected shutdown'))
            with self.assertRaisesRegex(RuntimeError, 'versioned scaffold'):
                s.preflight()

    def test_aggregate_rejects_legacy_shards(self):
        from worker import aggregate
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'shards').mkdir()
            (root / 'train.jsonl').write_text('')
            row = {'record_id': 0, 'sample_id': 0, 'valid': True,
                   'generated': 'Cc1ccccc1', 'core_match': False, 'core_retention': True}
            (root / 'shards/000000000.json').write_text(json.dumps([row]))
            cfg = {'expected_records': {'test': 1}, 'samples_per_record': 1}
            with self.assertRaisesRegex(RuntimeError, 'legacy/mixed'):
                aggregate(cfg, root, 1)
            self.assertFalse((root / 'metrics.json').exists())

    def test_cpu_capacity_uses_cgroup_quota_not_omp_nproc(self):
        self.assertEqual(cpu_capacity_from_limits(128, '1600000 100000'), 16)
        self.assertEqual(cpu_capacity_from_limits(128, '100000 100000'), 1)
        self.assertEqual(cpu_capacity_from_limits(16, 'max 100000'), 16)

    def test_attempts(self):
        rows = [{'record_id': i, 'sample_id': j, 'valid': False} for i in range(2) for j in range(3)]
        self.assertEqual(validate_attempts(rows, 2, 3), 6)
        for broken in (rows[:-1], rows + rows[:1], rows[:-1] + [{'record_id': 9, 'sample_id': 0}]):
            with self.assertRaises(ValueError):
                validate_attempts(broken, 2, 3)

    def run_case(self, exception):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            s = Supervisor({'run_root': d}, shutdown=lambda: calls.append('shutdown'))
            s.monitor = lambda: None
            s.preflight = lambda: None
            def pipeline():
                if exception:
                    raise exception
                Path(d, 'result.json').write_text('{}')
            s.pipeline = pipeline
            with patch('stage2.os.sync'):
                if exception:
                    with self.assertRaises(type(exception)):
                        s.run()
                else:
                    s.run()
            state = json.loads(Path(d, 'state.json').read_text())
            self.assertEqual(state['status'], 'FAILED' if exception else 'COMPLETE')
            self.assertEqual(calls, ['shutdown'])
            if exception:
                self.assertIsNotNone(state['error'])

    def test_success_shutdown(self): self.run_case(None)
    def test_failure_shutdown(self): self.run_case(RuntimeError('simulated failure'))
    def test_interrupt_shutdown(self): self.run_case(InterruptedError('simulated signal'))
    def test_shutdown_failure_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            s = Supervisor({'run_root': d}, shutdown=lambda: (_ for _ in ()).throw(RuntimeError('shutdown denied')))
            s.monitor = lambda: None
            s.preflight = lambda: None
            s.pipeline = lambda: None
            with patch('stage2.os.sync'), self.assertRaises(RuntimeError): s.run()
            self.assertTrue(Path(d, 'shutdown_failed.json').exists())

    def test_pipeline_advances_both_methods(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {'run_root': d, 'python': {'libinvent': 'unused', 'promptsmiles': 'unused'},
                   'cpu_workers': 16, 'overlap_workers': 14, 'expected_records': {'test': 4}, 'samples_per_record': 2}
            s = Supervisor(cfg, shutdown=lambda: None)
            s.preflight = lambda: None
            calls = []
            def task(method, task, workers):
                calls.append((method, task))
                root = Path(d, method)
                root.mkdir(exist_ok=True)
                if task == 'prepare':
                    for name in ('train.jsonl', 'validation.jsonl', 'test.jsonl', 'dataset_manifest.json'):
                        root.joinpath(name).write_text('{}')
                if task == 'aggregate':
                    root.joinpath('metrics.json').write_text(json.dumps({'num_generated': 8, 'record_coverage': 4}))
            s.task = task
            s.pipeline()
            self.assertTrue(Path(d, 'artifacts.json').exists())
            for method in cfg['python']:
                for task_name in ('probe', 'train', 'rl', 'evaluate', 'aggregate'):
                    self.assertEqual(calls.count((method, task_name)), 1)

    def test_child_nonzero_exit_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            s = Supervisor({'run_root': d, 'task_timeout_hours': 1}, shutdown=lambda: None)
            with self.assertRaises(RuntimeError):
                s.command([sys.executable, '-c', 'raise SystemExit(7)'], Path(d, 'task.log'), timeout=5)
            self.assertFalse(s.children)

    def test_startup_failure_never_shuts_down(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            s = Supervisor({'run_root': d}, shutdown=lambda: calls.append('shutdown'))
            def fail():
                raise RuntimeError('no GPU / incompatible data')
            s.preflight = fail
            with self.assertRaises(RuntimeError):
                s.run()
            self.assertEqual(calls, [])
            self.assertFalse(Path(d, 'state.json').exists())
            self.assertFalse(json.loads(Path(d, 'startup_failed.json').read_text())['shutdown'])

    def test_child_exit_race_does_not_abort_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            s = Supervisor({'run_root': d}, shutdown=lambda: None)
            p = Mock(pid=123456)
            p.poll.return_value = None
            p.wait.return_value = 0
            s.children.add(p)
            with patch('stage2.os.killpg', side_effect=ProcessLookupError):
                s.terminate_children()
            p.wait.assert_called_once()

    def test_failure_diagnostics_precede_delayed_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            s = Supervisor({'run_root': d, 'failure_shutdown_delay_seconds': 600},
                           shutdown=lambda: self.assertTrue(Path(d, 'failure.json').exists()))
            s.preflight = lambda: None
            s.monitor = lambda: None
            s.last_task = {'method': 'libinvent', 'task': 'prepare'}
            Path(d, 'prepare.log').write_text('specific traceback record 15596')
            def fail():
                raise ValueError('conversion failed')
            s.pipeline = fail
            with patch('stage2.time.sleep') as sleep, patch('stage2.os.sync'), self.assertRaises(ValueError):
                s.run()
            sleep.assert_called_once_with(600)
            failure = json.loads(Path(d, 'failure.json').read_text())
            self.assertEqual(failure['task']['task'], 'prepare')
            self.assertIn('15596', failure['log_tails']['prepare.log'])

    def test_configured_failure_keeps_server_running(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            s = Supervisor({'run_root': d, 'shutdown_on_failure': False}, shutdown=lambda: calls.append('shutdown'))
            s.preflight = lambda: None
            s.monitor = lambda: None
            s.pipeline = lambda: (_ for _ in ()).throw(RuntimeError('diagnostic failure'))
            with patch('stage2.os.sync'), self.assertRaises(RuntimeError):
                s.run()
            self.assertEqual(calls, [])
            pending = json.loads(Path(d, 'shutdown_pending.json').read_text())
            self.assertFalse(pending['shutdown_on_failure'])

    def test_configured_success_defers_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            s = Supervisor({'run_root': d, 'shutdown_on_complete': False},
                           shutdown=lambda: calls.append('shutdown'))
            s.preflight = lambda: None
            s.monitor = lambda: None
            s.pipeline = lambda: None
            with patch('stage2.os.sync'):
                s.run()
            self.assertEqual(calls, [])
            state = json.loads(Path(d, 'state.json').read_text())
            self.assertEqual(state['status'], 'COMPLETE')
            self.assertFalse(state['shutdown'])

    def test_reuse_supervised_is_checksum_guarded(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d, 'source')
            target = Path(d, 'target')
            method_dir = source / 'libinvent'
            method_dir.mkdir(parents=True)
            for name, content in [('supervised.done.json', '{}'), ('supervised_best.pt', 'checkpoint')]:
                (method_dir / name).write_text(content)
            from common import sha256
            artifacts = {name: sha256(method_dir / name) for name in ('supervised.done.json', 'supervised_best.pt')}
            (method_dir / 'train.task.json').write_text(json.dumps({'artifacts': artifacts}))
            s = Supervisor({'run_root': str(target), 'reuse_supervised_from': str(source)}, shutdown=lambda: None)
            s.reuse_supervised('libinvent')
            marker = json.loads((target / 'libinvent/train.task.json').read_text())
            self.assertEqual(marker['artifacts'], artifacts)
            (method_dir / 'supervised_best.pt').write_text('changed')
            with self.assertRaises(RuntimeError):
                s.reuse_supervised('libinvent')


if __name__ == '__main__': unittest.main()
