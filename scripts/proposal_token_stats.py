"""Count token usage for saved Crafter proposal prompt/response logs."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import hydra
from omegaconf import DictConfig


_SUPPORTED_MODELS = ('gpt-5.4', 'qwen3_vl')
_QWEN3_VL_TOKENIZER = 'Qwen/Qwen3-VL-235B-A22B-Instruct'
_PROPOSAL_DIR_RE = re.compile(r'^after-episode-(\d+)$')


def _gpt54_counter() -> Callable[[str], int]:
  try:
    import tiktoken  # noqa: PLC0415
  except ImportError as exc:
    raise ImportError(
      'model=gpt-5.4 requires the optional `tiktoken` package. '
      'Install it in this environment, for example: `uv add tiktoken`.'
    ) from exc
  encoding = tiktoken.encoding_for_model('gpt-5-4')
  return lambda text: len(encoding.encode(text))


def _qwen3_vl_counter() -> Callable[[str], int]:
  try:
    from transformers import AutoTokenizer  # noqa: PLC0415
  except ImportError as exc:
    raise ImportError(
      'model=qwen3_vl requires the optional `transformers` package. '
      'Install it in this environment, for example: `uv add transformers`.'
    ) from exc
  try:
    tokenizer = AutoTokenizer.from_pretrained(
      _QWEN3_VL_TOKENIZER,
      trust_remote_code=True,
    )
  except Exception as exc:
    raise RuntimeError(
      f'Failed to load Qwen3-VL tokenizer `{_QWEN3_VL_TOKENIZER}`. '
      'Make sure the tokenizer is cached locally or this environment can '
      'download it from Hugging Face.'
    ) from exc
  return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _token_counter(model: str) -> Callable[[str], int]:
  if model == 'gpt-5.4':
    return _gpt54_counter()
  if model == 'qwen3_vl':
    return _qwen3_vl_counter()
  raise AssertionError(
    f'Unsupported model: {model!r}. Expected one of: {", ".join(_SUPPORTED_MODELS)}.'
  )


def _timestamp_dir(input_root: Path, timestamp: str) -> Path:
  timestamp_dir = input_root / timestamp
  assert timestamp_dir.exists(), f'Timestamp directory does not exist: {timestamp_dir}'
  return timestamp_dir


def _proposal_episode_index(proposal_dir: Path) -> int:
  match = _PROPOSAL_DIR_RE.match(proposal_dir.name)
  assert match is not None, f'Invalid proposal directory name: {proposal_dir}'
  return int(match.group(1))


def _complete_proposal_dirs(timestamp_dir: Path) -> list[Path]:
  proposals_dir = timestamp_dir / 'proposals'
  proposal_dirs = []
  if proposals_dir.exists():
    proposal_dirs = [
      path
      for path in proposals_dir.iterdir()
      if path.is_dir()
      and _PROPOSAL_DIR_RE.match(path.name)
      and (path / 'input.txt').exists()
      and (path / 'output.txt').exists()
    ]
  proposal_dirs.sort(key=_proposal_episode_index)
  assert proposal_dirs, f'No complete proposal input/output pairs found under {proposals_dir}'
  return proposal_dirs


def _proposal_record(
    timestamp: str,
    proposal_dir: Path,
    count_tokens: Callable[[str], int],
) -> dict:
  input_text = (proposal_dir / 'input.txt').read_text()
  output_text = (proposal_dir / 'output.txt').read_text()
  input_tokens = int(count_tokens(input_text))
  output_tokens = int(count_tokens(output_text))
  return {
    'timestamp': timestamp,
    'proposal_dir': str(proposal_dir),
    'episode_index': _proposal_episode_index(proposal_dir),
    'input_tokens': input_tokens,
    'output_tokens': output_tokens,
    'total_tokens': input_tokens + output_tokens,
  }


def _summary(records: list[dict], timestamp: str) -> dict:
  proposal_count = len(records)
  input_tokens = sum(int(record['input_tokens']) for record in records)
  output_tokens = sum(int(record['output_tokens']) for record in records)
  total_tokens = input_tokens + output_tokens
  assert proposal_count > 0, f'Cannot summarize empty proposal record list for {timestamp}'
  return {
    'timestamp': timestamp,
    'proposal_count': proposal_count,
    'input_tokens': input_tokens,
    'output_tokens': output_tokens,
    'total_tokens': total_tokens,
    'avg_input_tokens': input_tokens / proposal_count,
    'avg_output_tokens': output_tokens / proposal_count,
    'avg_total_tokens': total_tokens / proposal_count,
  }


def _collect_stats(config: DictConfig) -> dict:
  input_root = Path(config.input_root).resolve()
  model = str(config.model)
  count_tokens = _token_counter(model)
  timestamp_summaries = []
  proposal_records = []
  for timestamp_value in config.timestamps:
    timestamp = str(timestamp_value)
    timestamp_records = [
      _proposal_record(timestamp, proposal_dir, count_tokens)
      for proposal_dir in _complete_proposal_dirs(_timestamp_dir(input_root, timestamp))
    ]
    timestamp_summaries.append(_summary(timestamp_records, timestamp))
    proposal_records.extend(timestamp_records)
  return {
    'model': model,
    'input_root': str(input_root),
    'timestamp_summaries': timestamp_summaries,
    'overall_summary': _summary(proposal_records, 'OVERALL'),
    'proposal_records': proposal_records,
  }


def _print_table(payload: dict) -> None:
  rows = list(payload['timestamp_summaries']) + [payload['overall_summary']]
  headers = (
    'timestamp',
    'proposal_count',
    'input_tokens',
    'output_tokens',
    'total_tokens',
    'avg_input_tokens',
    'avg_output_tokens',
    'avg_total_tokens',
  )
  widths = {
    header: max(len(header), *(len(_format_cell(row[header])) for row in rows))
    for header in headers
  }
  print('  '.join(header.rjust(widths[header]) for header in headers))
  print('  '.join('-' * widths[header] for header in headers))
  for row in rows:
    print('  '.join(_format_cell(row[header]).rjust(widths[header]) for header in headers))


def _format_cell(value) -> str:
  if isinstance(value, float):
    return f'{value:.2f}'
  return str(value)


def _write_output(config: DictConfig, payload: dict) -> None:
  output_path_value = config.get('output_path', None)
  if output_path_value is None:
    return
  output_path = Path(str(output_path_value)).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open('w') as file:
    json.dump(payload, file, indent=2, sort_keys=True)
  print(f'Wrote token stats to {output_path}')


@hydra.main(
  version_base=None,
  config_path='conf',
  config_name='proposal_token_stats',
)
def main(config: DictConfig):
  """Count proposal prompt and response tokens from saved text logs."""
  assert config.timestamps, 'Expected at least one timestamp.'
  try:
    payload = _collect_stats(config)
  except (AssertionError, ImportError, RuntimeError) as exc:
    print(f'Error: {exc}', file=sys.stderr)
    raise SystemExit(1) from None
  _print_table(payload)
  _write_output(config, payload)


if __name__ == '__main__':
  main()
