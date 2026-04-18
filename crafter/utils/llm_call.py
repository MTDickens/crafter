"""LLM call functions."""

import base64
from io import BytesIO
from pathlib import Path
from typing import cast

from omegaconf import DictConfig
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential


def encode_image_path_to_base64(image_path: Path) -> str:
  """Encode an image file (including SVG) to a base64 string."""
  if str(image_path).lower().endswith('.svg'):
    try:
      import cairosvg  # noqa: PLC0415
    except Exception as e:
      msg = (
        'SVG encoding requires `cairosvg` (and a working Cairo backend). '
        f'Failed to import cairosvg for image_path={image_path}.'
      )
      raise ImportError(msg) from e

    # Convert SVG -> raster bytes (PNG) the same way we'd load an SVG into an image

    with open(image_path, 'rb') as f:
      svg_bytes = f.read()

    png_bytes_raw = cairosvg.svg2png(bytestring=svg_bytes)
    if png_bytes_raw is None:
      raise ValueError(f'Failed to convert SVG to PNG bytes: {image_path}')
    png_bytes = cast(bytes, png_bytes_raw)

    # (Optional) ensure it can be loaded as an image
    _ = Image.open(BytesIO(png_bytes))

    return base64.b64encode(png_bytes).decode('utf-8')

  with open(image_path, 'rb') as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_path_to_formatted_base64(image_path: Path) -> str:
  """Encode an image file to a base64 string formatted as a data URL."""
  return f'data:image/jpeg;base64,{encode_image_path_to_base64(image_path)}'


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_completion_text(queries: list[str | Path], llm_cfg: DictConfig) -> str:
  """Get completion text from LLM using Hydra DictConfig."""
  REASONING_MODELS = ['gpt-5.2', 'gemini-3.1-thinking']

  model_name = str(llm_cfg.model_name)
  provider = llm_cfg.get('model_provider', None)
  api_type = str(llm_cfg.api_type).lower()
  temperature = float(llm_cfg.temperature)
  max_completion_tokens = int(llm_cfg.max_completion_tokens)
  reasoning_effort: str = llm_cfg.get('reasoning_effort', None)
  use_openai_sdk: bool = llm_cfg.use_openai_sdk

  assert api_type == 'completion', (
    f"We only support 'completion' api_type for now. Got '{api_type}'."
  )
  assert use_openai_sdk, (
    'Currently we only support using the OpenAI SDK for LLM calls. Set use_openai_sdk=True in your config.'
  )

  if reasoning_effort is not None:
    assert any(x in model_name for x in REASONING_MODELS), (
      'reasoning_effort requires a reasoning model. '
      f"Got model_name='{model_name}', expected one containing one of {REASONING_MODELS}."
    )

  if any(x in model_name for x in REASONING_MODELS):
    assert reasoning_effort is not None, (
      f"A reasoning model was specified but reasoning_effort is None. Got model_name='{model_name}'."
    )

  def construct_content(
    items: list[str | Path],
  ) -> list[dict[str, str | dict[str, str]]]:
    return [
      (
        {'type': 'text', 'text': q}
        if isinstance(q, str)
        else {
          'type': 'image_url',
          'image_url': {'url': encode_image_path_to_formatted_base64(q)},
        }
      )
      for q in items
    ]

  client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)
  content = construct_content(queries)

  provider_extra = {'only': [provider]} if provider else None

  kwargs = {
    'model': model_name,
    'messages': [{'role': 'user', 'content': content}],
    'temperature': temperature,
    'max_completion_tokens': max_completion_tokens,
  }
  if provider_extra is not None:
    kwargs['extra_body'] = provider_extra
  if reasoning_effort is not None:
    kwargs['reasoning_effort'] = reasoning_effort

  completion = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]

  msg = completion.choices[0].message
  if reasoning_effort is None:
    assert msg.content is not None, 'Expected content in completion choices'
    return msg.content

  possible_reasoning_keys = ['reasoning_content', 'reasoning', 'reasoning_text']
  reasoning_content = None
  for key in possible_reasoning_keys:
    reasoning_content = getattr(msg, key, None)
    if reasoning_content:
      break

  if reasoning_content:
    return f'{reasoning_content}\n{msg.content or ""}'.strip()

  assert msg.content is not None, 'Expected content in completion choices'
  return msg.content
