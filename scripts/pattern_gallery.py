"""Render Crafter cross patterns as a simple image gallery.

This script reads a Hydra config whose ``initial_patterns`` subtree follows the
same shape as ``crafter/conf/tamp_planner/pattern_learning.yaml`` and renders
every configured pattern using the sprite images in ``crafter/assets``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont


_PLACEHOLDER_MATERIALS = (
  'water',
  'grass',
  'stone',
  'path',
  'sand',
  'tree',
  'lava',
  'coal',
  'iron',
  'diamond',
)


def _load_font(font_size: int) -> ImageFont.ImageFont:
  """Load a readable font for labels.

  Parameters
  ----------
  font_size : int
      Requested font size in pixels.

  Returns
  -------
  ImageFont.ImageFont
      Loaded truetype font when available, otherwise PIL's default font.
  """
  for font_name in (
      'DejaVuSans.ttf',
      '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
      '/System/Library/Fonts/Supplemental/Arial.ttf',
  ):
    try:
      return ImageFont.truetype(font_name, font_size)
    except OSError:
      continue
  return ImageFont.load_default()


def _asset_image(asset_dir: Path, name: str, tile_size: int) -> Image.Image:
  path = asset_dir / f'{name}.png'
  assert path.exists(), f'Missing asset image: {path}'
  image = Image.open(path).convert('RGBA')
  return image.resize((tile_size, tile_size), resample=Image.NEAREST)


def _pattern_specs(config: DictConfig) -> list[dict[str, object]]:
  patterns = []
  if bool(config.initial_patterns.add_center_only_placeholders):
    for material in _PLACEHOLDER_MATERIALS:
      patterns.append({
        'kind': 'placeholder',
        'center': material,
        'top': 'unknown',
        'bottom': 'unknown',
        'left': 'unknown',
        'right': 'unknown',
        'label_lines': [
          'placeholder',
          f'center={material}',
          'top=unknown',
          'bottom=unknown',
          'left=unknown',
          'right=unknown',
        ],
      })
  for preset in config.initial_patterns.presets:
      patterns.append({
      'kind': 'preset',
      'center': str(preset.center),
      'top': str(preset.top),
      'bottom': str(preset.bottom),
      'left': str(preset.left),
      'right': str(preset.right),
      'label_lines': [
        f'center={preset.center}',
        f'top={preset.top}',
        f'bottom={preset.bottom}',
        f'left={preset.left}',
        f'right={preset.right}',
      ],
    })
  assert patterns, 'Expected at least one pattern to render.'
  return patterns


def _render_pattern_card(
    asset_dir: Path,
    pattern: dict[str, object],
    tile_size: int,
    card_padding: int,
    label_height: int,
    font_size: int,
) -> Image.Image:
  card_size = tile_size * 3 + card_padding * 2
  card = Image.new('RGBA', (card_size, card_size + label_height), (255, 255, 255, 255))
  draw = ImageDraw.Draw(card)
  font = _load_font(font_size)
  origin = card_padding
  positions = {
    'top': (origin + tile_size, origin),
    'left': (origin, origin + tile_size),
    'center': (origin + tile_size, origin + tile_size),
    'right': (origin + tile_size * 2, origin + tile_size),
    'bottom': (origin + tile_size, origin + tile_size * 2),
  }
  for key, pos in positions.items():
    tile = _asset_image(asset_dir, str(pattern[key]), tile_size)
    card.alpha_composite(tile, pos)
  draw.rectangle((0, 0, card_size - 1, card_size - 1), outline=(180, 180, 180, 255), width=1)
  label_y = card_size + 2
  for line in pattern['label_lines']:
    draw.text((card_padding, label_y), str(line), fill=(0, 0, 0, 255), font=font)
    bbox = draw.textbbox((card_padding, label_y), str(line), font=font)
    label_y += (bbox[3] - bbox[1]) + 2
  return card


def _render_gallery(config: DictConfig) -> Image.Image:
  asset_dir = Path(config.asset_dir).resolve()
  tile_size = int(config.tile_size)
  card_padding = int(config.card_padding)
  label_height = int(config.label_height)
  columns = int(config.columns)
  resolution_scale = int(config.resolution_scale)
  assert resolution_scale >= 1, (
    f'resolution_scale must be >= 1, got {resolution_scale}'
  )
  scaled_tile_size = tile_size * resolution_scale
  scaled_card_padding = card_padding * resolution_scale
  scaled_label_height = label_height * resolution_scale
  font_size = int(config.font_size) * resolution_scale
  patterns = _pattern_specs(config)
  cards = [
    _render_pattern_card(
      asset_dir=asset_dir,
      pattern=pattern,
      tile_size=scaled_tile_size,
      card_padding=scaled_card_padding,
      label_height=scaled_label_height,
      font_size=font_size,
    )
    for pattern in patterns
  ]
  card_width, card_height = cards[0].size
  rows = (len(cards) + columns - 1) // columns
  gallery = Image.new(
    'RGBA',
    (columns * card_width, rows * card_height),
    tuple(config.background_rgba),
  )
  for index, card in enumerate(cards):
    row = index // columns
    col = index % columns
    gallery.alpha_composite(card, (col * card_width, row * card_height))
  return gallery.convert('RGB')


@hydra.main(
  version_base=None,
  config_path='conf',
  config_name='pattern_gallery',
)
def main(config: DictConfig):
  """Render the configured Crafter patterns to one gallery image."""
  gallery = _render_gallery(config)
  output_path = Path(config.output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  gallery.save(output_path)
  print(f'Saved pattern gallery to {output_path}')


if __name__ == '__main__':
  main()
