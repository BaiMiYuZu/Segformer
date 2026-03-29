import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def convert_key(key, value, converted):
    if key.startswith('head.'):
        return

    parts = key.split('.')
    if len(parts) < 4 or parts[0] != 'layers':
        converted[key] = value
        return

    stage_idx = int(parts[1]) + 1
    block_group = parts[2]

    if block_group == '0':
        if parts[3] == 'projection':
            converted[f'patch_embed{stage_idx}.proj.{parts[4]}'] = value
            return
        if parts[3] == 'norm':
            converted[f'patch_embed{stage_idx}.norm.{parts[4]}'] = value
            return

    if block_group == '2':
        converted[f'norm{stage_idx}.{parts[3]}'] = value
        return

    if block_group != '1':
        return

    block_idx = parts[3]
    prefix = f'block{stage_idx}.{block_idx}'

    if parts[4] in ('norm1', 'norm2'):
        converted[f'{prefix}.{parts[4]}.{parts[5]}'] = value
        return

    if parts[4] == 'attn':
        if parts[5] == 'attn':
            if parts[6] == 'in_proj_weight':
                embed_dim = value.shape[0] // 3
                converted[f'{prefix}.attn.q.weight'] = value[:embed_dim]
                converted[f'{prefix}.attn.kv.weight'] = value[embed_dim:]
                return
            if parts[6] == 'in_proj_bias':
                embed_dim = value.shape[0] // 3
                converted[f'{prefix}.attn.q.bias'] = value[:embed_dim]
                converted[f'{prefix}.attn.kv.bias'] = value[embed_dim:]
                return
            if parts[6] == 'out_proj':
                converted[f'{prefix}.attn.proj.{parts[7]}'] = value
                return
        if parts[5] in ('sr', 'norm'):
            converted[f'{prefix}.attn.{parts[5]}.{parts[6]}'] = value
            return

    if parts[4] == 'ffn' and parts[5] == 'layers':
        layer_idx = parts[6]
        mapping = {
            '0': 'mlp.fc1',
            '1': 'mlp.dwconv.dwconv',
            '4': 'mlp.fc2',
        }
        if layer_idx in mapping:
            if layer_idx in ('0', '4') and value.ndim == 4:
                value = value.squeeze(-1).squeeze(-1)
            converted[f'{prefix}.{mapping[layer_idx]}.{parts[7]}'] = value


def convert_state_dict(state_dict):
    converted = OrderedDict()
    for key, value in state_dict.items():
        convert_key(key, value, converted)
    return converted


def main():
    parser = argparse.ArgumentParser(
        description='Convert official SegFormer MiT checkpoint to this repo format.')
    parser.add_argument('src', help='Path to the source checkpoint.')
    parser.add_argument('dst', help='Path to save the converted checkpoint.')
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(src_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    converted = convert_state_dict(state_dict)
    torch.save(converted, dst_path)

    print(f'Converted checkpoint saved to: {dst_path}')
    print(f'Converted parameter tensors: {len(converted)}')


if __name__ == '__main__':
    main()
