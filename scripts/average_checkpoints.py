import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def _load_checkpoint(path):
    return torch.load(path, map_location='cpu')


def _average_state_dicts(state_dicts):
    if not state_dicts:
        raise ValueError('No state_dicts to average.')

    common_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        common_keys &= set(sd.keys())
    if not common_keys:
        raise ValueError('No common parameter keys found across checkpoints.')

    averaged = OrderedDict()
    for key in sorted(common_keys):
        tensors = [sd[key].float() for sd in state_dicts]
        ref = tensors[0]
        if not all(t.shape == ref.shape for t in tensors[1:]):
            continue
        stacked = torch.stack(tensors, dim=0)
        averaged[key] = stacked.mean(dim=0).to(dtype=ref.dtype)
    return averaged


def main():
    parser = argparse.ArgumentParser(description='Average BasicSR checkpoints saved with params/params_ema.')
    parser.add_argument('checkpoints', nargs='+', help='Checkpoint paths to average.')
    parser.add_argument('--output', required=True, help='Output checkpoint path.')
    parser.add_argument(
        '--keys',
        nargs='+',
        default=['params_ema', 'params'],
        help='Parameter groups to average if present. Default: params_ema params',
    )
    args = parser.parse_args()

    checkpoints = [Path(p) for p in args.checkpoints]
    payloads = [_load_checkpoint(str(path)) for path in checkpoints]

    output = {}
    for key in args.keys:
        state_dicts = [payload[key] for payload in payloads if key in payload]
        if not state_dicts:
            continue
        output[key] = _average_state_dicts(state_dicts)
        print(f'Averaged key={key} from {len(state_dicts)} checkpoints, tensors={len(output[key])}')

    if not output:
        raise ValueError(f'None of the requested keys {args.keys} were found in the provided checkpoints.')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, str(output_path))
    print(f'Saved averaged checkpoint to {output_path}')


if __name__ == '__main__':
    main()
