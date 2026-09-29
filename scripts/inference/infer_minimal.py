"""Run the official four-base forward example with Evo-2 7B."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', default='ACGT')
    args = parser.parse_args()
    sequence = args.sequence.upper()
    if not sequence or set(sequence) - set('ACGT'):
        parser.error('sequence must contain only A, C, G, T')

    import torch
    from evo2 import Evo2

    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU unavailable. Run on the GPU server after sourcing env.sh.')
    started = time.monotonic()
    model = Evo2('evo2_7b')
    model.model.eval()
    ids = torch.tensor(model.tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0).to('cuda:0')
    with torch.inference_mode():
        outputs, _ = model(ids)
        logits = outputs[0]
    torch.cuda.synchronize()
    if not torch.isfinite(logits).all().item():
        raise RuntimeError('Non-finite logits')
    if tuple(logits.shape[:2]) != (1, len(sequence)):
        raise RuntimeError(f'Unexpected output shape: {logits.shape}')
    dna_ids = torch.tensor(model.tokenizer.tokenize('ACGT'), dtype=torch.long, device=logits.device)
    probs = logits[0, -1, dna_ids].float().softmax(dim=-1).tolist()
    result = {
        'model': 'evo2_7b',
        'sequence': sequence,
        'gpu': torch.cuda.get_device_name(0),
        'logits_shape': list(logits.shape),
        'all_logits_finite': True,
        'next_base_probabilities_conditional_on_ACGT': dict(zip('ACGT', probs)),
        'peak_allocated_gpu_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'elapsed_seconds_including_model_load': time.monotonic() - started,
    }
    destination = Path(__file__).resolve().parents[2] / 'results' / 'minimal.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'Saved: {destination}')


if __name__ == '__main__':
    main()
