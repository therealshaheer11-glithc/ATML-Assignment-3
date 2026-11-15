"""Approved architecture, curriculum and optimizer settings."""
import math

SEED = 6304
MICROBATCH = 8
ACCUMULATION = 2
EFFECTIVE_BATCH = 16
VISUAL_TOKENS = 196
VISION_WIDTH = 768
LANGUAGE_WIDTH = 1024
STAGES = {
    1: {'updates': 2250, 'warmup': 113, 'connector_lr': 1e-3, 'lora_lr': None,
        'mixture': [('caption', 1.0)]},
    2: {'updates': 1500, 'warmup': 75, 'connector_lr': 1e-4, 'lora_lr': 2e-5,
        'mixture': [('presence', 0.5), ('counting', 0.25), ('caption', 0.25)]},
    3: {'updates': 750, 'warmup': 38, 'connector_lr': 5e-5, 'lora_lr': 1e-5,
        'mixture': [('natural', 0.7), ('presence', 0.1), ('counting', 0.1), ('caption', 0.1)]},
}
DEV_UPDATES = (0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500)
MODEL_REVISIONS = {
    'qwen': {'repo_id': 'Qwen/Qwen3-0.6B', 'revision': 'c1899de289a04d12100db370d81485cdf75e47ca'},
    'siglip': {'repo_id': 'google/siglip-base-patch16-224', 'revision': '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'},
}


def learning_rate(peak, stage, successful_stage_update):
    """Return the rate to set before this one-based successful optimizer update."""
    config = STAGES[stage]
    update = successful_stage_update
    if type(update) is not int or not 1 <= update <= config['updates'] or peak <= 0:
        raise ValueError('Invalid learning-rate update or peak.')
    warmup = config['warmup']
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (config['updates'] - warmup)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def vqa_user_text(question):
    return 'Answer with a short phrase.\n' + question
