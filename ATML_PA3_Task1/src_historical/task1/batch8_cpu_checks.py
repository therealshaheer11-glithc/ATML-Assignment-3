"""Real native-harness CPU integration, with explicitly synthetic model outputs."""
import json
import time
from pathlib import Path


def run(root, work):
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from batch8_hflm import BatchJournalHFLM, semantic_tokens
    from batch8_pilot import make_requests, prefill_logits
    from baseline_runtime import atomic
    root, work = Path(root), Path(work); work.mkdir(parents=True)
    assert not torch.cuda.is_available()
    torch.set_num_threads(2); torch.manual_seed(6304)
    tokenizer = AutoTokenizer.from_pretrained(root / 'sources/original_model', local_files_only=True)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=4096, tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    questions = ['CPU fixture ' + ('word ' * (i + 1)) + str(i) for i in range(8)]
    requests = make_requests(tokenizer, questions, cap=16)
    answers = {}
    for index, request in enumerate(requests):
        text = '7 Q: extra text' if index == 0 else 'value ' + str(index)
        answers[tuple(tokenizer.encode(request.args[0], add_special_tokens=False))] = (
            tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id])
    state = work / 'state.json'
    atomic(state, {'deadline_wall': time.time() + 600, 'progress_wall': time.time(), 'status': 'running'})
    job = {'root': str(work), 'state': str(state), 'soft_wall': time.time() + 500}

    class Fixture(BatchJournalHFLM):
        calls = 0
        fail_after = None

        def _uncached_generate(self, context, max_length, stop, **kwargs):
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError('INTENTIONAL_CPU_INTERRUPTION_FIXTURE')
            mask = kwargs['attention_mask'].tolist(); generated = []
            for padded, keep in zip(context.tolist(), mask):
                key = tuple(t for t, flag in zip(padded, keep) if flag)
                continuation = answers[key][:max_length - context.shape[1]]
                if context.shape[0] == 1:
                    continuation, _ = semantic_tokens(tokenizer, continuation, stop, [tokenizer.eos_token_id])
                generated.append(continuation)
            width = max(map(len, generated))
            return torch.tensor([prefix + row + [tokenizer.pad_token_id] * (width - len(row))
                for prefix, row in zip(context.tolist(), generated)], dtype=context.dtype)

    def fixture(journal):
        return Fixture(pretrained=model, tokenizer=tokenizer, dtype='float32', batch_size=8,
            enable_thinking=False, add_bos_token=False, job_config=job,
            audit_contract={'condition': 'CPU_FIXTURE_ONLY'}, journal=work / journal, cpu_fixture=True)

    first = fixture('one')
    one_text, one_meta, _ = first.generate_journaled(requests, 1)
    assert first.calls == 8
    many = fixture('eight')
    many_text, many_meta, _ = many.generate_journaled(requests, 8)
    assert many.calls == 1 and one_text == many_text
    assert [r['semantic_new_ids'] for r in one_meta] == [r['semantic_new_ids'] for r in many_meta]
    assert [r['input_ids'] for r in many_meta] == [first.tok_encode(r.args[0]) for r in requests]
    assert many_meta[0]['stock_finish_reason'] == 'stock_stop'
    assert many_meta[0]['semantic_generated_tokens'] < many_meta[0]['raw_generated_tokens']
    again, _, cached = many.generate_journaled(requests, 8)
    assert again == many_text and many.calls == 1 and cached['cache_hits'] == 1
    # More than eight rows with repeats: freeze full native chunk on interruption.
    chunk = [requests[i % 8] for i in range(25)]
    partial = fixture('interrupted'); partial.fail_after = 1
    try: partial.generate_journaled(chunk, 8)
    except RuntimeError as error: assert str(error) == 'INTENTIONAL_CPU_INTERRUPTION_FIXTURE'
    else: raise AssertionError('Interruption fixture did not execute')
    resumed = fixture('interrupted')
    recovered, metadata, times = resumed.generate_journaled(chunk, 8)
    assert len(recovered) == len(metadata) == 25 and resumed.calls == 3 and times['cache_hits'] == 1
    repeated, _, times = resumed.generate_journaled(chunk, 8)
    assert recovered == repeated and resumed.calls == 3 and times['cache_hits'] == 4
    assert all(metadata[i]['input_ids'] == resumed.tok_encode(chunk[i].args[0]) for i in range(25))
    logits = prefill_logits(many, [request.args[0] for request in requests])
    assert logits.shape == (8, len(tokenizer)) and torch.isfinite(logits).all()
    return {'native_harness_batching_and_reordering_checked': True,
        'left_padding_excluded_from_prompt_identity': True,
        'EOS_padding_and_stock_stop_semantics_preserved': True,
        'complete_batch_reused_without_new_inference_calls': True,
        'interrupted_25_question_chunk_preserves_physical_groups': True,
        'all_25_source_order_metadata_rows_recovered': True,
        'native_generation_position_ID_preparation_checked': True,
        'scope': 'CPU synthetic outputs and tiny random FP32 model; NOT original-model numeric/speed evidence',
        'actual_FP16_L4_pilot_passed': False}
