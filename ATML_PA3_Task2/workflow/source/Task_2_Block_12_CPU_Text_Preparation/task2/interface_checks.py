"""CPU checks for prompt boundaries, masks, loss weighting and frozen-Qwen gradients."""
def run_interface_checks(tokenizer):
    import copy
    import torch
    import torch.nn.functional as F
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from .text import prepare_text,splice_batch,example_mean_ce,causal_example_loss,IGNORE_INDEX
    checks=[]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6304)
        records=[prepare_text(tokenizer,'Answer with a short phrase.\nIs this a bus?','yes',16,question='Is this a bus?'),
            prepare_text(tokenizer,'Describe this image.','\nA red bus on a street.\n',64)]
        for record in records:
            assert tokenizer.decode(record['target_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==record['target']
            assert record['user_prefix_length']==3
        checks.append('Native thinking-disabled prompt boundaries and verbatim newline-bearing targets round-trip exactly')
        config=Qwen3Config(vocab_size=len(tokenizer),hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=1024,pad_token_id=tokenizer.pad_token_id,
            bos_token_id=151643,eos_token_id=151645,attention_dropout=0.0,tie_word_embeddings=True)
        config._attn_implementation='eager'
        model=Qwen3ForCausalLM(config).requires_grad_(False).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        visual=torch.randn(2,196,32,requires_grad=True)
        batch=splice_batch(records,model.get_input_embeddings(),visual)
        assert batch['padding_side']=='right'
        for index,record in enumerate(records):
            prefix=record['user_prefix_length']
            assert torch.equal(batch['inputs_embeds'][index,prefix:prefix+196],visual[index])
            start=batch['target_starts'][index]
            assert start==len(record['prompt_ids'])+196
            assert batch['labels'][index,:start].eq(IGNORE_INDEX).all()
            assert batch['labels'][index,start:start+len(record['target_ids'])+1].tolist()==record['target_ids']+[151645]
            assert batch['content_mask'][index].sum().item()==len(record['target_ids'])
            assert not batch['content_mask'][index,start+len(record['target_ids'])]
            assert batch['position_ids'][index,:batch['lengths'][index]].tolist()==list(range(batch['lengths'][index]))
        assert batch['labels'][batch['attention_mask']==0].eq(IGNORE_INDEX).all()
        checks.append('All 196 continuous tokens sit inside the user message before its text, while only original target tokens and EOT carry training labels')
        hidden=model.model(inputs_embeds=batch['inputs_embeds'],attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids'],use_cache=False,return_dict=True).last_hidden_state
        selected=batch['labels'][:,1:]!=IGNORE_INDEX
        targets=batch['labels'][:,1:][selected]
        logits=model.lm_head(hidden[:,:-1][selected])
        example_ids=torch.arange(2)[:,None].expand_as(selected)[selected]
        loss=example_mean_ce(logits,targets,example_ids,2)
        loss.backward()
        assert visual.grad is not None and torch.isfinite(visual.grad).all() and visual.grad.abs().sum().item()>0
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
        checks.append('A real checkpointed Qwen3 model with frozen parameters passes finite nonzero gradients into continuous visual inputs without parameter gradients')

        noimage=splice_batch(records,model.get_input_embeddings(),None)
        assert [a-b for a,b in zip(batch['lengths'],noimage['lengths'])]==[196,196]
        assert [a-b for a,b in zip(batch['target_starts'],noimage['target_starts'])]==[196,196]
        for i in range(2):
            assert batch['labels'][i][batch['labels'][i]!=IGNORE_INDEX].tolist()==noimage['labels'][i][noimage['labels'][i]!=IGNORE_INDEX].tolist()
        checks.append('No-image inputs retain exactly the same text and target labels while removing the 196 visual positions')

        model.eval()
        generation=splice_batch(records,model.get_input_embeddings(),visual.detach(),include_targets=False)
        assert generation['padding_side']=='left' and generation['attention_mask'][:,-1].eq(1).all()
        assert generation['labels'].eq(IGNORE_INDEX).all() and not generation['content_mask'].any()
        assert (generation['attention_mask']==0).any()
        changed=generation['inputs_embeds'].clone()
        changed[generation['attention_mask']==0]=1234
        with torch.no_grad():
            left=model.model(inputs_embeds=generation['inputs_embeds'],attention_mask=generation['attention_mask'],
                position_ids=generation['position_ids'],use_cache=False).last_hidden_state[:,-1]
            right=model.model(inputs_embeds=changed,attention_mask=generation['attention_mask'],
                position_ids=generation['position_ids'],use_cache=False).last_hidden_state[:,-1]
        assert torch.allclose(left,right,rtol=0,atol=1e-6)
        checks.append('Generation uses left padding and correct position/attention masks, with last-prompt states unaffected by masked padding values')

        test_logits=torch.tensor([[4.,0.,0.],[0.,3.,0.],[0.,3.,0.],[0.,3.,0.]],requires_grad=True)
        test_targets=torch.tensor([0,2,2,2])
        ids=torch.tensor([0,1,1,1])
        per_token=F.cross_entropy(test_logits,test_targets,reduction='none')
        expected=(per_token[0]+per_token[1:].mean())/2
        actual=example_mean_ce(test_logits,test_targets,ids,2)
        assert torch.allclose(actual,expected) and not torch.allclose(actual,per_token.mean())
        actual.backward()
        assert test_logits.grad is not None and torch.isfinite(test_logits.grad).all()
        checks.append('Unequal-length examples contribute equal weights through per-example token means, with valid CE gradients')

        toy_logits=torch.randn(2,6,5,requires_grad=True)
        labels=torch.tensor([[-100,-100,1,2,3,-100],[-100,-100,-100,2,3,-100]])
        content=torch.tensor([[False,False,True,True,False,False],[False,False,False,True,False,False]])
        full=causal_example_loss(toy_logits,labels)
        nll=causal_example_loss(toy_logits,labels,content)
        expected_full=(F.cross_entropy(toy_logits[0,1:4],torch.tensor([1,2,3]))+
            F.cross_entropy(toy_logits[1,2:4],torch.tensor([2,3])))/2
        expected_nll=(F.cross_entropy(toy_logits[0,1:3],torch.tensor([1,2]))+
            F.cross_entropy(toy_logits[1,2:3],torch.tensor([2])))/2
        assert torch.allclose(full,expected_full) and torch.allclose(nll,expected_nll)
        checks.append('Causal shifting is correct, training CE includes EOT, and the separate content-only NLL mask excludes EOT')
        try:
            prepare_text(tokenizer,'Describe this image.','<|im_end|>',64)
        except ValueError:
            pass
        else:
            raise AssertionError('Reserved target control token was accepted')
        try:
            prepare_text(tokenizer,'Describe this image.','x '*100,16)
        except ValueError:
            pass
        else:
            raise AssertionError('An over-cap target was accepted')
        checks.append('Reserved control tokens and over-cap original content cause failure without truncation')
    return {'status':'passed','checks':checks,'model':'Real Qwen3 class with one synthetic 32-wide layer on CPU',
        'full_original_Qwen_weights_loaded':False,'gpu_used':False,'training_updates':0}
