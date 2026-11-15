"""Approved live prompt/extraction/generation helpers, without model allocation."""
import math

def initial_messages(question,protocol):
    return [{'role':'system','content':protocol['system']},{'role':'user','content':question+protocol['problem_suffix']}]
def revision_messages(initial,response,protocol):
    return list(initial)+[{'role':'assistant','content':response},{'role':'user','content':protocol['revision_instruction']}]
def expected_native_text(messages):
    return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
def encode_messages(tokenizer,messages):
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    if text!=expected_native_text(messages):raise ValueError('Native chat template rendering changed')
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False,padding=False,truncation=False)
    if hasattr(ids,'keys'):ids=ids['input_ids']
    ids=list(ids)
    if not ids or not all(type(i)==int and i>=0 for i in ids):raise ValueError('Unexpected native tokenization shape')
    return {'rendered_prompt':text,'input_ids':ids}
def extract_last_box(text):
    marker='\\boxed{';start=text.rfind(marker)
    if start<0:return {'answer':None,'reason':'no_box'}
    pos=start+len(marker);depth=1;begin=pos
    while pos<len(text):
        char=text[pos]
        if char=='{':depth+=1
        elif char=='}':
            depth-=1
            if depth==0:
                value=text[begin:pos].strip();return {'answer':value or None,'reason':None if value else 'empty_last_box'}
        pos+=1
    return {'answer':None,'reason':'unclosed_last_box'}
def generation_options(protocol,cap):
    return {'max_new_tokens':cap,'do_sample':True,'temperature':protocol['temperature'],'top_p':protocol['top_p'],'top_k':protocol['top_k'],'repetition_penalty':protocol['repetition_penalty'],'num_beams':1,'num_return_sequences':1,'use_cache':True,'cache_implementation':'dynamic','disable_compile':True,'pad_token_id':151643,'bos_token_id':151643,'eos_token_id':[151645,151643],'return_dict_in_generate':False,'output_scores':False,'output_logits':False,'output_attentions':False,'output_hidden_states':False}
def install_finite_guard(model,torch):
    def check(_module,_inputs,output):
        if not bool(torch.isfinite(output).all()):raise FloatingPointError('Nonfinite model logits; stop session')
    return model.lm_head.register_forward_hook(check)
def check_context(input_ids,cap,max_context=32768):
    if len(input_ids)==0 or len(input_ids)+cap>max_context:raise ValueError('Prompt plus output cap exceeds model context; no silent truncation')
