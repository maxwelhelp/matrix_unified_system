#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
from transformers import AutoModel,AutoModelForCausalLM,AutoModelForSequenceClassification,AutoTokenizer

def get(obj,path):
    cur=obj
    for p in path.split('.'):
        if not hasattr(cur,p): return None
        cur=getattr(cur,p)
    return cur

def find_layers(model):
    paths=['model.layers','model.model.layers','transformer.h','gpt_neox.layers','bert.encoder.layer','roberta.encoder.layer','encoder.layer']
    for p in paths:
        x=get(model,p)
        if x is not None: return p,x
    return None,None

def find_head(model):
    paths=['lm_head','classifier','score','cls.predictions.decoder','qa_outputs']
    for p in paths:
        x=get(model,p)
        if x is not None: return p,x
    return None,None

def layer_map(layer):
    names=dict(layer.named_modules())
    keys=list(names.keys())
    def first(cands):
        for c in cands:
            if c in names: return c
        for k in keys:
            if any(c in k for c in cands): return k
        return None
    return {
      'attn': first(['self_attn','attention','attn','self_attention']),
      'q': first(['q_proj','query','c_attn']),
      'k': first(['k_proj','key','c_attn']),
      'v': first(['v_proj','value','c_attn']),
      'o': first(['o_proj','dense','c_proj']),
      'mlp': first(['mlp','feed_forward','ffn','intermediate']),
      'mlp_down': first(['down_proj','c_proj','dense_4h_to_h','output.dense']),
      'mlp_gate': first(['gate_proj','w1']),
      'mlp_up': first(['up_proj','w3','intermediate.dense']),
    }

def shape(x):
    if hasattr(x,'weight'): return list(x.weight.shape)
    return str(type(x).__name__)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--task',default='causal_lm'); ap.add_argument('--out-dir',default='runs/universal_model_probe_v1'); ap.add_argument('--trust-remote-code',action='store_true')
    a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    cls=AutoModelForCausalLM if a.task=='causal_lm' else AutoModelForSequenceClassification if a.task=='seq_cls' else AutoModel
    model=cls.from_pretrained(a.model,trust_remote_code=a.trust_remote_code,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    lp,layers=find_layers(model); hp,head=find_head(model)
    rec={'model':a.model,'class':model.__class__.__name__,'layers_path':lp,'n_layers':len(layers) if layers is not None else 0,'head_path':hp,'head_shape':shape(head) if head is not None else None,'layers':[]}
    if layers is not None:
        for i,l in enumerate(layers[:min(4,len(layers))]):
            m=layer_map(l); info={'layer':i,'map':m,'shapes':{}}
            mods=dict(l.named_modules())
            for k,v in m.items():
                if v and v in mods: info['shapes'][k]=shape(mods[v])
            rec['layers'].append(info)
    (out/'model_probe.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# Universal model probe v1\n',f"model={a.model}\nclass={rec['class']}\n",f"layers_path={lp} n_layers={rec['n_layers']}\n",f"head_path={hp} head_shape={rec['head_shape']}\n\n"]
    for r in rec['layers']:
        md.append(f"## Layer {r['layer']}\nmap={r['map']}\nshapes={r['shapes']}\n")
    (out/'model_probe.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps(rec,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
