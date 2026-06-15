#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

def DT(x): return {'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}.get(x,torch.float16)
def mkdir(p): Path(p).mkdir(parents=True,exist_ok=True)
def wjson(p,o): p=Path(p); mkdir(p.parent); p.write_text(json.dumps(o,indent=2,ensure_ascii=False),encoding='utf-8')
def wcsv(p,rows):
    p=Path(p); mkdir(p.parent); keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: keys.append(k); seen.add(k)
    with p.open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys); wr.writeheader()
        for r in rows: wr.writerow({k:r.get(k,'') for k in keys})
def fmt(x):
    try: x=float(x)
    except Exception: return 'n/a'
    return f'{x:.3e}' if abs(x)>0 and (abs(x)<1e-3 or abs(x)>1e4) else f'{x:.4f}'
def table(h,rs):
    if not rs: return '_No rows._\n'
    s=['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']
    s += ['| '+' | '.join(map(str,r))+' |' for r in rs]
    return '\n'.join(s)+'\n'
def parse_layers(spec,n):
    if spec in ('all','*'): return list(range(n))
    out=[]
    for p in spec.split(','):
        p=p.strip()
        if not p: continue
        if '-' in p:
            a,b=map(int,p.split('-',1)); out+=list(range(a,b+1))
        else: out.append(int(p))
    return sorted({x for x in out if 0<=x<n})
def logits(model,ids):
    with torch.no_grad(): return model(input_ids=ids,use_cache=False).logits[:,-1,:].detach()
def kl(a,b):
    la=F.log_softmax(a.float(),-1); lb=F.log_softmax(b.float(),-1); p=la.exp()
    return float((p*(la-lb)).sum().detach().cpu())
def rel(a,b): return float(((a.float()-b.float()).norm()/(a.float().norm()+1e-9)).detach().cpu())
def dec(tok,i):
    try: return tok.decode([int(i)])
    except Exception: return str(i)
def prompts(kind):
    base={
      'code':['Write a Python function that reverses a linked list.','Implement binary search over a sorted list in Python.','Find the bug: def f(x): return x.append(1)'],
      'math':['Solve step by step: if x + 7 = 19, what is x?','Explain the derivative of sin(x) * exp(x).','What is the quadratic formula used for?'],
      'text':['Explain why the sky appears blue in simple words.','Summarize photosynthesis in one paragraph.','Give three reasons exercise helps health.'],
      'dna':['ACGT motif analysis: explain what a promoter-like sequence might contain.','DNA sequence task: identify why GC-rich regions may matter.'],
      'physics':['Explain Newton second law with a simple example.','Why does energy conservation matter in mechanics?']}
    out=[]
    for k in kind.split(','):
        out += [(k,x) for x in base.get(k,[])]
    return out or [('custom',kind)]
def capture(model,ids,layers):
    cap={}; hs=[]
    for l in layers:
        def mk(layer):
            def hook(m,inp): cap[layer]=inp[0][:,-1,:].detach().float().cpu()
            return hook
        hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)))
    try: base=logits(model,ids)
    finally:
        for h in hs: h.remove()
    return base,cap
def patch_group(model,ids,l,neurons):
    mod=model.model.layers[l].mlp.down_proj
    idx=torch.tensor(neurons,dtype=torch.long,device=model.device)
    def hook(m,inp):
        x=inp[0]; y=x.clone(); y[:,-1,:].index_fill_(1,idx,0); return (y,)+tuple(inp[1:])
    h=mod.register_forward_pre_hook(hook)
    try: return logits(model,ids)
    finally: h.remove()
def run(args):
    out=Path(args.out_dir); mkdir(out/'tables')
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=DT(args.dtype),trust_remote_code=True,attn_implementation=args.attn_implementation).to(args.device).eval()
    cfg=model.config; nl=int(cfg.num_hidden_layers); layers=parse_layers(args.layers,nl)
    lm=model.lm_head.weight.detach().float().cpu()
    rows=[]; stable={}; patch=[]
    for suite,text in prompts(args.prompt_suites):
        ids=tok(text,return_tensors='pt').input_ids.to(args.device)
        if ids.shape[1]>args.max_length: ids=ids[:,-args.max_length:]
        base,cap=capture(model,ids,layers); prob=F.softmax(base.float(),-1)[0]
        tid=int(torch.argmax(prob).cpu()); tv=lm[tid]; token=dec(tok,tid); base_logit=float(base[0,tid].float().cpu())
        print(f'[scan] {suite} token={token!r} p={float(prob[tid]):.4f}')
        for l in layers:
            if l not in cap: continue
            h=cap[l][0]
            W=model.model.layers[l].mlp.down_proj.weight.detach().float().cpu()
            col=W.norm(dim=0); direct=torch.mv(W.t(),tv)*h; mag=h.abs()*col
            vals,idx=torch.topk(direct.abs(),min(args.top_k,direct.numel()))
            for rank,j in enumerate(idx.tolist(),1):
                j=int(j); eff=float(direct[j]); m=float(mag[j])
                rec={'suite':suite,'text':text,'layer':l,'neuron':j,'rank_abs_effect':rank,'token':token,'token_id':tid,'direct_logit_effect':eff,'abs_effect':abs(eff),'magnitude_score':m,'hidden':float(h[j]),'wdown_norm':float(col[j])}
                rows.append(rec); key=(l,j); st=stable.setdefault(key,{'layer':l,'neuron':j,'n':0,'sum':0.0,'abs_sum':0.0,'pos':0,'neg':0,'examples':[]})
                st['n']+=1; st['sum']+=eff; st['abs_sum']+=abs(eff); st['pos']+=1 if eff>0 else 0; st['neg']+=1 if eff<0 else 0
                if len(st['examples'])<4: st['examples'].append({'suite':suite,'token':token,'effect':eff})
            # patch top positive and negative groups for selected layers
            if l in layers and args.patch_groups:
                pos=torch.topk(direct,min(args.patch_k,direct.numel())).indices.tolist()
                neg=torch.topk(-direct,min(args.patch_k,direct.numel())).indices.tolist()
                for name,ns in [('positive',pos),('negative',neg)]:
                    patched=patch_group(model,ids,l,[int(x) for x in ns]); pl=float(patched[0,tid].float().cpu()); top=int(torch.argmax(patched[0]).cpu())
                    patch.append({'suite':suite,'text':text,'layer':l,'group':name,'token':token,'token_id':tid,'base_logit':base_logit,'patched_logit':pl,'delta_logit':base_logit-pl,'logit_rel':rel(base,patched),'kl':kl(base,patched),'top1_match':1.0 if top==tid else 0.0,'patched_top1_token':dec(tok,top),'neurons':json.dumps([int(x) for x in ns])})
    stable_rows=[]
    for st in stable.values():
        n=st['n']; sign=max(st['pos'],st['neg'])/max(1,n)
        stable_rows.append({'layer':st['layer'],'neuron':st['neuron'],'n':n,'mean_effect':st['sum']/n,'mean_abs_effect':st['abs_sum']/n,'sign_consistency':sign,'examples':json.dumps(st['examples'],ensure_ascii=False)})
    stable_rows=sorted(stable_rows,key=lambda r:(r['mean_abs_effect'],r['sign_consistency']),reverse=True)
    rows=sorted(rows,key=lambda r:r['abs_effect'],reverse=True)
    patch=sorted(patch,key=lambda r:abs(float(r['delta_logit'])),reverse=True)
    wcsv(out/'tables/full_neuron_top_effects.csv',rows); wcsv(out/'tables/full_neuron_stable.csv',stable_rows); wcsv(out/'tables/full_neuron_patch_groups.csv',patch)
    summary={'model':args.model,'layers':layers,'rows':len(rows),'stable_neurons':len(stable_rows),'patch_rows':len(patch),'top_effects':rows[:30],'top_stable':stable_rows[:30],'top_patch':patch[:30]}
    wjson(out/'full_neuron_scan_v9_summary.json',summary)
    md=['# Full-neuron scan v9\n\n','Scans all MLP neurons for selected layers/prompts, ranks direct token effects, estimates stability, and patches top positive/negative groups.\n\n','## Top direct effects\n',table(['rank','suite','layer','neuron','token','effect','hidden','w_norm'],[[i+1,r['suite'],r['layer'],r['neuron'],r['token'],fmt(r['direct_logit_effect']),fmt(r['hidden']),fmt(r['wdown_norm'])] for i,r in enumerate(rows[:40])]),'\n## Stable neurons\n',table(['rank','layer','neuron','n','mean_abs','sign'],[[i+1,r['layer'],r['neuron'],r['n'],fmt(r['mean_abs_effect']),fmt(r['sign_consistency'])] for i,r in enumerate(stable_rows[:40])]),'\n## Patch groups\n',table(['rank','suite','layer','group','token','delta_logit','KL','top1'],[[i+1,r['suite'],r['layer'],r['group'],r['token'],fmt(r['delta_logit']),fmt(r['kl']),fmt(r['top1_match'])] for i,r in enumerate(patch[:40])])]
    (out/'FULL_NEURON_SCAN_REPORT.md').write_text(''.join(md),encoding='utf-8')
    print('[done]',out)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct'); p.add_argument('--device',default='cuda'); p.add_argument('--dtype',default='fp16'); p.add_argument('--attn-implementation',default='eager'); p.add_argument('--prompt-suites',default='code,math,text'); p.add_argument('--layers',default='all'); p.add_argument('--max-length',type=int,default=192); p.add_argument('--top-k',type=int,default=64); p.add_argument('--patch-k',type=int,default=32); p.add_argument('--patch-groups',action='store_true'); p.add_argument('--out-dir',default='runs/full_neuron_scan_v9')
    run(p.parse_args())
