#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def dt(x):
    return {"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}.get(x,torch.float16)

def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def wjson(p,o):
    p=Path(p); mkdir(p.parent); p.write_text(json.dumps(o,indent=2,ensure_ascii=False),encoding="utf-8")

def wcsv(p,rows):
    p=Path(p); mkdir(p.parent)
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with p.open("w",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=keys); wr.writeheader()
        for r in rows: wr.writerow({k:r.get(k,"") for k in keys})

def fmt(x):
    try: x=float(x)
    except Exception: return "n/a"
    return f"{x:.3e}" if abs(x)>0 and (abs(x)<1e-3 or abs(x)>1e4) else f"{x:.4f}"

def table(h,rows):
    if not rows: return "_No rows._\n"
    s=["| "+" | ".join(h)+" |","| "+" | ".join(["---"]*len(h))+" |"]
    s += ["| "+" | ".join(map(str,r))+" |" for r in rows]
    return "\n".join(s)+"\n"

def parse_layers(spec,n):
    if spec in ("all","*"): return list(range(n))
    out=[]
    for part in spec.split(","):
        part=part.strip()
        if not part: continue
        if "-" in part:
            a,b=map(int,part.split("-",1)); out+=list(range(a,b+1))
        else: out.append(int(part))
    return sorted({x for x in out if 0<=x<n})

def parse_heads(spec,nl,nh):
    out=[]
    if spec in ("all","*"):
        return [(l,h) for l in range(nl) for h in range(nh)]
    for part in spec.split(","):
        part=part.strip()
        if not part: continue
        l,h=part.split(":",1); l=int(l)
        if h=="all": out += [(l,i) for i in range(nh)]
        else: out.append((l,int(h)))
    return sorted({(l,h) for l,h in out if 0<=l<nl and 0<=h<nh})

def dec(tok,i):
    try: return tok.decode([int(i)])
    except Exception: return str(i)

def logits(model,ids):
    with torch.no_grad():
        return model(input_ids=ids,use_cache=False).logits[:,-1,:].detach()

def kl(a,b):
    la=F.log_softmax(a.float(),-1); lb=F.log_softmax(b.float(),-1); p=la.exp()
    return float((p*(la-lb)).sum().detach().cpu())

def rel(a,b):
    return float(((a.float()-b.float()).norm()/(a.float().norm()+1e-9)).detach().cpu())

def dims(model):
    c=model.config
    hs=int(c.hidden_size); nh=int(c.num_attention_heads)
    hd=int(getattr(c,"head_dim",hs//nh)); nl=int(c.num_hidden_layers)
    return hs,nh,hd,nl

def patch_head(model,ids,l,h,hd,pos):
    mod=model.model.layers[l].self_attn.o_proj
    a=h*hd; b=a+hd
    def hook(m,inp):
        x=inp[0]; y=x.clone()
        if pos=="all": y[:,:,a:b]=0
        else: y[:,-1:,a:b]=0
        return (y,)+tuple(inp[1:])
    handle=mod.register_forward_pre_hook(hook)
    try: return logits(model,ids)
    finally: handle.remove()

def collect_mlp_inputs(model,ids,layers):
    cap={}; handles=[]
    for l in layers:
        def make(layer):
            def hook(m,inp):
                cap[layer]=inp[0][:,-1,:].detach().float().cpu()
            return hook
        handles.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(make(l)))
    try: _=logits(model,ids)
    finally:
        for h in handles: h.remove()
    return cap

def top_neurons(model,l,hid,k):
    W=model.model.layers[l].mlp.down_proj.weight.detach().float().cpu()
    sc=hid[0].abs()*W.norm(dim=0)
    vals,idx=torch.topk(sc,min(k,sc.numel()))
    return [int(i) for i in idx.tolist()],[float(v) for v in vals.tolist()]

def patch_mlp_group(model,ids,l,neurons,pos):
    mod=model.model.layers[l].mlp.down_proj
    idx=torch.tensor(neurons,dtype=torch.long,device=model.device)
    def hook(m,inp):
        x=inp[0]; y=x.clone()
        if pos=="all": y.index_fill_(2,idx,0)
        else: y[:,-1,:].index_fill_(1,idx,0)
        return (y,)+tuple(inp[1:])
    handle=mod.register_forward_pre_hook(hook)
    try: return logits(model,ids)
    finally: handle.remove()

def run(args):
    out=Path(args.out_dir); mkdir(out/"tables")
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dt(args.dtype),
        trust_remote_code=True,
        attn_implementation=args.attn_implementation
    ).to(args.device).eval()

    hs,nh,hd,nl=dims(model)
    heads=parse_heads(args.heads,nl,nh)
    mlp_layers=parse_layers(args.mlp_layers,nl)

    ids=tok(args.prompt,return_tensors="pt").input_ids.to(args.device)
    rows=[]; steps=[]; groups=[]

    for step in range(args.generate_steps):
        if ids.shape[1] > args.max_length:
            ids=ids[:,-args.max_length:]

        base=logits(model,ids)
        p=F.softmax(base.float(),-1)[0]
        tid=int(torch.argmax(p).detach().cpu())
        token=dec(tok,tid)
        base_logit=float(base[0,tid].float().detach().cpu())
        base_top=int(torch.argmax(base[0]).detach().cpu())

        print(f"[step {step}] token={token!r} id={tid} p={float(p[tid]):.4f}")
        steps.append({
            "step":step,
            "prefix_len":int(ids.shape[1]),
            "token":token,
            "token_id":tid,
            "prob":float(p[tid].detach().cpu()),
            "base_logit":base_logit
        })

        cap=collect_mlp_inputs(model,ids,mlp_layers)
        mlp_cache={}
        for l in mlp_layers:
            if l not in cap: continue
            ns,sc=top_neurons(model,l,cap[l],args.mlp_top_k)
            mlp_cache[l]=(ns,sc)
            groups.append({
                "step":step,"layer":l,
                "neuron_ids":json.dumps(ns),
                "score_sum":sum(sc),
                "score_max":max(sc) if sc else 0,
                "hidden_abs_mean":float(cap[l].abs().mean())
            })

        for l,h in heads:
            patched=patch_head(model,ids,l,h,hd,args.patch_position)
            pl=float(patched[0,tid].float().detach().cpu())
            top=int(torch.argmax(patched[0]).detach().cpu())
            rows.append({
                "step":step,"token":token,"token_id":tid,
                "component":"head","layer":l,"head":h,"group":"",
                "base_logit":base_logit,"patched_logit":pl,
                "causal_logit_contribution":base_logit-pl,
                "logit_rel":rel(base,patched),
                "kl_orig_to_patch":kl(base,patched),
                "top1_match":1.0 if top==base_top else 0.0,
                "patched_top1_token":dec(tok,top)
            })

        for l,(ns,sc) in mlp_cache.items():
            patched=patch_mlp_group(model,ids,l,ns,args.patch_position)
            pl=float(patched[0,tid].float().detach().cpu())
            top=int(torch.argmax(patched[0]).detach().cpu())
            rows.append({
                "step":step,"token":token,"token_id":tid,
                "component":"mlp_group","layer":l,"head":"","group":f"top{len(ns)}_neurons",
                "base_logit":base_logit,"patched_logit":pl,
                "causal_logit_contribution":base_logit-pl,
                "logit_rel":rel(base,patched),
                "kl_orig_to_patch":kl(base,patched),
                "top1_match":1.0 if top==base_top else 0.0,
                "patched_top1_token":dec(tok,top),
                "neuron_ids":json.dumps(ns),
                "neuron_score_sum":sum(sc),
                "neuron_score_max":max(sc) if sc else 0
            })

        ids=torch.cat([ids,torch.tensor([[tid]],device=ids.device,dtype=ids.dtype)],dim=1)

    wcsv(out/"tables/deep_patch_attribution.csv",rows)
    wcsv(out/"tables/deep_generation_steps.csv",steps)
    wcsv(out/"tables/mlp_neuron_groups.csv",groups)

    pos=sorted(rows,key=lambda r:float(r["causal_logit_contribution"]),reverse=True)
    neg=sorted(rows,key=lambda r:float(r["causal_logit_contribution"]))

    summary={
        "model":args.model,
        "prompt":args.prompt,
        "generate_steps":args.generate_steps,
        "patch_position":args.patch_position,
        "n_rows":len(rows),
        "n_heads":len(heads),
        "n_mlp_layers":len(mlp_layers),
        "steps":steps,
        "top_positive":pos[:30],
        "top_negative":neg[:30],
    }
    wjson(out/"deep_v8_summary.json",summary)

    md=["# Deep why-token report v8\n\n"]
    md.append("Real per-generated-token causal attribution: selected heads and MLP neuron groups are patched, then actual Δlogit/KL/top1 are measured for every generated token.\n\n")
    md.append("## Generated steps\n")
    md.append(table(["step","token","token_id","prob","base_logit"],[
        [r["step"],r["token"],r["token_id"],fmt(r["prob"]),fmt(r["base_logit"])] for r in steps
    ]))
    md.append("\n## Strong positive contributors\n")
    md.append(table(["rank","step","component","layer","head/group","token","Δlogit","KL","top1"],[
        [i+1,r["step"],r["component"],r["layer"],r.get("head") or r.get("group"),r["token"],fmt(r["causal_logit_contribution"]),fmt(r["kl_orig_to_patch"]),fmt(r["top1_match"])]
        for i,r in enumerate(pos[:40])
    ]))
    md.append("\n## Strong negative/suppressing contributors\n")
    md.append(table(["rank","step","component","layer","head/group","token","Δlogit","KL","top1"],[
        [i+1,r["step"],r["component"],r["layer"],r.get("head") or r.get("group"),r["token"],fmt(r["causal_logit_contribution"]),fmt(r["kl_orig_to_patch"]),fmt(r["top1_match"])]
        for i,r in enumerate(neg[:40])
    ]))

    for step in range(args.generate_steps):
        rs=[r for r in rows if int(r["step"])==step]
        pp=sorted(rs,key=lambda r:float(r["causal_logit_contribution"]),reverse=True)[:10]
        nn=sorted(rs,key=lambda r:float(r["causal_logit_contribution"]))[:10]
        tok_step=steps[step]["token"] if step < len(steps) else "?"
        md.append(f"\n## Step {step}: `{tok_step}`\n")
        md.append("Positive:\n")
        md.append(table(["component","layer","head/group","Δlogit","KL"],[
            [r["component"],r["layer"],r.get("head") or r.get("group"),fmt(r["causal_logit_contribution"]),fmt(r["kl_orig_to_patch"])] for r in pp
        ]))
        md.append("Negative:\n")
        md.append(table(["component","layer","head/group","Δlogit","KL"],[
            [r["component"],r["layer"],r.get("head") or r.get("group"),fmt(r["causal_logit_contribution"]),fmt(r["kl_orig_to_patch"])] for r in nn
        ]))

    (out/"WHY_TOKEN_DEEP_REPORT.md").write_text("".join(md),encoding="utf-8")

    mp=[r for r in pos if r["component"]=="mlp_group"][:40]
    mn=[r for r in neg if r["component"]=="mlp_group"][:40]
    mmd=["# MLP group patch report v8\n\n"]
    mmd.append("Top MLP neurons are selected by `abs(hidden_neuron) * norm(W_down[:, neuron])`, patched as a group, then measured by actual Δlogit/KL/top1.\n\n")
    mmd.append("## Positive MLP groups\n")
    mmd.append(table(["rank","step","layer","token","Δlogit","KL","top neurons"],[
        [i+1,r["step"],r["layer"],r["token"],fmt(r["causal_logit_contribution"]),fmt(r["kl_orig_to_patch"]),str(json.loads(r.get("neuron_ids","[]"))[:10])]
        for i,r in enumerate(mp)
    ]))
    mmd.append("\n## Negative/suppressing MLP groups\n")
    mmd.append(table(["rank","step","layer","token","Δlogit","KL","top neurons"],[
        [i+1,r["step"],r["layer"],r["token"],fmt(r["causal_logit_contribution"]),fmt(r["kl_orig_to_patch"]),str(json.loads(r.get("neuron_ids","[]"))[:10])]
        for i,r in enumerate(mn)
    ]))
    (out/"MLP_GROUP_PATCH_REPORT.md").write_text("".join(mmd),encoding="utf-8")

    print("[done]", out)
    print(" - WHY_TOKEN_DEEP_REPORT.md")
    print(" - MLP_GROUP_PATCH_REPORT.md")
    print(" - tables/deep_patch_attribution.csv")
    print(" - deep_v8_summary.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--dtype",default="fp16")
    ap.add_argument("--attn-implementation",default="eager")
    ap.add_argument("--prompt",default="Write a Python function that reverses a linked list.")
    ap.add_argument("--generate-steps",type=int,default=4)
    ap.add_argument("--max-length",type=int,default=192)
    ap.add_argument("--heads",default="23:1,23:4,21:9,23:8,3:6,4:8,16:1,11:11")
    ap.add_argument("--mlp-layers",default="all")
    ap.add_argument("--mlp-top-k",type=int,default=32)
    ap.add_argument("--patch-position",choices=["last","all"],default="last")
    ap.add_argument("--out-dir",default="runs/deep_v8_python")
    run(ap.parse_args())
