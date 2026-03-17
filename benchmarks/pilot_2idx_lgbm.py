#!/usr/bin/env python3
"""2-index fast path with LightGBM + diagonal hints.

Tests the minimal config: std k=3 + red k=5 (33s scoring) with
LightGBM two-tier selection and diagonal-hinted SW alignment.
"""

import ctypes, json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.benchmark_scop_search import (
    classify_hit, compute_roc_n, read_fasta, write_fasta, RankedHit,
)
import numba; numba.set_num_threads(8)
from clustkit.database import load_database
from clustkit.io import read_sequences
from clustkit.kmer_index import (
    build_kmer_index, compute_freq_threshold,
    REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
)
from clustkit.pairwise import BLOSUM62
from clustkit.search import (
    _merge_sequences_for_alignment, _remap_pairs_to_merged, _collect_top_k_hits,
)

_BASE = Path(__file__).resolve().parent.parent / "clustkit" / "csrc"
_klib = ctypes.cdll.LoadLibrary(str(_BASE / "kmer_score.so"))
_klib.batch_score_queries_c.restype = None
_klib.batch_score_queries_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
_swlib = ctypes.cdll.LoadLibrary(str(_BASE / "sw_align.so"))
_swlib.batch_sw_align_c.restype = None
_swlib.batch_sw_align_c.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

def c_score(q, qo, ql, nq, k, alpha, ko, ke, kf, ft, nd, mc, topk):
    ot=np.empty((nq,mc),dtype=np.int32); oc=np.zeros(nq,dtype=np.int32)
    os_=np.zeros((nq,mc),dtype=np.int32); od=np.zeros((nq,mc),dtype=np.int32)
    _klib.batch_score_queries_c(q.ctypes.data,qo.ctypes.data,ql.ctypes.data,nq,k,alpha,
        ko.ctypes.data,ke.ctypes.data,kf.ctypes.data,ft,nd,2,2,10,mc,topk,
        ot.ctypes.data,oc.ctypes.data,os_.ctypes.data,od.ctypes.data)
    return ot,oc,os_,od

def c_sw(mp, fs, off, lens, bw, dh=None):
    M=len(mp); pf=np.ascontiguousarray(mp.flatten())
    si=np.empty(M,dtype=np.float32); sc=np.empty(M,dtype=np.int32)
    mk=np.empty(M,dtype=np.uint8); sm=BLOSUM62.astype(np.int8)
    dh_ptr=dh.ctypes.data if dh is not None else None
    _swlib.batch_sw_align_c(pf.ctypes.data,fs.ctypes.data,off.ctypes.data,
        lens.ctypes.data,M,bw,sm.ctypes.data,0.1,dh_ptr,
        si.ctypes.data,sc.ctypes.data,mk.ctypes.data)
    return si,sc,mk.astype(np.bool_)

def evaluate_roc1(hits_list, metadata):
    di=metadata["domain_info"]; fs=metadata["family_sizes_in_db"]
    qs=set(metadata["query_sids"]); hbq=defaultdict(list)
    for qh in hits_list:
        for h in qh:
            hbq[h.query_id].append((h.target_id, h.score if h.score!=0 else h.identity))
    vals=[]
    for qid in qs:
        qi=di.get(qid)
        if qi is None: continue
        tp=fs.get(str(qi["family"]),1)-1
        if tp<=0: continue
        qh=sorted(hbq.get(qid,[]),key=lambda x:-x[1])
        ranked=[RankedHit(target_id=t,score=s,label=classify_hit(qid,t,di))
                for t,s in qh if classify_hit(qid,t,di)!="IGNORE"]
        vals.append(compute_roc_n(ranked,1,tp))
    return float(np.mean(vals)) if vals else 0.0


def main():
    scop_dir = Path("benchmarks/data/scop_search_results")
    out_dir = Path("benchmarks/data/speed_sensitivity_results")
    with open(scop_dir / "metadata.json") as f: fm=json.load(f)
    random.seed(42)
    qsids=sorted(random.sample(fm["query_sids"],min(2000,len(fm["query_sids"]))))
    all_seqs=dict(read_fasta(scop_dir/"queries.fasta"))
    sub=[(s,all_seqs[s]) for s in qsids if s in all_seqs]
    qf=str(out_dir/"queries_subset.fasta"); write_fasta(sub,qf)
    metadata=dict(fm); metadata["query_sids"]=qsids

    print("Loading database...",flush=True)
    db_index=load_database(out_dir/"clustkit_db")
    query_ds=read_sequences(qf,"protein"); db_ds=db_index.dataset
    nq,nd=query_ds.num_sequences,db_ds.num_sequences
    q_flat=query_ds.flat_sequences; q_off=query_ds.offsets.astype(np.int64)
    q_lens=query_ds.lengths.astype(np.int32)
    mc=8000; topk=200000
    k=int(db_index.params.get("kmer_index_k",db_index.params["kmer_size"]))
    ft=int(compute_freq_threshold(db_index.kmer_freqs,nd,99.5))
    print(f"Loaded {nq} queries, {nd} db\n",flush=True)

    # Build reduced k=5 index
    rq=_remap_flat(q_flat,REDUCED_ALPHA,len(q_flat))
    rd=_remap_flat(db_ds.flat_sequences,REDUCED_ALPHA,len(db_ds.flat_sequences))
    r5o,r5e,r5f=build_kmer_index(rd,db_ds.offsets,db_ds.lengths,5,"protein",alpha_size=REDUCED_ALPHA_SIZE)
    r5ft=int(compute_freq_threshold(r5f,nd,99.5))

    # ── C Scoring: 2 indices ─────────────────────────────────────────
    print("="*80)
    print("Step 1: C scoring (2 indices: std k=3 + red k=5)")
    print("="*80,flush=True)

    t_total=time.perf_counter()
    t0=time.perf_counter()
    ot1,oc1,os1,od1=c_score(q_flat,q_off,q_lens,nq,k,20,
        db_index.kmer_offsets,db_index.kmer_entries,db_index.kmer_freqs,ft,nd,mc,topk)
    t_std=time.perf_counter()-t0

    t0=time.perf_counter()
    ot2,oc2,os2,od2=c_score(rq,q_off,q_lens,nq,5,REDUCED_ALPHA_SIZE,
        r5o,r5e,r5f,r5ft,nd,mc,topk)
    t_r5=time.perf_counter()-t0
    t_score=t_std+t_r5
    print(f"  std k=3: {t_std:.1f}s, red k=5: {t_r5:.1f}s, total: {t_score:.1f}s\n",flush=True)

    # ── Union + features ─────────────────────────────────────────────
    print("="*80)
    print("Step 2: Union + features + diagonal hints")
    print("="*80,flush=True)

    t0=time.perf_counter()
    # Flatten with scores and diags
    def flatten_all(ot,oc,os_,od,nq,nd):
        total=int(oc.sum())
        pk=np.empty(total,dtype=np.int64); sc=np.empty(total,dtype=np.int32)
        dg=np.empty(total,dtype=np.int32)
        p=0
        for qi in range(nq):
            nc=int(oc[qi])
            if nc>0:
                pk[p:p+nc]=np.int64(qi)*nd+ot[qi,:nc].astype(np.int64)
                sc[p:p+nc]=os_[qi,:nc]; dg[p:p+nc]=od[qi,:nc]; p+=nc
        return pk,sc,dg

    pk1,sc1,dg1=flatten_all(ot1,oc1,os1,od1,nq,nd)
    pk2,sc2,dg2=flatten_all(ot2,oc2,os2,od2,nq,nd)

    # Union with score/diag tracking
    all_pk=np.concatenate([pk1,pk2])
    all_sc=np.concatenate([sc1,sc2]).astype(np.float32)
    all_dg=np.concatenate([dg1,dg2])
    all_id=np.concatenate([np.zeros(len(pk1),dtype=np.int8),np.ones(len(pk2),dtype=np.int8)])

    order=np.argsort(all_pk,kind='mergesort')
    all_pk=all_pk[order]; all_sc=all_sc[order]; all_dg=all_dg[order]; all_id=all_id[order]

    changes=np.empty(len(all_pk),dtype=np.bool_)
    changes[0]=True; changes[1:]=all_pk[1:]!=all_pk[:-1]
    upos=np.nonzero(changes)[0]
    n_unique=len(upos)
    pair_ids=np.cumsum(changes)-1

    per_idx=np.zeros((n_unique,2),dtype=np.float32)
    n_indices=np.zeros(n_unique,dtype=np.float32)
    max_score=np.zeros(n_unique,dtype=np.float32)
    best_diag=np.zeros(n_unique,dtype=np.int32)
    best_diag_sc=np.zeros(n_unique,dtype=np.float32)
    np.add.at(n_indices,pair_ids,1.0)
    np.maximum.at(max_score,pair_ids,all_sc)
    for i in range(len(all_pk)):
        pid=pair_ids[i]
        per_idx[pid,all_id[i]]=all_sc[i]
        if all_sc[i]>best_diag_sc[pid]:
            best_diag_sc[pid]=all_sc[i]
            best_diag[pid]=all_dg[i]

    sum_score=per_idx.sum(axis=1)
    unique_pk=all_pk[upos]
    pairs=np.empty((n_unique,2),dtype=np.int32)
    pairs[:,0]=(unique_pk//nd).astype(np.int32)
    pairs[:,1]=(unique_pk%nd).astype(np.int32)

    q_lens_f=q_lens[pairs[:,0]].astype(np.float32)
    t_lens_f=db_ds.lengths[pairs[:,1]].astype(np.float32)
    shorter=np.minimum(q_lens_f,t_lens_f)
    longer=np.maximum(q_lens_f,t_lens_f)
    len_ratio=np.where(longer>0,shorter/longer,0).astype(np.float32)
    len_diff=np.abs(q_lens_f-t_lens_f)

    # Pad to 5 columns for compatibility (zero for missing indices)
    per_idx_5=np.zeros((n_unique,5),dtype=np.float32)
    per_idx_5[:,0]=per_idx[:,0]  # std k=3
    per_idx_5[:,2]=per_idx[:,1]  # red k=5

    features=np.column_stack([per_idx_5,n_indices,max_score,sum_score,
                               len_ratio,len_diff,q_lens_f,t_lens_f])
    t_union=time.perf_counter()-t0
    print(f"  {n_unique} pairs, features {features.shape}, diags tracked, {t_union:.1f}s\n",flush=True)

    # ── Calibration + LightGBM ───────────────────────────────────────
    print("="*80)
    print("Step 3: Calibration (1000 queries) + LightGBM")
    print("="*80,flush=True)

    merged=_merge_sequences_for_alignment(query_ds,db_ds)
    m_off=merged["offsets"].astype(np.int64); m_lens=merged["lengths"].astype(np.int32)

    cal_mask=pairs[:,0]<1000
    cal_p=pairs[cal_mask]; cal_f=features[cal_mask]; cal_d=best_diag[cal_mask]
    cal_m=_remap_pairs_to_merged(cal_p,merged["nq"])
    sk=np.minimum(cal_m[:,0],cal_m[:,1]); so=np.argsort(sk,kind="mergesort")
    cal_m=cal_m[so]; cal_p=cal_p[so]; cal_f=cal_f[so]; cal_d=cal_d[so]

    t0=time.perf_counter()
    _,cal_scores,_=c_sw(cal_m,merged["flat_sequences"],m_off,m_lens,20,cal_d.astype(np.int32))
    t_cal=time.perf_counter()-t0
    print(f"  Cal SW: {len(cal_p)} pairs in {t_cal:.1f}s",flush=True)

    import lightgbm as lgb
    cal_qi=cal_p[:,0]; cal_uq=np.unique(cal_qi)
    np.random.seed(42); np.random.shuffle(cal_uq)
    cal_tq=set(cal_uq[:int(len(cal_uq)*0.8)].tolist())
    cal_tm=np.array([int(q) in cal_tq for q in cal_qi])

    model=lgb.LGBMRegressor(n_estimators=50,max_depth=4,learning_rate=0.05,
                             n_jobs=-1,random_state=42,verbose=-1)
    model.fit(cal_f[cal_tm],cal_scores[cal_tm])
    pred_val=model.predict(cal_f[~cal_tm])
    r=np.corrcoef(cal_scores[~cal_tm],pred_val)[0,1]
    print(f"  LGB-50-d4: r={r:.4f}\n",flush=True)

    # ── Two-tier + C SW with diagonal hints ──────────────────────────
    print("="*80)
    print("Step 4: LGB two-tier + diagonal-hinted C SW")
    print("="*80,flush=True)

    t0=time.perf_counter()
    predicted=model.predict(features)
    t_lgb=time.perf_counter()-t0
    print(f"  LGB inference: {t_lgb:.1f}s\n",flush=True)

    qi_arr=pairs[:,0]

    print(f"  {'Config':35s} {'Score':>6s} {'Union':>6s} {'LGB':>5s} {'SW':>5s} "
          f"{'Total':>6s} {'ROC1':>7s} {'vs MMseqs2':>10s}",flush=True)
    print("  "+"-"*85,flush=True)

    for N,bw in [(3000,20),(5000,20),(5000,30),(8000,20)]:
        sort_key=np.lexsort((-predicted,qi_arr))
        sorted_qi=qi_arr[sort_key]
        keep=np.empty(min(nq*N,len(pairs)),dtype=np.int64)
        out_pos=0; prev=-1; cnt=0
        for i in range(len(sorted_qi)):
            qi=int(sorted_qi[i])
            if qi!=prev: prev=qi; cnt=0
            if cnt<N: keep[out_pos]=sort_key[i]; out_pos+=1; cnt+=1
        keep=keep[:out_pos]

        sel=pairs[keep]; sel_dg=best_diag[keep]
        sel_m=_remap_pairs_to_merged(sel,merged["nq"])
        sk2=np.minimum(sel_m[:,0],sel_m[:,1])
        so2=np.argsort(sk2,kind="mergesort")
        sel_m=sel_m[so2]; sel=sel[so2]; sel_dg=sel_dg[so2]

        t0=time.perf_counter()
        sims,scores,mask=c_sw(sel_m,merged["flat_sequences"],m_off,m_lens,bw,sel_dg.astype(np.int32))
        t_sw=time.perf_counter()-t0

        total=t_score+t_union+t_lgb+t_sw
        p=scores>0
        hits=_collect_top_k_hits(sel[p],sims[p],nq,500,query_ds,db_ds,
                                 passing_scores=scores[p].astype(np.float32))
        roc1=evaluate_roc1(hits,metadata)
        vs=roc1-0.7942

        label=f"N={N}, bw={bw}"
        print(f"  {label:35s} {t_score:5.0f}s {t_union:5.0f}s {t_lgb:4.0f}s {t_sw:4.0f}s "
              f"{total:5.0f}s {roc1:7.4f} {vs:+10.4f}",flush=True)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")
    print(f"  Previous v7.4 (5 indices): 129s, ROC1=0.802")


if __name__ == "__main__":
    main()
