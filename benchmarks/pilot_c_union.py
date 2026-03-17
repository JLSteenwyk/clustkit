#!/usr/bin/env python3
"""C union + optimized 3-index and 5-index configs.

Tests:
A) 5-index + C union + LGB + diag hints bw=20 (max sensitivity)
B) 3-index + C union + LGB (1000 cal) + diag hints bw=20 (speed target)
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
    build_kmer_index, compute_freq_threshold, build_kmer_index_spaced,
    REDUCED_ALPHA, REDUCED_ALPHA_SIZE, _remap_flat,
)
from clustkit.pairwise import BLOSUM62
from clustkit.search import (
    _merge_sequences_for_alignment, _remap_pairs_to_merged, _collect_top_k_hits,
)

# ── C extensions ─────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent.parent / "clustkit" / "csrc"

_klib = ctypes.cdll.LoadLibrary(str(_BASE / "kmer_score.so"))
_klib.batch_score_queries_c.restype = None
_klib.batch_score_queries_c.argtypes = [
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,
    ctypes.c_int32,ctypes.c_int32,
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
]
_klib.batch_score_queries_spaced_c.restype = None
_klib.batch_score_queries_spaced_c.argtypes = [
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int32,
    ctypes.c_void_p,ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,ctypes.c_int32,
    ctypes.c_int32,ctypes.c_int32,
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
]

_ulib = ctypes.cdll.LoadLibrary(str(_BASE / "union_features.so"))
_ulib.union_build_features_c.restype = ctypes.c_int32
_ulib.union_build_features_c.argtypes = [
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int64,ctypes.c_int32,ctypes.c_int64,
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
]

_swlib = ctypes.cdll.LoadLibrary(str(_BASE / "sw_align.so"))
_swlib.batch_sw_align_c.restype = None
_swlib.batch_sw_align_c.argtypes = [
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int32,ctypes.c_int32,ctypes.c_void_p,ctypes.c_float,
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
]


def c_score(q,qo,ql,nq,k,alpha,ko,ke,kf,ft,nd,mc,topk):
    ot=np.empty((nq,mc),dtype=np.int32);oc=np.zeros(nq,dtype=np.int32)
    os_=np.zeros((nq,mc),dtype=np.int32);od=np.zeros((nq,mc),dtype=np.int32)
    _klib.batch_score_queries_c(q.ctypes.data,qo.ctypes.data,ql.ctypes.data,nq,k,alpha,
        ko.ctypes.data,ke.ctypes.data,kf.ctypes.data,ft,nd,2,2,10,mc,topk,
        ot.ctypes.data,oc.ctypes.data,os_.ctypes.data,od.ctypes.data)
    return ot,oc,os_,od

def c_score_sp(q,qo,ql,nq,so,w,sp,alpha,ko,ke,kf,ft,nd,mc,topk):
    ot=np.empty((nq,mc),dtype=np.int32);oc=np.zeros(nq,dtype=np.int32)
    os_=np.zeros((nq,mc),dtype=np.int32)
    _klib.batch_score_queries_spaced_c(q.ctypes.data,qo.ctypes.data,ql.ctypes.data,nq,
        so.ctypes.data,w,sp,alpha,ko.ctypes.data,ke.ctypes.data,kf.ctypes.data,ft,nd,
        2,2,10,mc,topk,ot.ctypes.data,oc.ctypes.data,os_.ctypes.data)
    return ot,oc,os_

def c_union(packed_list, scores_list, diags_list, idx_ids_list, n_indices, nd):
    all_pk=np.concatenate(packed_list)
    all_sc=np.concatenate(scores_list).astype(np.int32)
    all_dg=np.concatenate(diags_list).astype(np.int32) if diags_list[0] is not None else np.zeros(len(all_pk),dtype=np.int32)
    all_id=np.concatenate(idx_ids_list).astype(np.int8)
    total=len(all_pk)
    # Pre-allocate outputs (max = total entries, will be trimmed)
    out_qi=np.empty(total,dtype=np.int32);out_ti=np.empty(total,dtype=np.int32)
    out_pidx=np.zeros((total,n_indices),dtype=np.float32)
    out_nidx=np.empty(total,dtype=np.float32);out_maxs=np.empty(total,dtype=np.float32)
    out_sums=np.empty(total,dtype=np.float32);out_diag=np.empty(total,dtype=np.int32)
    n_unique=_ulib.union_build_features_c(
        all_pk.ctypes.data,all_sc.ctypes.data,all_dg.ctypes.data,all_id.ctypes.data,
        ctypes.c_int64(total),n_indices,ctypes.c_int64(nd),
        out_qi.ctypes.data,out_ti.ctypes.data,out_pidx.ctypes.data,
        out_nidx.ctypes.data,out_maxs.ctypes.data,out_sums.ctypes.data,out_diag.ctypes.data)
    return (out_qi[:n_unique],out_ti[:n_unique],out_pidx[:n_unique],
            out_nidx[:n_unique],out_maxs[:n_unique],out_sums[:n_unique],
            out_diag[:n_unique],n_unique)

def c_sw(mp,fs,off,lens,bw,dh=None):
    M=len(mp);pf=np.ascontiguousarray(mp.flatten())
    si=np.empty(M,dtype=np.float32);sc=np.empty(M,dtype=np.int32)
    mk=np.empty(M,dtype=np.uint8);sm=BLOSUM62.astype(np.int8)
    dh_ptr=dh.ctypes.data if dh is not None else None
    _swlib.batch_sw_align_c(pf.ctypes.data,fs.ctypes.data,off.ctypes.data,
        lens.ctypes.data,M,bw,sm.ctypes.data,0.1,dh_ptr,
        si.ctypes.data,sc.ctypes.data,mk.ctypes.data)
    return si,sc,mk.astype(np.bool_)

def flatten_all(ot,oc,os_,od,nq,nd,idx_id):
    total=int(oc.sum())
    pk=np.empty(total,dtype=np.int64);sc=np.empty(total,dtype=np.int32)
    dg=np.empty(total,dtype=np.int32);ids=np.full(total,idx_id,dtype=np.int8)
    p=0
    for qi in range(nq):
        nc=int(oc[qi])
        if nc>0:
            pk[p:p+nc]=np.int64(qi)*nd+ot[qi,:nc].astype(np.int64)
            sc[p:p+nc]=os_[qi,:nc]
            dg[p:p+nc]=od[qi,:nc] if od is not None else 0
            p+=nc
    return pk,sc,dg,ids

def evaluate_roc1(hits_list,metadata):
    di=metadata["domain_info"];fs=metadata["family_sizes_in_db"]
    qs=set(metadata["query_sids"]);hbq=defaultdict(list)
    for qh in hits_list:
        for h in qh:
            hbq[h.query_id].append((h.target_id,h.score if h.score!=0 else h.identity))
    vals=[]
    for qid in qs:
        qi=di.get(qid)
        if qi is None:continue
        tp=fs.get(str(qi["family"]),1)-1
        if tp<=0:continue
        qh=sorted(hbq.get(qid,[]),key=lambda x:-x[1])
        ranked=[RankedHit(target_id=t,score=s,label=classify_hit(qid,t,di))
                for t,s in qh if classify_hit(qid,t,di)!="IGNORE"]
        vals.append(compute_roc_n(ranked,1,tp))
    return float(np.mean(vals)) if vals else 0.0


def run_config(label, pk_list, sc_list, dg_list, id_list, n_idx, nd, nq,
               q_lens, db_ds, query_ds, metadata, merged, m_off, m_lens,
               t_score, bw=20):
    """Run union + LGB calibration + two-tier SW for a given set of index outputs."""
    print(f"\n  {label}", flush=True)
    print(f"  {'='*70}", flush=True)

    # C union
    t0=time.perf_counter()
    qi_arr,ti_arr,pidx,nidx,maxs,sums,diags,n_union=c_union(
        pk_list,sc_list,dg_list,id_list,n_idx,nd)
    t_union=time.perf_counter()-t0
    print(f"    C union: {n_union} pairs in {t_union:.1f}s", flush=True)

    # Build feature matrix
    pairs=np.column_stack([qi_arr,ti_arr]).astype(np.int32)
    ql_f=q_lens[qi_arr].astype(np.float32)
    tl_f=db_ds.lengths[ti_arr].astype(np.float32)
    shorter=np.minimum(ql_f,tl_f);longer=np.maximum(ql_f,tl_f)
    lr=np.where(longer>0,shorter/longer,0).astype(np.float32)
    ld=np.abs(ql_f-tl_f)
    # Pad per_idx to 5 columns
    pidx5=np.zeros((n_union,5),dtype=np.float32)
    pidx5[:,:n_idx]=pidx[:,:n_idx]
    features=np.column_stack([pidx5,nidx,maxs,sums,lr,ld,ql_f,tl_f])

    # Calibration on first 1000 queries
    cal_mask=qi_arr<1000
    cal_p=pairs[cal_mask];cal_f=features[cal_mask];cal_d=diags[cal_mask]
    cal_m=_remap_pairs_to_merged(cal_p,merged["nq"])
    sk=np.minimum(cal_m[:,0],cal_m[:,1]);so=np.argsort(sk,kind="mergesort")
    cal_m=cal_m[so];cal_p=cal_p[so];cal_f=cal_f[so];cal_d=cal_d[so]

    t0=time.perf_counter()
    _,cal_scores,_=c_sw(cal_m,merged["flat_sequences"],m_off,m_lens,bw,cal_d.astype(np.int32))
    t_cal=time.perf_counter()-t0
    print(f"    Calibration SW: {len(cal_p)} pairs in {t_cal:.1f}s", flush=True)

    import lightgbm as lgb
    cal_qi=cal_p[:,0];cal_uq=np.unique(cal_qi)
    np.random.seed(42);np.random.shuffle(cal_uq)
    cal_tq=set(cal_uq[:int(len(cal_uq)*0.8)].tolist())
    cal_tm=np.array([int(q) in cal_tq for q in cal_qi])
    model=lgb.LGBMRegressor(n_estimators=50,max_depth=4,learning_rate=0.05,
                             n_jobs=-1,random_state=42,verbose=-1)
    model.fit(cal_f[cal_tm],cal_scores[cal_tm])
    pred_val=model.predict(cal_f[~cal_tm])
    r=np.corrcoef(cal_scores[~cal_tm],pred_val)[0,1]
    print(f"    LGB r={r:.4f}", flush=True)

    # Predict + top-N
    t0=time.perf_counter()
    predicted=model.predict(features)
    t_lgb=time.perf_counter()-t0

    results=[]
    for N in [3000,5000,8000]:
        sort_key=np.lexsort((-predicted,qi_arr))
        sorted_qi=qi_arr[sort_key]
        keep=np.empty(min(nq*N,len(pairs)),dtype=np.int64)
        out_pos=0;prev=-1;cnt=0
        for i in range(len(sorted_qi)):
            qi=int(sorted_qi[i])
            if qi!=prev:prev=qi;cnt=0
            if cnt<N:keep[out_pos]=sort_key[i];out_pos+=1;cnt+=1
        keep=keep[:out_pos]
        sel=pairs[keep];sel_dg=diags[keep]
        sel_m=_remap_pairs_to_merged(sel,merged["nq"])
        sk2=np.minimum(sel_m[:,0],sel_m[:,1])
        so2=np.argsort(sk2,kind="mergesort")
        sel_m=sel_m[so2];sel=sel[so2];sel_dg=sel_dg[so2]

        t0=time.perf_counter()
        sims,scores,mask=c_sw(sel_m,merged["flat_sequences"],m_off,m_lens,bw,sel_dg.astype(np.int32))
        t_sw=time.perf_counter()-t0

        total=t_score+t_union+t_lgb+t_sw
        p=scores>0
        hits=_collect_top_k_hits(sel[p],sims[p],nq,500,query_ds,db_ds,
                                 passing_scores=scores[p].astype(np.float32))
        roc1=evaluate_roc1(hits,metadata)
        vs=roc1-0.7942
        print(f"    N={N}: score={t_score:.0f}s union={t_union:.0f}s lgb={t_lgb:.0f}s "
              f"sw={t_sw:.0f}s total={total:.0f}s ROC1={roc1:.4f} ({vs:+.4f})", flush=True)
        results.append({"N":N,"total":total,"roc1":roc1})
    return results


def main():
    scop_dir=Path("benchmarks/data/scop_search_results")
    out_dir=Path("benchmarks/data/speed_sensitivity_results")
    with open(scop_dir/"metadata.json") as f:fm=json.load(f)
    random.seed(42)
    qsids=sorted(random.sample(fm["query_sids"],min(2000,len(fm["query_sids"]))))
    all_seqs=dict(read_fasta(scop_dir/"queries.fasta"))
    sub=[(s,all_seqs[s]) for s in qsids if s in all_seqs]
    qf=str(out_dir/"queries_subset.fasta");write_fasta(sub,qf)
    metadata=dict(fm);metadata["query_sids"]=qsids

    print("Loading database...",flush=True)
    db_index=load_database(out_dir/"clustkit_db")
    query_ds=read_sequences(qf,"protein");db_ds=db_index.dataset
    nq,nd=query_ds.num_sequences,db_ds.num_sequences
    q_flat=query_ds.flat_sequences;q_off=query_ds.offsets.astype(np.int64)
    q_lens=query_ds.lengths.astype(np.int32)
    mc=8000;topk=200000
    k=int(db_index.params.get("kmer_index_k",db_index.params["kmer_size"]))
    ft=int(compute_freq_threshold(db_index.kmer_freqs,nd,99.5))

    # Build indices
    rq=_remap_flat(q_flat,REDUCED_ALPHA,len(q_flat))
    rd=_remap_flat(db_ds.flat_sequences,REDUCED_ALPHA,len(db_ds.flat_sequences))
    r4o,r4e,r4f=build_kmer_index(rd,db_ds.offsets,db_ds.lengths,4,"protein",alpha_size=REDUCED_ALPHA_SIZE)
    r5o,r5e,r5f=build_kmer_index(rd,db_ds.offsets,db_ds.lengths,5,"protein",alpha_size=REDUCED_ALPHA_SIZE)
    s1d=build_kmer_index_spaced(rd,db_ds.offsets,db_ds.lengths,"11011","protein",alpha_size=REDUCED_ALPHA_SIZE)
    s2d=build_kmer_index_spaced(rd,db_ds.offsets,db_ds.lengths,"110011","protein",alpha_size=REDUCED_ALPHA_SIZE)
    r4ft=int(compute_freq_threshold(r4f,nd,99.5))
    r5ft=int(compute_freq_threshold(r5f,nd,99.5))
    s1ft=int(compute_freq_threshold(s1d[2],nd,99.5))
    s2ft=int(compute_freq_threshold(s2d[2],nd,99.5))

    merged=_merge_sequences_for_alignment(query_ds,db_ds)
    m_off=merged["offsets"].astype(np.int64);m_lens=merged["lengths"].astype(np.int32)

    print(f"Loaded {nq} queries, {nd} db\n",flush=True)

    # ── Score all indices ────────────────────────────────────────────
    print("="*80)
    print("C scoring: all 5 indices")
    print("="*80,flush=True)

    t0=time.perf_counter()
    ot1,oc1,os1,od1=c_score(q_flat,q_off,q_lens,nq,k,20,
        db_index.kmer_offsets,db_index.kmer_entries,db_index.kmer_freqs,ft,nd,mc,topk)
    t_std=time.perf_counter()-t0;print(f"  std k=3: {t_std:.1f}s",flush=True)

    t0=time.perf_counter()
    ot2,oc2,os2,od2=c_score(rq,q_off,q_lens,nq,4,REDUCED_ALPHA_SIZE,r4o,r4e,r4f,r4ft,nd,mc,topk)
    t_r4=time.perf_counter()-t0;print(f"  red k=4: {t_r4:.1f}s",flush=True)

    t0=time.perf_counter()
    ot3,oc3,os3,od3=c_score(rq,q_off,q_lens,nq,5,REDUCED_ALPHA_SIZE,r5o,r5e,r5f,r5ft,nd,mc,topk)
    t_r5=time.perf_counter()-t0;print(f"  red k=5: {t_r5:.1f}s",flush=True)

    t0=time.perf_counter()
    ot4,oc4,os4=c_score_sp(rq,q_off,q_lens,nq,s1d[3],int(s1d[4]),int(s1d[5]),
        REDUCED_ALPHA_SIZE,s1d[0],s1d[1],s1d[2],s1ft,nd,mc,topk)
    t_s1=time.perf_counter()-t0;print(f"  sp 11011: {t_s1:.1f}s",flush=True)

    t0=time.perf_counter()
    ot5,oc5,os5=c_score_sp(rq,q_off,q_lens,nq,s2d[3],int(s2d[4]),int(s2d[5]),
        REDUCED_ALPHA_SIZE,s2d[0],s2d[1],s2d[2],s2ft,nd,mc,topk)
    t_s2=time.perf_counter()-t0;print(f"  sp 110011: {t_s2:.1f}s",flush=True)

    t_all=t_std+t_r4+t_r5+t_s1+t_s2
    t_3idx=t_std+t_r5+t_s2
    print(f"  Total 5-idx: {t_all:.1f}s, 3-idx (std+r5+sp2): {t_3idx:.1f}s\n",flush=True)

    # Flatten all
    pk1,sc1,dg1,id1=flatten_all(ot1,oc1,os1,od1,nq,nd,0)
    pk2,sc2,dg2,id2=flatten_all(ot2,oc2,os2,od2,nq,nd,1)
    pk3,sc3,dg3,id3=flatten_all(ot3,oc3,os3,od3,nq,nd,2)
    # Spaced seeds don't output diags from current C — use zeros
    od4_z=np.zeros_like(os4);od5_z=np.zeros_like(os5)
    pk4,sc4,dg4,id4=flatten_all(ot4,oc4,os4,od4_z,nq,nd,3)
    pk5,sc5,dg5,id5=flatten_all(ot5,oc5,os5,od5_z,nq,nd,4)

    # ── Config A: 5 indices + C union ────────────────────────────────
    print("="*80)
    print("Config A: 5 indices + C union + LGB + diag hints bw=20")
    print("="*80,flush=True)

    run_config("5-index pipeline",
        [pk1,pk2,pk3,pk4,pk5],[sc1,sc2,sc3,sc4,sc5],
        [dg1,dg2,dg3,dg4,dg5],[id1,id2,id3,id4,id5],
        5,nd,nq,q_lens,db_ds,query_ds,metadata,merged,m_off,m_lens,t_all)

    # ── Config B: 3 indices + C union ────────────────────────────────
    print("\n"+"="*80)
    print("Config B: 3 indices (std+r5+sp2) + C union + LGB + diag hints bw=20")
    print("="*80,flush=True)

    run_config("3-index pipeline",
        [pk1,pk3,pk5],[sc1,sc3,sc5],
        [dg1,dg3,dg5],[id1,id3,id5],
        3,nd,nq,q_lens,db_ds,query_ds,metadata,merged,m_off,m_lens,t_3idx)

    print(f"\n  Reference: MMseqs2=0.7942 (14s)  DIAMOND=0.7963 (13s)")
    print(f"  Previous v7.4 (numpy union): 129s, ROC1=0.802")


if __name__ == "__main__":
    main()
