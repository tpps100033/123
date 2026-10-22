"""Create the two-domain summary without changing score definitions."""
import ast, csv, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    tables=sorted((ROOT/'04_merged_results').glob('merged_*/benchmark_table.csv'))
    if not tables: raise FileNotFoundError('Run scripts/merge_results.py first.')
    with tables[-1].open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    by={}
    for row in rows: by.setdefault(row['model_id'],{})[row['dataset_id']]=row
    summary=[]; latency=[]
    for model,domains in sorted(by.items()):
        if 'coco_singleperson_val2017' not in domains or 'syrip_test100' not in domains: continue
        adult,infant=domains['coco_singleperson_val2017'],domains['syrip_test100']
        def n(x): return float(x) if x not in ('',None) else None
        row={'model_id':model}
        for key in ('coverage','oks_e2e','pck10_e2e','oks_conditional','pck10_conditional'):
            av,iv=n(adult.get(key)),n(infant.get(key)); row['adult_'+key]=av;row['infant_'+key]=iv
            row['absolute_drop_'+key]=None if av is None or iv is None else av-iv
            row['relative_drop_pct_'+key]=None if av in (None,0) or iv is None else (av-iv)/av*100
        summary.append(row)
        for domain,run in (('adult',adult),('infant',infant)):
            item={'model_id':model,'domain':domain,'dataset_id':run['dataset_id'],'timing_comparability':run.get('timing_comparability') in ('True',True),'fair_ranking_eligible':run.get('fair_ranking_eligible') in ('True',True)}
            try: timing=ast.literal_eval(run.get('latency_ms','{}'))
            except Exception: timing={}
            for stat in ('median','p95','mean'): item[f'latency_{stat}_ms']=timing.get(stat) if item['timing_comparability'] else None
            latency.append(item)
    out=ROOT/'05_analysis_results'/f'analysis_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';out.mkdir(parents=True)
    if summary:
        with (out/'cross_domain_summary.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    if latency:
        with (out/'latency_by_model_domain.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(latency[0]));w.writeheader();w.writerows(latency)
    report=['# Two-domain fair-reference summary','',f"Input table: `{tables[-1].relative_to(ROOT)}`",'',f'Paired models: {len(summary)}','', 'Scores are reported as stored by the source metric implementation. No missing conditional score is inferred. CPU or failed runs are excluded from cross-model timing eligibility.']
    if not summary: report += ['', 'No model has a complete pair of adult and infant runs yet.']
    (out/'comparison_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (out/'analysis_manifest.json').write_text(json.dumps({'input_table':str(tables[-1]),'paired_models':len(summary),'score_rows':summary,'latency_rows':latency,'limits':'Only paired model/domain results are reported; no missing metric is inferred.'},ensure_ascii=False,indent=2),encoding='utf-8')
    if summary:
        try:
            import matplotlib.pyplot as plt
            labels=[x['model_id'] for x in summary]; adult=[x['adult_oks_e2e'] or 0 for x in summary]; infant=[x['infant_oks_e2e'] or 0 for x in summary]
            fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.5),5)); pos=list(range(len(labels))); width=.38
            ax.bar([x-width/2 for x in pos],adult,width,label='COCO val2017');ax.bar([x+width/2 for x in pos],infant,width,label='SyRIP Test100')
            ax.set_xticks(pos,labels,rotation=20,ha='right');ax.set_ylabel('End-to-end OKS');ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(out/'cross_domain_oks.png',dpi=160);plt.close(fig)
            timed=[x for x in latency if x['timing_comparability'] and x['latency_median_ms'] is not None]
            if timed:
                fig,ax=plt.subplots(figsize=(max(8,len(timed)*1.2),5)); ax.bar([x['model_id']+'\n'+x['domain'] for x in timed],[x['latency_median_ms'] for x in timed]);ax.set_ylabel('Median end-to-end latency (ms)');fig.tight_layout();fig.savefig(out/'latency_median_two_domains.png',dpi=160);plt.close(fig)
        except ImportError:
            report.append('Plot output unavailable because matplotlib is not installed.')
            (out/'comparison_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'paired_models':len(summary),'timing_rows':len(latency)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
