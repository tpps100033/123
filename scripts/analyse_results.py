"""Summarize official COCOeval keypoint AP across COCO, infant, and child domains."""
import ast, csv, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOMAINS=(
    ('coco_singleperson_val2017','coco_subset','COCO subset'),
    ('syrip_test100','infant_syrip','SyRIP infant'),
    ('coco_unihuman_child_val134','child_unihuman','COCO-UniHuman child'),
)
METRICS=('AP','AP50','AP75','AP_medium','AP_large','AR','AR50','AR75','AR_medium','AR_large')

def number(value):
    return float(value) if value not in ('',None,'None') else None

def main():
    tables=sorted((ROOT/'04_merged_results').glob('merged_*/benchmark_table.csv'))
    if not tables: raise FileNotFoundError('Run scripts/merge_results.py first.')
    with tables[-1].open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    by={}
    for row in rows: by.setdefault(row['model_id'],{})[row['dataset_id']]=row
    score_rows=[]; latency=[]
    for model, datasets in sorted(by.items()):
        present=[(key,label,datasets.get(key)) for key,slug,label in DOMAINS]
        if not all(row for _,_,row in present): continue
        item={'model_id':model}
        values={}
        for key,slug,label in DOMAINS:
            run=datasets[key]; values[slug]=number(run.get('AP'))
            for metric in METRICS: item[f'{slug}_{metric}']=number(run.get(metric))
            item[f'{slug}_pose_coverage_diagnostic']=number(run.get('pose_coverage_diagnostic'))
            try: timing=ast.literal_eval(run.get('latency_ms','{}'))
            except Exception: timing={}
            eligible=run.get('timing_comparability') in ('True',True)
            latency.append({'model_id':model,'domain':slug,'dataset_id':key,'timing_comparability':eligible,
                            'latency_mean_ms':timing.get('mean') if eligible else None,
                            'latency_median_ms':timing.get('median') if eligible else None,
                            'latency_p95_ms':timing.get('p95') if eligible else None})
        item['difference_child_minus_coco_AP']=None if values['child_unihuman'] is None or values['coco_subset'] is None else values['child_unihuman']-values['coco_subset']
        item['difference_infant_minus_coco_AP']=None if values['infant_syrip'] is None or values['coco_subset'] is None else values['infant_syrip']-values['coco_subset']
        score_rows.append(item)

    out=ROOT/'05_analysis_results'/f'analysis_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';out.mkdir(parents=True)
    if score_rows:
        with (out/'three_domain_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(score_rows[0]));w.writeheader();w.writerows(score_rows)
    if latency:
        with (out/'latency_by_model_domain.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(latency[0]));w.writeheader();w.writerows(latency)

    note=('All AP/AR fields use official cocodataset/cocoapi COCOeval keypoint defaults. COCO values use the selected '
          '921-image single-person subset; SyRIP uses 100 infant images; the child domain uses 134 COCO-UniHuman '
          'apparent-age crops. Child source image IDs do not overlap the 921-image COCO subset, but both cohorts use '
          'COCO val2017 source imagery. SyRIP and child AP are custom-domain COCO-OKS evaluations, not leaderboard results.')
    report=['# Three-domain COCO keypoint evaluation','',f"Input table: `{tables[-1].relative_to(ROOT)}`",'',f'Complete three-domain model pairs: {len(score_rows)}','',note,'','Pose coverage is diagnostic only. Latency is reported only for complete GPU runs.']
    if not score_rows: report += ['', 'No model has a complete set of COCO, SyRIP, and child runs yet.']
    else:
        report += ['', '| Model | COCO AP | SyRIP infant AP | Child AP | Child − COCO AP |', '|---|---:|---:|---:|---:|']
        for row in score_rows:
            report.append('| '+row['model_id']+' | '+fmt(row['coco_subset_AP'])+' | '+fmt(row['infant_syrip_AP'])+' | '+fmt(row['child_unihuman_AP'])+' | '+fmt(row['difference_child_minus_coco_AP'])+' |')
        report += ['', 'Domain AP differences are descriptive because the datasets have different populations and source protocols. A paired original Colab run with official COCOeval metrics was not available, so a numerical Colab-to-local delta cannot be calculated. Historical project-defined OKS/PCK outputs remain separate and are not treated as COCOeval scores.']
    (out/'comparison_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (out/'analysis_manifest.json').write_text(json.dumps({'input_table':str(tables[-1]),'complete_three_domain_models':len(score_rows),'score_rows':score_rows,'latency_rows':latency,'metric_protocol':'cocodataset/cocoapi.COCOeval(iouType=keypoints)'},ensure_ascii=False,indent=2),encoding='utf-8')
    if score_rows:
        try:
            import matplotlib.pyplot as plt
            labels=[x['model_id'] for x in score_rows]; pos=list(range(len(labels))); width=.26
            fig,ax=plt.subplots(figsize=(max(9,len(labels)*1.6),5))
            for i,(key,slug,label) in enumerate(DOMAINS):
                ax.bar([x+(i-1)*width for x in pos],[row[f'{slug}_AP'] or 0 for row in score_rows],width,label=label)
            ax.set_xticks(pos,labels,rotation=20,ha='right');ax.set_ylabel('COCO Keypoint AP');ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(out/'three_domain_keypoint_ap.png',dpi=160);plt.close(fig)
            timed=[x for x in latency if x['timing_comparability'] and x['latency_median_ms'] is not None]
            if timed:
                fig,ax=plt.subplots(figsize=(max(9,len(timed)*.7),5));ax.bar([x['model_id']+'\n'+x['domain'] for x in timed],[x['latency_median_ms'] for x in timed]);ax.set_ylabel('Median end-to-end latency (ms)');fig.tight_layout();fig.savefig(out/'latency_median_three_domains.png',dpi=160);plt.close(fig)
        except ImportError:
            pass
    print(json.dumps({'output':str(out),'complete_three_domain_models':len(score_rows),'timing_rows':len(latency)},ensure_ascii=False,indent=2))

def fmt(value): return '—' if value is None else f'{value:.3f}'

if __name__=='__main__': main()
