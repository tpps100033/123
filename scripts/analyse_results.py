"""Compare official COCOeval keypoint AP on COCO and SyRIP handoffs."""
import ast, csv, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METRICS=('AP','AP50','AP75','AP_medium','AP_large','AR','AR50','AR75','AR_medium','AR_large')

def main():
    tables=sorted((ROOT/'04_merged_results').glob('merged_*/benchmark_table.csv'))
    if not tables: raise FileNotFoundError('Run scripts/merge_results.py first.')
    with tables[-1].open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    by={}
    for row in rows: by.setdefault(row['model_id'],{})[row['dataset_id']]=row
    summary=[]; latency=[]
    for model,domains in sorted(by.items()):
        if 'coco_singleperson_val2017' not in domains or 'syrip_test100' not in domains: continue
        coco,syrip=domains['coco_singleperson_val2017'],domains['syrip_test100']
        def number(value): return float(value) if value not in ('',None,'None') else None
        item={'model_id':model}
        for metric in METRICS:
            cv,sv=number(coco.get(metric)),number(syrip.get(metric))
            item['coco_subset_'+metric]=cv; item['syrip_coco_oks_'+metric]=sv
            item['difference_'+metric]=None if cv is None or sv is None else cv-sv
        item['coco_pose_coverage_diagnostic']=number(coco.get('pose_coverage_diagnostic'))
        item['syrip_pose_coverage_diagnostic']=number(syrip.get('pose_coverage_diagnostic'))
        summary.append(item)
        for domain,run in (('coco_subset',coco),('syrip_coco_oks',syrip)):
            eligible=run.get('timing_comparability') in ('True',True)
            try: timing=ast.literal_eval(run.get('latency_ms','{}'))
            except Exception: timing={}
            latency.append({'model_id':model,'domain':domain,'dataset_id':run['dataset_id'],'timing_comparability':eligible,'latency_mean_ms':timing.get('mean') if eligible else None,'latency_median_ms':timing.get('median') if eligible else None,'latency_p95_ms':timing.get('p95') if eligible else None})
    out=ROOT/'05_analysis_results'/f'analysis_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}';out.mkdir(parents=True)
    if summary:
        with (out/'cross_domain_summary.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    if latency:
        with (out/'latency_by_model_domain.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(latency[0]));w.writeheader();w.writerows(latency)
    note='AP/AR fields are the official cocodataset/cocoapi COCOeval keypoint summary. COCO values use the selected 921-image single-person subset; SyRIP values use the same COCO OKS evaluator and are not official COCO leaderboard results.'
    report=['# Two-domain COCO keypoint evaluation','',f"Input table: `{tables[-1].relative_to(ROOT)}`",'',f'Paired models: {len(summary)}','',note,'','Pose coverage is diagnostic only. Timing is compared only for complete GPU runs.']
    if not summary: report += ['', 'No model has a complete pair of COCO and SyRIP runs yet.']
    else:
        names={
            'mediapipe_pose_heavy':'MediaPipe Pose Heavy',
            'movenet_thunder':'MoveNet Thunder',
            'openpose_coco18':'OpenPose COCO18',
            'rtmpose_m':'RTMPose-M',
            'sapiens_1b_coco':'Sapiens 1B COCO',
            'yolo11x_pose':'YOLO11x Pose',
        }
        def fmt(value): return '—' if value is None else f'{value:.3f}'
        report += ['', '| Model | COCO subset AP | SyRIP COCO-OKS AP | Δ AP (COCO − SyRIP) | COCO AP50 | SyRIP AP50 | COCO AP75 | SyRIP AP75 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for item in summary:
            model=names.get(item['model_id'],item['model_id'])
            report.append('| '+model+' | '+fmt(item['coco_subset_AP'])+' | '+fmt(item['syrip_coco_oks_AP'])+' | '+fmt(item['difference_AP'])+' | '+fmt(item['coco_subset_AP50'])+' | '+fmt(item['syrip_coco_oks_AP50'])+' | '+fmt(item['coco_subset_AP75'])+' | '+fmt(item['syrip_coco_oks_AP75'])+' |')
        report += ['', 'Δ AP is descriptive only because SyRIP is a custom-domain dataset. A paired original Colab run with official COCOeval metrics was not available, so a numerical Colab-to-local delta cannot be calculated. Historical project-defined OKS/PCK outputs remain separate and are not treated as COCOeval scores.']
    (out/'comparison_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (out/'analysis_manifest.json').write_text(json.dumps({'input_table':str(tables[-1]),'paired_models':len(summary),'score_rows':summary,'latency_rows':latency,'metric_protocol':'cocodataset/cocoapi.COCOeval(iouType=keypoints)'},ensure_ascii=False,indent=2),encoding='utf-8')
    if summary:
        try:
            import matplotlib.pyplot as plt
            labels=[x['model_id'] for x in summary]; coco=[x['coco_subset_AP'] or 0 for x in summary]; syrip=[x['syrip_coco_oks_AP'] or 0 for x in summary]
            fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.5),5)); pos=list(range(len(labels))); width=.38
            ax.bar([x-width/2 for x in pos],coco,width,label='COCO 921-image subset AP');ax.bar([x+width/2 for x in pos],syrip,width,label='SyRIP COCO-OKS AP')
            ax.set_xticks(pos,labels,rotation=20,ha='right');ax.set_ylabel('COCO Keypoint AP');ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(out/'cross_domain_keypoint_ap.png',dpi=160);plt.close(fig)
            timed=[x for x in latency if x['timing_comparability'] and x['latency_median_ms'] is not None]
            if timed:
                fig,ax=plt.subplots(figsize=(max(8,len(timed)*1.2),5)); ax.bar([x['model_id']+'\n'+x['domain'] for x in timed],[x['latency_median_ms'] for x in timed]);ax.set_ylabel('Median end-to-end latency (ms)');fig.tight_layout();fig.savefig(out/'latency_median_two_domains.png',dpi=160);plt.close(fig)
        except ImportError:
            report.append('Plot output unavailable because matplotlib is not installed.')
            (out/'comparison_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'paired_models':len(summary),'timing_rows':len(latency)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
