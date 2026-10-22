import math
SIGMAS=(.026,.025,.025,.035,.035,.079,.079,.072,.072,.062,.062,.107,.107,.087,.087,.089,.089)
def score(row,pred):
    if pred.status!='ok': return {'sample_id':row['sample_id'],'metric_status':'failed','oks_e2e':0.,'pck10_e2e':0.,'oks_conditional':None,'pck10_conditional':None}
    gt=__import__('json').loads(row['gt_xy']); vis=__import__('json').loads(row['visibility']); b=__import__('json').loads(row['bbox_xywh']); scale=max(float(b[2])*float(b[3]),1.); diagonal=math.hypot(float(b[2]),float(b[3])); oks=[]; pck=[]
    for i,v in enumerate(vis):
        if int(v)<=0: continue
        d=math.dist(gt[i],pred.keypoints_xy[i]); oks.append(math.exp(-(d*d)/(2*scale*SIGMAS[i]**2))); pck.append(float(d<=.1*diagonal))
    value_oks=sum(oks)/len(oks); value_pck=sum(pck)/len(pck)
    return {'sample_id':row['sample_id'],'metric_status':'ok','oks_e2e':value_oks,'pck10_e2e':value_pck,'oks_conditional':value_oks,'pck10_conditional':value_pck}
def summarise(records):
    n=len(records); ok=[r for r in records if r['metric_status']=='ok']; avg=lambda k:sum(r[k] for r in records)/max(n,1)
    conditional=lambda k:(sum(r[k] for r in ok)/len(ok)) if ok else None
    return {'images':n,'coverage':len(ok)/max(n,1),'oks_e2e':avg('oks_e2e'),'pck10_e2e':avg('pck10_e2e'),'oks_conditional':conditional('oks_conditional'),'pck10_conditional':conditional('pck10_conditional')}
