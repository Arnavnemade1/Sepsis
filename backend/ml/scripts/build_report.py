"""Render artifacts/rigorous_eval.json into a self-contained results report.

Produces webui/validation.html — a single theme-aware HTML page (data inlined, no
external assets) with the honest, judge-facing evaluation: a forest plot of
cross-hospital AUROC with 95% CIs, ROC/PR/calibration/decision/alarm-burden
curves, the ablation table, and disclosed limitations.

    python -m backend.ml.scripts.build_report
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(ROOT, "artifacts", "rigorous_eval.json")
AUDIT_SRC = os.path.join(ROOT, "artifacts", "model_audit.json")
OUT = os.path.join(ROOT, "webui", "validation.html")

TEMPLATE = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AegisOS — Detection Validation</title>
<style>
:root{
  --bg:#f7f7f4; --surface:#fcfcfb; --card:#ffffff; --line:#e6e5df; --line2:#efeee9;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#8a897f;
  --s1:#2a78d6; --s2:#1baf7a; --ref:#b9b8b0; --good:#1a7f4b; --warn:#b8860b; --bad:#c0392b;
  --s1soft:rgba(42,120,214,.12); --s2soft:rgba(27,175,122,.14);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#111110; --surface:#1a1a19; --card:#1f1f1d; --line:#33332f; --line2:#2a2a27;
  --ink:#f4f3ee; --ink2:#c3c2b7; --muted:#8f8e84;
  --s1:#3987e5; --s2:#1fbd86; --ref:#55554e; --good:#41c98a; --warn:#e0b64b; --bad:#e66767;
  --s1soft:rgba(57,135,229,.16); --s2soft:rgba(31,189,134,.16);
}}
:root[data-theme=dark]{
  --bg:#111110; --surface:#1a1a19; --card:#1f1f1d; --line:#33332f; --line2:#2a2a27;
  --ink:#f4f3ee; --ink2:#c3c2b7; --muted:#8f8e84;
  --s1:#3987e5; --s2:#1fbd86; --ref:#55554e; --good:#41c98a; --warn:#e0b64b; --bad:#e66767;
  --s1soft:rgba(57,135,229,.16); --s2soft:rgba(31,189,134,.16);}
:root[data-theme=light]{
  --bg:#f7f7f4; --surface:#fcfcfb; --card:#ffffff; --line:#e6e5df; --line2:#efeee9;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#8a897f;
  --s1:#2a78d6; --s2:#1baf7a; --ref:#b9b8b0; --good:#1a7f4b; --warn:#b8860b; --bad:#c0392b;
  --s1soft:rgba(42,120,214,.12); --s2soft:rgba(27,175,122,.14);}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto;padding:40px 22px 80px}
.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace}
h1{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0 0 6px}
.sub{color:var(--ink2);font-size:15px;max-width:70ch}
.tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:var(--s2);border:1px solid var(--line);border-radius:20px;padding:3px 11px;margin-bottom:14px}
.protocol{margin:16px 0 6px;padding:12px 15px;border-left:3px solid var(--s1);background:var(--card);
  border-radius:0 10px 10px 0;font-size:13.5px;color:var(--ink2)}
.protocol b{color:var(--ink)}
h2{font-size:17px;font-weight:700;margin:38px 0 4px;letter-spacing:-.01em}
h2 .n{color:var(--muted);font-weight:600;margin-right:8px}
.lede{color:var(--ink2);font-size:13.5px;margin:0 0 14px;max-width:74ch}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0 8px}
@media(max-width:720px){.tiles{grid-template-columns:repeat(2,1fr)}}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 15px}
.tile .k{font-size:11px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-size:26px;font-weight:800;letter-spacing:-.02em;margin-top:4px;font-variant-numeric:tabular-nums}
.tile .ci{font-size:11.5px;color:var(--ink2);margin-top:2px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 18px 12px;margin-top:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.ch-h{font-size:13.5px;font-weight:700;margin-bottom:2px}
.ch-s{font-size:12px;color:var(--muted);margin-bottom:8px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:2px 0 6px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
svg{display:block;width:100%;height:auto;overflow:hidden}
.axl{fill:var(--muted);font-size:10px;font-family:ui-monospace,Menlo,monospace}
.axt{fill:var(--ink2);font-size:11px}
.gl{stroke:var(--line2);stroke-width:1}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line2);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;font-weight:600}
tbody tr:hover{background:var(--line2)}
.win{color:var(--s2);font-weight:700}
.lim{font-size:13.5px;color:var(--ink2)}
.lim li{margin:6px 0}
.foot{margin-top:30px;font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:14px}
.toggle{position:fixed;top:14px;right:14px;font-size:12px;font-weight:600;color:var(--ink2);
  background:var(--card);border:1px solid var(--line);border-radius:20px;padding:6px 12px;cursor:pointer}
.scrollx{overflow-x:auto}
</style></head><body>
<button class="toggle" onclick="tg()">◐ theme</button>
<div class="wrap" id="app"></div>
<script>window.DATA=__DATA__;</script>
<script>
function tg(){var r=document.documentElement,c=r.getAttribute('data-theme');
 var d=matchMedia('(prefers-color-scheme:dark)').matches;var cur=c||(d?'dark':'light');
 r.setAttribute('data-theme',cur==='dark'?'light':'dark');draw();}
function cv(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
var NS='http://www.w3.org/2000/svg';
function el(t,a,kids){var e=document.createElementNS(NS,t);for(var k in(a||{}))e.setAttribute(k,a[k]);
 (kids||[]).forEach(function(c){e.appendChild(c)});return e;}
function txt(t,a){var e=el('text',a);e.textContent=t;return e;}
// generic line/scatter chart: series=[{name,color,pts:[[x,y]],dash,fill}], refs=[{pts,label}]
function chart(host,opt){
 var W=opt.w||440,H=opt.h||300,m={l:44,r:14,t:10,b:34};
 var xr=opt.xr,yr=opt.yr;
 var cl=function(v,r){return Math.max(r[0],Math.min(r[1],v));};
 var sx=function(x){return m.l+(cl(x,xr)-xr[0])/(xr[1]-xr[0])*(W-m.l-m.r);};
 var sy=function(y){return H-m.b-(cl(y,yr)-yr[0])/(yr[1]-yr[0])*(H-m.t-m.b);};
 var g=el('svg',{viewBox:'0 0 '+W+' '+H,role:'img'});
 // gridlines + ticks
 var xt=opt.xticks||tk(xr),yt=opt.yticks||tk(yr);
 yt.forEach(function(v){g.appendChild(el('line',{x1:m.l,y1:sy(v),x2:W-m.r,y2:sy(v),'class':'gl'}));
  g.appendChild(txt(fmt(v),{x:m.l-7,y:sy(v)+3,'text-anchor':'end','class':'axl'}));});
 xt.forEach(function(v){g.appendChild(txt(fmt(v),{x:sx(v),y:H-m.b+15,'text-anchor':'middle','class':'axl'}));});
 if(opt.xlab)g.appendChild(txt(opt.xlab,{x:(m.l+W-m.r)/2,y:H-4,'text-anchor':'middle','class':'axt'}));
 if(opt.ylab)g.appendChild(txt(opt.ylab,{x:12,y:(m.t+H-m.b)/2,'text-anchor':'middle','class':'axt',transform:'rotate(-90 12 '+((m.t+H-m.b)/2)+')'}));
 (opt.refs||[]).forEach(function(r){
  var d=r.pts.map(function(p,i){return(i?'L':'M')+sx(p[0])+' '+sy(p[1])}).join(' ');
  g.appendChild(el('path',{d:d,fill:'none',stroke:'var(--ref)','stroke-width':1.4,'stroke-dasharray':r.dash||'4 4'}));
  if(r.label)g.appendChild(txt(r.label,{x:sx(r.pts[r.pts.length-1][0])-2,y:sy(r.pts[r.pts.length-1][1])-5,'text-anchor':'end','class':'axl'}));
 });
 (opt.series||[]).forEach(function(s){
  if(s.fill){var fd=s.fill.map(function(p,i){return(i?'L':'M')+sx(p[0])+' '+sy(p[1])}).join(' ')+' Z';
   g.appendChild(el('path',{d:fd,fill:s.color,opacity:.12,stroke:'none'}));}
  var d=s.pts.map(function(p,i){return(i?'L':'M')+sx(p[0])+' '+sy(p[1])}).join(' ');
  g.appendChild(el('path',{d:d,fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round',
    'stroke-dasharray':s.dash||''}));
  if(s.dots)s.pts.forEach(function(p){g.appendChild(el('circle',{cx:sx(p[0]),cy:sy(p[1]),r:3.2,fill:s.color,
    stroke:'var(--card)','stroke-width':1.5}))});
  if(s.label){var lp=s.pts[Math.max(0,Math.round(s.pts.length*(s.lat||0.6))-1)];
   g.appendChild(txt(s.name,{x:sx(lp[0]),y:sy(lp[1])-7,'text-anchor':'middle',fill:s.color,'font-size':11,'font-weight':700}));}
 });
 host.appendChild(g);
}
function tk(r){var a=[],n=5,s=(r[1]-r[0])/n;for(var i=0;i<=n;i++)a.push(r[0]+s*i);return a;}
function fmt(v){if(Math.abs(v)>=1000)return(v/1000).toFixed(0)+'k';if(v===0)return'0';
 if(Math.abs(v)<1)return(+v.toFixed(2)).toString();return(+v.toFixed(v%1?1:0)).toString();}
function pct(v){return(v*100).toFixed(0)+'%';}
function f2(v){return(v==null||isNaN(v))?'—':(+v).toFixed(2);}
function f3(v){return(v==null||isNaN(v))?'—':(+v).toFixed(3);}

// forest plot of AUROC±CI
function forest(host,rows){
 var W=580,rh=36,H=rows.length*rh+30,m={l:150,r:190};
 var xr=[0.5,0.8];var sx=function(x){return m.l+(x-xr[0])/(xr[1]-xr[0])*(W-m.l-m.r);};
 var g=el('svg',{viewBox:'0 0 '+W+' '+H});
 [0.5,0.6,0.7,0.8].forEach(function(v){g.appendChild(el('line',{x1:sx(v),y1:20,x2:sx(v),y2:H-10,'class':'gl'}));
  g.appendChild(txt(v.toFixed(1),{x:sx(v),y:14,'text-anchor':'middle','class':'axl'}));});
 g.appendChild(el('line',{x1:sx(0.5),y1:20,x2:sx(0.5),y2:H-10,stroke:'var(--ref)','stroke-dasharray':'3 3','stroke-width':1.3}));
 rows.forEach(function(r,i){var y=30+i*rh;
  g.appendChild(txt(r.label,{x:8,y:y+4,'class':'axt','font-weight':r.bold?700:400,fill:r.bold?r.color:'var(--ink2)'}));
  g.appendChild(el('line',{x1:sx(r.ci[0]),y1:y,x2:sx(r.ci[1]),y2:y,stroke:r.color,'stroke-width':2}));
  g.appendChild(el('line',{x1:sx(r.ci[0]),y1:y-4,x2:sx(r.ci[0]),y2:y+4,stroke:r.color,'stroke-width':2}));
  g.appendChild(el('line',{x1:sx(r.ci[1]),y1:y-4,x2:sx(r.ci[1]),y2:y+4,stroke:r.color,'stroke-width':2}));
  g.appendChild(el('circle',{cx:sx(r.v),cy:y,r:5,fill:r.color,stroke:'var(--card)','stroke-width':1.5}));
  g.appendChild(txt(f3(r.v)+'  ['+f3(r.ci[0])+'–'+f3(r.ci[1])+']',{x:W-4,y:y+4,'text-anchor':'end','class':'axl'}));
 });
 host.appendChild(g);
}

function card(title,sub){var c=document.createElement('div');c.className='card';
 if(title){var h=document.createElement('div');h.className='ch-h';h.textContent=title;c.appendChild(h);}
 if(sub){var s=document.createElement('div');s.className='ch-s';s.textContent=sub;c.appendChild(s);}
 return c;}
function legend(items){var d=document.createElement('div');d.className='legend';
 items.forEach(function(it){var s=document.createElement('span');
  s.innerHTML='<i style="background:'+it.c+'"></i>'+it.t;d.appendChild(s);});return d;}

function draw(){
 var app=document.getElementById('app');app.innerHTML='';
 var D=window.DATA, M=D.meta, dir=D.directions;
 var C1=cv('--s1'),C2=cv('--s2');
 var MODEL={gbm:{n:'Calibrated GBM (baseline)',short:'GBM',c:C1},stacked:{n:'Stacked ensemble (ours)',short:'Stacked',c:C2}};
 var models=M.models;
 // header
 var head=document.createElement('div');
 head.innerHTML='<div class="tag">Real-data validation · reproducible</div>'+
  '<h1>Does it actually work? — an honest evaluation</h1>'+
  '<div class="sub">Every number below is <b>out-of-hospital</b>: the detector is trained on one hospital and tested on a <b>different</b> one it has never seen (PhysioNet/CinC-2019 — set A = Beth Israel, set B = Emory), in both directions. This is the gap between "works on our data" and "works."</div>'+
  '<div class="protocol"><b>Protocol.</b> '+M.protocol+'. Patient-grouped (no patient spans train/test). '+
  'Set A: '+M.setA.patients.toLocaleString()+' patients / '+M.setA.rows.toLocaleString()+' hourly rows ('+pct(M.setA.prevalence)+' septic-hours). '+
  'Set B: '+M.setB.patients.toLocaleString()+' patients / '+M.setB.rows.toLocaleString()+' rows ('+pct(M.setB.prevalence)+'). '+
  '95% CIs by patient-level bootstrap ('+M.boot+'×). Generated '+M.generated+'.</div>';
 app.appendChild(head);

 // headline tiles (mean over both directions, stacked)
 var st=[dir.A_to_B.models.stacked,dir.B_to_A.models.stacked];
 var meanAuroc=(st[0].auroc+st[1].auroc)/2;
 var ciLo=Math.min(st[0].auroc_ci[0],st[1].auroc_ci[0]),ciHi=Math.max(st[0].auroc_ci[1],st[1].auroc_ci[1]);
 var meanEce=(st[0].ece+st[1].ece)/2;
 // Lead time is deliberately NOT headlined. It was audited on 2026-08-30 and found to
 // carry no early-warning information: at MATCHED DETECTION RATE a pure random score
 // reproduces this detector's median lead, and beats it at 70% detection. See
 // docs/LEAD_TIME_NULL.md. The old tile took Math.max over the two directions, i.e. it
 // published the more flattering of two numbers that were both inside the null.
 var lead=(( st[0].early_warning.median_lead_time_hours||0)+(st[1].early_warning.median_lead_time_hours||0))/2;
 var util=(st[0].normalized_utility+st[1].normalized_utility)/2;
 var tiles=document.createElement('div');tiles.className='tiles';
 function tile(k,v,ci){return '<div class="tile"><div class="k">'+k+'</div><div class="v">'+v+'</div><div class="ci">'+(ci||'')+'</div></div>';}
 tiles.innerHTML=
  tile('Cross-hospital AUROC',f3(meanAuroc),'95% CI '+f3(ciLo)+'–'+f3(ciHi))+
  tile('Calibration error',f3(meanEce),'ECE · lower is better')+
  tile('Median lead time','RETRACTED','not evidence of early warning — see below')+
  tile('CinC-2019 utility',f2(util),'official normalized score');
 app.appendChild(tiles);

 // The retraction, stated where the number used to be rather than in a footnote.
 var retr=document.createElement('div');
 retr.className='protocol';
 retr.style.borderLeft='3px solid #c0392b';
 retr.innerHTML='<b>Why lead time is retracted.</b> This page used to headline a median '+
  'lead time of '+lead.toFixed(1)+' h. On 2026-08-30 we tested that number against the '+
  'right null and it did not survive. A detector alarms at the <i>first</i> crossing of '+
  'its threshold, which for any low threshold lands near the start of a stay, so a long '+
  'lead is manufactured by how long patients happen to lie in the ICU and by how often '+
  'the alarm fires \u2014 not by skill. Measured at <b>matched detection rate</b> on the '+
  'development cohort, a <b>pure random score</b> produced 35.2 / 39.0 / 38.0 / 39.0 h '+
  'against this detector\u2019s 40.0 / 40.0 / 38.5 / 38.5 h, and random <b>beat</b> it at '+
  '70% detection. At the shipped operating point our excess over the null is '+
  '<b>\u22122.0 h</b>. Every lead time this project published before that audit '+
  '(26/29/34/37/39/40 h) sits inside the same null. The number above is kept only so the '+
  'retraction is auditable; it is not a claim. Reproduce: '+
  '<span class="mono">python -m backend.ml.scripts.attack_clock_claim</span>. '+
  'The corrected metric is <span class="mono">metrics.lead_time_vs_null()</span>, which '+
  'reports observed lead beside the random-score null at matched detection.';
 app.appendChild(retr);

 // ---- MODEL AUDIT -------------------------------------------------------------
 // What this detector's performance is actually MADE of. Pointed at our own model
 // first: an instrument for finding soft numbers in other people's work has no
 // standing until it has been used on ours.
 var AUD=/*__AUDIT__*/null;
 if(AUD&&AUD.detector){
  var au=document.createElement('div');
  au.className='protocol';
  au.style.borderLeft='3px solid #2c7a7b';
  function arow(k,v,note){return '<tr><td style="padding:4px 14px 4px 0">'+k+
    '</td><td class="mono" style="padding:4px 14px 4px 0;text-align:right"><b>'+v+
    '</b></td><td style="padding:4px 0;opacity:.72">'+(note||'')+'</td></tr>';}
  function af3(x){return (x===null||x===undefined||isNaN(x))?'\u2014':Number(x).toFixed(3);}
  function af1(x){return (x===null||x===undefined||isNaN(x))?'\u2014':Number(x).toFixed(1);}
  au.innerHTML='<b>Model audit \u2014 what this number is made of.</b> '+
   'A pooled AUROC averages over every positive/negative hour pair and silently mixes '+
   'two different questions. Applied here to <span class="mono">'+AUD.detector+
   '</span> on '+(AUD.n_patients||0).toLocaleString()+' patients.'+
   '<table style="margin:10px 0 6px;border-collapse:collapse;font-size:13px">'+
   arow('pooled AUROC',af3(AUD.pooled_auroc),'what is normally reported')+
   arow('BETWEEN \u2014 is this a septic patient?',af3(AUD.between_auroc),
       'patient selection \u2014 triage')+
   arow('WITHIN \u2014 is this the hour?',af3(AUD.within_auroc_strict),
       'per-stay, each admission its own control \u2014 the early-warning question')+
   arow('\u2003 same, pooled',af3(AUD.within_auroc_pooled),
       'disagrees by construction (Simpson\u2019s paradox); the stratified row above is honest')+
   arow('lead time observed',af1(AUD.lead_observed_h)+' h',
       'at detection '+af3(AUD.lead_detection_rate))+
   arow('lead time, RANDOM null',af1(AUD.lead_null_h)+' h',
       'a shuffled score at the SAME detection rate')+
   arow('excess over null',(AUD.lead_excess_h>0?'+':'')+af1(AUD.lead_excess_h)+' h',
       AUD.lead_excess_h>0?'':'no early-warning excess')+
   arow('suspicion echo',(AUD.suspicion_drop>0?'+':'')+af3(AUD.suspicion_drop)+' AUROC',
       'lost when the care team\u2019s ordering behaviour is deleted')+
   '</table>'+
   '<div style="margin-top:8px"><b>Verdict.</b> '+(AUD.verdict||'')+'</div>'+
   '<div style="margin-top:8px;opacity:.72"><b>The suspicion-echo row is the one to read '+
   'against the literature.</b> The published critique of the Epic Sepsis Model \u2014 '+
   'deployed across hundreds of hospitals, later found to miss about two thirds of cases '+
   '\u2014 was that because antibiotics are ordered once a clinician already suspects '+
   'sepsis, the model partly detected <i>the clinician\u2019s own suspicion</i>. That was '+
   'argued qualitatively. Here it is measured, and this detector loses only '+
   af3(AUD.suspicion_drop)+' AUROC when those features go, so it is not running on that '+
   'echo. The probe is an UPPER BOUND, not an attribution: the deleted features carry '+
   'real information too (a recently measured channel is also fresher data).</div>'+
   '<div style="margin-top:8px;opacity:.72">A low within-patient score is not '+
   'automatically a flaw \u2014 a model may be honestly aimed at triage. It is a flaw '+
   'only when the system is sold as early warning. No clock-amplification measure is '+
   'reported: that claim was attacked and failed. Reproduce: '+
   '<span class="mono">python -m backend.ml.scripts.run_audit</span> \u00b7 live at '+
   '<span class="mono">/aegis/audit</span>.</div>';
  app.appendChild(au);
 }

 // Forest plot
 var h2a=document.createElement('h2');h2a.innerHTML='<span class="n">1</span>Generalization — AUROC with 95% CI';app.appendChild(h2a);
 var l1=document.createElement('div');l1.className='lede';
 l1.textContent='Discrimination on the held-out hospital, both directions and both models. Overlapping intervals mean the difference is within noise — reported honestly rather than cherry-picked.';
 app.appendChild(l1);
 var fc=card();var frows=[];
 [['A_to_B','A→B'],['B_to_A','B→A']].forEach(function(dd){
  models.forEach(function(mk){var r=dir[dd[0]].models[mk];
   frows.push({label:MODEL[mk].n.split(' (')[0]+' · '+dd[1],v:r.auroc,ci:r.auroc_ci,color:MODEL[mk].c,bold:mk==='stacked'});});
 });
 forest(fc,frows);app.appendChild(fc);

 // ROC + PR (A->B)
 var h2b=document.createElement('h2');h2b.innerHTML='<span class="n">2</span>Discrimination curves (A→B)';app.appendChild(h2b);
 var g2=document.createElement('div');g2.className='grid2';
 var roc=card('ROC','true-positive vs false-positive rate');
 roc.appendChild(legend([{c:C2,t:'Stacked'},{c:C1,t:'GBM'},{c:cv('--ref'),t:'chance'}]));
 chart(roc,{w:460,h:300,xr:[0,1],yr:[0,1],xlab:'false-positive rate',ylab:'sensitivity',
  refs:[{pts:[[0,0],[1,1]],label:'chance'}],
  series:models.map(function(mk){var r=dir.A_to_B.models[mk];
   return {name:MODEL[mk].short,color:MODEL[mk].c,label:1,lat:0.5,
     pts:r.roc.fpr.map(function(x,i){return [x,r.roc.tpr[i]]})};})});
 g2.appendChild(roc);
 var pr=card('Precision–Recall','the honest view at 2% prevalence');
 pr.appendChild(legend([{c:C2,t:'Stacked'},{c:C1,t:'GBM'},{c:cv('--ref'),t:'prevalence'}]));
 var prev=dir.A_to_B.models.stacked.prevalence;
 // ignore the degenerate recall->0 spike (precision->1 on a single sample) so the axis
 // frames the informative region, not the endpoint. Take max precision at recall>=0.05.
 var maxP=prev*2;models.forEach(function(mk){var P=dir.A_to_B.models[mk].pr;
  P.recall.forEach(function(rc,j){if(rc>=0.05)maxP=Math.max(maxP,P.precision[j]);});});
 var prTop=Math.min(1,Math.max(0.12,maxP*1.2));
 chart(pr,{w:460,h:300,xr:[0,1],yr:[0,prTop],xlab:'recall',ylab:'precision',
  refs:[{pts:[[0,prev],[1,prev]],label:'base '+pct(prev)}],
  series:models.map(function(mk){var r=dir.A_to_B.models[mk];
   return {name:MODEL[mk].short,color:MODEL[mk].c,
     pts:r.pr.recall.map(function(x,i){return [x,r.pr.precision[i]]})};})});
 g2.appendChild(pr);
 app.appendChild(g2);

 // Calibration + Decision curve
 var h2c=document.createElement('h2');h2c.innerHTML='<span class="n">3</span>Are the probabilities trustworthy & useful?';app.appendChild(h2c);
 var g3=document.createElement('div');g3.className='grid2';
 var cal=card('Calibration (A→B)','predicted vs observed frequency');
 cal.appendChild(legend([{c:C2,t:'Stacked'},{c:cv('--ref'),t:'perfect'}]));
 var rel=dir.A_to_B.models.stacked.reliability;
 var mx=Math.max.apply(null,rel.mean_pred.concat(rel.obs_freq,[0.05]));
 chart(cal,{w:460,h:300,xr:[0,mx],yr:[0,mx],xlab:'mean predicted',ylab:'observed frequency',
  refs:[{pts:[[0,0],[mx,mx]],label:'ideal'}],
  series:[{name:'Stacked',color:C2,dots:1,pts:rel.mean_pred.map(function(x,i){return [x,rel.obs_freq[i]]})}]});
 g3.appendChild(cal);
 var dcCard=card('Decision curve (A→B)','net clinical benefit vs act-on-all / act-on-none');
 dcCard.appendChild(legend([{c:C2,t:'Stacked'},{c:C1,t:'GBM'},{c:cv('--ref'),t:'treat all / none'}]));
 var dcs=dir.A_to_B.models.stacked.decision_curve;
 var allnb=dcs.treat_all,thr=dcs.thresholds;
 var ymin=Math.min(0,Math.min.apply(null,allnb));var ymax=Math.max.apply(null,dir.A_to_B.models.stacked.decision_curve.model.concat([prev]));
 chart(dcCard,{w:460,h:300,xr:[thr[0],thr[thr.length-1]],yr:[ymin,ymax],xlab:'threshold probability',ylab:'net benefit',
  refs:[{pts:thr.map(function(t,i){return [t,allnb[i]]}),label:'all'},{pts:[[thr[0],0],[thr[thr.length-1],0]],label:'none',dash:'2 3'}],
  series:models.map(function(mk){var r=dir.A_to_B.models[mk];
   return {name:MODEL[mk].short,color:MODEL[mk].c,lat:0.35,
     pts:r.decision_curve.thresholds.map(function(t,i){return [t,r.decision_curve.model[i]]})};})});
 g3.appendChild(dcCard);
 app.appendChild(g3);

 // Alarm burden
 var h2d=document.createElement('h2');h2d.innerHTML='<span class="n">4</span>Alarm burden — the alarm-fatigue axis';app.appendChild(h2d);
 var l4=document.createElement('div');l4.className='lede';
 l4.textContent='Sweeping the alarm threshold: how many septic patients we catch vs how many false alerts we raise per 100 non-septic patient-hours. The tradeoff a bedside nurse lives with.';
 app.appendChild(l4);
 var ab=card();
 ab.appendChild(legend([{c:C2,t:'Stacked'},{c:C1,t:'GBM'}]));
 var abd=dir.A_to_B.models.stacked.alarm_burden;
 chart(ab,{w:900,h:300,xr:[0,Math.max.apply(null,abd.false_alerts_per_100h)],yr:[0,1],
  xlab:'false alerts per 100 non-septic patient-hours',ylab:'detection rate',
  series:models.map(function(mk){var r=dir.A_to_B.models[mk].alarm_burden;
   return {name:MODEL[mk].short,color:MODEL[mk].c,label:1,lat:0.5,
     pts:r.false_alerts_per_100h.map(function(x,i){return [x,r.detection_rate[i]]})};})});
 app.appendChild(ab);

 // Ablation table
 var h2e=document.createElement('h2');h2e.innerHTML='<span class="n">5</span>Full results — does stacking earn its complexity?';app.appendChild(h2e);
 var tc=card();tc.className='card scrollx';
 var rowsHTML='';
 [['A_to_B','A→B (train BIDMC, test Emory)'],['B_to_A','B→A (train Emory, test BIDMC)']].forEach(function(dd){
  models.forEach(function(mk){var r=dir[dd[0]].models[mk];var op=r.operating_points['sens@spec>=0.90']||{};
   rowsHTML+='<tr><td>'+dd[1]+'</td><td>'+MODEL[mk].n.split(' (')[0]+'</td>'+
    '<td class="'+(mk==='stacked'?'win':'')+'">'+f3(r.auroc)+'</td>'+
    '<td>'+f3(r.auprc)+'</td><td>'+f3(r.ece)+'</td>'+
    '<td>'+f2(r.normalized_utility)+'</td>'+
    '<td>'+((r.early_warning.median_lead_time_hours||0).toFixed(1))+'h</td>'+
    '<td>'+pct(r.early_warning.false_alarm_rate||0)+'</td></tr>';});
 });
 tc.innerHTML='<table><thead><tr><th>Test</th><th>Model</th><th>AUROC</th><th>AUPRC</th><th>ECE</th><th>Utility</th><th>Lead</th><th>False-alarm</th></tr></thead><tbody>'+rowsHTML+'</tbody></table>';
 app.appendChild(tc);

 // limitations
 var h2f=document.createElement('h2');h2f.innerHTML='<span class="n">6</span>What this does <i>not</i> show (read this)';app.appendChild(h2f);
 var lim=card();lim.className='card lim';
 lim.innerHTML='<ul>'+
  '<li><b>Wearable-only signal.</b> These numbers use only vitals a wearable can plausibly capture (HR, SpO₂, resp, temp, BP) — no labs. Adding labs raises AUROC but abandons the ambulatory use-case; that ablation is separate.</li>'+
  '<li><b>Modest absolute discrimination.</b> Cross-hospital AUROC ≈0.68–0.70 is honest for wearable-only sepsis prediction and consistent with the literature — it is <i>not</i> a state-of-the-art detection claim. The contribution is the <b>safety + personalization system around</b> the model, evaluated elsewhere.</li>'+
  '<li><b>Low AUPRC is expected, not a bug.</b> At ~2% per-hour prevalence, precision is intrinsically low; that is why we report calibration, utility, lead time, and decision-curve net benefit instead of leaning on AUROC alone.</li>'+
  '<li><b>Retrospective.</b> Two hospitals, replayed records — not a prospective deployment. Generalization beyond these two sites is unmeasured.</li>'+
  '</ul>';
 app.appendChild(lim);

 var foot=document.createElement('div');foot.className='foot';
 foot.innerHTML='Reproduce: <span class="mono">python -m backend.ml.scripts.rigorous_eval</span> → <span class="mono">build_report</span>. '+
  'Metrics adapt the official PhysioNet/CinC-2019 evaluation. AegisOS · HyperSepsisTwin.';
 app.appendChild(foot);
}
draw();
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
</script></body></html>"""


def _audit_blob() -> str:
    """The model audit, embedded so the page stands alone on a static host."""
    try:
        with open(AUDIT_SRC) as fh:
            return json.dumps(json.load(fh))
    except (OSError, ValueError):
        return "null"


def main() -> None:
    with open(SRC) as fh:
        data = json.load(fh)
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    html = html.replace("/*__AUDIT__*/null", _audit_blob())
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"wrote {OUT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
