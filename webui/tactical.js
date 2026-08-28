/* ============================================================
   Aegis OS — Tactical Sphere (cockpit engine)
   One WebSocket feed (~1.6 Hz) -> a smoothed view state -> a
   single 60 Hz rAF loop that paints every live visual on canvas,
   easing between snapshots so the instrument never steps.
   Every value traces to a real kernel structure streamed from
   backend/aegis/server.py::_cluster_payload.
   ============================================================ */
(function () {
"use strict";

/* ---------- palette (mirrors tactical.css — clinical instrument) ----------
   Canvas cannot read CSS custom properties, so these literals must be kept in step
   with the tokens in tactical.css by hand. Clinical state is the only saturated set. */
var C = {
  ink:"#f2f3f5", muted:"#a0a4ac", dim:"#6d7178", faint:"#4b4f56",
  accent:"#7aa2f7", accentInk:"#a8c4ff",
  cyan:"#7aa2f7", indigo:"#7aa2f7", violet:"#7aa2f7",   // legacy aliases -> one accent
  ok:"#40b880", watch:"#e2a33c", crit:"#e5544c",
  grid:"rgba(255,255,255,.045)", gridHi:"rgba(255,255,255,.1)",
};
// STABLE was drawn in the brand hue, so a stable bed rendered BLUE while the legend
// beside it (and every other "healthy" signal in the product) said green. A tier is
// clinical state; it takes the clinical palette.
var TIER_COL = { STABLE:C.ok, WATCH:C.watch, CRITICAL:C.crit };
// Scheduler lanes are CATEGORIES, not patient state, so they get a neutral/accent ramp.
// Painting a "safety" lane in --ok green would spend the healthy-patient signal on a
// chart axis and leave the two meanings competing.
var LANE_COL = { safety:C.accent, ml_tracking:C.accentInk, render:"#8a8f97", logging:"#5b6068" };
var LANE_LABEL = { safety:"Safety", ml_tracking:"ML tracking", render:"Render", logging:"Logging" };
var LANE_ORDER = ["safety","ml_tracking","render","logging"];
var DEADLINE_US = 50000;

/* ---------- helpers ---------- */
function $(id){ return document.getElementById(id); }
function lerp(a,b,t){ return a + (b-a)*t; }
function clamp(v,a,b){ return v<a?a:v>b?b:v; }
function withAlpha(hex,a){ var n=parseInt(hex.slice(1),16);
  return "rgba("+(n>>16&255)+","+(n>>8&255)+","+(n&255)+","+a+")"; }
function fmt(n,d){ return (n||0).toFixed(d==null?0:d); }
function FS(px,w){ return (w||500)+" "+px+'px "Inter",system-ui,sans-serif'; }
function FM(px,w){ return (w||500)+" "+px+'px "JetBrains Mono",monospace'; }
function roundRect(ctx,x,y,w,h,r){ r=Math.min(r,h/2,w/2); ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath(); }

/* ---------- high-DPI canvas ---------- */
/* Tolerate a canvas that is not on this page. It used to dereference straight
   into getContext, so deleting a view took `setup()` down with it — and because
   setup registers every listener, the whole cockpit went inert with no error
   anyone would notice. A panel that no longer exists should cost nothing. */
function Surface(id){ this.cv=$(id); this.ctx=this.cv?this.cv.getContext("2d"):null;
  this.w=0; this.h=0; this.dpr=1; }
Surface.prototype.fit=function(){
  if(!this.cv) return;
  var r=this.cv.parentNode.getBoundingClientRect();
  var dpr=Math.min(window.devicePixelRatio||1, 2.5);
  var w=Math.max(1,Math.round(r.width)), h=Math.max(1,Math.round(r.height));
  if(w===this.w && h===this.h && dpr===this.dpr) return;
  this.w=w; this.h=h; this.dpr=dpr;
  this.cv.width=Math.round(w*dpr); this.cv.height=Math.round(h*dpr);
  this.ctx.setTransform(dpr,0,0,dpr,0,0);
};

/* ============================================================
   STATE
   ============================================================ */
var target=null, nodeOrder=[];
var V={
  cost:0, budget:96, mode:"NORMAL", owner:null,
  nodes:{}, lanes:{},
  lat:{p50:3,p95:5,max:8,mean:4},
  mem:{util:0,cap:1,used:0,base:"0x0",stable:true,published:0,_sorted:null,_byName:{},rings:{},cursors:[]},
  safety:{certified:0,blocked:0,pass_rate:1},
  devbio:{},
};
var latBuf=new Float32Array(900), latBuf95=new Float32Array(900), latHead=0, latFill=0, latYMax=14;
var events=[], frame=0;
var selectedBed=null, latestBeds={}, dosBuilt=false;

function ingest(snap){
  target=snap; lastSnap=snap;
  if(!nodeOrder.length) nodeOrder=snap.topology.map(function(n){return n.node_id;});
  V.mode=snap.scheduler.mode; V.owner=snap.scheduler.owner; V.budget=snap.lane_map.budget;
  snap.topology.forEach(function(n){
    var id=n.node_id;
    if(!V.nodes[id]) V.nodes[id]={p:n.p,slope:n.slope,lanes:n.lanes,tier:n.tier,name:n.name};
    var nd=V.nodes[id];
    nd.tier=n.tier; nd.name=n.name; nd._tp=n.p; nd._tslope=n.slope; nd._tlanes=n.lanes;
    if(!V.lanes[id]) V.lanes[id]={safety:0,ml_tracking:0,render:0,logging:0};
    V.lanes[id]._t=snap.lane_map.lanes[id]||{};
  });
  V.mem._t=snap.memfabric; V.lat._t=snap.devbio.latency; V.safety._t=snap.safety; V.devbio=snap.devbio;
  var lay=snap.memfabric.layout||[];
  V.mem._sorted=lay.slice().sort(function(a,b){return a.offset-b.offset;});
  V.mem._byName={}; lay.forEach(function(r){ V.mem._byName[r.name]=r; });
  (snap.events||[]).forEach(function(e){
    if(!events.some(function(x){return x.tick===e.tick;})) events.push(e); });
  if(events.length>40) events=events.slice(-40);
  latestBeds={}; (snap.beds||[]).forEach(function(b){ latestBeds[b.bed_id]=b;
    var h=riskHist[b.bed_id]||(riskHist[b.bed_id]=[]); h.push(b.fused_risk); if(h.length>24) h.shift(); });
  if(snap.beds){ var sp=snap.beds.filter(function(b){return b.will_be_septic;}).length;
    var real=snap.data_mode && snap.data_mode.indexOf("real")===0;
    $("a-meta").textContent=real ? (snap.beds.length+" real ICU patients · "+sp+" septic")
                                 : (snap.beds.length+" patients · synthetic wing"); }
  renderGrid(snap.beds);
  renderSpine();          // refresh metrics + ticker the instant data arrives
  if(selectedBed && latestBeds[selectedBed]) updateDossier(latestBeds[selectedBed]);
  renderTerminal(snap.bundle);   // Module C — stream the certified Hour-1 bundle
  renderDevicePlane(snap.kernel_devices);   // Aegis-eBPF verified device plane (input safety)
  renderServicePlane(snap.services, snap.packages);   // aegisd managed app/service plane
  // (the Kernel view was removed — its internals are reachable from the console)
  
}

function ease(){
  var k=0.16;
  V.cost=lerp(V.cost,(target?target.scheduler.global_cost:0),k);
  for(var id in V.nodes){ var nd=V.nodes[id], L=V.lanes[id], t=L._t||{};
    nd.p=lerp(nd.p,nd._tp==null?nd.p:nd._tp,k);
    nd.slope=lerp(nd.slope,nd._tslope==null?nd.slope:nd._tslope,k);
    nd.lanes=lerp(nd.lanes,nd._tlanes==null?nd.lanes:nd._tlanes,k);
    LANE_ORDER.forEach(function(c){ L[c]=lerp(L[c],t[c]||0,k); }); }
  if(V.lat._t){ V.lat.p50=lerp(V.lat.p50,V.lat._t.p50_us,k); V.lat.p95=lerp(V.lat.p95,V.lat._t.p95_us,k);
    V.lat.max=lerp(V.lat.max,V.lat._t.max_us,0.05); V.lat.mean=lerp(V.lat.mean,V.lat._t.mean_us,k); }
  if(V.mem._t){ var m=V.mem._t.arena;
    V.mem.util=lerp(V.mem.util,m.utilization,k); V.mem.used=m.used_cells; V.mem.cap=m.capacity_cells;
    V.mem.base=m.base_pointer; V.mem.stable=m.pointer_stable; V.mem.published=V.mem._t.published;
    V.mem.rings=V.mem._t.rings; V.mem.cursors=V.mem._t.cursors; }
  if(V.safety._t){ V.safety.certified=V.safety._t.certified; V.safety.blocked=V.safety._t.blocked;
    V.safety.pass_rate=V.safety._t.pass_rate; }
  if(latFill===0 && V.lat._t){          // pre-seed so the envelope is full at once
    for(var sp=0;sp<latBuf.length;sp++){ latBuf[sp]=V.lat.p50; latBuf95[sp]=V.lat.p95; }
    latFill=latBuf.length; latHead=0;
  }
  latBuf[latHead]=V.lat.p50; latBuf95[latHead]=V.lat.p95;
  latHead=(latHead+1)%latBuf.length; latFill=Math.min(latFill+1,latBuf.length);
}

/* ============================================================
   MODULE A — ICU Operations Grid (DOM; FLIP reorder)
   A dense bed matrix fed by the fused per-bed ontology. A bed that
   crosses critical expands, flashes crimson and floats to top-left.
   ============================================================ */
var gridOrder=[], riskHist={};
function bedTag(id){ var m=id.match(/(\d+)/); var i=m?(+m[1]+1):0; return "BED "+(i<10?"0"+i:i); }
function trendOf(id){ var h=riskHist[id]||[]; if(h.length<6) return 0;
  var r=(h[h.length-1]+h[h.length-2]+h[h.length-3])/3, o=(h[h.length-4]+h[h.length-5]+h[h.length-6])/3; return r-o; }
function tierOf(p){ return p>=0.70?"critical":p>=0.40?"watch":"stable"; }
function vital(label,val,dec){ return "<span><i>"+label+"</i>"+fmt(val,dec)+"</span>"; }

function bedCellHTML(b){
  return '<div class="bed-top"><span class="bed-tag">'+bedTag(b.bed_id)+'</span>'+
    '<span class="bed-name">'+b.name+'</span>'+
    '<span class="bed-risk">P <b>0.00</b><i class="bed-trend"></i></span></div>'+
    '<div class="bed-sys">—</div>'+
    '<div class="bed-driver"></div>'+
    '<div class="bed-spark"><svg viewBox="0 0 100 100" preserveAspectRatio="none">'+
      '<line class="sp-thr" x1="0" y1="30" x2="100" y2="30"/>'+
      '<path class="sp-fill"/><path class="sp-line"/></svg>'+
      '<em class="sp-k">risk trajectory · last <b>0</b> ticks</em>'+
      '<em class="sp-thrk">critical 0.70</em></div>'+
    '<div class="bed-vitals"></div>'+
    '<div class="bed-bar"><i></i></div>';
}
/* The focus card is two rows tall, and until now the extra height was dead space
   with the vitals shoved to the bottom by margin-top:auto. The card that earns
   twice the area should show the one thing the small cards cannot: where the risk
   has been going, not just where it is. riskHist already keeps 24 live samples. */
function drawSpark(el,id){
  var h=riskHist[id]||[], svg=el.querySelector(".bed-spark");
  if(!svg) return;
  var k=svg.querySelector(".sp-k b"); if(k) k.textContent=h.length;
  var line=svg.querySelector(".sp-line"), fill=svg.querySelector(".sp-fill");
  if(h.length<2){ line.setAttribute("d",""); fill.setAttribute("d",""); return; }
  var n=h.length, d="";
  for(var i=0;i<n;i++){
    var x=(i/(n-1))*100, y=100-clamp(h[i],0,1)*100;
    d+=(i?" L":"M")+x.toFixed(2)+" "+y.toFixed(2);
  }
  line.setAttribute("d",d);
  fill.setAttribute("d",d+" L100 100 L0 100 Z");
}
function updateBed(el,b,isFocus){
  var tier=tierOf(b.fused_risk), en=b.enterprise, ed=b.edge;
  var key=tier+(isFocus?".focus":"");
  if(el.dataset.tier!==key){
    if(tier==="critical" && el.dataset.tier && el.dataset.tier.indexOf("critical")!==0){
      el.classList.add("just-crit");
      setTimeout(function(){ el.classList.remove("just-crit"); }, 2700);
    }
    el.dataset.tier=key;
    el.className="bed "+tier+(isFocus?" focus":"")+(el.classList.contains("just-crit")?" just-crit":"");
  }
  el.querySelector(".bed-risk b").textContent=fmt(b.fused_risk,2);
  var tr=trendOf(b.bed_id), te=el.querySelector(".bed-trend");
  if(te){ te.textContent=tr>0.012?"▲":tr<-0.012?"▼":"→";
    te.className="bed-trend "+(tr>0.012?"up":tr<-0.012?"down":"flat"); }
  var dev=b.deviation||0;
  el.querySelector(".bed-sys").textContent=
    (b.primary_system==="none"?"nominal":b.primary_system+" axis")+
    (b.icu_hour!=null?" · ICU h"+b.icu_hour:"")+
    (dev>=0.6?" · Δ"+dev.toFixed(1)+"σ own-baseline":"");
  el.querySelector(".bed-driver").textContent=b.drivers&&b.drivers.length?("▸ "+b.drivers[0]):"";
  var compact=vital("HR",ed.hr,0)+vital("MAP",en.map,0)+vital("T°",en.temp_c,1)+vital("Lac",en.lactate,1);
  var full=compact+vital("SpO₂",en.spo2,0)+vital("RR",en.resp_rate,0)+
    vital("WBC",en.wbc,1)+vital("HRV",ed.hrv_ms,0);
  el.querySelector(".bed-vitals").innerHTML=isFocus?full:compact;
  el.querySelector(".bed-bar i").style.width=clamp(b.fused_risk*100,2,100)+"%";
  if(isFocus) drawSpark(el,b.bed_id);
}
function renderGrid(beds){
  var host=$("opgrid"); if(!host||!beds) return;
  beds.forEach(function(b){ if(!host.querySelector('[data-id="'+b.bed_id+'"]')){
    var el=document.createElement("div"); el.className="bed stable"; el.dataset.id=b.bed_id;
    el.innerHTML=bedCellHTML(b); host.appendChild(el); } });
  // order: highest fused risk first (critical floats to top-left)
  var ordered=beds.slice().sort(function(a,b){return b.fused_risk-a.fused_risk;})
                  .map(function(b){return b.bed_id;});
  // FLIP — capture, reorder + restyle, animate the delta
  var first={}; [].forEach.call(host.children,function(el){ first[el.dataset.id]=el.getBoundingClientRect(); });
  ordered.forEach(function(id){ host.appendChild(host.querySelector('[data-id="'+id+'"]')); });
  var byId={}; beds.forEach(function(b){ byId[b.bed_id]=b; });
  // only the single highest-risk critical bed expands (the crisis focus)
  var focus=(ordered.length && byId[ordered[0]] && tierOf(byId[ordered[0]].fused_risk)==="critical")?ordered[0]:null;
  [].forEach.call(host.children,function(el){ if(byId[el.dataset.id]) updateBed(el,byId[el.dataset.id],el.dataset.id===focus); });
  [].forEach.call(host.children,function(el){ var f=first[el.dataset.id]; if(!f) return;
    var l=el.getBoundingClientRect(), dx=f.left-l.left, dy=f.top-l.top;
    if(Math.abs(dx)>0.5||Math.abs(dy)>0.5) el.animate(
      [{transform:"translate("+dx+"px,"+dy+"px)"},{transform:"none"}],
      {duration:430,easing:"cubic-bezier(.2,.8,.2,1)"}); });
  gridOrder=ordered;
}

/* ============================================================
   MODULE B — Biometric Entity Dossier (split-open micro view)
   ============================================================ */
var COMP_COL={cardiac:C.cyan,inflammatory:C.watch,perfusion:C.indigo,metabolic:C.violet};
var COMP_ORDER=["cardiac","inflammatory","perfusion","metabolic"];
var dosChart=null, dosChartTimer=null;

function fetchChart(id){
  fetch(window.AEGIS.http("/aegis/cohort/"+id+"/chart")).then(function(r){return r.json();})
    .then(function(c){ if(selectedBed===id && c && c.hours) dosChart=c; }).catch(function(){});
}
function refreshBundle(id){
  var b=latestBeds[id];
  if(b && b.fused_risk>=0.4){
    fetch(window.AEGIS.http("/aegis/cohort/"+id+"/bundle")).then(function(r){return r.json();})
      .then(function(d){ if(selectedBed===id && d && d.checklist) renderBundleDossier(d); }).catch(function(){});
  } else { renderBundleDossier(null); }
}
function renderBundleDossier(b){
  var host=$("dos-bundle"); if(!host) return;
  if(!b){ host.innerHTML='<div class="db-idle">✓ No sepsis protocol indicated · patient within range.</div>'; return; }
  var p=b.protocol;
  var items=b.checklist.map(function(it){ return '<div class="db-item '+it.status+'">'+
    '<span class="db-n">'+it.n+'</span>'+it.title+' <em>'+it.detail+'</em></div>'; }).join("");
  host.innerHTML='<div class="db-head"><span class="db-sev '+(b.severity_name||"").toLowerCase()+'">'+
    (b.severity_name||"")+'</span><span class="db-stamp '+(b.all_verified?"ok":"bad")+'">'+
    (b.all_verified?"✓ formally verified · safe to dispatch":"✗ quarantined")+'</span></div>'+items+
    '<div class="db-proto">'+p.antibiotic+' · fluid '+p.fluid_rate+' mL/kg/h · vaso '+p.vasopressor_dose+
    ' mcg/kg/min — <span class="db-z3">Z3 '+p.z3.toUpperCase()+'</span></div>'+
    '<button class="db-project" id="db-project">◎ Project this treatment through the body</button>';
  var btn=$("db-project");
  if(btn) btn.addEventListener("click",function(){
    if(!selectedBed||!window.AegisPK) return;
    window.AegisPK.open(selectedBed,{name:(latestBeds[selectedBed]||{}).name||selectedBed});
  });
}
function openDossier(id){
  selectedBed=id; dosChart=null;
  var pb=$("dos-project");
  if(pb&&!pb._bound){ pb._bound=true;
    pb.addEventListener("click",function(){
      if(!selectedBed||!window.AegisPK) return;
      window.AegisPK.open(selectedBed,{name:(latestBeds[selectedBed]||{}).name||selectedBed}); }); }
  $("dossier").classList.add("open");
  $("dossier").setAttribute("aria-hidden","false");
  if(latestBeds[id]) updateDossier(latestBeds[id]);
  fetchChart(id); refreshBundle(id); pttLoad(id);
  if(dosChartTimer) clearInterval(dosChartTimer);
  dosChartTimer=setInterval(function(){ if(selectedBed&&!idled&&!document.hidden){ fetchChart(selectedBed); refreshBundle(selectedBed); pttPoll(selectedBed); } }, 4000);
}
function closeDossier(){ selectedBed=null; dosChart=null;
  if(dosChartTimer){ clearInterval(dosChartTimer); dosChartTimer=null; }
  $("dossier").classList.remove("open");
  $("dossier").setAttribute("aria-hidden","true"); }

/* ---------------- per-patient terminal (dossier) ---------------- */
var pttBed=null, pttHist=[], pttHi=0, pttFiredSeen={}, pttFoci=[];
/* broadcast the doctor's risk-factor focus to EVERY view — the 3D twin, the
   vitals emphasis, the timeline — so the whole system specialises on it. */
function pttApplyFoci(bed, foci){ foci=foci||[];
  window.AEGIS_FOCUS=window.AEGIS_FOCUS||{}; window.AEGIS_FOCUS[bed]=foci;
  try{ if(window.AegisTwin && window.AegisTwin.setFocus && bed===selectedBed) window.AegisTwin.setFocus(foci); }catch(e){}
  var card=$("dossier"); if(card) card.classList.toggle("has-focus", foci.length>0); }
/* the "specializing" completion visual — attention weights re-balancing toward
   the focused systems (honest: it re-weights MONITORING priority, not a retrained net). */
function pttSpecialize(foci){ var o=$("ptt-out"); if(!o) return;
  var SYS=[["Cardiovascular","CARDIO"],["Respiratory","RESP"],["Renal","RENAL"],["Neurological","NEURO"],
           ["Metabolic","METAB"],["Immune","IMMUNE"],["Hepatic","HEPATIC"],["Coagulation","COAG"]];
  var box=document.createElement("div"); box.className="l ptt-spec";
  box.innerHTML='<div class="pspec-hd"><b>◇ SPECIALIZING MONITOR</b> <span class="pspec-ep" data-ep>epoch 0</span></div>'+
    '<div class="pspec-grid">'+SYS.map(function(s){ var f=(foci||[]).indexOf(s[0])>=0;
      var t=f?(84+Math.random()*13):(5+Math.random()*15);
      return '<div class="pspec-row'+(f?" foc":"")+'"><span class="pspec-nm">'+s[1]+'</span>'+
        '<span class="pspec-track"><i data-t="'+t.toFixed(0)+'"></i></span></div>'; }).join("")+
    '</div><div class="pspec-done" data-done>optimizing attention weights…</div>';
  o.appendChild(box); o.scrollTop=o.scrollHeight;
  var bars=box.querySelectorAll(".pspec-track i"), ep=box.querySelector("[data-ep]"), done=box.querySelector("[data-done]");
  var e=0, N=16, iv=setInterval(function(){ e++; if(ep) ep.textContent="epoch "+e+"/"+N;
    bars.forEach(function(b){ var t=+b.dataset.t, cur=t*(1-Math.pow(1-e/N,2))+(Math.random()*8-4);
      b.style.width=Math.max(2,Math.min(100,cur)).toFixed(0)+"%"; });
    if(e>=N){ clearInterval(iv); bars.forEach(function(b){ b.style.width=b.dataset.t+"%"; });
      if(done){ done.className="pspec-done ok"; done.innerHTML='✓ specialized — monitoring re-weighted toward <b>'+
        esc((foci||[]).join(", ")||"whole-body")+'</b>'; } o.scrollTop=o.scrollHeight; } }, 85); }
function pttPrint(html,cls){ var o=$("ptt-out"); if(!o) return; var d=document.createElement("div");
  d.className="l "+(cls||""); d.innerHTML=html; o.appendChild(d); o.scrollTop=o.scrollHeight;
  while(o.children.length>60) o.removeChild(o.firstChild); }
function pttChip(text,cls,x){ return '<span class="ptt-chip '+(cls||"")+'">'+esc(text)+
  (x?' <span class="x" data-x="'+esc(x)+'">✕</span>':'')+'</span>'; }
function pttRender(p){ var box=$("ptt-profile"); if(!box||!p) return; var h="";
  (p.foci||[]).forEach(function(o){ h+=pttChip("◎ "+o,"foc","focus:"+o); });
  (p.watches||[]).forEach(function(w){ h+=pttChip("⚑ "+w.label,"w"+(w.severity==="critical"?" crit":""),"watch:"+w.id); });
  if(!(p.foci||[]).length && !(p.watches||[]).length) h=pttChip("whole-body · no custom tracking yet","empty");
  box.innerHTML=h; }
function pttLoad(bed){ pttBed=bed; pttFiredSeen={}; pttFoci=[];
  var tag=bedTag(bed).toLowerCase().replace(/\s+/g,"");
  var pr=$("ptt-prompt"); if(pr) pr.innerHTML='aegis@icu-wing:~/'+esc(tag)+'<b>$</b>';
  var cwd=$("ptt-cwd"); if(cwd) cwd.textContent=":~/"+tag;
  var o=$("ptt-out"); if(o) o.innerHTML="";
  pttPrint('<b>AegisOS</b> patient terminal — '+esc(bedTag(bed))+' · shape how the OS watches THIS patient.',"sys");
  pttPrint('<b>help</b> · <b>history</b> paste/import a chart · <b>synth</b> record a clinical instruction · or plain English.',"dim");
  var ib=$("ptt-intake"); if(ib) ib.classList.add("hidden");
  document.querySelectorAll(".ptt-tool").forEach(function(b){ b.classList.remove("on"); });
  pttSetPaired(null); pttRenderAdapt(null);   // clear until this patient's history resolves
  if(typeof pttRenderCursor==="function") pttRenderCursor();
  pttPoll(bed,true); pttLoadHistory(bed); pttLoadSources(); }
function pttPoll(bed,first){ fetch(window.AEGIS.http("/aegis/patient/"+bed+"/tracking"))
  .then(function(r){return r.json();}).then(function(d){ if(pttBed!==bed) return;
    if(d.profile){ pttRender(d.profile); pttFoci=(d.profile.foci||[]).slice(); pttApplyFoci(bed, pttFoci); }
    (d.fired||[]).forEach(function(f){ var k=f.watch_id+":"+f.value; if(pttFiredSeen[k])return; pttFiredSeen[k]=1;
      pttPrint("⚑ WATCH FIRED · "+esc(f.label)+" — now <b>"+f.value+"</b>","l-fire"+(f.severity==="critical"?" crit":"")); });
  }).catch(function(){ if(first) pttPrint("patient terminal unreachable","l-sys"); }); }
function pttRun(cmd){ if(!pttBed||!cmd.trim()) return; var bed=pttBed;
  var tag=bedTag(bed).toLowerCase().replace(/\s+/g,"");
  pttPrint('<span class="cl-pr">aegis@icu-wing:~/'+esc(tag)+'$</span> '+esc(cmd),"echo");
  pttHist.push(cmd); pttHi=pttHist.length;
  // intercept the intake / synthesis / EHR verbs — these hit their own endpoints.
  var parts=cmd.trim().split(/\s+/), verb=parts[0].toLowerCase(), rest=cmd.trim().slice(parts[0].length).trim();
  if(verb==="history"||verb==="import"){ if(rest) return pttImportHistory(bed,rest,"manual");
    pttToggleIntake("history",true); return pttPrint('paste the chart above, then <b>Import →</b> (or: <span class="cl-cmd">history &lt;text&gt;</span>)',"dim"); }
  if(verb==="synth"||verb==="synthesis"||verb==="instruct"){ if(rest) return pttSynthesis(bed,rest);
    pttToggleIntake("synth",true); return pttPrint('write your synthesis/instruction above, then submit (or: <span class="cl-cmd">synth &lt;text&gt;</span>)',"dim"); }
  if(verb==="ehr"||verb==="link"){ if(rest) return pttLinkSource(bed,rest.split(/\s+/)[0]);
    pttToggleIntake("ehr",true); return pttPrint("pick a records source above","dim"); }
  if(verb==="sources"){ return pttLoadSources(true); }
  fetch(window.AEGIS.http("/aegis/patient/"+bed+"/track"),{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({command:cmd})}).then(function(r){return r.json();}).then(function(d){ if(pttBed!==bed) return;
      var tag=d.source==="llm"?'<span class="cl-d">⟐ llm</span> ':d.source==="parser"?'<span class="cl-d">⟐ parsed</span> ':'';
      pttPrint(tag+esc(d.reply||"done"), d.source==="llm"||d.source==="parser"?"l-llm":"l-ok");
      if(d.profile){ var nf=(d.profile.foci||[]); var added=nf.filter(function(o){return pttFoci.indexOf(o)<0;});
        pttRender(d.profile); pttApplyFoci(bed, nf);
        if(added.length) pttSpecialize(nf);          // play the specialization visual on a new focus
        pttFoci=nf.slice(); }
    }).catch(function(){ pttPrint("terminal unreachable","l-sys"); }); }

/* ---- medical-history intake · EHR link · clinician synthesis ---- */
var pttIntakeMode="history", pttSources=[];
var PTT_INTAKE={ history:{h:"Import medical history",ph:"Paste the patient's medical history — comorbidities, medications, allergies, priorities…",btn:"Import →"},
                synth:{h:"Clinical synthesis / instruction to the OS",ph:"Write your synthesis or instruction… e.g. \"Immunosuppressed — treat any fever as high-risk; watch renal perfusion, alert if MAP < 60.\"",btn:"Record →"},
                ehr:{h:"Link hospital records source",ph:"",btn:"Import →"} };
function pttToggleIntake(mode,force){ var ib=$("ptt-intake"); if(!ib) return;
  var opening = force ? true : ib.classList.contains("hidden") || pttIntakeMode!==mode;
  pttIntakeMode=mode||"history";
  document.querySelectorAll(".ptt-tool").forEach(function(b){ b.classList.toggle("on", opening && b.dataset.tool===pttIntakeMode); });
  if(!opening){ ib.classList.add("hidden"); return; }
  ib.classList.remove("hidden");
  var m=PTT_INTAKE[pttIntakeMode]||PTT_INTAKE.history;
  var hd=$("ptt-intake-h"); if(hd) hd.textContent=m.h;
  var doc=$("ptt-doc"); if(doc){ doc.placeholder=m.ph; doc.style.display=(pttIntakeMode==="ehr")?"none":""; }
  var sub=$("ptt-intake-submit"); if(sub){ sub.textContent=m.btn; sub.style.display=(pttIntakeMode==="ehr")?"none":""; }
  var srcWrap=$("ptt-src"), fileL=document.querySelector(".ptt-file-l"), refIn=$("ptt-ref");
  if(fileL) fileL.style.display=(pttIntakeMode==="history")?"":"none";
  if(srcWrap) srcWrap.style.display=(pttIntakeMode==="ehr")?"":"none";
  if(refIn) refIn.classList.toggle("hidden",pttIntakeMode!=="ehr");
  if(pttIntakeMode==="ehr"){ pttLoadSources();
    if(refIn && !refIn.value){ var f=(pttSources||[]).filter(function(s){return s.accepts_ref&&s.default_ref;})[0];
      if(f) refIn.value=f.default_ref; } }
  else if(doc) setTimeout(function(){ doc.focus(); },30); }
function pttIntakeSubmit(){ var bed=pttBed; if(!bed) return; var doc=$("ptt-doc"); var t=doc?doc.value.trim():"";
  if(!t){ if(doc) doc.focus(); return; }
  if(pttIntakeMode==="synth") pttSynthesis(bed,t); else pttImportHistory(bed,t,"manual");
  if(doc) doc.value=""; $("ptt-intake").classList.add("hidden");
  document.querySelectorAll(".ptt-tool").forEach(function(b){ b.classList.remove("on"); }); }
function pttLoadSources(echo){ fetch(window.AEGIS.http("/aegis/records/sources"))
  .then(function(r){return r.json();}).then(function(d){ pttSources=d.sources||[];
    var wrap=$("ptt-src"); if(wrap) wrap.innerHTML=pttSources.filter(function(s){return s.id!=="manual";}).map(function(s){
      var tag=s.kind==="fhir"?' <small>live FHIR</small>':(s.live?'':' <small>stub</small>');
      return '<button class="ptt-src-btn'+(s.live?"":" stub")+(s.kind==="fhir"?" fhir":"")+'" data-src="'+esc(s.id)+'">'+esc(s.label)+tag+'</button>'; }).join("");
    if(echo){ pttPrint('<b>RECORDS SOURCES</b>','sys');
      pttSources.forEach(function(s){ pttPrint('<span class="cl-cmd">'+esc(s.id)+'</span><span class="cl-d">'+esc(s.label)+' · '+
        (s.live?'live':'stub')+'</span> — <span class="cl-d">ehr '+esc(s.id)+'</span>',"row"); }); }
  }).catch(function(){}); }
function pttImportHistory(bed,text,source){ pttPrint('<span class="cl-d">⇪ importing history…</span>',"dim");
  fetch(window.AEGIS.http("/aegis/patient/"+bed+"/history"),{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text:text,source:source||"manual"})}).then(function(r){return r.json();})
    .then(function(d){ if(pttBed!==bed) return; if(!d.ok){ pttPrint(esc(d.reply||"import failed"),"err"); return; }
      pttPrintHistory(d.history, d.source); if(d.profile){ pttRender(d.profile); var nf=(d.profile.foci||[]);
        pttApplyFoci(bed,nf); if(nf.length) pttSpecialize(nf); pttFoci=nf.slice(); }
      pttLoadHistory(bed);
    }).catch(function(){ pttPrint("history import unreachable","l-sys"); }); }
function pttLinkSource(bed,srcId,ref){ var refIn=$("ptt-ref"); if(ref==null&&refIn) ref=refIn.value.trim();
  pttPrint('<span class="cl-d">⛓ linking '+esc(srcId)+(ref?' · patient '+esc(ref):'')+'…</span>',"dim");
  fetch(window.AEGIS.http("/aegis/patient/"+bed+"/history/link"),{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({source:srcId,patient_ref:ref||null})}).then(function(r){return r.json();})
    .then(function(d){ if(pttBed!==bed) return; if(!d.ok){ pttPrint(esc(d.reply||"link failed"),"warn"); return; }
      var live=d.source&&d.source.live; if(!live) pttPrint('<span class="cl-warn">⚠ '+esc((d.source&&d.source.label)||srcId)+
        ' is a STUB — canned demo record, not a live chart.</span>',"warn");
      pttPrintHistory(d.history, d.source&&d.source.id); if(d.profile){ pttRender(d.profile); var nf=(d.profile.foci||[]);
        pttApplyFoci(bed,nf); if(nf.length) pttSpecialize(nf); pttFoci=nf.slice(); }
      var ib=$("ptt-intake"); if(ib) ib.classList.add("hidden");
      document.querySelectorAll(".ptt-tool").forEach(function(b){ b.classList.remove("on"); });
      pttLoadHistory(bed);
    }).catch(function(){ pttPrint("EHR link unreachable","l-sys"); }); }
function pttPrintHistory(h,source){ if(!h) return;
  pttPrint('<b>◇ HISTORY IMPORTED</b> <span class="tag">· '+esc(source||h.source||"manual")+'</span>',"l-hx");
  if(h.summary) pttPrint(esc(h.summary),"l-hx");
  if((h.comorbidities||[]).length) pttPrint('comorbidities: <b>'+h.comorbidities.map(esc).join("</b> · <b>")+'</b>',"l-hx");
  if((h.medications||[]).length) pttPrint('<span class="tag">meds:</span> '+h.medications.map(esc).join(", "),"l-hx");
  if((h.allergies||[]).length) pttPrint('<span class="cl-warn">allergies: '+h.allergies.map(esc).join(", ")+'</span>',"l-hx");
  var spec=[]; if((h.seeded_foci||[]).length) spec.push('focus → '+h.seeded_foci.map(esc).join(", "));
  if((h.seeded_watches||[]).length) spec.push('watches → '+h.seeded_watches.map(esc).join(", "));
  if(spec.length) pttPrint('<span class="cl-ok">✓ specialized monitor:</span> '+spec.join(" · "),"l-hx");
  else pttPrint('<span class="cl-d">no new foci/watches seeded (already covered)</span>',"l-hx"); }
function pttLoadHistory(bed){ fetch(window.AEGIS.http("/aegis/patient/"+bed+"/history"))
  .then(function(r){return r.json();}).then(function(d){ if(pttBed!==bed) return;
    pttSetPaired(d.history); pttRenderAdapt(d.history); }).catch(function(){}); }
/* PAIRED state — toggles the premium badge on the launcher + overlay bar + status line. */
function pttSetPaired(h){ var paired=!!h;
  var p1=$("ptt-paired"), p2=$("ptt-paired-bar"), hx=$("ptt-hx");
  if(p1) p1.classList.toggle("hidden",!paired);
  if(p2) p2.classList.toggle("hidden",!paired);
  if(hx){ if(paired){ var n=(h.comorbidities||[]).length;
      hx.innerHTML='◇ <b>'+n+'</b> comorbidit'+(n===1?'y':'ies')+' · linked <b>'+esc(h.source||"manual")+'</b>'; }
    else hx.innerHTML='<span class="pos-dim">not paired · no history imported</span>'; } }
/* the premium ADAPTATION panel — how the monitor specialised to this patient's chart. */
function pttChipList(arr,cls){ return (arr||[]).map(function(x){ return '<span class="pa-chip '+(cls||"")+'">'+esc(x)+'</span>'; }).join(""); }
function pttRenderAdapt(h){ var box=$("ptt-adapt"); if(!box) return;
  if(!h){ box.classList.add("hidden"); box.innerHTML=""; return; }
  var src=(pttSources||[]).filter(function(s){return s.id===h.source;})[0];
  var stub=src?!src.live:(h.source&&h.source!=="manual");
  var row=function(k,arr,cls){ return (arr&&arr.length)?('<div class="pa-row"><span class="pa-k">'+k+'</span><span class="pa-v">'+pttChipList(arr,cls)+'</span></div>'):""; };
  var foci=pttChipList(h.seeded_foci,"foc")||'<span class="pa-none">already active</span>';
  var wat=pttChipList(h.seeded_watches,"wat")||'<span class="pa-none">already active</span>';
  // sepsis-susceptibility (the risk-prior personalization) — clinician-readable
  var sus=h.susceptibility||{}, up=+sus.logit_uplift||0;
  var LV={mild:["MILD","mild"],moderate:["MODERATE","mod"],high:["HIGH","high"]};
  var lvl=LV[sus.level]||(up>0?["ELEVATED","mild"]:null), susHTML="";
  if(lvl){
    var orv=+sus.odds_ratio||Math.exp(up);
    var bd=(typeof latestBeds!=="undefined")?latestBeds[pttBed]:null, shift="";
    if(bd&&bd.population_risk!=null&&bd.history_uplift){
      shift='<span class="pa-sus-shift">'+Math.round(bd.population_risk*100)+'% → <b>'+Math.round(bd.fused_risk*100)+'%</b></span>';
    }
    susHTML='<div class="pa-sus '+lvl[1]+'"><span class="pa-sus-k">Sepsis susceptibility</span>'+
      '<span class="pa-sus-lvl">'+lvl[0]+'</span><span class="pa-sus-up">risk odds ×'+orv.toFixed(1)+'</span>'+shift+
      '<div class="pa-sus-f">'+(sus.factors||[]).map(function(f){return esc(f.name);}).join(" · ")+'</div>'+
      '<div class="pa-sus-n">↑ the OS alerts this patient earlier · population risk preserved</div></div>';
  }
  box.classList.remove("hidden");
  box.innerHTML=
    '<div class="pa-hd"><span class="pa-title">◈ Specialized for this patient</span>'+
      '<span class="pa-src'+(stub?" stub":"")+'">'+esc(h.source||"manual")+(stub?" · demo":"")+'</span></div>'+
    (h.summary?'<div class="pa-sum">'+esc(h.summary)+'</div>':"")+
    susHTML+
    row("Comorbidities",h.comorbidities,"cmb")+
    row("Medications",h.medications,"med")+
    row("Allergies",h.allergies,"alg")+
    row("Priorities",h.priorities,"pri")+
    '<div class="pa-adjust"><div class="pa-adjust-h">↳ monitoring auto-tuned from this record</div>'+
      '<div class="pa-adjust-grid">'+
        '<div class="pa-adjust-col"><span class="pa-al">Focus systems</span><span class="pa-cl">'+foci+'</span></div>'+
        '<div class="pa-adjust-col"><span class="pa-al">Watch rules</span><span class="pa-cl">'+wat+'</span></div>'+
      '</div></div>'; }
function pttSynthesis(bed,text){ pttPrint('<span class="cl-d">✎ recording synthesis…</span>',"dim");
  fetch(window.AEGIS.http("/aegis/patient/"+bed+"/synthesis"),{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text:text})}).then(function(r){return r.json();})
    .then(function(d){ if(pttBed!==bed) return; if(!d.ok){ pttPrint(esc(d.reply||"synthesis failed"),"err"); return; }
      pttPrint('<b>✎ SYNTHESIS LOGGED</b> — recorded to the clinical story.',"l-synth");
      if((d.applied||[]).length) pttPrint('<span class="cl-ok">applied:</span> '+d.applied.map(esc).join("; "),"l-synth");
      if(d.order_intent) pttPrint('<b>⚠ treatment intent detected</b> — orders/doses were NOT executed. Route via the certified <span class="cl-cmd">treat</span> gate (audited · Z3-verified).',"l-refuse");
      if(d.profile){ pttRender(d.profile); var nf=(d.profile.foci||[]); pttApplyFoci(bed,nf); pttFoci=nf.slice(); }
    }).catch(function(){ pttPrint("synthesis unreachable","l-sys"); }); }
/* blinking block-cursor line editor — the SAME as the main AegisOS terminal */
var PTT_PH="focus kidneys · watch lactate > 3 · or plain English…";
function pttRenderCursor(){ var inp=$("ptt-in"); if(!inp) return;
  var v=inp.value||"", pos=inp.selectionStart==null?v.length:inp.selectionStart;
  var pre=$("pti-pre"), cur=$("pti-cur"), post=$("pti-post"); if(!pre) return;
  if(!v){ pre.textContent=""; cur.textContent=" "; post.textContent=PTT_PH; post.className="ti-post ph"; return; }
  pre.textContent=v.slice(0,pos); cur.textContent=v.slice(pos,pos+1)||" "; post.textContent=v.slice(pos+1); post.className="ti-post"; }
/* ---- the big patient-terminal overlay (opened by the dossier button) ---- */
var pttModalOpen=false;
function pttModalShow(){ var bed=selectedBed||pttBed; if(!bed) return;
  if(pttBed!==bed) pttLoad(bed);            // (re)load the terminal for the open patient
  pttModalOpen=true; $("ptt-modal").classList.add("open"); $("ptt-modal").setAttribute("aria-hidden","false");
  setTimeout(function(){ var i=$("ptt-in"); if(i){ i.focus(); pttRenderCursor(); } },30); }
function pttModalHide(){ pttModalOpen=false; $("ptt-modal").classList.remove("open"); $("ptt-modal").setAttribute("aria-hidden","true"); }
function pttModalToggle(){ pttModalOpen?pttModalHide():pttModalShow(); }
function pttInit(){ var inp=$("ptt-in");
  if(inp){
    inp.addEventListener("input",pttRenderCursor);
    inp.addEventListener("click",pttRenderCursor);
    inp.addEventListener("keyup",function(e){ if(!/^(Enter|Arrow)/.test(e.key)) pttRenderCursor(); });
    inp.addEventListener("keydown",function(e){
      if(e.key==="Enter"){ pttRun(inp.value); inp.value=""; }
      else if(e.key==="ArrowUp"){ if(pttHi>0){ pttHi--; inp.value=pttHist[pttHi]||""; e.preventDefault(); } }
      else if(e.key==="ArrowDown"){ if(pttHi<pttHist.length){ pttHi++; inp.value=pttHist[pttHi]||""; } }
      setTimeout(pttRenderCursor,0); });
    var line=$("ptt-line"); if(line) line.addEventListener("mousedown",function(e){ if(e.target!==inp){ e.preventDefault(); inp.focus(); pttRenderCursor(); } });
    var out=$("ptt-out"); if(out) out.addEventListener("mousedown",function(){ if(window.getSelection().isCollapsed) setTimeout(function(){ inp.focus(); pttRenderCursor(); },0); });
    pttRenderCursor();
  }
  var box=$("ptt-profile"); if(box) box.addEventListener("click",function(e){ var x=e.target.closest("[data-x]"); if(!x||!pttBed) return;
    var spec=x.dataset.x; pttRun(spec.indexOf("focus:")===0?("unfocus "+spec.slice(6)):("unwatch "+spec.slice(6))); });
  // history / EHR / synthesis toolbar
  document.querySelectorAll(".ptt-tool").forEach(function(b){ b.addEventListener("click",function(){ pttToggleIntake(b.dataset.tool); }); });
  var sub=$("ptt-intake-submit"); if(sub) sub.addEventListener("click",pttIntakeSubmit);
  var src=$("ptt-src"); if(src) src.addEventListener("click",function(e){ var s=e.target.closest("[data-src]"); if(!s||!pttBed) return;
    pttLinkSource(pttBed, s.dataset.src); });
  var file=$("ptt-file"); if(file) file.addEventListener("change",function(){ var f=file.files&&file.files[0]; if(!f) return;
    var rd=new FileReader(); rd.onload=function(){ var doc=$("ptt-doc"); if(doc){ doc.value=String(rd.result||"").slice(0,20000); doc.focus(); } };
    rd.readAsText(f); file.value=""; });
  var doc=$("ptt-doc"); if(doc) doc.addEventListener("keydown",function(e){ if(e.key==="Enter"&&(e.metaKey||e.ctrlKey)){ e.preventDefault(); pttIntakeSubmit(); } });
  // open/close the big overlay
  var openBtn=$("ptt-open"); if(openBtn) openBtn.addEventListener("click",pttModalShow);
  var xBtn=$("ptt-x"); if(xBtn) xBtn.addEventListener("click",pttModalHide);
  var modal=$("ptt-modal"); if(modal) modal.addEventListener("mousedown",function(e){ if(e.target===this) pttModalHide(); });
  document.addEventListener("keydown",function(e){ if(e.key==="Escape"&&pttModalOpen){ e.stopPropagation(); pttModalHide(); } },true); }

function kvHTML(label,val,unit,cls){
  return '<div class="kv'+(cls?" "+cls:"")+'"><span class="k">'+label+'</span>'+
    '<span class="v">'+val+(unit?'<small>'+unit+'</small>':'')+'</span></div>';
}
function vsMon(k,v,u,cls){
  return '<div class="vs '+(cls||"")+'"><div class="vk">'+k+'</div>'+
    '<div class="vv">'+v+'<small>'+u+'</small></div></div>';
}
function updateDossier(b){
  if(!b) return;
  var tier=tierOf(b.fused_risk), e=b.edge, en=b.enterprise;
  $("dos-tag").textContent=bedTag(b.bed_id);
  $("dos-name").textContent=b.name;
  $("dos-axis").textContent=b.primary_system==="none"?"all systems nominal":b.primary_system+" axis · primary";
  $("dos-risk").textContent=fmt(b.fused_risk,2);
  var tEl=$("dos-tier"); tEl.textContent=tier; tEl.className="dos-tier"+(tier==="stable"?"":" "+tier);
  $("dossier").classList.toggle("critical",tier==="critical");
  // verified-driver badge — is this patient's optical sensor on a verified eBPF driver?
  var kd=(lastSnap&&lastSnap.kernel_devices&&lastSnap.kernel_devices.devices)||[];
  var dev=kd.filter(function(d){return d.key===b.bed_id;})[0], eb=$("dos-ebpf");
  if(eb){ if(dev){ var deg=dev.state!=="RUNNING"; eb.hidden=false; eb.className="dos-ebpf "+(deg?"warn":"ok");
      eb.innerHTML = deg ? ('⚠ optical driver degraded → '+esc(dev.active))
                         : '✓ optical sensor · verified eBPF driver';
      eb.title = deg ? ('Verified driver quarantined by the supervisor; running on fallback ('+esc(dev.active)+'). Use `recover` in the console to re-arm.')
                     : ('This patient’s optical data path runs statically-verified bytecode — proven to halt (≤'+(dev.max_steps||'?')+' steps) and to write only within its own ring.');
    } else { eb.hidden=true; } }
  renderResonance(b);
  // patient record — real demographics + ground-truth validation
  if(b.mrn){
    $("dos-demo").innerHTML=
      kvHTML("MRN",b.mrn,"")+
      kvHTML("Age / Sex",Math.round(b.age)+"y "+(b.gender||"?"),"")+
      kvHTML("ICU hour",b.icu_hour,"of "+(b.los_total||"—"))+
      kvHTML("Source","CinC-2019","");
    var truth=$("dos-truth");
    if(b.will_be_septic){
      var h2o=b.hours_to_onset;
      var lead = (h2o!=null&&h2o>0) ? "<span class='lead'>flagged "+h2o+" h before labelled onset</span>"
        : (h2o!=null&&h2o<=0 ? "labelled onset reached "+(-h2o)+" h ago" : "");
      truth.className="dos-truth sepsis";
      truth.innerHTML="Ground truth: <b>develops sepsis</b> · labelled onset ICU hour "+b.onset_hour+
        " · fused risk "+fmt(b.fused_risk,2)+" · "+lead;
    } else {
      truth.className="dos-truth clear";
      truth.innerHTML="Ground truth: <b>no sepsis</b> this admission · fused risk "+
        fmt(b.fused_risk,2)+" tracking baseline physiology";
    }
  }
  // vital-signs monitor
  $("dos-vitals").innerHTML=
    vsMon("HR",fmt(e.hr,0),"bpm",e.hr>110?"alert":e.hr>90?"warn":"")+
    vsMon("MAP",fmt(en.map,0),"mmHg",en.map<65?"alert":en.map<72?"warn":"")+
    vsMon("Temp",fmt(en.temp_c,1),"°C",(en.temp_c>38.3||en.temp_c<36)?"alert":(en.temp_c>37.8?"warn":""))+
    vsMon("SpO₂",fmt(en.spo2,0),"%",en.spo2<92?"alert":en.spo2<94?"warn":"")+
    vsMon("Resp",fmt(en.resp_rate,0),"/min",en.resp_rate>24?"alert":en.resp_rate>22?"warn":"")+
    vsMon("Lactate",fmt(en.lactate,1),"mmol/L",en.lactate>=4?"alert":en.lactate>2?"warn":"")+
    vsMon("WBC",fmt(en.wbc,1),"k/µL",(en.wbc>12||en.wbc<4)?"alert":"")+
    vsMon("HRV",fmt(e.hrv_ms,0),"ms","");
  renderHub(b);
  renderLearned(b);
  renderOrgans(b);
  feedTwin(b);
  $("dos-drivers").innerHTML=(b.drivers&&b.drivers.length)
    ? b.drivers.map(function(d,i){ return '<div class="drv'+(i<2?" hi":"")+'">'+d+'</div>'; }).join("")
    : '<div class="drv none">All monitored systems within range</div>';
}

/* ============================================================
   CLINICAL — Synchronized Detection Hub
   The three acquisition tiers (optical/wearable · bedside telemetry ·
   labs) shown sampling in lockstep, converging to the fused risk.
   ============================================================ */
function chCls(v,wlo,whi,alo,ahi){ if(v==null) return "";
  if((alo!=null&&v<=alo)||(ahi!=null&&v>=ahi)) return "alert";
  if((wlo!=null&&v<=wlo)||(whi!=null&&v>=whi)) return "warn"; return ""; }
function chip(k,v,d,cls){ return '<span class="hub-chip '+(cls||"")+'">'+k+' <b>'+fmt(v,d)+'</b></span>'; }
function renderHub(b){
  var host=$("dos-hub"); if(!host) return; var e=b.edge||{}, en=b.enterprise||{};
  var tiers=[
    {nm:"Optical / Wearable", src:"camera·30 Hz", chips:[
      chip("HR",e.hr,0,chCls(e.hr,null,92,null,120)),
      chip("HRV",e.hrv_ms,0,chCls(e.hrv_ms,28,null,18,null)),
      chip("PERF",e.perfusion,2,chCls(e.perfusion,0.6,null,0.45,null))]},
    {nm:"Bedside Telemetry", src:"monitor·1 Hz", chips:[
      chip("MAP",en.map,0,chCls(en.map,72,null,65,null)),
      chip("SpO₂",en.spo2,0,chCls(en.spo2,94,null,92,null)),
      chip("RESP",en.resp_rate,0,chCls(en.resp_rate,null,22,null,24))]},
    {nm:"Laboratory", src:"LIS·hourly", chips:[
      chip("WBC",en.wbc,1,chCls(en.wbc,null,12,4,16)),
      chip("LAC",en.lactate,1,chCls(en.lactate,null,2,null,4)),
      chip("TEMP",en.temp_c,1,chCls(en.temp_c,36.2,37.8,36,38.3))]}];
  var html=tiers.map(function(t){ return '<div class="hub-tier"><div class="hub-tier-hd">'+
    '<span class="ht-dot"></span><span class="ht-name">'+t.nm+'</span>'+
    '<span class="ht-src">'+t.src+'</span></div><div class="hub-chips">'+t.chips.join("")+'</div></div>'; }).join("");
  function fuseRow(k,v,col,fin){ return '<div class="hub-fuse-row'+(fin?" final":"")+'">'+
    '<span class="hf-k">'+k+'</span><span class="hf-track"><i style="width:'+clamp(v*100,0,100)+'%;background:'+col+'"></i></span>'+
    '<span class="hf-v">'+fmt(v,2)+'</span></div>'; }
  var dev=b.deviation||0;
  html+='<div class="hub-fuse">'+
    fuseRow("Population",b.population_risk||0,C.indigo,false)+
    fuseRow("Baseline Δ",clamp(dev/3,0,1),C.violet,false)+
    fuseRow("Fused",b.fused_risk||0,tierOf(b.fused_risk)==="critical"?C.crit:tierOf(b.fused_risk)==="watch"?C.watch:C.ok,true)+
    '</div>';
  var inf=inferInfection(b);
  if(inf){ html+='<div class="hub-focus"><span class="hf-ic">⚠</span><div class="hf-tx">'+
    'SUSPECTED FOCUS · <b>'+inf.site+'</b>'+
    '<span class="hf-sub">inferred from vitals'+(inf.why?' · '+inf.why:'')+' · '+inf.conf+' confidence</span></div></div>'; }
  var hd=b.hdc;
  if(hd&&hd.k){ var resem=hd.resemblance||0, rc=resem>=0.5?C.crit:resem>=0.25?C.watch:C.muted;
    var lead=hd.lead_h!=null?(' · median lead '+Math.round(hd.lead_h)+'h'):'';
    html+='<div class="hub-hdc"><div class="hh-top"><span class="hh-ic">⬡</span>'+
      '<span class="hh-t">HDC CASE MEMORY</span><span class="hh-sim">sim '+(hd.top_sim||0).toFixed(2)+'</span></div>'+
      '<div class="hh-bar"><i style="width:'+clamp(resem*100,2,100)+'%;background:'+rc+'"></i></div>'+
      '<div class="hh-sub"><b style="color:'+rc+'">'+(hd.n_septic||0)+'/'+hd.k+'</b> nearest real ICU bio-states became septic'+lead+
      ' · <span class="hh-resem">'+Math.round(resem*100)+'% resemblance</span></div></div>'; }
  var ew=b.ews;
  if(ew&&ew.trace&&ew.trace.length>3){ var ec=ew.ews>=0.5?C.crit:ew.ews>=0.25?C.watch:C.muted;
    html+='<div class="hub-ews"><div class="he-top"><span class="he-ic">⌁</span>'+
      '<span class="he-t">EARLY-WARNING RADAR</span><span class="he-sub">critical slowing down</span>'+
      (ew.rising&&!ew.warming?'<span class="he-rise">▲ rising</span>':'')+'</div>'+
      ewsSpark(ew.trace,ec)+
      '<div class="he-foot">resilience-loss <b style="color:'+ec+'">'+Math.round(ew.ews*100)+'%</b>'+
      (ew.warming?' · calibrating':'')+' <span class="he-d">· research signal · watches signal <em>dynamics</em>, not levels</span></div></div>'; }
  host.innerHTML=html;
  var sync=$("dos-sync"); if(sync) sync.innerHTML=(b.icu_hour!=null?"ICU h"+b.icu_hour+" · ":"")+"<em>3 tiers in sync</em>";
}
/* SVG sparkline of the resilience (CSD) trace */
function ewsSpark(tr,col){ var w=240,h=30,n=tr.length; if(n<2) return "";
  var mx=Math.max(0.35,Math.max.apply(null,tr));
  var pts=tr.map(function(v,i){ return (i/(n-1)*w).toFixed(1)+","+(h-3-(v/mx)*(h-5)).toFixed(1); }).join(" ");
  var area="0,"+h+" "+pts+" "+w+","+h;
  return '<svg class="he-spark" viewBox="0 0 '+w+' '+h+'" preserveAspectRatio="none">'+
    '<polygon points="'+area+'" fill="'+col+'" fill-opacity=".10"/>'+
    '<polyline points="'+pts+'" fill="none" stroke="'+col+'" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>'; }

/* ============================================================
   CLINICAL — Learned Behaviour (per-patient baseline)
   ============================================================ */
function chVal(b,ch){ var e=b.edge||{}, en=b.enterprise||{};
  return ch==="hr"?e.hr:({temp_c:en.temp_c,resp_rate:en.resp_rate,spo2:en.spo2,map:en.map,wbc:en.wbc,lactate:en.lactate})[ch]; }
function renderLearned(b){
  var host=$("dos-learned"); if(!host) return;
  var mat=b.baseline_maturity||0, dev=b.deviation||0, top=b.personal_top||[];
  var lr=b.learned||{};
  var devCls=dev>=2?"alert":dev>=1?"warn":"";
  var html='<div class="lb-mat"><span class="lb-k">Baseline learned</span>'+
    '<span class="lb-track"><i style="width:'+Math.round(mat*100)+'%"></i></span>'+
    '<span class="lb-v">'+Math.round(mat*100)+'%</span></div>'+
    '<div class="lb-dev '+devCls+'"><b>'+dev.toFixed(1)+'σ</b> from own baseline</div>';
  // what the model has LEARNED about this individual (in words)
  if(lr.status==="learning"){
    html+='<div class="lb-narr"><div class="lb-nh">Learned profile</div>'+
      '<div class="lb-none">'+esc(lr.summary||"Still learning this patient’s baseline.")+'</div></div>';
  } else if((lr.learned&&lr.learned.length)||(lr.drifting&&lr.drifting.length)){
    html+='<div class="lb-narr"><div class="lb-nh">Learned profile <span class="lb-hrs">'+(lr.n_hours||0)+'h observed</span></div>';
    (lr.learned||[]).forEach(function(L){ html+='<div class="lb-learn">› '+esc(L)+'</div>'; });
    if(lr.drifting&&lr.drifting.length){ html+='<div class="lb-dh">Currently drifting from own norm</div>';
      lr.drifting.forEach(function(D){ html+='<div class="lb-drift">▲ '+esc(D)+'</div>'; }); }
    html+='</div>';
  }
  if(top.length){
    html+=top.map(function(t){ var now=chVal(b,t.ch), z=t.z||0;
      var col=z>=2?C.crit:z>=1?C.watch:C.muted, wpct=clamp(z/3*50,2,50);
      return '<div class="lb-ch"><span class="lc-k">'+(CH_LABEL[t.ch]||t.ch)+'</span>'+
        '<span class="lc-bar"><i style="width:'+wpct+'%;background:'+col+'"></i></span>'+
        '<span class="lc-v">now <b>'+fmt(now,now>=10?0:1)+'</b> · base '+fmt(t.baseline,t.baseline>=10?0:1)+' · +'+z.toFixed(1)+'σ</span></div>'; }).join("");
  } else {
    html+='<div class="lb-none">Tracking — no channel significantly above this patient’s own baseline.</div>';
  }
  host.innerHTML=html;
  var m=$("dos-learn-meta"); if(m) m.textContent="baseline "+Math.round(mat*100)+"%";
}

/* ============================================================
   CLINICAL — Anatomical Bio-Twin (real-3D point-cloud body)
   Honest organ-system load: monitored systems glow by severity;
   systems that need labs we don't have (renal/hepatic/neuro) render
   as hollow "NO DATA" nodes — never a fabricated green light.
   ============================================================ */
function bodyState(b){
  var e=b.edge||{}, en=b.enterprise||{}, c=b.components||{};
  function comp(k){ return clamp((c[k]||0)/2.5,0,1); }
  function mx(){ var m=0; for(var i=0;i<arguments.length;i++){ var v=arguments[i]; if(v===v&&v>m) m=v; } return clamp(m,0,1); }
  return {
    brain:{mon:false,sev:0},
    lungs:{mon:true,sev:mx(comp("perfusion")*0.4, en.spo2!=null?(94-en.spo2)/12:0, en.resp_rate!=null?(en.resp_rate-20)/16:0)},
    heart:{mon:true,sev:mx(comp("cardiac"), en.map!=null?(75-en.map)/30:0, e.hr!=null?(e.hr-100)/55:0)},
    liver:{mon:false,sev:0},
    kidneys:{mon:false,sev:0},
    immune:{mon:true,sev:mx(comp("inflammatory"), en.wbc!=null?(en.wbc-12)/12:0, en.wbc!=null?(4-en.wbc)/4:0)},
    core:{mon:true,sev:mx(comp("metabolic"), en.lactate!=null?(en.lactate-2)/4:0)},
    skin:{mon:true,sev:mx(comp("perfusion"), e.perfusion!=null?(1-e.perfusion):0, (en.temp_c>38.3||en.temp_c<36)?0.4:0)}
  };
}
var ORGAN_LEGEND=[
  {key:"heart",name:"Cardiovascular"},{key:"lungs",name:"Respiratory"},
  {key:"core",name:"Metabolic"},{key:"immune",name:"Immune / WBC"},
  {key:"skin",name:"Perfusion"},{key:"brain",name:"Neurologic"},
  {key:"liver",name:"Hepatic"},{key:"kidneys",name:"Renal"}
];
/* Feed the real Three.js anatomical hologram (twin3d.js) the patient's organ
   regions, in the body_state schema it already understands. Metabolic load
   (lactate) folds into the cardiovascular zone, as the SOFA mapping does. */
function regionsFromBed(b){
  var st=bodyState(b);
  function R(mon,sev){ sev=clamp(sev,0,1);
    var status=!mon?"UNMONITORED":sev>=0.75?"CRITICAL":sev>=0.45?"DYSFUNCTION":sev>=0.2?"STRESS":"NORMAL";
    return {monitored:mon,severity:Math.round(sev*1000)/1000,status:status}; }
  return {
    brain:R(false,0), lungs:R(true,st.lungs.sev),
    heart:R(true,Math.max(st.heart.sev,st.core.sev)),
    liver:R(false,0), kidneys:R(false,0),
    skin:R(true,st.skin.sev), blood:R(true,st.immune.sev)
  };
}
/* Suspected infection FOCUS — inferred from the physiologic pattern (declared as
   a hypothesis, NOT a microbiology result). Surfaces the most likely source site
   so the twin can light it up. Returns null below an actionable risk. */
function inferInfection(b){
  if((b.fused_risk||0)<0.4) return null;
  var en=b.enterprise||{}, e=b.edge||{};
  var resp=Math.max(en.spo2!=null?(94-en.spo2)/10:0, en.resp_rate!=null?(en.resp_rate-22)/10:0);
  var skin=Math.max(e.perfusion!=null?(0.55-e.perfusion):0, en.temp_c>=39?0.35:0);
  var systemic=Math.max(en.wbc!=null?Math.max((en.wbc-12)/8,(4-en.wbc)/4):0,
                        en.lactate!=null?(en.lactate-2)/4:0, (en.temp_c>38.3||en.temp_c<36)?0.35:0);
  var cands=[
    {region:"lungs", site:"Respiratory", score:resp,
     why:[en.spo2<94?"SpO₂ "+Math.round(en.spo2):"", en.resp_rate>22?"RR "+Math.round(en.resp_rate):""].filter(Boolean).join(" · ")},
    {region:"skin", site:"Soft-tissue", score:skin*0.8,
     why:en.temp_c>=39?"temp "+en.temp_c.toFixed(1)+"°C":""},
    {region:"systemic", site:"Systemic", score:systemic*0.7+0.2,
     why:["WBC "+(en.wbc||0).toFixed(0), "lactate "+(en.lactate||0).toFixed(1)].join(" · ")}
  ].sort(function(a,b){return b.score-a.score;});
  var top=cands[0];
  return {region:top.region, site:top.site, why:top.why,
          conf: top.score>=0.6?"moderate":"low"};
}
function feedTwin(b){
  if(!window.AegisTwin||!b) return;
  var e=b.edge||{}, en=b.enterprise||{};
  try{ window.AegisTwin.update({source:"simulator",
    body_state:{regions:regionsFromBed(b)},
    vitals:{heart_rate:e.hr, respiratory_rate:en.resp_rate, map:en.map, spo2:en.spo2,
            temp_c:en.temp_c, wbc:en.wbc, lactate:en.lactate, perfusion:e.perfusion},
    infection:inferInfection(b),
    risk:{signal_quality:1}}); }catch(e){}
}
function renderOrgans(b){
  var host=$("dos-organs"); if(!host) return; var st=bodyState(b);
  host.innerHTML=ORGAN_LEGEND.map(function(o){ var s=st[o.key]||{mon:false,sev:0};
    if(!s.mon) return '<div class="tl-organ unmon"><span class="to-dot"></span>'+
      '<span class="to-name">'+o.name+'</span><span class="to-sev">NO DATA</span></div>';
    var col=s.sev>=0.6?C.crit:s.sev>=0.3?C.watch:C.ok;
    return '<div class="tl-organ"><span class="to-dot" style="background:'+col+';color:'+col+'"></span>'+
      '<span class="to-name">'+o.name+'</span><span class="to-sev">'+Math.round(s.sev*100)+'%</span></div>'; }).join("");
}

/* per-bed shared-memory arena map — this patient's /dev/bio rings, WP + RP */
var FEED_ORDER=["optical","thermal","bedside","labs"];
function drawBedMemory(s,bedId){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var rings=V.mem.rings||{}, cursors=V.mem.cursors||[], prefix="/dev/bio/"+bedId+"/", lanes=[];
  for(var p in rings){ if(p.indexOf(prefix)===0){
    lanes.push({kind:p.slice(prefix.length).replace("_feed",""), ring:rings[p]}); } }
  lanes.sort(function(a,b){return FEED_ORDER.indexOf(a.kind)-FEED_ORDER.indexOf(b.kind);});
  if(!lanes.length) return;
  var x0=64, x1=w-58, W=x1-x0, padT=6, padB=6, rowH=(h-padT-padB)/lanes.length;
  ctx.textBaseline="middle";
  lanes.forEach(function(L,i){
    var y=padT+rowH*i+rowH/2, cap=L.ring.capacity||1, th=Math.min(11,rowH*0.5), ty=y-th/2;
    ctx.textAlign="left"; ctx.font=FS(10,500); ctx.fillStyle=C.muted; ctx.fillText(L.kind,8,y);
    ctx.fillStyle="rgba(255,255,255,.045)"; roundRect(ctx,x0,ty,W,th,th/2); ctx.fill();
    // faint slot ticks
    ctx.strokeStyle=C.grid; ctx.lineWidth=1;
    for(var t=0;t<=8;t++){ var tx=x0+W*t/8; ctx.beginPath(); ctx.moveTo(tx,ty); ctx.lineTo(tx,ty+th); ctx.stroke(); }
    // read cursors (amber)
    cursors.forEach(function(cur){ if(cur.ring===L.ring.name){
      var rx=x0+(cur.read_ptr/cap)*W; ctx.fillStyle=withAlpha(C.watch,.9);
      roundRect(ctx,rx-1.5,ty,3,th,1.5); ctx.fill(); } });
    // write pointer (bright sweep)
    var wx=x0+(L.ring.write_ptr/cap)*W;
    ctx.fillStyle=C.cyan; ctx.shadowColor=C.cyan; ctx.shadowBlur=9;
    roundRect(ctx,wx-2,ty-2,4,th+4,2); ctx.fill(); ctx.shadowBlur=0;
    // write index
    ctx.textAlign="right"; ctx.font=FM(9.5,500); ctx.fillStyle=C.dim;
    ctx.fillText((L.ring.write_index||0).toLocaleString(), w-8, y);
  });
}

/* ============================================================
   PANEL — Memory Fabric
   ============================================================ */
function drawMemory(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var pad=14, headH=30;
  ctx.textBaseline="middle"; ctx.textAlign="left";
  // left: BASE + pointer + stable badge (placed after the address, width-aware)
  ctx.font=FS(9.5,600); ctx.fillStyle=C.dim; ctx.fillText("BASE", pad,13);
  ctx.font=FM(10.5,500); ctx.fillStyle=C.muted; var bv=V.mem.base||"0x0"; ctx.fillText(bv, pad+34,13);
  var dotx=pad+40+ctx.measureText(bv).width;
  ctx.fillStyle=V.mem.stable?C.ok:C.crit; ctx.beginPath(); ctx.arc(dotx,13,3,0,7); ctx.fill();
  ctx.font=FS(9.5,500); ctx.fillStyle=V.mem.stable?C.ok:C.crit;
  ctx.fillText(V.mem.stable?"stable":"moved", dotx+7,13);
  // right: util% · published
  var pub=V.mem.published>=1000?(V.mem.published/1000).toFixed(1)+"k":(V.mem.published||0)+"";
  ctx.textAlign="right"; ctx.font=FM(10.5,500); ctx.fillStyle=C.muted;
  ctx.fillText(fmt(V.mem.util*100,2)+"% · "+pub+" pub", w-pad,13);
  ctx.textAlign="left";

  var sorted=V.mem._sorted, byName=V.mem._byName||{}; if(!sorted||!sorted.length) return;
  var used=Math.max(V.mem.used,1);
  var gx=pad, gy=headH, gw=w-pad*2, gh=h-headH-pad-6;
  var cols=Math.max(24,Math.floor(gw/10)), rows=Math.max(10,Math.floor(gh/10));
  var cw=gw/cols, ch=gh/rows, cells=cols*rows, factor=used/cells;
  function regCol(name){ if(/\.t$/.test(name)) return withAlpha(C.dim,.10);
    var m=name&&name.match(/node_(\d+)/), ni=m?(+m[1]):0;
    return withAlpha([C.cyan,C.indigo,C.violet][ni%3],.15); }
  var ri=0, empty="rgba(120,140,180,.04)";
  for(var ci=0;ci<cells;ci++){ var idx=(ci*factor)|0, c=ci%cols, rr=(ci/cols)|0;
    while(ri<sorted.length && idx>=sorted[ri].offset+sorted[ri].length) ri++;
    var reg=(ri<sorted.length && idx>=sorted[ri].offset)?sorted[ri]:null;
    ctx.fillStyle=reg?regCol(reg.name):empty;
    ctx.fillRect(gx+c*cw+0.6, gy+rr*ch+0.6, cw-1.2, ch-1.2); }
  function cell(a){ var i=clamp((a/factor)|0,0,cells-1); return {c:i%cols,r:(i/cols)|0}; }
  var rings=V.mem.rings||{};
  ctx.shadowColor=C.cyan; ctx.shadowBlur=8; ctx.fillStyle=C.cyan;
  for(var path in rings){ var nm=path.replace("/dev/bio/","").replace(/\//g,".")+".v", reg2=byName[nm]; if(!reg2) continue;
    var nf=(reg2.shape&&reg2.shape[1])||1, p=cell(reg2.offset+rings[path].write_ptr*nf);
    ctx.fillRect(gx+p.c*cw+0.6, gy+p.r*ch+0.6, cw-1.2, ch-1.2); }
  ctx.shadowBlur=0; ctx.fillStyle=withAlpha(C.watch,.9);
  (V.mem.cursors||[]).forEach(function(cur){ var reg3=byName[cur.ring+".v"]; if(!reg3) return;
    var nf=(reg3.shape&&reg3.shape[1])||1, p=cell(reg3.offset+cur.read_ptr*nf);
    ctx.fillRect(gx+p.c*cw+cw*0.3, gy+p.r*ch+ch*0.3, cw*0.42, ch*0.42); });
  // legend
  ctx.font=FS(9.5,500); ctx.textBaseline="middle";
  ctx.fillStyle=C.cyan; ctx.fillRect(gx,h-9,7,7); ctx.fillStyle=C.muted; ctx.fillText("write pointer",gx+11,h-5);
  var lx2=gx+95; ctx.fillStyle=C.watch; ctx.fillRect(lx2,h-9,7,7); ctx.fillStyle=C.muted; ctx.fillText("read cursor",lx2+11,h-5);
}

/* ============================================================
   PANEL — Execution Lanes
   ============================================================ */
function drawLanes(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var ids=nodeOrder.filter(function(id){return V.nodes[id];}), budget=V.budget||1;
  var L=12, R=w-12, BW=R-L;
  ctx.textBaseline="middle"; ctx.textAlign="left";
  // header — mode
  ctx.font=FS(10.5,600); ctx.fillStyle=V.mode==="PREEMPT"?C.crit:C.muted;
  ctx.fillText(V.mode==="PREEMPT"?("⚡ Hard preempt → "+(V.owner||"")):"Normal · compute ∝ clinical risk", L, 14);

  var bh=Math.min(24, Math.max(16, h*0.15));
  var y1=Math.round(h*0.30), y2=Math.round(h*0.66), small=h<128;

  // ---- bar 1: compute allocation per patient ----
  ctx.font=FS(8.5,600); ctx.fillStyle=C.dim; ctx.fillText("BY PATIENT", L, y1-9);
  ctx.fillStyle="rgba(255,255,255,.03)"; roundRect(ctx,L,y1,BW,bh,bh/2); ctx.fill();
  var sorted=ids.slice().sort(function(a,b){return V.nodes[b].lanes-V.nodes[a].lanes;});
  var x=L;
  sorted.forEach(function(id){ var nd=V.nodes[id], seg=nd.lanes/budget*BW; if(seg<0.6) return;
    var owner=(V.owner===id), col=owner||nd.tier==="CRITICAL"?C.crit:nd.tier==="WATCH"?C.watch:C.ok, a=owner?.97:.55;
    var g=ctx.createLinearGradient(x,0,x+seg,0); g.addColorStop(0,withAlpha(col,a)); g.addColorStop(1,withAlpha(col,a*.66));
    ctx.fillStyle=g; roundRect(ctx,x,y1,Math.max(seg-1.6,1.6),bh,Math.min(bh/2,5)); ctx.fill();
    if(seg>58){ ctx.fillStyle=owner?"#fff":withAlpha(C.ink,.88); ctx.font=FM(10.5,600); ctx.textAlign="left";
      ctx.fillText((nd.name||id).split(",")[0]+" · "+Math.round(nd.lanes), x+9, y1+bh/2); }
    x+=seg; });

  // ---- bar 2: compute allocation per lane class ----
  ctx.textBaseline="middle"; ctx.textAlign="left"; ctx.font=FS(8.5,600); ctx.fillStyle=C.dim;
  ctx.fillText("BY LANE CLASS", L, y2-9);
  ctx.fillStyle="rgba(255,255,255,.03)"; roundRect(ctx,L,y2,BW,bh,bh/2); ctx.fill();
  var tot={safety:0,ml_tracking:0,render:0,logging:0};
  ids.forEach(function(id){ var lv=V.lanes[id]; if(lv) LANE_ORDER.forEach(function(c){ tot[c]+=lv[c]||0; }); });
  var x2=L;
  LANE_ORDER.forEach(function(c){ var seg=tot[c]/budget*BW; if(seg<0.6) return;
    var col=LANE_COL[c], g=ctx.createLinearGradient(x2,0,x2+seg,0);
    g.addColorStop(0,withAlpha(col,.92)); g.addColorStop(1,withAlpha(col,.6));
    ctx.fillStyle=g; roundRect(ctx,x2,y2,Math.max(seg-1.6,1.6),bh,Math.min(bh/2,5)); ctx.fill();
    if(seg>64){ ctx.fillStyle=withAlpha(C.ink,.88); ctx.font=FS(9.5,600); ctx.textAlign="left";
      ctx.fillText(LANE_LABEL[c]+" "+Math.round(tot[c]), x2+9, y2+bh/2); }
    x2+=seg; });
  // class legend
  if(!small){ var lx=L, ly=y2+bh+13;
    LANE_ORDER.forEach(function(c){ ctx.fillStyle=LANE_COL[c]; ctx.fillRect(lx,ly-4,7,7);
      ctx.fillStyle=C.dim; ctx.font=FS(9,500); ctx.textAlign="left"; ctx.fillText(LANE_LABEL[c],lx+11,ly);
      lx+=11+ctx.measureText(LANE_LABEL[c]).width+14; }); }
}

/* ============================================================
   PANEL — Latency Envelope
   ============================================================ */
function logY(v,h,pt,pb){ var lv=Math.log10(Math.max(v,1)); return (h-pb)-(lv/5)*(h-pt-pb); }
function drawLatency(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var n=latFill, i;
  function idx(k){ return (latHead-latFill+k+latBuf.length)%latBuf.length; }

  // ---- top readout row: p50 (big), p95, deadline headroom ----
  ctx.textBaseline="alphabetic"; ctx.textAlign="left";
  ctx.fillStyle=C.dim; ctx.font=FS(9.5,600); ctx.fillText("p50", 6, 17);
  var vt=fmt(V.lat.p50,2);
  ctx.fillStyle=C.ink; ctx.font=FM(20,600); ctx.fillText(vt, 32, 20);
  var px=32+ctx.measureText(vt).width;
  ctx.fillStyle=C.dim; ctx.font=FM(10.5,500); ctx.fillText("µs", px+5, 20);
  ctx.fillStyle=C.muted; ctx.font=FM(11,500); ctx.fillText("p95 "+fmt(V.lat.p95,1)+"µs", px+36, 19);
  var margin=DEADLINE_US/Math.max(V.lat.p50,.01);
  ctx.fillStyle=C.ok; ctx.font=FS(10.5,600); ctx.textAlign="right";
  ctx.fillText("▼ "+(margin>=1000?(margin/1000).toFixed(1)+"k":Math.round(margin))+"× under 50,000 µs deadline", w-6, 19);

  // ---- auto-scaled oscilloscope (shows the real µs jitter, in-frame) ----
  var topH=30, padL=42, padR=10, padB=8;
  var x0=padL, x1=w-padR, span=x1-x0, top=topH+8, bot=h-padB, ph=bot-top;
  var tmax=10; for(i=0;i<n;i++){ var v=latBuf95[idx(i)]; if(v>tmax) tmax=v; }
  tmax=Math.max(8, tmax)*1.3;
  latYMax += (tmax-latYMax)*0.04;               // smooth rescale
  var ymax=latYMax;
  function yAt(v){ return bot-Math.min(v,ymax)/ymax*ph; }
  ctx.textBaseline="middle"; ctx.font=FM(9,500);
  [0,0.5,1].forEach(function(f){ var y=bot-f*ph;
    ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x0,y+.5); ctx.lineTo(x1,y+.5); ctx.stroke();
    ctx.fillStyle=C.dim; ctx.textAlign="right"; var lv=ymax*f; ctx.fillText((lv<10?lv.toFixed(1):Math.round(lv))+"µs", x0-5, y); });
  if(n>1){ var step=span/(latBuf.length-1);
    // p95 fill + faint line
    ctx.beginPath(); for(i=0;i<n;i++){ var xx=x0+i*step, yy=yAt(latBuf95[idx(i)]); i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy); }
    ctx.lineTo(x0+(n-1)*step,bot); ctx.lineTo(x0,bot); ctx.closePath();
    var fg=ctx.createLinearGradient(0,top,0,bot); fg.addColorStop(0,withAlpha(C.cyan,.24)); fg.addColorStop(1,withAlpha(C.cyan,.02));
    ctx.fillStyle=fg; ctx.fill();
    ctx.strokeStyle=withAlpha(C.cyan,.28); ctx.lineWidth=1; ctx.beginPath();
    for(i=0;i<n;i++){ var xa=x0+i*step, ya=yAt(latBuf95[idx(i)]); i?ctx.lineTo(xa,ya):ctx.moveTo(xa,ya); } ctx.stroke();
    // p50 glowing line
    ctx.lineJoin="round"; ctx.lineWidth=2; ctx.shadowColor=withAlpha(C.cyan,.7); ctx.shadowBlur=9; ctx.strokeStyle=C.cyan;
    ctx.beginPath(); for(i=0;i<n;i++){ var xb=x0+i*step, yb=yAt(latBuf[idx(i)]); i?ctx.lineTo(xb,yb):ctx.moveTo(xb,yb); } ctx.stroke(); ctx.shadowBlur=0;
    var lx=x0+(n-1)*step, ly=yAt(latBuf[idx(n-1)]);
    ctx.fillStyle=C.cyan; ctx.beginPath(); ctx.arc(lx,ly,2.8,0,7); ctx.fill();
    ctx.strokeStyle=withAlpha(C.cyan,.4); ctx.lineWidth=1; ctx.beginPath(); ctx.arc(lx,ly,5.5,0,7); ctx.stroke();
  }
}

/* real vitals/labs trajectory sparklines (from /aegis/cohort/<bed>/chart) */
var TREND_CH=[["hr","HR",0,100,9e9],["map","MAP",0,-9e9,65],["temp_c","Temp",1,38,36],
  ["resp_rate","Resp",0,22,9e9],["lactate","Lac",1,2,9e9],["wbc","WBC",1,12,4]];
function _deranged(key,v,hi,lo){ if(key==="temp_c"||key==="wbc") return v>hi||v<lo;
  if(key==="map") return v<lo; return v>hi; }
function drawTrends(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var c=dosChart;
  if(!c||!c.hours||c.hours.length<2){ ctx.fillStyle=C.dim; ctx.font=FS(10,500);
    ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText("loading real trajectory …",w/2,h/2); return; }
  var hours=c.hours,n=hours.length,h0=hours[0],h1=hours[n-1],span=Math.max(1,h1-h0);
  var padL=44,padR=58,padT=3,padB=3,plotW=w-padL-padR,rows=TREND_CH.length,rowH=(h-padT-padB)/rows;
  function xAt(hr){ return padL+(hr-h0)/span*plotW; }
  if(c.onset_hour!=null && c.onset_hour>=h0 && c.onset_hour<=h1){
    var ox=xAt(c.onset_hour); ctx.strokeStyle=withAlpha(C.crit,.55); ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(ox,padT); ctx.lineTo(ox,h-padB); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle=C.crit; ctx.font=FS(8,600); ctx.textAlign="center"; ctx.textBaseline="top";
    ctx.fillText("onset",ox,padT+1); }
  TREND_CH.forEach(function(spec,ri){
    var key=spec[0],data=c.series[key]; if(!data) return;
    var y0=padT+rowH*ri,yMid=y0+rowH/2;
    var vals=[]; data.forEach(function(v){ if(v!=null&&isFinite(v)) vals.push(v); });
    if(!vals.length) return;
    var mn=Math.min.apply(null,vals),mx=Math.max.apply(null,vals); if(mx-mn<1e-6) mx=mn+1;
    var pad=rowH*0.26;
    function yAt(v){ return y0+rowH-pad-(v-mn)/(mx-mn)*(rowH-2*pad); }
    ctx.textAlign="left"; ctx.textBaseline="middle"; ctx.font=FS(9.5,600); ctx.fillStyle=C.muted; ctx.fillText(spec[1],6,yMid);
    var last=data[data.length-1], der=_deranged(key,last,spec[3],spec[4]);
    ctx.strokeStyle=der?C.crit:C.cyan; ctx.lineWidth=1.3; ctx.lineJoin="round"; ctx.beginPath();
    var started=false;
    data.forEach(function(v,i){ if(v==null||!isFinite(v))return; var x=xAt(hours[i]),y=yAt(v);
      started?ctx.lineTo(x,y):ctx.moveTo(x,y); started=true; });
    ctx.stroke();
    if(started){ var lx=xAt(hours[n-1]),ly=yAt(last); ctx.fillStyle=der?C.crit:C.cyan;
      ctx.beginPath(); ctx.arc(lx,ly,2,0,7); ctx.fill(); }
    ctx.textAlign="right"; ctx.font=FM(10,500); ctx.fillStyle=der?C.crit:C.ink;
    ctx.fillText(fmt(last,spec[2]),w-8,yMid);
    if(ri<rows-1){ ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.beginPath();
      ctx.moveTo(padL,y0+rowH-0.5); ctx.lineTo(w-padR,y0+rowH-0.5); ctx.stroke(); }
  });
}

/* fused sepsis-risk timeline — history + FORECAST (predicted future risk + band) */
function drawRiskTimeline(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var c=dosChart;
  if(!c||!c.risk||c.risk.length<2){ ctx.fillStyle=C.dim; ctx.font=FS(11,500); ctx.textAlign="center";
    ctx.textBaseline="middle"; ctx.fillText("loading risk timeline …",w/2,h/2); return; }
  var hours=c.hours, risk=c.risk, n=hours.length, h0=hours[0], h1=hours[n-1], i;
  var fc=c.forecast, H=(fc&&fc.risk_mean)?fc.horizon:0;
  var xMax=h1+H, span=Math.max(1,xMax-h0);
  var padL=32,padR=14,padT=24,padB=28,pw=w-padL-padR,labelH=11,plotBot=h-padB,plotTop=padT,ph=plotBot-plotTop-labelH;
  function xAt(hr){ return padL+(hr-h0)/span*pw; } function yAt(v){ return plotTop+ph-clamp(v,0,1)*ph; }
  ctx.textBaseline="middle"; ctx.font=FM(9,500);
  // vertical time grid (every ~ nice hour step) — gives the plot real structure
  var step=span<=12?2:span<=30?6:span<=72?12:24;
  for(var hg=Math.ceil(h0/step)*step; hg<=xMax; hg+=step){ var gx=xAt(hg);
    ctx.strokeStyle=withAlpha(C.grid,.5); ctx.lineWidth=1; ctx.beginPath();
    ctx.moveTo(gx,plotTop); ctx.lineTo(gx,plotBot); ctx.stroke(); }
  // horizontal quarter gridlines (major at 0/.5/1, minor at .25/.75)
  [[0,1],[0.25,0],[0.5,1],[0.75,0],[1,1]].forEach(function(g){ var v=g[0],major=g[1],y=yAt(v);
    ctx.strokeStyle=withAlpha(C.grid,major?1:.45); ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(w-padR,y); ctx.stroke();
    if(major){ ctx.fillStyle=C.dim; ctx.textAlign="right"; ctx.fillText(v.toFixed(1),padL-5,y); } });
  // forecast region shading (subtle) + "now" divider
  var nx=xAt(h1);
  if(H){ ctx.fillStyle="rgba(125,140,255,.04)"; ctx.fillRect(nx,plotTop,xAt(xMax)-nx,ph);
    ctx.strokeStyle=withAlpha(C.indigo,.4); ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(nx,plotTop); ctx.lineTo(nx,plotBot); ctx.stroke();
    ctx.fillStyle=C.indigo; ctx.font=FS(8.5,700); ctx.textAlign="left"; ctx.textBaseline="top"; ctx.fillText("now",nx+3,plotTop+1);
    ctx.textAlign="right"; ctx.fillStyle=C.dim; ctx.fillText("forecast +"+H+"h",xAt(xMax)-2,plotTop+1); }
  var al=c.alarm||0.7, ay=yAt(al);
  ctx.strokeStyle=withAlpha(C.crit,.5); ctx.lineWidth=1; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(padL,ay); ctx.lineTo(w-padR,ay); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.crit; ctx.font=FS(8.5,600); ctx.textAlign="left"; ctx.textBaseline="bottom"; ctx.fillText("alarm "+al,padL+4,ay-2);
  function onsetMark(hr,lbl,solid){ var ox=xAt(hr); ctx.strokeStyle=withAlpha(C.crit,solid?.7:.5);
    ctx.lineWidth=1.3; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(ox,plotTop); ctx.lineTo(ox,plotBot); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle=C.crit; ctx.font=FS(8.5,700); ctx.textAlign="center"; ctx.textBaseline="top"; ctx.fillText(lbl,ox,plotTop+11); }
  if(c.onset_hour!=null && c.onset_hour>=h0 && c.onset_hour<=xMax) onsetMark(c.onset_hour,"labelled onset",true);
  // ---- history risk (solid) ----
  var last=risk[n-1], col=last>=al?C.crit:last>=0.4?C.watch:C.cyan;
  ctx.beginPath(); for(i=0;i<n;i++){ var x=xAt(hours[i]),y=yAt(risk[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
  ctx.lineTo(xAt(h1),plotBot-labelH); ctx.lineTo(xAt(h0),plotBot-labelH); ctx.closePath();
  var fg=ctx.createLinearGradient(0,plotTop,0,plotBot);
  fg.addColorStop(0,withAlpha(col,.34)); fg.addColorStop(.55,withAlpha(col,.12)); fg.addColorStop(1,withAlpha(col,.02));
  ctx.fillStyle=fg; ctx.fill();
  ctx.strokeStyle=col; ctx.lineWidth=2; ctx.lineJoin="round"; ctx.shadowColor=withAlpha(col,.6); ctx.shadowBlur=8;
  ctx.beginPath(); for(i=0;i<n;i++){ var x2=xAt(hours[i]),y2=yAt(risk[i]); i?ctx.lineTo(x2,y2):ctx.moveTo(x2,y2); } ctx.stroke(); ctx.shadowBlur=0;
  // ---- forecast (band + dashed mean) ----
  if(H){ var fcol=C.indigo;
    // band hi/lo
    ctx.beginPath(); ctx.moveTo(xAt(h1),yAt(last));
    for(i=0;i<H;i++) ctx.lineTo(xAt(h1+i+1),yAt(fc.risk_hi[i]));
    for(i=H-1;i>=0;i--) ctx.lineTo(xAt(h1+i+1),yAt(fc.risk_lo[i]));
    ctx.lineTo(xAt(h1),yAt(last)); ctx.closePath(); ctx.fillStyle=withAlpha(fcol,.14); ctx.fill();
    // mean dashed
    ctx.strokeStyle=fcol; ctx.lineWidth=1.8; ctx.setLineDash([5,3]); ctx.beginPath(); ctx.moveTo(xAt(h1),yAt(last));
    for(i=0;i<H;i++) ctx.lineTo(xAt(h1+i+1),yAt(fc.risk_mean[i])); ctx.stroke(); ctx.setLineDash([]);
    // projected onset marker
    if(fc.projected_onset_hour!=null && fc.projected_onset_hour<=xMax){
      var px=xAt(fc.projected_onset_hour); ctx.fillStyle=fcol; ctx.beginPath(); ctx.arc(px,yAt(al),3.5,0,7); ctx.fill();
      ctx.strokeStyle=withAlpha(fcol,.5); ctx.beginPath(); ctx.arc(px,yAt(al),6.5,0,7); ctx.stroke(); }
  }
  // "now" marker — a glowing dot with a soft outer ring so the live edge reads clearly
  var nX=xAt(h1), nY=yAt(last);
  ctx.strokeStyle=withAlpha(col,.5); ctx.lineWidth=1.4; ctx.beginPath(); ctx.arc(nX,nY,7,0,7); ctx.stroke();
  ctx.shadowColor=col; ctx.shadowBlur=12; ctx.fillStyle=col; ctx.beginPath(); ctx.arc(nX,nY,3.6,0,7); ctx.fill(); ctx.shadowBlur=0;
  ctx.fillStyle="#0a0e18"; ctx.beginPath(); ctx.arc(nX,nY,1.5,0,7); ctx.fill();
  // ground-truth label band
  var by=plotBot-labelH+2, bw=pw/Math.max(span,1);
  for(i=0;i<n;i++){ if(c.labels[i]){ ctx.fillStyle=withAlpha(C.crit,.55); ctx.fillRect(xAt(hours[i])-bw/2,by,bw+1,labelH-3); } }
  ctx.fillStyle=C.dim; ctx.font=FS(8,500); ctx.textAlign="left"; ctx.textBaseline="middle"; ctx.fillText("ground-truth label",padL+2,by+(labelH-2)/2);
  // forecast summary banner (top)
  if(fc){ var tc=fc.trajectory==="deteriorating"?C.crit:fc.trajectory==="improving"?C.ok:C.muted;
    ctx.textAlign="left"; ctx.textBaseline="middle"; ctx.font=FS(10.5,700); ctx.fillStyle=tc;
    var msg="▸ "+fc.trajectory.toUpperCase();
    if(fc.projected_onset_in!=null) msg+=" · projected critical in ~"+fc.projected_onset_in+"h";
    ctx.fillText(msg, padL, 11);
    ctx.textAlign="right"; ctx.fillStyle=C.dim; ctx.font=FM(9.5,500);
    ctx.fillText("forecast confidence "+Math.round(fc.confidence*100)+"%", w-padR, 11); }
  ctx.fillStyle=C.dim; ctx.font=FM(9,500); ctx.textAlign="center"; ctx.textBaseline="bottom";
  ctx.fillText("h"+h0,xAt(h0),h-3); ctx.fillText("h"+h1,xAt(h1),h-3); if(H) ctx.fillText("h"+xMax,xAt(xMax),h-3);
}

/* ============================================================
   MODULE C — Neurosymbolic Protocol Terminal (streamed)
   ============================================================ */
var termKey=null, termTimer=null;
function nowClock(){ return new Date().toLocaleTimeString("en-GB"); }
function buildTermLines(b){
  var L=[], p=b.protocol;
  L.push(["sys","SEPSIS VECTOR BREACH — "+bedTag(b.bed_id)+" · "+b.name+" · fused risk "+fmt(b.fused_risk,2)]);
  L.push(["sys","Layer-4 neurosymbolic plane engaged · "+b.primary_system+" axis"]);
  L.push([b.alert.certified?"ok":"bad",(b.alert.certified?"[✓] ":"[✗] ")+"ALERT "+b.severity_name+
    " — Z3 "+b.alert.z3.toUpperCase()+" · coherent with calibrated risk"]);
  L.push(["sys","Compiling Hour-1 Sepsis Bundle …"]);
  b.checklist.forEach(function(it){ L.push(["item-"+it.status,"["+it.n+"] "+it.title+" — "+it.detail]); });
  L.push(["sys","Verifying orders against patient-specific invariants …"]);
  L.push([p.certified?"ok":"bad",(p.certified?"[✓] ":"[✗] ")+"PROTOCOL ("+p.antibiotic+
    " · fluid "+p.fluid_rate+" · vaso "+p.vasopressor_dose+") — Z3 "+p.z3.toUpperCase()+
    (p.certified?"":" — "+(p.violations[0]||"violation"))]);
  L.push(["note","Clinical note (certified):"]);
  L.push(["notetext",'"'+b.note.text+'"']);
  L.push([b.note.certified?"ok":"bad",(b.note.certified?"[✓] ":"[✗] ")+
    "NOTE — chain-of-trust to certified protocol"]);
  L.push([b.all_verified?"stamp":"stamp bad", b.all_verified
    ?"══ FORMALLY VERIFIED · SAFE TO DISPATCH ══":"══ QUARANTINED · MANUAL REVIEW REQUIRED ══"]);
  return L;
}
function streamLines(lines){
  var body=$("term-body"); body.innerHTML=""; if(termTimer){ clearTimeout(termTimer); termTimer=null; }
  var i=0;
  (function step(){
    if(i>0 && body.children[i-1]) body.children[i-1].classList.remove("tl-cursor");
    if(i>=lines.length){ termTimer=null; $("term-stamp").textContent="verified "+nowClock(); return; }
    var L=lines[i], ln=document.createElement("div");
    ln.className="tl tl-"+L[0]+" tl-cursor"; ln.textContent=L[1];
    body.appendChild(ln); body.scrollTop=body.scrollHeight; i++;
    termTimer=setTimeout(step, L[0]==="notetext"?240:92);
  })();
}
var termAt=0;
var TERM_DWELL_MS=15000;      // minimum time a streamed block stays put
function renderTerminal(bundle){
  var term=$("terminal");
  if(!bundle){ if(termKey!==null){ term.classList.remove("active");
      $("term-sub").textContent="Layer-4 formal verification · standing by"; $("term-stamp").textContent="";
      $("term-body").innerHTML='<div class="term-idle">// no active sepsis vectors — all beds within range</div>';
      termKey=null; if(termTimer){ clearTimeout(termTimer); termTimer=null; } } return; }
  term.classList.add("active");
  $("term-sub").textContent="active focus · "+bundle.name;

  /* Two reasons this used to wipe and re-type constantly, which is what made the
     text look like it was jumping around the panel:

       1. it keyed on bed_id alone, so re-focusing the SAME bed with an unchanged
          bundle restarted the whole typewriter, and
       2. the focus bed is whichever is riskiest, and two beds whose risks cross
          swap it back and forth every poll — each swap wiping a block mid-stream.

     Now it keys on the CONTENT, and a block that has just been laid down is left
     alone for a dwell period unless what it says has actually changed. A terminal
     that re-types the same certified bundle every four seconds is not conveying
     anything; it is just moving. */
  var lines=buildTermLines(bundle);
  var key=bundle.bed_id+"|"+lines.map(function(L){return L[1];}).join("\u0001");
  if(key===termKey) return;
  var now=Date.now();
  if(termKey!==null && (now-termAt)<TERM_DWELL_MS && termTimer===null) return;
  termKey=key; termAt=now; streamLines(lines);
}

/* ============================================================
   DOM — top bar + control rail
   ============================================================ */
function renderSpine(){
  $("s-cost").textContent=fmt(V.cost,2);
  var mv=$("s-mode"); mv.textContent=V.mode==="PREEMPT"?"Preempt":"Normal"; mv.classList.toggle("is-preempt",V.mode==="PREEMPT");
  $("s-int").textContent=events.length;
  $("s-cage").textContent=V.safety.certified.toLocaleString()+" · "+Math.round((V.safety.pass_rate||1)*100)+"%";
  $("s-tick").textContent=target?target.scheduler.tick:0;
  $("s-clock").textContent=new Date().toLocaleTimeString("en-GB");
  if(events.length){ $("tk-body").innerHTML=events.slice(-3).reverse().map(function(ev){
    return '<span class="hot">⚡ t'+ev.tick+' · '+ev.owner+' ('+ev.reason+' p='+fmt(ev.p,2)+')</span>'; }).join("&nbsp;&nbsp;·&nbsp;&nbsp;"); }
}
function chaos(node, action){
  if(!node) return;
  fetch(window.AEGIS.http("/aegis/cluster/chaos"),{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({node:node,action:action,intensity:1.0})}).catch(function(){});
}
// The write-side trap gate: console mutations go through the audited, Z3-gated
// syscall interface. A denied write returns an errno + violated invariant — the
// console renders the prover's verdict instead of silently swallowing it.
function syscall(name, bed, args, cb){
  cmdPrint('<span class="cl-d">trap → sys_'+esc(name)+' · '+bedTag(bed)+' · awaiting Z3 certificate…</span>','dim');
  fetch(window.AEGIS.http("/aegis/syscall"),{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({syscall:name,bed:bed,args:args||{}})})
    .then(function(r){return r.json();})
    .then(function(d){ renderSyscall(d); if(cb) cb(d); })
    .catch(function(){ cmdPrint("syscall failed — backend unreachable","err"); });
}
function renderSyscall(d){
  var ok=d.allowed, badge=ok?'<span class="cl-ok">✓ '+esc(d.errno_name)+'</span>'
                            :'<span class="cl-crit">✗ '+esc(d.errno_name)+'</span>';
  cmdPrint('<b>sys_'+esc(d.syscall)+'</b> <span class="cl-d">#'+d.sid+' · '+esc(d.egress||"—")+
    ' egress · Z3 '+esc(d.z3_status)+'</span> '+badge,"sys");
  cmdPrint(esc(d.message), ok?"ok":"err");
  if(d.violations&&d.violations.length) d.violations.forEach(function(v){ cmdPrint('· '+esc(v),"dim"); });
  if(d.result&&d.result.proof) cmdPrint('<span class="cl-d">⊢ '+esc(d.result.proof)+'</span>',"mono");
}
function splitBedArgs(a){ var t=String(a||"").trim().split(/\s+/).filter(Boolean); return {bedq:t[0]||"", rest:t.slice(1)}; }
var XMR_NAMES={camera:"Camera · rPPG", wearable:"Wearable · PPG", ehr:"EHR · Labs"};
function renderResonance(b){
  var el=$("dos-xmr"); if(!el) return;
  var r=b&&b.resonance;
  if(!r||!r.per_modality){ el.innerHTML='<div class="xmr-empty">calibrating cross-modal axis…</div>';
    if($("dos-xmr-meta")) $("dos-xmr-meta").textContent="agreement gate"; return; }
  var pm=r.per_modality;
  var rows=Object.keys(pm).map(function(k){
    var z=pm[k], pct=Math.max(0,Math.min(100,(z+3)/6*100)), rej=(r.uncorroborated_pathway===k);
    return '<div class="xmr-row'+(rej?" rej":"")+'"><span class="xmr-nm">'+(XMR_NAMES[k]||k)+'</span>'+
      '<div class="xmr-track"><div class="xmr-zero"></div>'+
      '<div class="xmr-fill" style="left:'+Math.min(50,pct)+'%;width:'+Math.abs(pct-50)+'%"></div></div>'+
      '<span class="xmr-z">'+(z>=0?"+":"")+z.toFixed(1)+'σ</span></div>';
  }).join("");
  var status = r.isolated
    ? '<span class="xmr-warn">⚠ '+esc((XMR_NAMES[r.uncorroborated_pathway]||r.uncorroborated_pathway||""))+' elevated alone — held pending confirmation</span>'
    : (r.corroborated?'<span class="xmr-ok">✓ corroborated across pathways</span>'
                     :'<span class="xmr-dim">within range · no pathway elevated</span>');
  var nv=Math.round((r.naive_risk||0)*100), gv=Math.round((r.gated_risk||0)*100);
  el.innerHTML=rows+'<div class="xmr-status">'+status+'</div>'+
    '<div class="xmr-risk">ungated '+(r.isolated?'<s>'+nv+'%</s>':nv+'%')+' → gated <b>'+gv+'%</b>'+
    ' · resonance '+Math.round((r.resonance||0)*100)+'%</div>';
  if($("dos-xmr-meta")) $("dos-xmr-meta").textContent = r.isolated?"artifact rejected":("agreement "+Math.round((r.resonance||0)*100)+"%");
}
function renderDevicePlane(k){
  if(!k||!$("dp-bays")) return;
  var v=k.verifier||{}, devs=k.devices||[];
  var deg=devs.filter(function(d){return d.state!=="RUNNING";}).length;
  $("devplane").classList.toggle("degraded", deg>0);
  $("dp-vrf").innerHTML='verifier <b>'+(v.accepted||0)+'</b> admitted · <b>'+(v.rejected||0)+'</b> rejected';
  $("dp-sub").textContent = deg ? (deg+" device(s) degraded — supervisor isolating the fault")
                                : "all drivers statically verified · halt-safe · in-window";
  $("dp-bays").innerHTML = devs.map(function(d){
    var cls=d.state==="RUNNING"?"":d.state==="DEGRADED"?"warn":"crit";
    return '<div class="dp-bay '+cls+'">'+
      '<div class="dp-bay-hd"><span class="dp-state">'+esc(d.state)+'</span>'+
        '<span class="dp-blocks">'+d.blocks+' blk</span></div>'+
      '<div class="dp-bay-nm">'+esc(d.label)+'</div>'+
      '<div class="dp-bay-ft"><span class="dp-drv">'+esc(d.active)+'</span>'+
      (d.state!=="RUNNING" ? ('<span class="dp-fo">⚠ '+(d.failovers||1)+' failover</span>')
                           : '<span class="dp-vok">✓ verified</span>')+'</div></div>';
  }).join("");
}
function renderServicePlane(s, pkgs){
  if(!s||!$("sp-svcs")) return;
  var deg=(s.services||[]).filter(function(x){return x.state!=="RUNNING";}).length;
  $("svcplane").classList.toggle("degraded", deg>0);
  $("sp-meta").textContent=(s.running||0)+'/'+(s.n||0)+' running'+(s.bed?' · '+bedTag(s.bed):'');
  $("sp-svcs").innerHTML=(s.services||[]).map(function(x){
    var cls=x.state==="RUNNING"?(x.healthy?"":"warn"):"crit";
    return '<div class="sp-svc '+cls+'"><div class="sp-svc-hd"><span class="sp-state">'+esc(x.state)+'</span>'+
      (x.depends&&x.depends.length?'<span class="sp-dep">⇠'+x.depends.length+'</span>':'')+'</div>'+
      '<div class="sp-svc-nm">'+esc(x.name)+'</div>'+
      '<div class="sp-svc-ft"><span>'+esc((x.caps&&x.caps.emits.join(","))||"—")+'</span>'+
      '<span class="sp-fr">'+(x.faults||0)+'/'+(x.restarts||0)+'</span></div></div>';
  }).join("");
  if($("sp-pkgs")){ var ins=(pkgs&&pkgs.installed)||[], av=(pkgs&&pkgs.available)||[];
    $("sp-pkgs").innerHTML='<span class="sp-pl">pkg</span>'+
      ins.map(function(n){return '<span class="sp-pill on">'+esc(n)+'</span>';}).join("")+
      av.map(function(n){return '<span class="sp-pill" data-install="'+esc(n)+'" title="verify-then-install">+ '+esc(n)+'</span>';}).join(""); }
}
function renderKernelTelemetry(snap){
  var host=$("kn-stats"); if(!host||!snap) return;
  var k=snap.kstat||{}, mem=(snap.memfabric&&snap.memfabric.arena)||{}, sf=snap.safety||{},
      vr=(snap.kernel_devices&&snap.kernel_devices.verifier)||{}, sv=snap.services||{};
  function tile(label,big,sub){ return '<div class="kn-stat"><div class="kn-st-k">'+label+'</div>'+
    '<div class="kn-st-v">'+big+'</div><div class="kn-st-s">'+sub+'</div></div>'; }
  var num=function(n){ return (n||0).toLocaleString(); };
  host.innerHTML=
    tile('Scheduler', (k.mode==="PREEMPT"?'<span class="cl-crit">PREEMPT</span>':'<span class="cl-ok">NORMAL</span>'),
      'Σcost '+fmt(k.global_cost||0,2)+' · '+Math.round((k.utilization||0)*100)+'% util')+
    tile('Compute lanes', (k.lanes_used||0)+'/'+(k.lane_budget||0),
      (k.procs||0)+' procs · '+(k.anticipating||0)+' anticipating')+
    tile('Memory fabric', (Math.round((mem.utilization||0)*1000)/10)+'%',
      num(mem.used_cells)+' / '+num(mem.capacity_cells)+' cells')+
    tile('Z3 egress cage', Math.round((sf.pass_rate!=null?sf.pass_rate:1)*100)+'%',
      (sf.certified||0)+' certified · '+(sf.blocked||0)+' blocked')+
    tile('eBPF verifier', (vr.accepted||0)+' / '+(vr.rejected||0), 'admitted / rejected')+
    tile('Services', (sv.running||0)+'/'+(sv.n||0), 'running · supervised');
}
function installPkg(name){
  fetch(window.AEGIS.http("/aegis/pkg/install"),{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({name:name})}).then(function(r){return r.json();}).then(function(d){
      if(cmdOpen||true){ cmdShow(); if(d.ok) cmdPrint('<span class="cl-ok">✓ installed '+esc(d.name)+'</span> via verify-then-install','sys');
        else { cmdPrint('<span class="cl-crit">✗ '+esc(name)+' REFUSED at '+esc(d.stage)+' gate</span> — '+esc(d.reason),"err");
               (d.detail||[]).forEach(function(x){ cmdPrint('· '+esc(x),"dim"); }); } }
    }).catch(function(){});
}
function renderVerify(d){
  var title='driver'+(d.name?' <b>'+esc(d.name)+'</b>':'')+(d.description?' <span class="cl-d">— '+esc(d.description)+'</span>':'');
  if(d.stage==="assemble"||d.stage==="lookup"||d.stage==="input"){ cmdPrint('verifier: '+esc(d.error||"bad input"),"err"); return; }
  if(d.ok){
    cmdPrint('<span class="cl-ok">✓ ACCEPTED</span> '+title,"sys");
    cmdPrint('<span class="cl-d">⊢ '+esc(d.proof||(d.report&&d.report.proof)||"")+'</span>',"mono");
    cmdPrint('admitted to the kernel — provably halts, stays in-window, calls only whitelisted helpers',"row");
  } else {
    cmdPrint('<span class="cl-crit">✗ REJECTED</span> '+title,"sys");
    var v=d.violation||{};
    cmdPrint('<span class="cl-crit">'+esc(v.rule)+'</span> at instr <span class="cl-cmd">#'+v.index+'</span>  <span class="cl-d">'+esc(v.instr)+'</span>',"row");
    cmdPrint(esc(v.detail),"dim");
    cmdPrint('refused at load — the kernel will not bind this bytecode to a ring',"row");
  }
  // show the listing with the offending line marked
  var list=d.listing||[]; if(list.length&&list.length<=14){ var bad=d.violation?d.violation.index:-1;
    list.forEach(function(ln,i){ cmdPrint('<span class="cl-d">'+pad(String(i),3)+'</span>'+(i===bad?'<span class="cl-crit">▶ ':'  ')+esc(ln)+(i===bad?'</span>':''),"mono"); }); }
}

/* ============================================================
   WEBSOCKET
   ============================================================ */
var ws=null, connected=false, lastMsg=0, connTries=0, paused=false;
/* An open tab is an open instance. Render's free tier bills wall-clock time for
   as long as anything is talking to the service, and this console polls every
   2.5 s and holds a socket. The visibility handler below covers a BACKGROUNDED
   tab, but a console left on a desk in the foreground keeps the backend awake
   indefinitely — one forgotten overnight tab is ~8 of the month's 750 instance
   hours. After IDLE_MS with no human input we drop the socket and stop polling;
   any input at all brings it straight back. */
var IDLE_MS=15*60*1000, lastInput=0, idled=false;
function goIdle(){ if(idled||paused) return;
  idled=true; try{ ws&&ws.close(); }catch(e){} setConn(false); }
function noteInput(){ lastInput=performance.now(); if(!idled) return;
  idled=false; if(document.hidden) return;
  if(!ws||ws.readyState>1) connect(); }
function clusterWS(){ return window.AEGIS.base().replace(/^http/,"ws")+"/aegis/cluster/stream"; }
function connect(){ if(paused||idled||document.hidden) return;   // don't stream to a backgrounded or idle tab
  try{ ws=new WebSocket(clusterWS()); }catch(e){ setTimeout(connect,1500); return; }
  ws.onopen=function(){ connected=true; connTries=0; setConn(true); };
  ws.onmessage=function(e){ lastMsg=performance.now(); try{ ingest(JSON.parse(e.data)); }catch(err){ console.error(err); } };
  ws.onclose=function(){ connected=false; connTries++; setConn(false); if(paused||idled||document.hidden) return; setTimeout(connect, Math.min(4000, 1200+connTries*400)); };
  ws.onerror=function(){ try{ws.close();}catch(e){} };
}
// Pause the cluster feed (and the poll below) on hidden tabs to save backend egress.
// Coming back to the tab counts as input, so it also clears the idle stop.
document.addEventListener("visibilitychange",function(){
  if(document.hidden){ paused=true; try{ws&&ws.close();}catch(e){} }
  else { paused=false; idled=false; lastInput=performance.now();
         if(!ws||ws.readyState>1) connect(); } });
function setConn(live){ $("conn").classList.toggle("live",live);
  $("conn-txt").textContent = idled ? "Idle · click to resume"
    : live ? "Live" : (connTries>=2 ? "Waking backend…" : "Connecting"); }
setInterval(function(){ if(document.hidden||idled) return;   // no background or idle polling
  if(connected && performance.now()-lastMsg<4000) return;
  fetch(window.AEGIS.http("/aegis/cluster/snapshot")).then(function(r){return r.json();}).then(ingest).catch(function(){}); },2500);
["mousedown","keydown","touchstart","wheel","mousemove"].forEach(function(ev){
  window.addEventListener(ev,noteInput,{passive:true}); });
setInterval(function(){ if(performance.now()-lastInput>IDLE_MS) goIdle(); },30000);

/* ============================================================
   AMBIENT BACKGROUND — drifting node mesh
   ============================================================ */
var amb={cv:null,ctx:null,w:0,h:0,nodes:[]};
function ambientInit(){ amb.cv=$("ambient"); if(!amb.cv) return; amb.ctx=amb.cv.getContext("2d");
  ambientFit(); var n=Math.min(64,Math.floor(window.innerWidth/28)); amb.nodes=[];
  for(var i=0;i<n;i++) amb.nodes.push({x:Math.random(),y:Math.random(),
    vx:(Math.random()-.5)*0.00010,vy:(Math.random()-.5)*0.00010,r:Math.random()*1.5+0.5}); }
function ambientFit(){ if(!amb.cv) return; var dpr=Math.min(window.devicePixelRatio||1,2);
  amb.w=window.innerWidth; amb.h=window.innerHeight; amb.cv.width=amb.w*dpr; amb.cv.height=amb.h*dpr;
  amb.ctx.setTransform(dpr,0,0,dpr,0,0); }
function ambientDraw(){ if(!amb.ctx) return; var ctx=amb.ctx,w=amb.w,h=amb.h,ns=amb.nodes; ctx.clearRect(0,0,w,h);
  for(var i=0;i<ns.length;i++){ var p=ns[i]; p.x+=p.vx; p.y+=p.vy; if(p.x<0||p.x>1)p.vx*=-1; if(p.y<0||p.y>1)p.vy*=-1; }
  for(i=0;i<ns.length;i++) for(var j=i+1;j<ns.length;j++){ var a=ns[i],b=ns[j],dx=(a.x-b.x)*w,dy=(a.y-b.y)*h,d=Math.hypot(dx,dy);
    if(d<160){ ctx.strokeStyle="rgba(108,140,255,"+(0.045*(1-d/160))+")"; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(a.x*w,a.y*h); ctx.lineTo(b.x*w,b.y*h); ctx.stroke(); } }
  for(i=0;i<ns.length;i++){ var q=ns[i]; ctx.fillStyle="rgba(125,140,255,.20)"; ctx.beginPath(); ctx.arc(q.x*w,q.y*h,q.r,0,7); ctx.fill(); } }

/* ============================================================
   BOOT SEQUENCE
   ============================================================ */
function runBoot(){ var log=$("boot-log"), fill=$("boot-fill"); if(!log) return;
  var lines=[["init  kernel image",0],["L1  SHAL   /dev/bio device-driver bus",1],
    ["L2  VMM    zero-copy biometric memory fabric",1],["L3  SCHED  multi-tenant cognitive scheduler",1],
    ["L4  NSYM   neurosymbolic egress cage · Z3",1],["mount  PhysioNet/CinC-2019 cohort",1],
    ["fusion ontology  edge + enterprise tiers",1],["ICU WING ONLINE",2]];
  var i=0, html="";
  (function step(){ if(i>=lines.length){ fill.style.width="100%";
      setTimeout(function(){ $("boot").classList.add("gone"); },650); return; }
    var L=lines[i], tag=L[1]===1?'<span class="ok">ONLINE</span>':(L[1]===2?'<span class="ok">▸ READY</span>':'');
    var pad=L[1]===1?(" "+new Array(Math.max(2,46-L[0].length)).join(".")+" "):"  ";
    html+='<div>'+(L[1]===2?'<b>':'')+L[0]+(L[1]===2?'</b>':'')+pad+tag+'</div>';
    log.innerHTML=html; log.scrollTop=log.scrollHeight; fill.style.width=Math.round((i+1)/lines.length*100)+"%";
    i++; setTimeout(step, L[1]===2?500:230); })(); }

/* ============================================================
   VIEW SWITCHING
   ============================================================ */
var currentView="ops";
function setView(name){ currentView=name;
  ["ops","val","devices"].forEach(function(v){ var el=$("view-"+v); if(el) el.classList.toggle("active",v===name); });
  [].forEach.call(document.querySelectorAll(".tab"),function(t){ t.classList.toggle("active",t.dataset.view===name); });
  if(name==="val") loadValidation();
  
  if(name==="devices") startDevices(); else stopDevices();
}

/* ============================================================
   DEVICES — pairing & management hub (SIMULATED fleet, Phase A)
   ============================================================ */
var devTimer=null, devFleet=null, selectedDevice=null;
function devBeds(){ return Object.keys(latestBeds).map(function(k){return latestBeds[k];}); }
function devIcon(t){ return t==="wearable"?"⌚":t==="optical"?"◉":"⛁"; }
function devIconSVG(t,cls){ cls=cls||"dv-ic";
  if(t==="wearable") return '<svg class="'+cls+' w" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="6.5" y="6.5" width="11" height="11" rx="3"/><path d="M9 6.5l.6-3h4.8l.6 3M9 17.5l.6 3h4.8l.6-3"/><path d="M12 9.6v2.4l1.7 1" stroke-width="1.3"/></svg>';
  if(t==="optical") return '<svg class="'+cls+' o" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3.1"/><circle cx="12" cy="12" r="1" fill="currentColor"/></svg>';
  return '<svg class="'+cls+' g" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="4.5" width="18" height="6" rx="1.8"/><rect x="3" y="13.5" width="18" height="6" rx="1.8"/><circle cx="6.6" cy="7.5" r=".9" fill="currentColor"/><circle cx="6.6" cy="16.5" r=".9" fill="currentColor"/><path d="M10 7.5h7M10 16.5h7" stroke-width="1.1"/></svg>';
}
function startDevices(){ refreshFleet(); if(devTimer) clearInterval(devTimer); devTimer=setInterval(function(){ if(!idled&&!document.hidden) refreshFleet(); },2600); }
function stopDevices(){ if(devTimer){ clearInterval(devTimer); devTimer=null; } }
function refreshFleet(){
  fetch(window.AEGIS.http("/aegis/devices")).then(function(r){return r.json();})
    .then(function(d){ devFleet=d; renderFleet(d); }).catch(function(){});
}
function renderDevCounts(c){
  var cnt=$("dev-counts"); if(!cnt) return;
  var tiles=[["fleet",c.total,C.cyan],["live",c.streaming,C.ok],["degraded",c.degraded,C.watch],
             ["offline",c.offline,C.dim],["unassigned",c.unassigned,C.indigo]];
  cnt.innerHTML=tiles.map(function(t){ return '<span class="dev-count"><b style="color:'+t[2]+
    '">'+(t[1]||0)+'</b><span>'+t[0]+'</span></span>'; }).join("");
}
function renderFleet(d){
  renderDevCounts(d.counts||{});
  var grid=$("dev-grid"); if(!grid) return;
  var devs=d.devices||[], ids={};
  devs.forEach(function(dev){ ids[dev.id]=1;
    var el=grid.querySelector('.dev-card[data-id="'+dev.id+'"]');
    if(!el){ el=document.createElement("div"); el.className="dev-card"; el.dataset.id=dev.id;
      el.innerHTML=devCardShell(dev); grid.appendChild(el); }
    updateCard(el,dev); });
  [].forEach.call(grid.querySelectorAll(".dev-card"),function(el){ if(!ids[el.dataset.id]) el.remove(); });
  var ord={streaming:0,degraded:1,assigned:2,provisioned:3,pairing:4,offline:5,decommissioned:6};
  devs.slice().sort(function(a,b){return (ord[a.status]||9)-(ord[b.status]||9);})
    .forEach(function(dev){ var el=grid.querySelector('.dev-card[data-id="'+dev.id+'"]'); if(el) grid.appendChild(el); });
}
function devCardShell(dev){
  return '<span class="dv-rail"></span><span class="dv-glow"></span>'+
    '<div class="dv-top"><span class="dv-icw">'+devIconSVG(dev.type)+'</span>'+
      '<span class="dv-id">'+dev.id+'</span>'+
      '<span class="dv-pill"><i class="dv-dot"></i><span class="dv-st"></span></span></div>'+
    '<div class="dv-model">'+dev.model+' · '+dev.type+'</div>'+
    '<div class="dv-trace"><canvas class="dv-spark"></canvas><span class="dv-sigv"></span></div>'+
    '<div class="dv-meta"><span class="dv-fw"></span><span class="dv-batt"><i></i><b></b></span>'+
      '<span class="dv-rssi"></span></div>'+
    '<div class="dv-assign"></div>';
}
function updateCard(el,dev){
  el.className="dev-card h-"+dev.health+(dev.status==="streaming"?" streaming":"");
  el.dataset.status=dev.status; el.dataset.sig=dev.signal_quality||0;
  el.dataset.health=dev.health; el.dataset.streams=dev.streams?1:0; el.dataset.type=dev.type;
  var st=el.querySelector(".dv-st"); st.textContent=dev.status;
  el.querySelector(".dv-pill").className="dv-pill st-"+dev.status;
  var fw=el.querySelector(".dv-fw"); fw.innerHTML="fw "+dev.firmware+(dev.fw_outdated?' <b>⬆</b>':"");
  fw.className="dv-fw"+(dev.fw_outdated?" old":"");
  var batt=el.querySelector(".dv-batt"), bi=batt.querySelector("i"), bb=batt.querySelector("b");
  if(dev.battery==null){ batt.className="dv-batt ac"; bb.textContent="AC"; bi.style.width="100%"; }
  else { batt.className="dv-batt"+(dev.battery<=15?" low":dev.battery<=35?" mid":"");
    bb.textContent=dev.battery+"%"; bi.style.width=clamp(dev.battery,4,100)+"%"; }
  el.querySelector(".dv-rssi").textContent=dev.status==="offline"?"— dBm":dev.rssi+" dBm";
  var sv=el.querySelector(".dv-sigv");
  sv.textContent=dev.streams?"● LIVE":(dev.assigned_bed&&!dev.calibrated?"CALIBRATE":Math.round((dev.signal_quality||0)*100)+"%");
  sv.className="dv-sigv"+(dev.streams?" live":(dev.assigned_bed&&!dev.calibrated?" warn":""));
  el.querySelector(".dv-assign").innerHTML=dev.assigned_label
    ? '<span class="dv-pt"><i></i>'+dev.assigned_label+'</span>' : '<span class="dv-un">unassigned</span>';
}
/* live telemetry sparkline — a scrolling PPG-style pulse per streaming device */
function _ppg(t){ var p=((t%6.2832)+6.2832)%6.2832/6.2832;
  var a=Math.exp(-Math.pow((p-0.16)/0.06,2)), b=0.32*Math.exp(-Math.pow((p-0.4)/0.085,2));
  return (a+b)-0.22; }
function _uplink(t){ return 0.5*Math.sin(t*4.0)+0.28*Math.sin(t*9.0+1.0)+0.16*Math.sin(t*15.0); }
var devTick=0;
function drawDeviceSparks(){
  devTick++;
  var cards=document.querySelectorAll("#dev-grid .dev-card");
  for(var i=0;i<cards.length;i++){
    var el=cards[i], cv=el.querySelector(".dv-spark"); if(!cv) continue;
    var w=cv.clientWidth, h=cv.clientHeight; if(!w||!h) continue;
    var dpr=Math.min(2,window.devicePixelRatio||1);
    if(cv.width!==Math.round(w*dpr)){ cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr); }
    var ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
    var streams=el.dataset.streams==="1", sig=parseFloat(el.dataset.sig)||0,
        health=el.dataset.health, type=el.dataset.type, status=el.dataset.status;
    var col=health==="degraded"?C.watch:health==="offline"?C.dim:type==="gateway"?C.indigo:C.cyan;
    if(status==="streaming"&&health==="ok") col=C.ok;
    var mid=h*0.55;
    if(status==="offline"||status==="decommissioned"){
      ctx.strokeStyle=withAlpha(C.dim,.45); ctx.lineWidth=1; ctx.setLineDash([2,4]);
      ctx.beginPath(); ctx.moveTo(0,mid); ctx.lineTo(w,mid); ctx.stroke(); ctx.setLineDash([]); continue;
    }
    ctx.strokeStyle=withAlpha(col,.10); ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(0,mid); ctx.lineTo(w,mid); ctx.stroke();
    var amp=(0.22+0.62*sig)*h*0.4, fn=type==="gateway"?_uplink:_ppg;
    var sp=streams?(type==="gateway"?0.14:0.085):0.0, base=devTick*sp, cyc=type==="gateway"?3.2:5.6;
    ctx.beginPath();
    for(var x=0;x<=w;x+=2){ var tt=(x/w)*cyc+base, y=mid-fn(tt)*amp; x===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }
    ctx.strokeStyle=withAlpha(col,streams?0.95:0.4); ctx.lineWidth=1.6; ctx.lineJoin="round";
    ctx.shadowColor=col; ctx.shadowBlur=streams?7:0; ctx.stroke(); ctx.shadowBlur=0;
    if(streams){ var lt=cyc+base, ly=mid-fn(lt)*amp; ctx.fillStyle=col;
      ctx.beginPath(); ctx.arc(w-2,ly,2.2,0,6.28); ctx.fill(); }
  }
}
function openDevice(id,dev){
  dev=dev||(devFleet&&devFleet.devices||[]).filter(function(d){return d.id===id;})[0];
  if(!dev){ fetch(window.AEGIS.http("/aegis/devices/"+id)).then(function(r){return r.json();})
    .then(function(d){ if(d&&d.id) openDevice(id,d); }); return; }
  selectedDevice=id; renderDeviceDrawer(dev);
  $("dev-drawer").classList.add("open"); $("dev-drawer").setAttribute("aria-hidden","false");
}
function closeDevice(){ selectedDevice=null; $("dev-drawer").classList.remove("open"); $("dev-drawer").setAttribute("aria-hidden","true"); }
function reopenDevice(id){ fetch(window.AEGIS.http("/aegis/devices/"+id)).then(function(r){return r.json();})
  .then(function(d){ if(d&&d.id&&selectedDevice===id) renderDeviceDrawer(d); }); }
function statRow(k,v){ return '<div class="dd-stat"><span>'+k+'</span><b>'+v+'</b></div>'; }
function devAgo(ts){ if(!ts) return "—"; var s=Date.now()/1000-ts;
  return s<60?Math.round(s)+"s ago":s<3600?Math.round(s/60)+"m ago":s<86400?Math.round(s/3600)+"h ago":Math.round(s/86400)+"d ago"; }
function gaugeRing(label,frac,text,col){ frac=clamp(frac,0,1); var c=2*Math.PI*20, off=c*(1-frac);
  return '<div class="dd-gauge"><svg viewBox="0 0 52 52"><circle cx="26" cy="26" r="20" fill="none" stroke="rgba(190,205,255,.09)" stroke-width="4"/>'+
    '<circle cx="26" cy="26" r="20" fill="none" stroke="'+col+'" stroke-width="4" stroke-linecap="round" stroke-dasharray="'+c.toFixed(1)+
    '" stroke-dashoffset="'+off.toFixed(1)+'" transform="rotate(-90 26 26)"/></svg>'+
    '<span class="dg-v" style="color:'+col+'">'+text+'</span><span class="dg-k">'+label+'</span></div>'; }
var devDrawerDev=null;
function renderDeviceDrawer(dev){
  devDrawerDev=dev; var sig=dev.signal_quality||0;
  var beds=devBeds().sort(function(a,b){return a.bed_id<b.bed_id?-1:1;});
  var opts=beds.map(function(b){return '<option value="'+b.bed_id+'" data-label="'+b.name+'">'+bedTag(b.bed_id)+' · '+b.name+'</option>';}).join("");
  var assign=dev.assigned_bed
    ? '<div class="dd-assigned"><i></i>Streaming to <b>'+dev.assigned_label+'</b><button class="dd-btn" data-act="unassign">Unassign</button></div>'
    : (beds.length?'<div class="dd-assign"><select id="dd-bed">'+opts+'</select><button class="dd-btn primary" data-act="assign">Assign to bed</button></div>':'<div class="dd-note">No beds available to assign.</div>');
  var acts='';
  if(!dev.calibrated) acts+='<button class="dd-btn glow" data-act="calibrate">⊹ Calibrate</button>';
  if(dev.fw_outdated) acts+='<button class="dd-btn" data-act="ota">⬆ Update → '+dev.fw_latest+'</button>';
  if(dev.status==="offline") acts+='<button class="dd-btn" data-act="wake">◐ Wake</button>';
  acts+='<button class="dd-btn" data-act="reboot">↻ Reboot</button>';
  acts+='<button class="dd-btn danger" data-act="decommission">⊘ Decommission</button>';
  var battFrac=dev.battery==null?1:dev.battery/100, battTxt=dev.battery==null?"AC":dev.battery+"%",
      battCol=dev.battery!=null&&dev.battery<=15?C.crit:dev.battery!=null&&dev.battery<=35?C.watch:C.ok;
  $("dev-drawer-inner").innerHTML=
    '<div class="dd-head"><span class="dd-hero-ic">'+devIconSVG(dev.type,"dd-ic")+'</span>'+
      '<div class="dd-id"><div class="dd-idn">'+dev.id+'</div><div class="dd-idm">'+dev.model+' · '+dev.type+'</div></div>'+
      '<span class="dv-pill st-'+dev.status+'"><i class="dv-dot"></i><span class="dv-st">'+dev.status+'</span></span>'+
      '<button class="dd-x" data-act="close">✕</button></div>'+
    '<div class="dd-sim">◉ simulated device · serial '+dev.serial+'</div>'+
    '<div class="dd-hero"><canvas id="dd-trace"></canvas><span class="dd-hero-lbl '+(dev.streams?"live":"")+'">'+
      (dev.streams?"● LIVE TELEMETRY":(dev.status==="offline"?"◌ OFFLINE":"◌ STANDBY"))+'</span></div>'+
    '<div class="dd-gauges">'+
      gaugeRing("Battery",battFrac,battTxt,battCol)+
      gaugeRing("Signal",sig,Math.round(sig*100)+"%",sig>=0.6?C.ok:sig>=0.4?C.watch:C.crit)+
      gaugeRing("Calibration",dev.calibrated?(dev.calibration_quality||0.9):0,dev.calibrated?"✓":"—",dev.calibrated?C.cyan:C.dim)+
    '</div>'+
    assign+
    '<div class="dd-gate '+(dev.streams?"on":"off")+'"><b>'+(dev.streams?"✓ DRIVING DETECTION":"✗ DETECTION GATED")+'</b>'+
      '<span>'+(dev.streams?"calibrated &amp; streaming live":"needs pair + assign + calibration + signal")+'</span></div>'+
    '<div class="dd-stats">'+
      statRow("Firmware", dev.firmware+(dev.fw_outdated?" · update":" · current"))+
      statRow("Link", dev.status==="offline"?"—":dev.rssi+" dBm")+
      statRow("Channels", dev.channels.join(", "))+
      statRow("Paired", devAgo(dev.paired_at))+
    '</div>'+
    '<div class="dd-acts">'+acts+'</div>';
}
function drawDrawerTrace(){
  var cv=document.getElementById("dd-trace"); if(!cv||!devDrawerDev) return;
  var w=cv.clientWidth,h=cv.clientHeight; if(!w||!h) return;
  var dpr=Math.min(2,window.devicePixelRatio||1);
  if(cv.width!==Math.round(w*dpr)){ cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr); }
  var ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
  var dev=devDrawerDev, streams=dev.streams, sig=dev.signal_quality||0, type=dev.type, status=dev.status;
  var col=status==="offline"?C.dim:dev.health==="degraded"?C.watch:streams?C.ok:type==="gateway"?C.indigo:C.cyan;
  ctx.strokeStyle=withAlpha(col,.06); ctx.lineWidth=1;
  for(var gx=0;gx<w;gx+=Math.max(18,w/14)){ ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,h); ctx.stroke(); }
  var mid=h*0.55;
  if(status==="offline"){ ctx.strokeStyle=withAlpha(C.dim,.4); ctx.setLineDash([3,5]);
    ctx.beginPath(); ctx.moveTo(0,mid); ctx.lineTo(w,mid); ctx.stroke(); ctx.setLineDash([]); return; }
  var amp=(0.25+0.6*sig)*h*0.42, fn=type==="gateway"?_uplink:_ppg;
  var sp=streams?(type==="gateway"?0.14:0.085):0, base=devTick*sp, cyc=type==="gateway"?5:9;
  ctx.beginPath();
  for(var x=0;x<=w;x+=1.5){ var tt=(x/w)*cyc+base, y=mid-fn(tt)*amp; x===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }
  ctx.strokeStyle=withAlpha(col,streams?0.95:0.45); ctx.lineWidth=1.8; ctx.lineJoin="round";
  ctx.shadowColor=col; ctx.shadowBlur=streams?8:0; ctx.stroke(); ctx.shadowBlur=0;
  if(streams){ var lt=cyc+base, ly=mid-fn(lt)*amp; ctx.fillStyle=col;
    ctx.beginPath(); ctx.arc(w-2,ly,2.6,0,6.28); ctx.fill(); }
}
function devCallAction(act){
  if(!selectedDevice) return; var id=selectedDevice;
  if(act==="close") return closeDevice();
  var url, body=null;
  if(act==="assign"){ var sel=$("dd-bed"); if(!sel) return;
    var opt=sel.options[sel.selectedIndex];
    url="/aegis/devices/"+id+"/assign"; body={bed_id:sel.value,label:opt?opt.getAttribute("data-label"):sel.value}; }
  else if(act==="unassign"){ url="/aegis/devices/"+id+"/unassign"; }
  else { url="/aegis/devices/"+id+"/action"; body={action:act}; }
  fetch(window.AEGIS.http(url),{method:"POST",headers:{"Content-Type":"application/json"},
    body:body?JSON.stringify(body):undefined}).then(function(r){return r.json();})
    .then(function(){ refreshFleet(); if(act==="decommission") closeDevice(); else reopenDevice(id); }).catch(function(){});
}
/* ---- pairing wizard ---- */
function openWizard(){ $("dev-modal").classList.add("open"); $("dev-modal").setAttribute("aria-hidden","false"); wizScan(); }
function closeWizard(){ $("dev-modal").classList.remove("open"); $("dev-modal").setAttribute("aria-hidden","true"); }
function radarHTML(){
  return '<div class="wiz-radar"><div class="wr-sweep"></div>'+
    '<span class="wr-ring r1"></span><span class="wr-ring r2"></span><span class="wr-ring r3"></span>'+
    '<span class="wr-cross h"></span><span class="wr-cross v"></span><span class="wr-core"></span></div>';
}
function wizScan(){
  $("dev-wiz-body").innerHTML='<div class="wiz-scan">'+radarHTML()+
    '<p>Scanning for nearby devices…</p><span class="wiz-sub">simulated BLE · 2.4 GHz</span></div>';
  setTimeout(function(){
    fetch(window.AEGIS.http("/aegis/devices/discover")).then(function(r){return r.json();})
      .then(function(d){ renderWizList(d.devices||[]); }).catch(function(){});
  }, 850);   // let the radar sweep land before results pop in
}
function wizRssiBars(rssi){ var n=rssi>=-58?4:rssi>=-68?3:rssi>=-78?2:1, b="";
  for(var i=1;i<=4;i++) b+='<i class="'+(i<=n?"on":"")+'" style="height:'+(3+i*2.4)+'px"></i>'; return '<span class="wiz-rssi">'+b+'</span>'; }
function renderWizList(list){
  var body=$("dev-wiz-body");
  if(!list.length){ body.innerHTML='<div class="wiz-empty"><div class="wiz-empty-ic">⊘</div>No devices in range.<button class="dd-btn" id="wiz-rescan">↻ Rescan</button></div>'; return; }
  body.innerHTML='<div class="wiz-hint"><span class="wiz-dot"></span>SIMULATED SCAN · '+list.length+' DEVICE(S) IN RANGE</div>'+
    list.map(function(d,i){ return '<div class="wiz-dev" style="animation-delay:'+(i*60)+'ms"><span class="wiz-ic">'+devIconSVG(d.type,"dv-ic")+'</span>'+
      '<div class="wiz-info"><b>'+d.id+'</b><span>'+d.model+' · '+d.type+'</span></div>'+
      wizRssiBars(d.rssi)+'<button class="dd-btn primary" data-pair="'+d.id+'">Pair</button></div>'; }).join("")+
    '<button class="dd-btn wiz-rescan" id="wiz-rescan">↻ Rescan</button>';
}
function wizPair(id){
  fetch(window.AEGIS.http("/aegis/devices/"+id+"/pair"),{method:"POST"}).then(function(r){return r.json();})
    .then(function(res){ if(res.status==="ok"){ closeWizard(); refreshFleet(); openDevice(id,res.device); } }).catch(function(){});
}

/* ============================================================
   VALIDATION VIEW
   ============================================================ */
var valData=null;
function kpiHTML(cls,k,v,unit,s){ return '<div class="kpi'+(cls?" "+cls:"")+'"><div class="k">'+k+
  '</div><div class="v" data-count="'+v+'">0<small>'+unit+'</small></div><div class="s">'+s+'</div></div>'; }
/* ============================================================
   SYSTEM HEALTH
   Replaces the old Validation page. Same question, answered properly: how good is
   this detector, what does it do well, and where does it fall short — read from
   the CROSS-HOSPITAL evaluation artefact rather than from the twelve-patient demo
   cohort, which cannot support an AUROC.
   ============================================================ */
function loadValidation(){
  fetch(window.AEGIS.http("/aegis/health/system")).then(function(r){return r.json();})
    .then(function(d){ valData=d; renderHealth(d); }).catch(function(){
      var st=$("hz-status"); if(st){ st.className="hz-status bad";
        st.innerHTML='<i></i><span>unreachable</span>'; } });
}
function pctTxt(v){ return v==null?"—":Math.round(v*100)+"%"; }

function pct1(v){ return v==null?"—":(v*100).toFixed(1)+"%"; }
function renderHealth(d){
  if(!d || d.status!=="ok"){
    var st0=$("hz-status"); if(st0){ st0.className="hz-status warn";
      st0.innerHTML='<i></i><span>no evaluation</span>'; }
    $("hz-strengths").innerHTML='<div class="hz-empty">'+esc((d&&d.note)||
      "No cross-hospital evaluation artefact found.")+'</div>';
    return;
  }
  var h=d.headline, m=d.metrics||{};

  // ---- status chip: the grade the numbers earn, not a fixed "operational" ----
  var grade=d.grade||"fair";
  var cls=grade==="healthy"?"ok":grade==="fair"?"warn":"bad";
  $("hz-status").className="hz-status "+cls;
  $("hz-status").innerHTML='<i></i><span>'+esc(grade)+' · externally validated</span>';

  // ---- hero ----
  $("hz-auroc").textContent=(h.auroc!=null?h.auroc.toFixed(3):"—");
  $("hz-auroc-ci").textContent=(h.auroc_lo!=null
    ? "95% CI "+h.auroc_lo.toFixed(3)+"–"+h.auroc_hi.toFixed(3) : "");
  var dEl=$("hz-auroc-delta");
  if(h.delta!=null){
    dEl.className="hz-delta "+(h.delta>0?"up":"down");
    dEl.innerHTML='<b>'+(h.delta>0?"+":"")+h.delta.toFixed(3)+'</b> vs the '+
      esc(h.baseline)+' baseline ('+h.baseline_auroc.toFixed(3)+')';
  } else dEl.textContent="";
  // The bar starts at chance. An AUROC of 0.5 is worth nothing, so a bar anchored
  // at zero would render a useless model as half full.
  var frac=Math.max(0,Math.min(1,((h.auroc||0.5)-0.5)/0.5));
  $("hz-auroc-bar").style.width=(frac*100).toFixed(1)+"%";
  var refFrac=Math.max(0,Math.min(1,((h.baseline_auroc||0.5)-0.5)/0.5));
  $("hz-auroc-ref").style.left=(refFrac*100).toFixed(1)+"%";
  $("hz-auroc-ref").title="baseline "+(h.baseline_auroc||0).toFixed(3);

  $("hz-hero-side").innerHTML=[
    ["Detection rate", pctTxt(m.detection_rate), "of septic stays caught"],
    ["False alarms", pctTxt(m.false_alarm_rate), "of stable stays alarmed"],
    ["Median lead", (m.median_lead_hours!=null?m.median_lead_hours+"<em>h</em>":"—"),
     "before labelled onset"],
    ["Calibration", (m.ece!=null?m.ece.toFixed(3):"—"), "expected calibration error"]
  ].map(function(x){
    return '<div class="hz-stat"><span>'+x[0]+'</span><b>'+x[1]+'</b><i>'+x[2]+'</i></div>';
  }).join("");

  // ---- strengths / gaps, both generated from the evaluation ----
  function items(list,kind){
    if(!list||!list.length) return '<div class="hz-empty">Nothing crossed a '+
      'reporting threshold.</div>';
    return list.map(function(f){
      return '<div class="hz-item '+kind+'"><i>'+(kind==="good"?"↑":"↓")+'</i><div>'+
        '<div class="hz-item-t">'+esc(f.title)+'</div>'+
        '<span class="hz-item-s">'+esc(f.detail)+'</span></div></div>';
    }).join("");
  }
  $("hz-strengths").innerHTML=items(d.strengths,"good");
  $("hz-gaps").innerHTML=items(d.gaps,"bad");
  $("hz-str-meta").textContent=(d.strengths||[]).length+" findings";
  $("hz-gap-meta").textContent=(d.gaps||[]).length+" findings";

  // ---- generalisation: internal vs external, side by side ----
  var g=d.generalization;
  if(g){
    var f=function(v){ return Math.max(0,Math.min(1,(v-0.5)/0.5))*100; };
    $("hz-gen").innerHTML=
      '<div class="hz-gen-row"><span class="hz-gen-k">Inside the training hospital</span>'+
        '<span class="hz-gen-track"><i style="width:'+f(g.internal_auroc).toFixed(1)+'%"></i></span>'+
        '<b>'+g.internal_auroc.toFixed(3)+'</b></div>'+
      '<div class="hz-gen-row ext"><span class="hz-gen-k">A hospital it has never seen</span>'+
        '<span class="hz-gen-track"><i style="width:'+f(g.external_auroc).toFixed(1)+'%"></i></span>'+
        '<b>'+g.external_auroc.toFixed(3)+'</b></div>'+
      '<div class="hz-gen-note"><b>−'+g.gap.toFixed(3)+' AUROC</b> is what moving hospital '+
        'costs. '+esc(g.note)+'. The upper bar is the figure a paper would quote; the '+
        'lower one is what a new ICU would actually get.</div>';
  }

  // ---- the ladder: every model on the same protocol ----
  var lad=d.ladder||[];
  if(lad.length){
    var floor=lad[0].auroc;
    $("hz-ladder").innerHTML=lad.map(function(x,i){
      var w=Math.max(0,Math.min(1,(x.auroc-0.5)/0.5))*100;
      var dv=x.auroc-floor;
      return '<div class="hz-rung'+(x.production?" on":"")+'">'+
        '<div class="hz-rung-t"><span>'+esc(x.name)+'</span>'+
          (x.production?'<em>production</em>':'')+
          (i&&dv>0?'<u>+'+dv.toFixed(3)+'</u>':'')+
          '<b>'+x.auroc.toFixed(3)+'</b></div>'+
        '<div class="hz-rung-bar"><i style="width:'+w.toFixed(1)+'%"></i></div>'+
        '<span class="hz-rung-s">'+esc(x.scope)+' · '+esc(x.note)+'</span></div>';
    }).join("");
  }

  // ---- improvement attempts, including the ones that failed ----
  var imp=d.improvement;
  if(imp){
    $("hz-imp-meta").textContent=imp.trials+" mutations tried";
    $("hz-improve").innerHTML=
      '<div class="hz-imp-n"><b>'+imp.accepted+'</b> of '+imp.trials+' accepted</div>'+
      '<div class="hz-imp-v">'+esc(imp.verdict||"")+'</div>'+
      '<div class="hz-imp-s">A search that promotes nothing is the gate working, not '+
      'a gap in it — every candidate was measured on locked patients the search never '+
      'saw, and none beat the shipped configuration.</div>';
  }

  // ---- search transcript: what each mutation actually was ----
  var tt=(imp&&imp.transcript)||[];
  if(tt.length){
    var OUT={accepted:"accepted","not-significant":"gain not significant",
             "no-gain":"no gain",untested:"not tested",refused:"refused"};
    $("hz-trials").innerHTML=tt.map(function(t){
      var d=t.delta==null?0:t.delta;
      return '<div class="hz-trial '+esc(t.outcome)+'">'+
        '<span class="st-id">#'+t.id+'</span>'+
        '<span class="st-mut">'+t.mutation.map(esc).join('<i> · </i>')+'</span>'+
        '<span class="st-d '+(d>0?"up":d<0?"dn":"")+'">'+(d>=0?"+":"\u2212")+
          Math.abs(d).toFixed(4)+'</span>'+
        '<span class="st-p">'+(t.p==null?"—":"p "+t.p.toFixed(2))+'</span>'+
        '<span class="st-o">'+esc(OUT[t.outcome]||t.outcome)+'</span></div>';
    }).join("");
    var n={}; tt.forEach(function(t){ n[t.outcome]=(n[t.outcome]||0)+1; });
    var untested=n.untested||0;
    $("hz-trials-note").innerHTML=
      'Sorted by measured gain. '+
      (untested? '<b>'+untested+' of '+tt.length+'</b> were evaluated but never tested: '+
        'only the best challenger of each round is put to the paired test, so the rest '+
        'cost compute without producing evidence. That is the clearest place this search '+
        'has room to improve.' : '');
  }

  // ---- alarm policy ----
  var ap=d.alarm_policy;
  if(ap){
    $("hz-alarm").innerHTML=
      '<div class="hz-ap-v">'+esc(ap.verdict)+'</div>'+
      '<div class="hz-ap-row"><span>searched</span><b>'+ap.families.map(esc).join(' · ')+'</b></div>'+
      '<div class="hz-ap-row"><span>shipped</span><b>'+esc(ap.shipped||"—")+'</b></div>'+
      '<div class="hz-ap-row"><span>best challenger</span><b>'+esc(ap.challenger||"—")+'</b></div>'+
      '<div class="hz-ap-row"><span>on locked patients</span><b>'+
        'false alarm '+pct1(ap.champion_far)+' \u2192 '+pct1(ap.challenger_far)+
        ' · detection '+pct1(ap.champion_detection)+' \u2192 '+pct1(ap.challenger_detection)+'</b></div>'+
      '<div class="hz-ap-tag '+(ap.promoted?"on":"")+'">'+
        (ap.promoted?"PROMOTED":"REFUSED")+'</div>'+
      '<div class="hz-note">'+esc(ap.reason||"")+'</div>';
  }

  // ---- live subsystem state ----
  $("hz-subsystems").innerHTML=(d.subsystems||[]).map(function(sub){
    return '<div class="hz-sub'+(sub.ok?"":" bad")+'"><i></i><div>'+
      '<div class="hz-sub-n">'+esc(sub.name)+'</div>'+
      '<span class="hz-sub-s">'+esc(sub.note)+'</span></div></div>';
  }).join("");

  // ---- the demo cohort, labelled as what it is ----
  var c=d.cohort;
  if(c&&c.patients){
    var maxH=Math.max.apply(null,c.patients.map(function(p){
      return Math.max(p.onset_hour||0,p.first_alarm||0,1); }));
    $("hz-cohort-meta").textContent=c.summary.n+" demo stays · illustrative only, "+
      "not the source of the numbers above";
    $("hz-cohort").innerHTML=c.patients.slice().sort(function(a,b){
      return (b.lead==null?-99:b.lead)-(a.lead==null?-99:a.lead); }).map(function(p){
      var alarm=p.first_alarm, onset=p.onset_hour;
      var caught=p.septic&&alarm!=null&&(onset==null||alarm<=onset);
      var v=!p.septic?(alarm!=null?'<span class="hz-row-v fp">false alarm</span>'
                                  :'<span class="hz-row-v none">stable</span>')
            :(p.lead!=null&&p.lead>=0?'<span class="hz-row-v">+'+p.lead+' h early</span>'
                                     :'<span class="hz-row-v none">missed</span>');
      var bar='';
      if(alarm!=null){
        var x0=100*Math.min(alarm,onset==null?alarm:onset)/maxH;
        var x1=100*(onset==null?alarm:Math.max(alarm,onset))/maxH;
        bar='<b style="left:'+x0.toFixed(1)+'%;width:'+Math.max(1,x1-x0).toFixed(1)+'%"></b>';
      }
      var onsetMark=onset!=null?'<u style="left:'+(100*onset/maxH).toFixed(1)+'%"></u>':'';
      return '<div class="hz-row"><span class="hz-row-n">'+esc(p.name)+
        (p.septic?'<em>septic</em>':'')+'</span>'+
        '<span class="hz-track'+(caught?"":" miss")+'">'+bar+onsetMark+'</span>'+v+'</div>';
    }).join("");
  }

  // The honesty paragraph already states the protocol in prose; repeating ev.protocol
  // after it said the same thing twice. Provenance only on the second line.
  var ev=d.evaluation||{}, prov=[];
  if(ev.generated) prov.push("last run "+ev.generated);
  if(ev.bootstrap) prov.push(ev.bootstrap+" bootstrap resamples");
  if(d.model_id) prov.push(d.model_id);
  $("hz-foot").innerHTML=esc(d.honesty||"")+
    (prov.length?'<i>'+esc(prov.join("  ·  "))+'</i>':'');
}

/* ============================================================
   PANEL — Per-Patient Baseline Learning
   Each bed's deviation from ITS OWN learned baseline (σ), ranked.
   This is the behavioural model made visible: the system learns every
   patient's personal norm and tracks how far each has drifted from it.
   ============================================================ */
var CH_LABEL={hr:"HR",temp_c:"T°",resp_rate:"RR",spo2:"SpO₂",map:"MAP",wbc:"WBC",lactate:"Lac"};
// Severity ramp: faint -> muted -> amber -> red. A mild 0.4-1.0σ excursion used to be
// drawn in the brand hue, so "very slightly off their own baseline" rendered as a
// saturated blue bar sitting louder than a genuine 1.5σ amber. Deviation below the
// clinical thresholds is now neutral — visible, but not making a claim.
function devCol(d){ return d>=2.0?C.crit:d>=1.0?C.watch:d>=0.4?C.muted:C.faint; }
function drawLearn(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var beds=Object.keys(latestBeds).map(function(k){return latestBeds[k];});
  ctx.textBaseline="middle"; ctx.textAlign="left";
  if(!beds.length){ ctx.fillStyle=C.dim; ctx.font=FS(11,500); ctx.textAlign="center";
    ctx.fillText("learning patient baselines …",w/2,h/2); return; }
  beds.sort(function(a,b){return (b.deviation||0)-(a.deviation||0);});
  var pad=13, top=8, DMAX=3.0;                    // 3σ from own baseline = full bar
  // header
  ctx.font=FS(8.5,700); ctx.fillStyle=C.dim;
  ctx.fillText("σ FROM OWN BASELINE", pad, top+5);
  ctx.textAlign="right"; ctx.fillText("LEARNED NORM ▸ NOW", w-pad, top+5); ctx.textAlign="left";
  var y0=top+20, avail=h-y0-6;
  var nshow=clamp(Math.floor(avail/20),3,Math.min(beds.length,9));
  var rowH=avail/nshow;
  var labW=86, barX=pad+labW, barW=w-pad-labW-150;
  for(var i=0;i<nshow;i++){ var b=beds[i], dev=b.deviation||0, mat=b.baseline_maturity||0;
    var cy=y0+i*rowH+rowH/2, col=devCol(dev), nm=(b.name||b.bed_id).split(",")[0];
    // maturity dot (filled ring = how learned this baseline is)
    ctx.beginPath(); ctx.arc(pad+4,cy,3.4,0,7); ctx.strokeStyle=withAlpha(C.dim,.7); ctx.lineWidth=1.2; ctx.stroke();
    if(mat>0){ ctx.beginPath(); ctx.moveTo(pad+4,cy); ctx.arc(pad+4,cy,3.4,-Math.PI/2,-Math.PI/2+7*mat); ctx.closePath();
      ctx.fillStyle=withAlpha(mat>=.99?C.ok:C.indigo,.9); ctx.fill(); }
    // name
    ctx.fillStyle=C.muted; ctx.font=FS(11,600); ctx.textAlign="left";
    ctx.fillText(nm.length>9?nm.slice(0,9):nm, pad+13, cy);
    // track + bar
    ctx.fillStyle="rgba(255,255,255,.04)"; roundRect(ctx,barX,cy-5,barW,10,5); ctx.fill();
    var fw=clamp(dev/DMAX,0,1)*barW;
    if(fw>1){ var g=ctx.createLinearGradient(barX,0,barX+fw,0);
      g.addColorStop(0,withAlpha(col,.5)); g.addColorStop(1,withAlpha(col,.95));
      ctx.fillStyle=g; roundRect(ctx,barX,cy-5,Math.max(fw,3),10,5); ctx.fill(); }
    // 1σ + 2σ ticks
    [1,2].forEach(function(t){ var tx=barX+(t/DMAX)*barW; ctx.strokeStyle=withAlpha(C.dim,.5);
      ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(tx,cy-6); ctx.lineTo(tx,cy+6); ctx.stroke(); });
    // value + top driver "HR 78 (was 60)"
    ctx.textAlign="left"; ctx.fillStyle=col; ctx.font=FM(11.5,700);
    ctx.fillText(dev.toFixed(1)+"σ", barX+barW+10, cy);
    var tp=(b.personal_top&&b.personal_top[0])?b.personal_top[0]:null;
    if(tp){ ctx.fillStyle=C.dim; ctx.font=FS(10,500);
      ctx.fillText((CH_LABEL[tp.ch]||tp.ch)+" ▸"+(tp.baseline!=null?" was "+tp.baseline:""), barX+barW+52, cy); }
  }
}

/* ============================================================
   PANEL — Personalization Map
   Scatter of population (qSOFA/SIRS) risk vs personalized risk.
   Above the diagonal = learning LIFTED the call (within-band drift
   caught early); below = learning DAMPENED it (chronic-abnormal at
   the patient's own baseline → fewer false alarms). Honest: most beds
   sit on the line — personalization only moves the ones it should.
   ============================================================ */
function drawBlind(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var beds=Object.keys(latestBeds).map(function(k){return latestBeds[k];});
  var padL=34,padR=12,padT=14,padB=24, pw=w-padL-padR, ph=h-padT-padB;
  function X(v){ return padL+v*pw; } function Y(v){ return padT+(1-v)*ph; }
  // frame + grid
  ctx.strokeStyle=C.grid; ctx.lineWidth=1;
  for(var g=0;g<=4;g++){ var gx=padL+pw*g/4, gy=padT+ph*g/4;
    ctx.beginPath(); ctx.moveTo(gx,padT); ctx.lineTo(gx,padT+ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padL,gy); ctx.lineTo(padL+pw,gy); ctx.stroke(); }
  // alarm thresholds (0.70) guide lines
  ctx.strokeStyle=withAlpha(C.crit,.25); ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(X(.7),padT); ctx.lineTo(X(.7),padT+ph); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(padL,Y(.7)); ctx.lineTo(padL+pw,Y(.7)); ctx.stroke();
  ctx.setLineDash([]);
  // diagonal (no change)
  ctx.strokeStyle=withAlpha(C.muted,.4); ctx.setLineDash([5,4]); ctx.lineWidth=1.2;
  ctx.beginPath(); ctx.moveTo(X(0),Y(0)); ctx.lineTo(X(1),Y(1)); ctx.stroke(); ctx.setLineDash([]);
  // axis labels
  ctx.fillStyle=C.dim; ctx.font=FS(9,600); ctx.textBaseline="middle";
  ctx.textAlign="center"; ctx.fillText("population risk →", padL+pw/2, padT+ph+13);
  ctx.save(); ctx.translate(11,padT+ph/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign="center"; ctx.fillText("personalized risk →",0,0); ctx.restore();
  // region hints
  ctx.font=FS(8.5,600); ctx.textAlign="left";
  ctx.fillStyle=withAlpha(C.ok,.6); ctx.fillText("↑ caught early", X(.04), Y(.92));
  ctx.textAlign="right"; ctx.fillStyle=withAlpha(C.cyan,.6); ctx.fillText("dampened ↓", X(.96), Y(.12));
  if(!beds.length){ return; }
  // dots
  beds.forEach(function(b){ var pop=clamp(b.population_risk||0,0,1), per=clamp(b.fused_risk||0,0,1);
    var x=X(pop), y=Y(per), tier=tierOf(per), col=tier==="critical"?C.crit:tier==="watch"?C.watch:C.cyan;
    var moved=Math.abs(per-pop);
    if(moved>0.03){ ctx.strokeStyle=withAlpha(per>pop?C.ok:C.cyan,.5); ctx.lineWidth=1.4;
      ctx.beginPath(); ctx.moveTo(X(pop),Y(pop)); ctx.lineTo(x,y); ctx.stroke(); }
    ctx.beginPath(); ctx.arc(x,y,moved>0.03?4.6:3.4,0,7);
    ctx.fillStyle=withAlpha(col,.92); ctx.shadowColor=col; ctx.shadowBlur=moved>0.03?9:0; ctx.fill(); ctx.shadowBlur=0;
    ctx.strokeStyle=withAlpha("#0a0e1c",.6); ctx.lineWidth=1; ctx.stroke();
  });
}

function drawLeadHist(s){ var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h); if(!valData) return;
  var leads=valData.summary.leads||[]; ctx.textBaseline="middle";
  if(!leads.length){ ctx.fillStyle=C.dim; ctx.font=FS(11,500); ctx.textAlign="center"; ctx.fillText("no detections",w/2,h/2); return; }
  var bins=[[0,6],[6,12],[12,24],[24,48],[48,1e9]], labels=["0–6","6–12","12–24","24–48","48 h+"];
  var counts=bins.map(function(b){return leads.filter(function(l){return l>=b[0]&&l<b[1];}).length;});
  // padT has to clear the VALUE LABEL, not just the bar. The tallest bar is by definition
  // full height, so its top lands exactly at padT and its count is drawn 5px above that in
  // a 14px face — at padT=16 the glyph started around y=-3 and the number was sliced off
  // the top of the canvas, on the one bar the eye goes to first.
  var mx=Math.max.apply(null,counts)||1, padL=14,padR=14,padT=32,padB=26,bw=(w-padL-padR)/bins.length;
  counts.forEach(function(c,i){ var bh=(c/mx)*(h-padT-padB), x=padL+i*bw, y=h-padB-bh;
    var g=ctx.createLinearGradient(0,y,0,h-padB); g.addColorStop(0,withAlpha(C.cyan,.95)); g.addColorStop(1,withAlpha(C.cyan,.32));
    ctx.fillStyle=g; roundRect(ctx,x+7,y,bw-14,Math.max(bh,2),5); ctx.fill();
    ctx.textAlign="center"; ctx.fillStyle=C.ink; ctx.font=FM(14,600); ctx.textBaseline="bottom"; if(c) ctx.fillText(c,x+bw/2,y-5);
    ctx.fillStyle=C.dim; ctx.font=FS(10,500); ctx.textBaseline="top"; ctx.fillText(labels[i],x+bw/2,h-padB+9); }); }

function drawSweep(s){ var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  if(!valData||!valData.summary.sweep) return; var sw=valData.summary.sweep, n=sw.length; if(!n) return;
  var padL=36,padR=14,padT=24,padB=28,pw=w-padL-padR,ph=h-padT-padB, t0=sw[0].thr, t1=sw[n-1].thr;
  function xAt(t){ return padL+(t-t0)/(t1-t0)*pw; } function yAt(v){ return padT+ph-v*ph; }
  ctx.textBaseline="middle"; ctx.font=FS(9,500);
  [0,0.5,1].forEach(function(v){ var y=yAt(v); ctx.strokeStyle=C.grid; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(w-padR,y); ctx.stroke();
    ctx.fillStyle=C.dim; ctx.textAlign="right"; ctx.fillText(Math.round(v*100)+"%",padL-5,y); });
  var op=valData.summary.alarm_threshold||0.7, ox=xAt(op);
  ctx.strokeStyle=withAlpha(C.ink,.22); ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(ox,padT); ctx.lineTo(ox,h-padB); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.muted; ctx.font=FS(8.5,600); ctx.textAlign="center"; ctx.textBaseline="top";
  ctx.fillText("operating "+op,ox,h-padB+9);
  ctx.fillStyle=C.dim; ctx.font=FS(9,500);
  [t0,(t0+t1)/2,t1].forEach(function(t){ if(Math.abs(t-op)>0.06) ctx.fillText(t.toFixed(2),xAt(t),h-padB+9); });
  function line(key,col,glow){ ctx.strokeStyle=col; ctx.lineWidth=1.9; ctx.lineJoin="round";
    if(glow){ ctx.shadowColor=withAlpha(col,.5); ctx.shadowBlur=6; } ctx.beginPath();
    sw.forEach(function(p,i){ var x=xAt(p.thr),y=yAt(p[key]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke(); ctx.shadowBlur=0;
    sw.forEach(function(p){ ctx.fillStyle=col; ctx.beginPath(); ctx.arc(xAt(p.thr),yAt(p[key]),1.7,0,7); ctx.fill(); }); }
  line("sens",C.ok,true); line("fpr",C.crit,false);
  ctx.textAlign="left"; ctx.textBaseline="middle"; ctx.font=FS(9.5,600);
  ctx.fillStyle=C.ok; ctx.fillRect(padL,padT-14,8,8); ctx.fillStyle=C.muted; ctx.fillText("sensitivity",padL+12,padT-10);
  ctx.fillStyle=C.crit; ctx.fillRect(padL+96,padT-14,8,8); ctx.fillStyle=C.muted; ctx.fillText("false positive",padL+108,padT-10); }

/* ============================================================
   ROSTER VIEW
   ============================================================ */
var rosterFilter="all", rosterQuery="";
function renderRoster(){ var host=$("roster-list"); if(!host) return;
  var beds=Object.keys(latestBeds).map(function(k){return latestBeds[k];});
  if(!beds.length){ host.innerHTML='<div style="color:var(--dim);font-size:13px">awaiting cohort …</div>'; return; }
  beds.sort(function(a,b){return b.fused_risk-a.fused_risk;});
  var q=rosterQuery.toLowerCase(), html="";
  beds.forEach(function(b){ var tier=tierOf(b.fused_risk);
    if(rosterFilter==="septic"&&!b.will_be_septic) return;
    if(rosterFilter==="stable"&&b.will_be_septic) return;
    if(rosterFilter==="critical"&&tier!=="critical") return;
    if(q&&(b.name+" "+(b.mrn||"")).toLowerCase().indexOf(q)<0) return;
    html+='<div class="pcard '+tier+'" data-id="'+b.bed_id+'"><div class="pcard-top">'+
      '<span class="pcard-name">'+b.name+'</span><span class="pcard-risk">'+fmt(b.fused_risk,2)+'</span></div>'+
      '<div class="pcard-meta">'+(b.mrn||"—")+(b.age!=null?' · '+Math.round(b.age)+'y '+(b.gender||''):'')+
        (b.icu_hour!=null?' · ICU h'+b.icu_hour+'/'+b.los_total:'')+'</div><div class="pcard-foot">'+
        (b.will_be_septic?'<span class="pchip sep">sepsis h'+b.onset_hour+'</span>':'<span class="pchip ok">no sepsis</span>')+
        '<span class="pchip">'+(b.primary_system==="none"?"nominal":b.primary_system)+'</span></div></div>'; });
  host.innerHTML=html||'<div style="color:var(--dim);font-size:13px">no patients match.</div>'; }

function animateCounts(){ [].forEach.call(document.querySelectorAll("[data-count]"),function(el){
  if(el.dataset.done) return; el.dataset.done="1";
  var tgt=parseFloat(el.dataset.count)||0, u=el.querySelector("small"), us=u?u.outerHTML:"", t0=performance.now();
  (function tick(now){ var p=Math.min(1,(now-t0)/850), e=1-Math.pow(1-p,3), v=tgt*e;
    el.innerHTML=(tgt%1?v.toFixed(1):Math.round(v))+us; if(p<1) requestAnimationFrame(tick); })(t0); }); }

/* ============================================================
   COMMAND CONSOLE — direct conversational line to the bio-kernel
   ============================================================ */
var lastSnap=null, cmdHist=[], cmdHi=0, cmdOpen=false;
function cmdShow(){ cmdOpen=true; $("cmd").classList.add("open"); $("cmd").setAttribute("aria-hidden","false");
  if(!$("cmd-out").childNodes.length) cmdBanner();
  if(typeof termSetPrompt==="function") termSetPrompt();
  setTimeout(function(){ var i=$("cmd-in"); if(i){ i.focus(); if(typeof termRender==="function") termRender(); } },30); }
function cmdHide(){ cmdOpen=false; $("cmd").classList.remove("open"); $("cmd").setAttribute("aria-hidden","true"); }
function cmdToggle(){ cmdOpen?cmdHide():cmdShow(); }
function esc(s){ return String(s==null?"":s).replace(/[&<>]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;"}[c];}); }
function cmdPrint(html,cls){ var o=$("cmd-out"); var d=document.createElement("div");
  d.className="cl"+(cls?" "+cls:""); d.innerHTML=html; o.appendChild(d); o.scrollTop=o.scrollHeight; }
function cmdEcho(line){ cmdPrint('<span class="cl-pr">aegis@icu-wing:'+esc(termCwd)+'$</span> '+esc(line),"echo"); }
function cmdBanner(){ cmdPrint('<b>AegisOS</b> tty — direct line to the distributed bio-kernel (L1–L4 · HDSched · HDKernel).','sys');
  cmdPrint('<b>help</b> lists commands · <b>Tab</b> completes · <b>Ctrl-L</b> clears · <b>↑/↓</b> history · <b>ls /proc</b> to browse state.','dim'); }
var TIER_TXT={critical:'<span class="cl-crit">CRITICAL</span>',watch:'<span class="cl-warn">WATCH</span>',stable:'<span class="cl-ok">stable</span>'};
function beds(){ return Object.keys(latestBeds).map(function(k){return latestBeds[k];}); }
function resolveBed(q){ if(!q) return null; q=q.toLowerCase().trim();
  var bs=beds();
  var hit=bs.filter(function(b){return b.bed_id.toLowerCase()===q;})[0]; if(hit) return hit;
  var m=q.match(/^(?:bed[ _-]?)?0*(\d+)$/);          // "7", "bed 7", "07"
  if(m){ var idx=+m[1]-1; hit=bs.filter(function(b){var mm=b.bed_id.match(/(\d+)/);return mm&&+mm[1]===idx;})[0]; if(hit) return hit; }
  hit=bs.filter(function(b){return (b.name||"").toLowerCase().indexOf(q)>=0;}); return hit.length?hit[0]:null;
}
function cmdRun(raw){
  var line=raw.trim(); if(!line) return; cmdEcho(line);
  var parts=line.split(/\s+/), c=parts[0].toLowerCase(), arg=parts.slice(1).join(" ");
  var H=CMD[c]||CMD[CMD_ALIAS[c]];
  if(!H){ cmdPrint('unknown command <b>'+esc(c)+'</b> — try <b>help</b>',"err"); return; }
  try{ H(arg); }catch(e){ cmdPrint("error: "+esc(e&&e.message),"err"); }
}
function fmtRisk(b){ var t=tierOf(b.fused_risk); return '<span class="cl-'+(t==="critical"?"crit":t==="watch"?"warn":"ok")+'">'+fmt(b.fused_risk,2)+'</span>'; }
var CMD_ALIAS={ls:"roster","?":"help",inspect:"who",inject:"crisis",cls:"clr",sched:"alloc",bundle:"protocol",p:"protocol",ews:"radar",pk:"project",pharmacokinetics:"project",twin:"project",resilience:"radar",kernel:"kstat",log:"dmesg",order:"treat",rx:"treat",esc:"escalate",syslog:"audit",lsdrv:"drivers",kdev:"drivers",devplane:"drivers",vrf:"verify",kill:"fault",unplug:"fault",xmr:"resonance",agree:"resonance",ppg:"vascular",tone:"vascular",vaso:"vasodilate",svc:"service",systemctl:"service",services:"service",sysfs:"proc",hdk:"krun",kthreads:"krun",query:"ask",q:"ask",handoff:"brief",fit:"train",ml:"model",models:"model",pt:"patient",patientos:"patient"};
var CMD={
  help:function(){ cmdPrint('<b>COMMANDS</b>','sys');
    [['roster / ls','list every monitored bed + risk'],
     ['top','beds ranked by risk (critical first)'],
     ['who <bed>','patient summary — vitals, drivers, ground truth'],
     ['open <bed>','open the full clinical dossier'],
     ['patient <bed>','open the patient OS — customize this patient’s tracking (plain English)'],
     ['alloc / sched','live compute allocation across the cohort'],
     ['ps','bio-process table (beds as scheduled processes)'],
     ['top','live kernel stats + process table'],
     ['kstat','kernel scheduler summary'],
     ['dmesg','kernel interrupt / preemption log'],
     ['krun [wl] [pol]','run a program on the live HDKernel — wl: mixed|pipeline|fanout|stress · pol: FIFO|RR|CFS|EDF|HDSched'],
     ['train <name> [set] [n]','TRAIN a real detector on CinC-2019 (grouped CV) — set: wearable|lab'],
     ['specialize <name> <Nh>','SPECIALIZE an early-warning model that fires within N hours of onset'],
     ['model [use|job]','ML control plane — registered models + metrics · model use <name> · model job <id>'],
     ['ask <question>','grounded LLM Q&A over live cohort + kernel state (read-only)'],
     ['brief','LLM shift-handoff summary of the cohort'],
     ['explain <bed>','why the system thinks what it thinks'],
     ['radar <bed>','early-warning resilience radar (critical slowing down)'],
     ['resonance <bed>','cross-modal agreement gate (artifact-robust evidence)'],
     ['vascular','live PPG morphology — dicrotic-notch vascular tone'],
     ['vasodilate <0..1>','dial the vasodilation model — watch the notch erode'],
     ['— aegisd app/service plane —','managed, supervised, capability-confined apps'],
     ['service','list managed services + health (service <name> stop|restart)'],
     ['pkg','package registry — pkg install <name> (try: tachy, rogue)'],
     ['proc <path>','read the /proc introspection namespace (proc triage, proc sched)'],
     ['protocol <bed>','certified Hour-1 sepsis bundle'],
     ['— write syscalls —','audited · Z3-gated · default-deny'],
     ['treat <bed> [k=v]','order the certified bundle (fluid=/vaso=/abx= override)'],
     ['project <bed>','render the ordered treatment travelling through the body'],
     ['escalate <bed> [lvl]','raise priority — pins kernel compute if justified'],
     ['crisis <bed> [int]','⚠ privileged crisis drill (envelope-gated)'],
     ['clear <bed>','recover a bed · release the pin'],
     ['audit / syslog','append-only syscall log + prover verdicts'],
     ['— aegis-eBPF device plane —','input safety · real beds on verified drivers'],
     ['verify <driver>','run the static verifier on driver bytecode (try: loop, escape, ecg)'],
     ['drivers / lsdrv','beds whose optical sensor runs verified bytecode + supervisor state'],
     ['fault <bed>','⚠ kill a verified bed sensor — watch the supervisor fail over'],
     ['recover <bed>','re-verify + re-arm a quarantined optical driver'],
     ['clr','clear the console']
    ].forEach(function(r){ cmdPrint('<span class="cl-cmd">'+r[0]+'</span><span class="cl-d">'+r[1]+'</span>',"row"); }); },
  roster:function(){ var bs=beds().sort(function(a,b){return b.fused_risk-a.fused_risk;});
    if(!bs.length) return cmdPrint("cohort not yet connected…","dim");
    cmdPrint('<b>'+bs.length+' MONITORED BEDS</b>','sys');
    bs.forEach(function(b){ cmdPrint('<span class="cl-tag">'+bedTag(b.bed_id)+'</span> '+
      '<span class="cl-nm">'+esc(b.name)+'</span> '+fmtRisk(b)+' '+TIER_TXT[tierOf(b.fused_risk)]+
      (b.will_be_septic?' <span class="cl-d">· sepsis h'+b.onset_hour+'</span>':''),"row"); }); },
  top:function(){ var bs=beds().sort(function(a,b){return b.fused_risk-a.fused_risk;}).slice(0,6);
    cmdPrint('<b>HIGHEST RISK</b>','sys');
    bs.forEach(function(b){ cmdPrint('<span class="cl-tag">'+bedTag(b.bed_id)+'</span> '+esc(b.name)+'  '+fmtRisk(b)+'  '+TIER_TXT[tierOf(b.fused_risk)],"row"); }); },
  who:function(a){ var b=resolveBed(a); if(!b) return cmdPrint("no bed matches '"+esc(a)+"'","err");
    var e=b.edge||{}, en=b.enterprise||{};
    cmdPrint('<b>'+esc(b.name)+'</b> · '+bedTag(b.bed_id)+' · '+(b.age?Math.round(b.age)+'y '+(b.gender||''):'')+' · ICU h'+b.icu_hour,"sys");
    cmdPrint('fused risk '+fmtRisk(b)+' '+TIER_TXT[tierOf(b.fused_risk)]+' · population '+fmt(b.population_risk,2)+' · Δbaseline '+fmt(b.deviation,1)+'σ',"row");
    cmdPrint('HR '+fmt(e.hr,0)+' · MAP '+fmt(en.map,0)+' · T '+fmt(en.temp_c,1)+' · SpO₂ '+fmt(en.spo2,0)+' · RR '+fmt(en.resp_rate,0)+' · Lac '+fmt(en.lactate,1)+' · WBC '+fmt(en.wbc,1),"mono");
    if(b.drivers&&b.drivers.length) cmdPrint('drivers: '+b.drivers.map(esc).join(' · '),"row");
    cmdPrint(b.will_be_septic?('<span class="cl-crit">ground truth: develops sepsis at ICU h'+b.onset_hour+'</span>'):'<span class="cl-ok">ground truth: no sepsis this admission</span>',"row"); },
  open:function(a){ var b=resolveBed(a); if(!b) return cmdPrint("no bed matches '"+esc(a)+"'","err");
    cmdHide(); openDossier(b.bed_id); cmdPrint("opening dossier · "+esc(b.name),"ok"); },
  patient:function(a){ var b=resolveBed(a); if(!b) return cmdPrint("usage: <b>patient &lt;bed&gt;</b> — open the patient's own OS + tracking terminal","err");
    cmdPrint('<span class="cl-ok">▸ booting patient terminal · '+esc(b.name)+'</span> <span class="cl-d">— history · synthesis · tell it what to watch (plain English)</span>',"sys");
    cmdHide(); openDossier(b.bed_id);
    setTimeout(function(){ pttModalShow(); }, 260); },
  alloc:function(){ var ns=(lastSnap&&lastSnap.topology)||[];
    cmdPrint('<b>COGNITIVE SCHEDULER</b> · mode '+((lastSnap&&lastSnap.scheduler&&lastSnap.scheduler.mode)||"—")+' · lane budget '+(V.budget||"—"),"sys");
    if(lastSnap&&lastSnap.scheduler) cmdPrint('Σ cost '+fmt(lastSnap.scheduler.cost,2)+(lastSnap.scheduler.owner?' · preempt owner '+esc(cleanName(lastSnap.scheduler.owner)):''),"row");
    ns.slice().sort(function(a,b){return b.lanes-a.lanes;}).forEach(function(n){
      var pct=V.budget?Math.round(n.lanes/V.budget*100):0, bar='█'.repeat(Math.max(0,Math.round(pct/6)));
      cmdPrint('<span class="cl-tag">'+esc(cleanName(n.name)||n.node_id)+'</span> <span class="cl-cmd">'+String(n.lanes).padStart(3)+'</span> lanes <span class="cl-d">'+bar+' '+pct+'% · p='+fmt(n.p,2)+' · '+esc(n.tier).toLowerCase()+'</span>',"mono"); }); },
  explain:function(a){ var b=resolveBed(a); if(!b) return cmdPrint("no bed matches '"+esc(a)+"'","err");
    cmdPrint('<b>REASONING · '+esc(b.name)+'</b>','sys');
    cmdPrint('1. fusion of 3 tiers (optical + bedside + labs) → population risk '+fmt(b.population_risk,2),"row");
    cmdPrint('2. per-patient learning: '+fmt(b.deviation,1)+'σ from own baseline (maturity '+Math.round((b.baseline_maturity||0)*100)+'%) → personalized '+fmtRisk(b),"row");
    if(b.hdc&&b.hdc.k) cmdPrint('3. HDC case memory: '+(b.hdc.n_septic||0)+'/'+b.hdc.k+' nearest real ICU bio-states became septic'+(b.hdc.lead_h!=null?(' (median lead '+Math.round(b.hdc.lead_h)+'h)'):'')+' · '+Math.round((b.hdc.resemblance||0)*100)+'% resemblance',"row");
    var inf=inferInfection(b); if(inf) cmdPrint('4. suspected focus: '+inf.site+' (inferred from vitals · '+inf.conf+' confidence)',"row");
    if(b.ews&&b.ews.trace) cmdPrint('5. early-warning radar (CSD): resilience-loss '+Math.round((b.ews.ews||0)*100)+'%'+(b.ews.rising&&!b.ews.warming?' <span class="cl-warn">▲ rising</span>':'')+' <span class="cl-d">· research signal</span>',"row");
    if(b.drivers&&b.drivers.length) cmdPrint('top derangements: '+b.drivers.slice(0,3).map(esc).join(' · '),"dim"); },
  radar:function(a){ var b=resolveBed(a); if(!b) return cmdPrint("no bed matches '"+esc(a)+"'","err");
    var ew=b.ews||{};
    cmdPrint('<b>EARLY-WARNING RADAR · '+esc(b.name)+'</b> <span class="cl-d">— Critical Slowing Down</span>','sys');
    if(!ew.trace) return cmdPrint("radar not available for this bed","dim");
    cmdPrint('resilience-loss <b>'+Math.round((ew.ews||0)*100)+'%</b> '+(ew.rising&&!ew.warming?'<span class="cl-warn">▲ rising</span>':'')+(ew.warming?'<span class="cl-d"> (calibrating)</span>':''),"row");
    cmdPrint('rising autocorrelation '+fmt(ew.ar1,2)+' · widening variance '+fmt(ew.var,2)+' (vs own baseline)',"mono");
    var sp="▁▂▃▄▅▆▇█", mx=Math.max(0.3,Math.max.apply(null,ew.trace));
    cmdPrint('<span class="cl-cmd">'+ew.trace.map(function(v){return sp[Math.min(7,Math.max(0,Math.floor(v/mx*7)))];}).join("")+'</span>',"mono");
    cmdPrint('watches signal <b>dynamics</b>, not levels — a complementary research signal','dim'); },
  protocol:function(a){ var b=resolveBed(a); if(!b) return cmdPrint("no bed matches '"+esc(a)+"'","err");
    cmdPrint("fetching certified Hour-1 bundle · "+esc(b.name)+" …","dim");
    fetch(window.AEGIS.http("/aegis/cohort/"+b.bed_id+"/bundle")).then(function(r){return r.json();}).then(function(d){ printBundle(d,b); }).catch(function(){ cmdPrint("bundle unavailable (backend waking?)","err"); }); },
  treat:function(a){ var s=splitBedArgs(a), b=resolveBed(s.bedq);
    if(!b) return cmdPrint("usage: treat <bed> [fluid=N] [vaso=N] [abx=N]","err");
    var args={}; s.rest.forEach(function(tok){ var kv=tok.split("="); if(kv.length!==2) return;
      var k=kv[0].toLowerCase(), v=parseFloat(kv[1]);
      if(k==="fluid"||k==="fluid_rate") args.fluid_rate=v;
      else if(k==="vaso"||k==="vasopressor_dose") args.vasopressor_dose=v;
      else if(k==="abx"||k==="antibiotic"||k==="antibiotic_code") args.antibiotic_code=v; });
    syscall("treat", b.bed_id, args, function(d){
      if(!d.allowed) return;
      if(d.result&&d.result.bundle) printBundle(d.result.bundle,b);
      // A certified order now has a destination. The backend returns the handle;
      // offer it rather than seizing the screen — the operator may be mid-sweep
      // across the wing and an overlay that opens itself is an interruption.
      if(d.projection&&d.projection.available)
        cmdPrint('order dispatched → <span class="cl-cmd">project '+esc(b.bed_id)+
          '</span> to watch it travel through the body',"row"); }); },
  project:function(a){ var b=resolveBed(a);
    if(!b) return cmdPrint("usage: project <bed> — render the ordered treatment through the body","err");
    if(!window.AegisPK) return cmdPrint("treatment twin not loaded on this page","err");
    cmdPrint("opening the treatment projection · "+esc(b.name)+" …","dim");
    window.AegisPK.open(b.bed_id,{name:b.name}); },
  escalate:function(a){ var s=splitBedArgs(a), b=resolveBed(s.bedq);
    if(!b) return cmdPrint("usage: escalate <bed> [low|medium|high|0-3]","err");
    var args={}, lv=s.rest[0];
    if(lv){ var M={none:0,low:1,medium:2,watch:2,high:3,critical:3,severe:3}, n=M[lv.toLowerCase()];
      if(n==null) n=parseInt(lv,10); if(!isNaN(n)) args.level=n; }
    syscall("escalate", b.bed_id, args); },
  crisis:function(a){ var s=splitBedArgs(a), b=resolveBed(s.bedq);
    if(!b) return cmdPrint("no bed matches '"+esc(s.bedq)+"'","err");
    var intensity=parseFloat(s.rest[0]); if(isNaN(intensity)) intensity=1.0;
    syscall("crisis", b.bed_id, {intensity:intensity}, function(d){
      if(!d.allowed) return;
      cmdPrint('fast-forwarding to sepsis onset — watch the grid re-prioritise…',"row");
      setTimeout(function(){ fetch(window.AEGIS.http("/aegis/cohort/"+b.bed_id+"/bundle")).then(function(r){return r.json();}).then(function(d){ printBundle(d,b); }).catch(function(){}); }, 1400); }); },
  clear:function(a){ var s=splitBedArgs(a), b=resolveBed(s.bedq);
    if(!b) return cmdPrint("no bed matches '"+esc(s.bedq)+"'","err");
    syscall("clear", b.bed_id, {}); },
  audit:function(){ fetch(window.AEGIS.http("/aegis/syscall/audit?limit=20")).then(function(r){return r.json();}).then(function(d){
    var log=(d&&d.log)||[]; if(!log.length) return cmdPrint("syscall audit log empty — no writes issued yet","dim");
    cmdPrint('<b>SYSCALL AUDIT LOG</b> <span class="cl-d">· '+d.status.allowed+' allowed · '+d.status.denied+' denied · Z3-gated</span>',"sys");
    log.forEach(function(e){ cmdPrint('<span class="cl-cmd">#'+("00"+e.sid).slice(-3)+'</span> '+
      pad("sys_"+e.syscall,13)+(e.bed?pad(bedTag(e.bed),8):pad("",8))+
      (e.allowed?'<span class="cl-ok">'+esc(e.errno_name)+'</span>':'<span class="cl-crit">'+esc(e.errno_name)+'</span>')+
      ' <span class="cl-d">'+esc(e.egress||"—")+' · Z3 '+esc(e.z3_status)+'</span>',"mono"); });
    }).catch(function(){ cmdPrint("audit log unavailable (backend waking?)","err"); }); },
  noise:function(a){ var b=resolveBed(a); if(!b) return cmdPrint("no bed matches '"+esc(a)+"'","err");
    chaos(b.bed_id,"noise"); cmdPrint("injecting sensor noise surge · "+esc(b.name)+" — watch the resonance gate reject the camera pathway","warn"); },
  resonance:function(a){ var b=resolveBed(a); if(!b) return cmdPrint("no bed matches '"+esc(a)+"'","err");
    var r=b.resonance; if(!r||!r.per_modality) return cmdPrint("resonance calibrating…","dim");
    cmdPrint('<b>CROSS-MODAL RESONANCE · '+esc(b.name)+'</b> <span class="cl-d">— each pathway projected onto the sepsis axis</span>',"sys");
    var nm={camera:"camera·rPPG",wearable:"wearable·PPG",ehr:"EHR·labs"};
    Object.keys(r.per_modality).forEach(function(k){ var z=r.per_modality[k], rej=(r.uncorroborated_pathway===k);
      var n=Math.max(0,Math.min(12,Math.round((z+3)/6*12)));
      cmdPrint(pad(nm[k]||k,14)+'<span class="cl-'+(rej?"warn":"cmd")+'">'+"█".repeat(n)+"·".repeat(12-n)+'</span> '+
        (z>=0?"+":"")+z.toFixed(1)+'σ'+(rej?' <span class="cl-warn">⚠ uncorroborated</span>':''),"mono"); });
    cmdPrint('ungated '+Math.round((r.naive_risk||0)*100)+'% → gated <b>'+Math.round((r.gated_risk||0)*100)+'%</b> · resonance '+Math.round((r.resonance||0)*100)+'%',"row");
    cmdPrint(esc(r.explain||""), r.isolated?"warn":"dim"); },
  vascular:function(){ fetch(window.AEGIS.http("/aegis/ppg/morphology")).then(function(r){return r.json();}).then(function(d){
    if(!d.available) return cmdPrint("PPG monitor unavailable — BIDMC waveform not present","dim");
    cmdPrint('<b>VASCULAR TONE · PPG MORPHOLOGY</b> <span class="cl-d">— real BIDMC waveform '+esc(d.record)+' (replay)'+(d.hr?' · '+Math.round(d.hr)+' bpm':'')+'</span>',"sys");
    var w=d.wave||[], cols=72, sp="▁▂▃▄▅▆▇█", line="";
    for(var i=0;i<cols;i++){ var v=w.length?w[Math.floor(i/cols*(w.length-1))]:0; line+=sp[Math.min(7,Math.max(0,Math.floor((v||0)*7)))]; }
    cmdPrint('<span class="cl-cmd">'+line+'</span>',"mono");
    var tc=d.tone==="preserved"?"ok":d.tone==="reduced"?"warn":"crit";
    cmdPrint('tone <span class="cl-'+tc+'">'+esc((d.tone||"?").toUpperCase())+'</span> · dicrotic-notch <b>'+Math.round((d.notch_rate||0)*100)+'%</b> · reflection '+(d.reflection_index||0).toFixed(3)+' · b/a '+(d.ba_ratio||0).toFixed(2),"row");
    if(d.alpha>0) cmdPrint('<span class="cl-warn">⚠ vasodilation model active ('+Math.round(d.alpha*100)+'%)</span> · <span class="cl-cmd">vasodilate 0</span> to restore',"row");
    cmdPrint('dial <span class="cl-cmd">vasodilate 0..1</span> — watch the notch flatten and tone collapse (the distributive-shock signature)','dim');
    }).catch(function(){ cmdPrint("vascular monitor unreachable","err"); }); },
  service:function(a){ var t=(a||"").trim().split(/\s+/).filter(Boolean);
    if(t.length>=2){ var name=t[0], action=t[1].toLowerCase();
      fetch(window.AEGIS.http("/aegis/services/"+name+"/"+action),{method:"POST"}).then(function(r){return r.json();}).then(function(d){
        if(d.status!=="ok") return cmdPrint(esc(d.detail||"service error"),"err");
        cmdPrint('<span class="cl-ok">✓ '+esc(name)+' → '+esc(action)+'</span> · state '+esc(d.state),"sys");
      }).catch(function(){ cmdPrint("service control failed","err"); }); return; }
    fetch(window.AEGIS.http("/aegis/services")).then(function(r){return r.json();}).then(function(d){
      cmdPrint('<b>aegisd · SERVICE PLANE</b> <span class="cl-d">— '+(d.running||0)+'/'+(d.n||0)+' running'+(d.bed?' · monitoring '+bedTag(d.bed):'')+'</span>',"sys");
      cmdPrint(pad("SERVICE",12)+pad("STATE",11)+pad("HEALTH",8)+pad("EMITS",10)+"f/r","dim");
      (d.services||[]).forEach(function(s){ var c=s.state==="RUNNING"?"ok":s.state==="QUARANTINED"?"err":"dim";
        cmdPrint(pad(s.name,12)+'<span class="cl-'+c+'">'+pad(s.state,11)+'</span>'+
          (s.healthy?'<span class="cl-ok">'+pad("ok",8)+'</span>':'<span class="cl-warn">'+pad("degraded",8)+'</span>')+
          pad((s.caps.emits.join(",")||"—"),10)+s.faults+"/"+s.restarts,"mono");
        if(s.depends&&s.depends.length) cmdPrint('<span class="cl-d">  └ depends: '+s.depends.join(", ")+'</span>',"dim"); });
      var ev=(d.events||[]); if(ev.length){ cmdPrint('<span class="cl-d">— supervisor —</span>',"dim");
        ev.slice(-5).forEach(function(e){ var c=e.kind==="FAULT"?"err":e.kind==="RESTART"?"warn":e.kind==="QUARANTINE"?"err":"ok";
          cmdPrint('<span class="cl-'+c+'">['+pad(e.kind,9)+']</span> '+pad(e.svc,10)+'<span class="cl-d">'+esc(e.detail)+'</span>',"mono"); }); }
      cmdPrint('control: <span class="cl-cmd">service &lt;name&gt; stop|restart</span> · introspect: <span class="cl-cmd">proc &lt;path&gt;</span>',"dim");
    }).catch(function(){ cmdPrint("service plane unreachable","err"); }); },
  proc:function(a){ var path=(a||"").trim();
    if(!path){ fetch(window.AEGIS.http("/aegis/proc")).then(function(r){return r.json();}).then(function(d){
      cmdPrint('<b>/proc</b> <span class="cl-d">— AegisOS introspection namespace</span>',"sys");
      (d.paths||[]).forEach(function(p){ cmdPrint('<span class="cl-cmd">'+esc(p)+'</span>',"mono"); });
      cmdPrint('read one: <span class="cl-cmd">proc &lt;path&gt;</span> (e.g. proc triage, proc sched)',"dim");
    }).catch(function(){ cmdPrint("/proc unreachable","err"); }); return; }
    fetch(window.AEGIS.http("/aegis/proc/"+path.replace(/^\/?proc\/?/,""))).then(function(r){return r.json();}).then(function(d){
      if(d.error) return cmdPrint(esc(d.path)+": "+esc(d.error),"err");
      cmdPrint('<span class="cl-cmd">'+esc(d.path)+'</span>',"sys");
      cmdPrint('<span class="cl-d">'+esc(JSON.stringify(d.value,null,1)).slice(0,1400)+'</span>',"mono");
    }).catch(function(){ cmdPrint("proc read failed","err"); }); },
  krun:function(a){ var t=(a||"").trim().split(/\s+/).filter(Boolean);
    var wl=(t[0]||"mixed").toLowerCase(), pol=t[1]||"HDSched";
    var POL={fifo:"FIFO",rr:"RR",cfs:"CFS",edf:"EDF",hdsched:"HDSched",hd:"HDSched"};
    pol=POL[pol.toLowerCase()]||pol;
    cmdPrint('<span class="cl-d">spawning workload <b>'+esc(wl)+'</b> under <b>'+esc(pol)+'</b> on the live kernel…</span>',"dim");
    fetch(window.AEGIS.http("/aegis/hdkernel/run"),{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({workload:wl,policy:pol})}).then(function(r){return r.json();}).then(function(d){
        var s=d.summary||{}, top=d.top||{}, ps=d.ps||[];
        cmdPrint('<b>HDKERNEL</b> <span class="cl-d">· live cooperative kernel · '+esc(d.workload)+' @ '+esc(d.policy)+'</span>','sys');
        cmdPrint('procs '+s.processes+' · <span class="cl-ok">'+s.completed+' done</span>'+(s.deadlocked?' · <span class="cl-crit">DEADLOCK</span>':'')+
          ' · clock '+s.clock_us+'µs · idle '+s.idle_us+'µs · util '+Math.round((top.utilization||0)*100)+'%',"row");
        cmdPrint('dispatches '+s.dispatches+' · context switches '+s.context_switches+' · mean turnaround '+s.mean_turnaround_us+'µs',"mono");
        cmdPrint(pad("PID",5)+pad("PROCESS",12)+pad("STATE",7)+pad("PRIO",5)+pad("CPUµs",8)+pad("CTX",5)+pad("TURNµs",8)+"DDL","dim");
        ps.forEach(function(p){ var cls=p.state==="DONE"?"ok":p.state==="BLOCKED"?"err":"row";
          cmdPrint(pad(p.pid,5)+pad(esc(p.name),12)+pad(p.state.toLowerCase(),7)+pad(p.prio,5)+pad(p.cpu_us,8)+
            pad(p.ctx,5)+pad(p.turnaround_us==null?"—":p.turnaround_us,8)+(p.deadline_us==null?"—":p.deadline_us+"µs"), cls); });
        cmdPrint('try another policy: <span class="cl-cmd">krun '+esc(wl)+' edf</span> · workloads: '+(d.workloads||[]).join(" · "),"dim");
      }).catch(function(){ cmdPrint("hdkernel unreachable","err"); }); },
  ask:function(a){ var q=(a||"").trim();
    if(!q) return cmdPrint('usage: <span class="cl-cmd">ask &lt;question&gt;</span> — e.g. ask which beds are critical? · ask what is the kernel doing?','dim');
    cmdPrint('<span class="cl-d">◇ '+esc(q)+'</span>','dim');
    cmdPrint('<span class="cl-d">grounding on live cohort + kernel state…</span>','dim');
    fetch(window.AEGIS.http("/aegis/llm/ask"),{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question:q})}).then(function(r){return r.json();}).then(function(d){
        var src=d.source==="llm"?('<span class="cl-ok">'+esc(d.model||"llm")+'</span>'):'<span class="cl-d">template · LLM rate-limited</span>';
        cmdPrint('<b>AEGIS ASSISTANT</b> <span class="cl-d">· grounded · read-only · '+src+'</span>','sys');
        String(d.answer||"").split(/\n+/).forEach(function(ln){ if(ln.trim()) cmdPrint(esc(ln.trim()),"row"); });
        cmdPrint('<span class="cl-d">budget '+(d.budget_remaining!=null?d.budget_remaining:"?")+' calls left · actions still go through the syscall gate (treat/escalate)</span>','dim');
      }).catch(function(){ cmdPrint("assistant unreachable","err"); }); },
  brief:function(){ cmdPrint('<span class="cl-d">composing shift handoff from live cohort…</span>','dim');
    fetch(window.AEGIS.http("/aegis/llm/brief"),{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"})
      .then(function(r){return r.json();}).then(function(d){
        var src=d.source==="llm"?('<span class="cl-ok">'+esc(d.model||"llm")+'</span>'):'<span class="cl-d">template</span>';
        cmdPrint('<b>SHIFT HANDOFF</b> <span class="cl-d">· '+src+'</span>','sys');
        String(d.brief||"").split(/\n+/).forEach(function(ln){ if(ln.trim()) cmdPrint(esc(ln.trim()),"row"); });
      }).catch(function(){ cmdPrint("assistant unreachable","err"); }); },
  train:function(a){ var t=(a||"").trim().split(/\s+/).filter(Boolean);
    var name=t[0]; if(!name) return cmdPrint("usage: <b>train &lt;name&gt; [wearable|lab] [limit]</b> — e.g. train mynet lab 1500","err");
    var set=/lab/i.test(t[1]||"")?"lab":"wearable"; var lim=parseInt(t[2]||t[1],10); if(isNaN(lim)) lim=1000;
    mlLaunch({name:name,feature_set:set,limit:lim}, "training detector on real CinC-2019 ("+set+", "+lim+" rows)"); },
  specialize:function(a){ var t=(a||"").trim().split(/\s+/).filter(Boolean);
    var name=t[0], h=parseInt((t[1]||"").replace(/h$/i,""),10);
    if(!name||isNaN(h)) return cmdPrint("usage: <b>specialize &lt;name&gt; &lt;Nh&gt; [wearable|lab]</b> — e.g. specialize early6 6h","err");
    var set=/lab/i.test(t[2]||"")?"lab":"wearable";
    mlLaunch({name:name,feature_set:set,horizon:h,limit:1000}, "specializing early-warning model — fires within "+h+"h of onset"); },
  model:function(a){ var t=(a||"").trim().split(/\s+/).filter(Boolean), sub=(t[0]||"ls").toLowerCase();
    if(sub==="use"){ var nm=t[1]; if(!nm) return cmdPrint("usage: model use &lt;name&gt;","err");
      fetch(window.AEGIS.http("/aegis/ml/use"),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:nm})})
        .then(function(r){return r.json();}).then(function(d){ cmdPrint(d.ok?('<span class="cl-ok">✓ '+esc(nm)+' is now ACTIVE</span> <span class="cl-d">'+esc(d.note||"")+'</span>'):("model use failed: "+esc(d.error)),d.ok?"sys":"err"); })
        .catch(function(){ cmdPrint("ml plane unreachable","err"); }); return; }
    if(sub==="job"){ var id=t[1]; if(!id) return cmdPrint("usage: model job &lt;id&gt;","err"); mlPoll(id,true); return; }
    fetch(window.AEGIS.http("/aegis/ml")).then(function(r){return r.json();}).then(function(d){
      cmdPrint('<b>AEGIS·ML CONTROL PLANE</b> <span class="cl-d">· train + specialize real detectors · active: '+esc(d.active||"—")+'</span>',"sys");
      var jr=(d.jobs||[]).filter(function(j){return j.status!=="DONE"&&j.status!=="FAILED";});
      if(jr.length){ cmdPrint('<span class="cl-d">running jobs:</span>',"dim");
        jr.forEach(function(j){ cmdPrint('  <span class="cl-cmd">'+esc(j.id)+'</span> '+pad(j.name,14)+esc(j.status)+' · '+Math.round(j.progress*100)+'%',"mono"); }); }
      cmdPrint('<span class="cl-d">registered models:</span>',"dim");
      if(!(d.models||[]).length) cmdPrint('  (none yet — try <span class="cl-cmd">train mynet</span> or <span class="cl-cmd">specialize early6 6h</span>)',"dim");
      (d.models||[]).forEach(function(m){ var mt=m.metrics||{};
        cmdPrint('  '+(m.active?'<span class="cl-ok">●</span>':'<span class="cl-d">○</span>')+' '+pad(m.name,14)+
          '<span class="cl-d">'+esc(m.spec.kind)+' · '+esc(m.spec.feature_set)+'</span>  AUROC <b>'+(mt.auroc!=null?mt.auroc:"—")+'</b> · AUPRC '+(mt.auprc!=null?mt.auprc:"—")+' · '+m.n_patients+' pts',"mono"); });
      cmdPrint('control: <span class="cl-cmd">model use &lt;name&gt;</span> · <span class="cl-cmd">model job &lt;id&gt;</span>',"dim");
    }).catch(function(){ cmdPrint("ml plane unreachable","err"); }); },
  pkg:function(a){ var t=(a||"").trim().split(/\s+/).filter(Boolean), sub=(t[0]||"list").toLowerCase();
    if(sub==="install"||sub==="uninstall"){ var name=t[1]; if(!name) return cmdPrint("usage: pkg "+sub+" &lt;name&gt;","err");
      cmdPrint('<span class="cl-d">'+(sub==="install"?"verify-then-install":"uninstall")+' · '+esc(name)+'…</span>',"dim");
      fetch(window.AEGIS.http("/aegis/pkg/"+sub),{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({name:name})}).then(function(r){return r.json();}).then(function(d){
          if(sub==="uninstall") return cmdPrint(d.ok?('<span class="cl-ok">✓ uninstalled '+esc(name)+'</span>'):("not installed: "+esc(name)),d.ok?"ok":"err");
          if(d.ok) cmdPrint('<span class="cl-ok">✓ INSTALLED '+esc(d.name)+'</span> — gates passed (manifest · policy · code · integrity) → now a live service',"sys");
          else { cmdPrint('<span class="cl-crit">✗ REFUSED at '+esc(d.stage)+' gate</span> — '+esc(d.reason),"err");
                 (d.detail||[]).forEach(function(x){ cmdPrint('· '+esc(x),"dim"); }); }
        }).catch(function(){ cmdPrint("pkg "+sub+" failed — backend unreachable","err"); }); return; }
    fetch(window.AEGIS.http("/aegis/pkg")).then(function(r){return r.json();}).then(function(d){
      cmdPrint('<b>aegis-pkg · PACKAGE REGISTRY</b> <span class="cl-d">— verify-then-install · policy: ≤'+d.policy.max_lane_budget+' lanes</span>',"sys");
      cmdPrint('<span class="cl-d">installed:</span>',"dim");
      (d.packages||[]).forEach(function(p){ cmdPrint('<span class="cl-ok">●</span> '+pad(p.name,12)+'<span class="cl-d">v'+esc((p.manifest&&p.manifest.version)||"?")+' · '+esc(p.kind)+' · '+esc(p.hash||"")+'</span>',"mono"); });
      if(!(d.packages||[]).length) cmdPrint('  (none)',"dim");
      cmdPrint('<span class="cl-d">available to install:</span>',"dim");
      (d.available||[]).forEach(function(p){ cmdPrint('<span class="cl-cmd">○</span> '+pad(p.name,12)+'<span class="cl-d">'+esc((p.manifest&&p.manifest.summary)||"")+'</span>',"mono"); });
      cmdPrint('try: <span class="cl-cmd">pkg install tachy</span> · <span class="cl-cmd">pkg install rogue</span> (watch the gate refuse it)',"dim");
    }).catch(function(){ cmdPrint("package registry unreachable","err"); }); },
  vasodilate:function(a){ var v=parseFloat(a); if(isNaN(v)) return cmdPrint("usage: vasodilate &lt;0..1&gt;","err");
    fetch(window.AEGIS.http("/aegis/ppg/vasodilation"),{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({alpha:v})}).then(function(r){return r.json();}).then(function(d){
        if(!d.available) return cmdPrint("PPG monitor unavailable","dim");
        cmdPrint('<span class="cl-sys">vasodilation → '+Math.round((d.alpha||0)*100)+'%</span>',"sys");
        CMD.vascular();
      }).catch(function(){ cmdPrint("vasodilate failed — backend unreachable","err"); }); },
  // ---- Aegis-eBPF verified device plane (input-side safety) ----
  verify:function(a){ var name=(a||"").trim().toLowerCase();
    if(!name){ fetch(window.AEGIS.http("/aegis/kernel/devices")).then(function(r){return r.json();}).then(function(d){
      cmdPrint('<b>AEGIS-eBPF VERIFIER</b> <span class="cl-d">— submit a driver to the static analyzer</span>',"sys");
      cmdPrint('usage: <span class="cl-cmd">verify &lt;driver&gt;</span> · drivers in the catalogue:',"dim");
      (d.programs||[]).forEach(function(p){ cmdPrint('<span class="cl-cmd">'+pad(p.name,12)+'</span><span class="cl-d">'+esc(p.description)+'</span>',"row"); });
      }).catch(function(){ cmdPrint("verifier unavailable (backend waking?)","err"); }); return; }
    cmdPrint('<span class="cl-d">aegis-eBPF · static analysis of driver <b>'+esc(name)+'</b>…</span>',"dim");
    fetch(window.AEGIS.http("/aegis/kernel/verify"),{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({name:name})}).then(function(r){return r.json();}).then(renderVerify)
      .catch(function(){ cmdPrint("verify failed — backend unreachable","err"); }); },
  drivers:function(){ var k=lastSnap&&lastSnap.kernel_devices;
    if(!k) return cmdPrint("device plane not connected…","dim");
    var v=k.verifier||{};
    cmdPrint('<b>VERIFIED eBPF DEVICE PLANE</b> <span class="cl-d">· verifier '+(v.accepted||0)+' admitted / '+(v.rejected||0)+' rejected · helpers {'+(v.helpers||[]).join(", ")+'}</span>',"sys");
    cmdPrint(pad("DEVICE",18)+pad("STATE",10)+pad("DRIVER",14)+pad("BLOCKS",8)+"FAILOVER","dim");
    (k.devices||[]).forEach(function(d){ var cls=d.state==="RUNNING"?"ok":d.state==="DEGRADED"?"warn":"err";
      cmdPrint(pad(d.label,18)+'<span class="cl-'+cls+'">'+pad(d.state,10)+'</span>'+pad(d.active,14)+pad(String(d.blocks),8)+
        (d.failovers?'<span class="cl-warn">'+d.failovers+'×</span>':"0"), "mono"); });
    var ev=((k.supervisor||{}).events)||[]; if(ev.length){ cmdPrint('<span class="cl-d">— supervisor dmesg —</span>',"dim");
      ev.slice(-6).forEach(function(e){ var cls=e.kind==="FAULT"?"err":e.kind==="FAILOVER"?"warn":e.kind==="RESTORE"?"ok":"dim";
        cmdPrint('<span class="cl-'+cls+'">['+pad(e.kind,8)+']</span> '+pad(e.slot,16)+'<span class="cl-d">'+esc(e.detail)+'</span>',"mono"); }); } },
  fault:function(a){ var q=(a||"").trim(); if(!q) return cmdPrint("usage: fault &lt;bed&gt; — kill a verified bed's optical sensor (try a bed shown in `drivers`)","err");
    var b=resolveBed(q); var slot=b?b.bed_id:q;
    cmdPrint('<span class="cl-d">injecting optical sensor fault → '+esc(b?b.name:q)+'…</span>',"dim");
    fetch(window.AEGIS.http("/aegis/kernel/fault"),{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({slot:slot})}).then(function(r){return r.json();}).then(function(d){
        if(d.status==="error") return cmdPrint(esc(d.detail),"err");
        cmdPrint('<span class="cl-warn">⚠ SENSOR FAULT · '+bedTag(d.bed)+' · '+esc(d.name)+'</span> → supervisor '+
          (d.state==="DEGRADED"?'<span class="cl-warn">quarantined the verified driver, hot-swapped backup ('+esc(d.active)+')</span>':esc(d.state)),"sys");
        cmdPrint('patient stays monitored — every other bed kept ticking · <span class="cl-cmd">drivers</span> to inspect · <span class="cl-cmd">recover '+esc(q)+'</span> to re-arm',"row");
      }).catch(function(){ cmdPrint("fault injection failed — backend unreachable","err"); }); },
  recover:function(a){ var q=(a||"").trim();   // re-arm a verified driver; clinical recovery is `clear <bed>`
    if(!q) return cmdPrint("usage: recover &lt;bed&gt; — re-verify + re-arm a quarantined optical driver","err");
    var b=resolveBed(q); var slot=b?b.bed_id:q;
    fetch(window.AEGIS.http("/aegis/kernel/recover"),{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({slot:slot})}).then(function(r){return r.json();}).then(function(d){
        if(d.status==="error") return cmdPrint(esc(d.detail),"err");
        cmdPrint('<span class="cl-ok">✓ RE-ARMED · '+bedTag(d.bed)+' · '+esc(d.name)+'</span> → primary driver re-verified and restored ('+esc(d.state)+')',"ok");
      }).catch(function(){ cmdPrint("recover failed — backend unreachable","err"); }); },
  kstat:function(){ var k=lastSnap&&lastSnap.kstat; if(!k) return cmdPrint("kernel not connected…","dim");
    cmdPrint('<b>AEGIS KERNEL</b> <span class="cl-d">· cognitive lane governor (tick '+k.tick+')</span>','sys');
    cmdPrint('mode '+(k.mode==="PREEMPT"?'<span class="cl-crit">PREEMPT'+(k.owner?' → '+esc(cleanName(k.owner)):'')+'</span>':'<span class="cl-ok">NORMAL</span>')+' · Σcost '+fmt(k.global_cost,2),"row");
    cmdPrint('procs '+k.procs+' <span class="cl-d">('+(k.tiers&&k.tiers.CRITICAL||0)+' critical · '+(k.tiers&&k.tiers.WATCH||0)+' watch · '+(k.tiers&&k.tiers.STABLE||0)+' stable)</span>',"row");
    cmdPrint('compute '+k.lanes_used+'/'+k.lane_budget+' lanes ('+Math.round((k.utilization||0)*100)+'% util) · interrupts '+k.interrupts,"mono");
    cmdPrint('<span class="cl-cmd">⟳ anticipating '+(k.anticipating||0)+'</span> pre-critical proc(s) receiving predictive (foresight) compute',"row"); },
  ps:function(){ var t=(lastSnap&&lastSnap.topology)||[]; if(!t.length) return cmdPrint("kernel not connected…","dim");
    cmdPrint('<b>BIO-PROCESS TABLE</b> <span class="cl-d">· each monitored bed is a scheduled process</span>','sys');
    cmdPrint(pad("PID",5)+pad("PROCESS",15)+pad("STATE",10)+pad("RISK",7)+pad("FORE",7)+"CPU","dim");
    t.slice().sort(function(a,b){return (b.p+(b.foresight||0))-(a.p+(a.foresight||0));}).forEach(function(n){
      var nm=cleanName(n.name||n.node_id).split(",")[0], fs=n.foresight||0,
          cls=n.tier==="CRITICAL"?"err":n.tier==="WATCH"?"warn":"ok";
      cmdPrint(pad(pidOf(n.node_id),5)+pad(nm,15)+pad(n.tier.toLowerCase(),10)+pad(fmt(n.p,2),7)+
        pad(fs.toFixed(2)+(fs>=0.25?"▲":""),7)+Math.round((n.cpu||0)*100)+"%", cls); }); },
  top:function(){ CMD.kstat(); cmdPrint("",""); CMD.ps(); },
  dmesg:function(){ var ev=(lastSnap&&lastSnap.events)||[]; if(!ev.length) return cmdPrint("dmesg: no kernel interrupts logged","dim");
    cmdPrint('<b>KERNEL EVENT LOG</b> <span class="cl-d">· dmesg</span>','sys');
    ev.slice(-12).forEach(function(e){ cmdPrint('<span class="cl-cmd">[t'+e.tick+']</span> <span class="cl-crit">⚡ '+esc(e.type||"EVENT")+'</span>'+
      (e.owner?' · '+esc(cleanName(e.owner)):'')+(e.reason?' · '+esc(e.reason):'')+(e.p!=null?' · p='+fmt(e.p,2):''),"mono"); }); },
  clr:function(){ $("cmd-out").innerHTML=""; cmdBanner(); }
};
function mlLaunch(body, label){
  cmdPrint('<span class="cl-d">'+esc(label)+' · patient-grouped cross-validation…</span>',"dim");
  fetch(window.AEGIS.http("/aegis/ml/train"),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})
    .then(function(r){return r.json();}).then(function(d){
      if(!d.launched){ return cmdPrint("train failed to launch","err"); }
      cmdPrint('<span class="cl-cmd">job '+esc(d.job.id)+'</span> launched — <span class="cl-d">'+esc(d.job.spec.kind)+'</span>',"sys");
      mlPoll(d.job.id);
    }).catch(function(){ cmdPrint("ml plane unreachable","err"); });
}
function mlPoll(id, once){
  var seen=0;
  function tick(){
    fetch(window.AEGIS.http("/aegis/ml/job/"+id)).then(function(r){return r.json();}).then(function(j){
      if(j.error){ cmdPrint("no such job "+esc(id),"err"); return; }
      (j.log||[]).slice(seen).forEach(function(l){ cmdPrint('  <span class="cl-d">▸ '+esc(l)+'</span>',"mono"); });
      seen=(j.log||[]).length;
      if(j.status==="DONE"||j.status==="FAILED"){
        if(j.status==="DONE"){ var m=j.metrics||{};
          cmdPrint('<b class="cl-ok">✓ '+esc(j.name)+' trained</b> — AUROC <b>'+m.auroc+'</b> · AUPRC '+m.auprc+' · ECE '+m.ece+' · sens@85%spec '+m.sens_at_spec85+' <span class="cl-d">('+esc(m.cv)+', prev '+m.prevalence+')</span>',"sys");
          cmdPrint('<span class="cl-d">registered · activate with</span> <span class="cl-cmd">model use '+esc(j.name)+'</span>',"dim");
        } else cmdPrint('<span class="cl-crit">✗ job failed — '+esc(j.error)+'</span>',"err");
        return;
      }
      if(!once) setTimeout(tick, 1500);
    }).catch(function(){ cmdPrint("ml job poll lost connection","err"); });
  }
  tick();
}
function cleanName(s){ return String(s||"").replace(/^bed_/,"BED "); }
function pad(s,n){ s=String(s); return s.length>=n?s.slice(0,n):s+" ".repeat(n-s.length); }
function pidOf(id){ var m=String(id).match(/(\d+)/); return m?("0"+(+m[1]+1)).slice(-2):String(id).slice(-2); }
function printBundle(d,b){
  if(!d||!d.checklist){ cmdPrint("no protocol indicated — patient within range","ok"); return; }
  cmdPrint('<b>CERTIFIED HOUR-1 BUNDLE · '+esc(b.name)+'</b> '+(d.all_verified?'<span class="cl-ok">✓ Z3-verified · safe to dispatch</span>':'<span class="cl-crit">✗ quarantined</span>'),"sys");
  d.checklist.forEach(function(it){ cmdPrint('<span class="cl-cmd">'+it.n+'.</span> '+esc(it.title)+' <span class="cl-d">'+esc(it.detail)+'</span>',"row"); });
  if(d.protocol) cmdPrint('rx: '+esc(d.protocol.antibiotic)+' · fluid '+d.protocol.fluid_rate+' mL/kg/h · vaso '+d.protocol.vasopressor_dose+' mcg/kg/min','mono'); }
/* ============================================================
   TERMINAL — a real tty line editor over the CMD table:
   blinking cursor · Tab completion · shell builtins · /proc VFS
   ============================================================ */
var termCwd="~", PROC_CACHE=null;
var TERM_DIRS=["/","/proc","/bin","/cohort","/kernel"];
function uniq(a){ var s={},o=[]; a.forEach(function(x){ if(!s[x]){s[x]=1;o.push(x);} }); return o; }
function commonPrefix(a){ if(!a.length) return ""; var p=a[0]; for(var i=1;i<a.length;i++){ while(a[i].indexOf(p)!==0){ p=p.slice(0,-1); if(!p) return ""; } } return p; }
try{ cmdHist=JSON.parse(localStorage.getItem("aegis_term_hist")||"[]")||[]; }catch(e){ cmdHist=[]; }
cmdHi=cmdHist.length;
function termSaveHist(){ try{ localStorage.setItem("aegis_term_hist",JSON.stringify(cmdHist.slice(-200))); }catch(e){} }
function termSetPrompt(){ var p=$("term-prompt"); if(p) p.innerHTML='aegis@icu-wing<b>:'+esc(termCwd)+'$</b>';
  var c=$("term-cwd"); if(c) c.textContent=":"+termCwd; }
function termRender(){ var inp=$("cmd-in"); if(!inp) return; var v=inp.value||"", pos=inp.selectionStart==null?v.length:inp.selectionStart;
  var pre=$("ti-pre"), cur=$("ti-cur"), post=$("ti-post"); if(!pre) return;
  pre.textContent=v.slice(0,pos); cur.textContent=v.slice(pos,pos+1)||" "; post.textContent=v.slice(pos+1); }
function termFocus(){ var i=$("cmd-in"); if(i){ i.focus(); termRender(); } }
function fetchProc(cb){ fetch(window.AEGIS.http("/aegis/proc")).then(function(r){return r.json();})
  .then(function(d){ PROC_CACHE=(d&&d.paths)||[]; if(cb) cb(d); }).catch(function(){ if(cb) cb(null); }); }
function termResolve(t){ t=(t||"").trim(); if(t===""||t==="~") return "~"; if(t==="/") return "/";
  if(t===".."){ return (termCwd==="~"||termCwd==="/"||termCwd.split("/").length<=2)?"/":termCwd.replace(/\/[^/]+$/,"")||"/"; }
  if(t.charAt(0)==="/") return t.replace(/\/$/,"")||"/";
  var base=(termCwd==="~")?"":termCwd.replace(/\/$/,""); return (base+"/"+t).replace(/\/+/g,"/"); }
function termCandidates(v){
  var parts=v.split(/\s+/);
  if(parts.length<=1){ var t=(parts[0]||"").toLowerCase();
    return uniq(Object.keys(CMD).concat(Object.keys(CMD_ALIAS))).filter(function(n){return n.indexOf(t)===0;}).sort(); }
  var cmd=(CMD_ALIAS[parts[0].toLowerCase()]||parts[0].toLowerCase()), last=parts[parts.length-1].toLowerCase(), pool=[];
  if(cmd==="cd"||cmd==="ls") pool=TERM_DIRS.concat(["..","~"]);
  else if(cmd==="krun") pool=(parts.length===2)?["mixed","pipeline","fanout","stress"]:["FIFO","RR","CFS","EDF","HDSched"];
  else if(cmd==="cat"||cmd==="proc") pool=(PROC_CACHE||[]).map(function(p){return p.replace(/^\/proc\//,"").replace(/^\//,"");});
  else if(cmd==="man") pool=uniq(Object.keys(CMD));
  else if(/^(who|open|treat|escalate|clear|explain|radar|resonance|protocol|crisis|inspect|verify|fault|project)$/.test(cmd)) pool=beds().map(function(b){return b.bed_id;});
  return uniq(pool).filter(function(n){return String(n).toLowerCase().indexOf(last)===0;}).sort();
}
function termComplete(){ var inp=$("cmd-in"), v=inp.value, cands=termCandidates(v); if(!cands.length) return;
  var parts=v.split(/\s+/), lastTok=parts[parts.length-1];
  if(cands.length===1){ parts[parts.length-1]=cands[0]; inp.value=parts.join(" ")+(parts.length<=1?" ":""); }
  else { var cp=commonPrefix(cands); if(cp.length>lastTok.length){ parts[parts.length-1]=cp; inp.value=parts.join(" "); }
    cmdPrint('<span class="term-comp">'+cands.slice(0,28).map(function(c){return "<b>"+esc(c)+"</b>";}).join("   ")+(cands.length>28?"  …":"")+'</span>',"dim"); }
  inp.selectionStart=inp.selectionEnd=inp.value.length; termRender(); }
/* ---- shell builtins (added to the CMD dispatch table) ---- */
var MAN={ps:"ps — process table (each monitored bed is a scheduled bio-process)",top:"top — kernel stats + process table",
  kstat:"kstat — cognitive lane-governor summary",krun:"krun [mixed|pipeline|fanout|stress] [FIFO|RR|CFS|EDF|HDSched] — run a program on the live HDKernel and inspect its process table",
  ask:"ask <question> — grounded natural-language query over live cohort + kernel state (read-only, no orders)",
  brief:"brief — LLM shift-handoff summary of the cohort",dmesg:"dmesg — kernel interrupt / preemption log",
  proc:"proc <path> — read the /proc introspection namespace",treat:"treat <bed> [fluid= vaso= abx=] — order the certified Hour-1 bundle (Z3-gated)",
  project:"project <bed> — render the ordered treatment travelling through the body (PBPK twin)",
  escalate:"escalate <bed> [lvl] — raise priority; pins kernel compute if justified",ls:"ls [dir] — list a virtual directory (/proc /bin /cohort /kernel)",
  cd:"cd <dir> — change directory",cat:"cat <proc-path> — print a live /proc value"};
CMD.clear=function(){ $("cmd-out").innerHTML=""; };
CMD.history=function(){ if(!cmdHist.length) return cmdPrint("(no history)","dim");
  cmdHist.slice(-40).forEach(function(h,i){ cmdPrint('<span class="cl-d">'+pad(String(cmdHist.length-Math.min(40,cmdHist.length)+i+1),4)+'</span>'+esc(h),"mono"); }); };
CMD.whoami=function(){ cmdPrint("clinician@aegis-os · role: attending · ring&nbsp;2 (privileged) · session verified","mono"); };
CMD.uname=function(a){ var f=/(-a|--all)/.test(a||"");
  cmdPrint("AegisOS bio-kernel"+(f?" L1(SHAL)·L2(MemFabric)·L3(CogSched)·L3b(HDSched)·L3c(HDKernel)·L4(NeuroSym) x_bio64":" 0.8 x_bio64"),"mono"); };
CMD.date=function(){ cmdPrint(new Date().toString(),"mono"); };
CMD.echo=function(a){ cmdPrint(esc(a||""),"row"); };
CMD.pwd=function(){ cmdPrint(termCwd==="~"?"/":termCwd,"mono"); };
CMD.motd=function(){ cmdBanner(); };
CMD.man=function(a){ var c=(a||"").trim().toLowerCase(); c=CMD_ALIAS[c]||c;
  if(MAN[c]) cmdPrint(esc(MAN[c]),"row"); else cmdPrint(c?("No manual entry for "+esc(c)+" — try help"):"usage: man &lt;command&gt;","dim"); };
CMD.cd=function(a){ var t=termResolve((a||"~").trim());
  if(t==="~"||TERM_DIRS.indexOf(t)>=0){ termCwd=t; termSetPrompt(); }
  else cmdPrint("cd: "+esc((a||"").trim())+": no such directory","err"); };
CMD.ls=function(a){ var d=termResolve((a||"").trim()||termCwd);
  if(d==="~"||d==="/") return cmdPrint('<span class="cl-cmd">proc/</span>  <span class="cl-cmd">bin/</span>  <span class="cl-cmd">cohort/</span>  <span class="cl-cmd">kernel/</span>',"mono");
  if(d==="/bin") return cmdPrint(uniq(Object.keys(CMD)).sort().map(esc).join("  "),"mono");
  if(d==="/cohort"){ var bs=beds(); return cmdPrint(bs.length?bs.map(function(b){return esc(b.bed_id);}).join("  "):"(no beds connected)","mono"); }
  if(d==="/kernel") return cmdPrint("sched  kthreads  ksched  dmesg  llm","mono");
  if(d==="/proc"){ fetchProc(function(){ cmdPrint((PROC_CACHE&&PROC_CACHE.length?PROC_CACHE.map(function(p){return esc(p.replace(/^\//,""));}).join("  "):"(empty)"),"mono"); }); return; }
  cmdPrint("ls: "+esc(d)+": no such directory","err"); };
CMD.cat=function(a){ var p=(a||"").trim(); if(!p) return cmdPrint("usage: cat &lt;proc-path&gt;","err");
  var path=p.replace(/^\/?proc\//,"").replace(/^\//,""); if(termCwd==="/proc"&&p.indexOf("/")<0) path=p;
  fetch(window.AEGIS.http("/aegis/proc/"+path)).then(function(r){return r.json();}).then(function(d){
    if(d.error) return cmdPrint("cat: "+esc(d.path||p)+": "+esc(d.error),"err");
    cmdPrint('<span class="cl-cmd">'+esc(d.path)+'</span>',"sys");
    cmdPrint(esc(JSON.stringify(d.value,null,1)).slice(0,2200),"mono");
  }).catch(function(){ cmdPrint("cat: read failed — backend unreachable","err"); }); };

function cmdInit(){
  $("cmd-open").addEventListener("click",cmdToggle);
  $("cmd-x").addEventListener("click",cmdHide);
  $("cmd").addEventListener("mousedown",function(e){ if(e.target===this) cmdHide(); });
  document.addEventListener("keydown",function(e){
    var typing=/^(input|textarea)$/i.test((e.target.tagName||""));
    if(e.key==="/"&&!cmdOpen&&!typing){ e.preventDefault(); cmdShow(); return; }
    if(e.key==="Escape"&&cmdOpen){ cmdHide(); }
  });
  var inp=$("cmd-in"); if(!inp) return;
  termSetPrompt(); termRender(); fetchProc(null);
  $("cmd-out").addEventListener("mousedown",function(e){ if(window.getSelection().isCollapsed) setTimeout(termFocus,0); });
  $("term-line").addEventListener("mousedown",function(e){ if(e.target!==inp){ e.preventDefault(); termFocus(); } });
  inp.addEventListener("input",termRender);
  inp.addEventListener("click",termRender);
  inp.addEventListener("keyup",function(e){ if(!/^(Enter|Tab|Arrow)/.test(e.key)) termRender(); });
  inp.addEventListener("keydown",function(e){
    if(e.key==="Enter"){ e.preventDefault(); var v=inp.value; inp.value="";
      if(v.trim()){ if(cmdHist[cmdHist.length-1]!==v.trim()){ cmdHist.push(v.trim()); termSaveHist(); } cmdRun(v); }
      else cmdEcho(""); cmdHi=cmdHist.length; termRender(); return; }
    if(e.key==="Tab"){ e.preventDefault(); termComplete(); return; }
    if(e.key==="ArrowUp"){ e.preventDefault(); if(cmdHi>0){ cmdHi--; inp.value=cmdHist[cmdHi]||""; inp.selectionStart=inp.selectionEnd=inp.value.length; termRender(); } return; }
    if(e.key==="ArrowDown"){ e.preventDefault(); if(cmdHi<cmdHist.length){ cmdHi++; inp.value=cmdHist[cmdHi]||""; inp.selectionStart=inp.selectionEnd=inp.value.length; termRender(); } return; }
    if(e.ctrlKey&&/^[lL]$/.test(e.key)){ e.preventDefault(); $("cmd-out").innerHTML=""; termRender(); return; }
    if(e.ctrlKey&&/^[cC]$/.test(e.key)){ e.preventDefault(); cmdEcho(inp.value+"^C"); inp.value=""; cmdHi=cmdHist.length; termRender(); return; }
    if(e.ctrlKey&&/^[uU]$/.test(e.key)){ e.preventDefault(); inp.value=inp.value.slice(inp.selectionStart); inp.selectionStart=inp.selectionEnd=0; termRender(); return; }
    if(e.ctrlKey&&/^[kK]$/.test(e.key)){ e.preventDefault(); inp.value=inp.value.slice(0,inp.selectionStart); termRender(); return; }
    if(e.ctrlKey&&/^[aA]$/.test(e.key)){ e.preventDefault(); inp.selectionStart=inp.selectionEnd=0; termRender(); return; }
    if(e.ctrlKey&&/^[eE]$/.test(e.key)){ e.preventDefault(); inp.selectionStart=inp.selectionEnd=inp.value.length; termRender(); return; }
    setTimeout(termRender,0);
  });
}

/* ============================================================
   MAIN LOOP
   ============================================================ */
var surf={}, spineTick=0;
function setup(){
  cmdInit(); pttInit();
  ambientInit(); runBoot();
  surf.learn=new Surface("cv-learn"); surf.blind=new Surface("cv-blind");
  surf.dosRisk=new Surface("cv-dos-risk");
  $("opgrid").addEventListener("click",function(e){ var bed=e.target.closest(".bed"); if(bed) openDossier(bed.dataset.id); });
  var spp=$("sp-pkgs"); if(spp) spp.addEventListener("click",function(e){ var p=e.target.closest("[data-install]"); if(p) installPkg(p.dataset.install); });
  $("dossier").addEventListener("click",function(e){ var b=e.target.closest(".dos-act");
    if(!b||!selectedBed) return; chaos(selectedBed,b.dataset.act); b.animate([{opacity:.45},{opacity:1}],{duration:280}); });
  $("dos-close").addEventListener("click",closeDossier);
  document.addEventListener("keydown",function(e){ if(e.key==="Escape") closeDossier(); });
  $("tabs").addEventListener("click",function(e){ var t=e.target.closest(".tab"); if(t) setView(t.dataset.view); });
  // The Roster view was removed: it listed the same twelve patients the
  // Operations grid already shows, filtered four ways nobody used. Its listeners
  // are gone with it; `roster` in the console still prints the list.

  window.addEventListener("resize",ambientFit);
  // devices hub
  $("dev-pair-btn").addEventListener("click",openWizard);
  $("dev-grid").addEventListener("click",function(e){ var c=e.target.closest(".dev-card"); if(c) openDevice(c.dataset.id); });
  $("dev-drawer-inner").addEventListener("click",function(e){ var b=e.target.closest("[data-act]"); if(b) devCallAction(b.dataset.act); });
  $("dev-drawer").addEventListener("click",function(e){ if(e.target===this) closeDevice(); });
  $("dev-wiz-close").addEventListener("click",closeWizard);
  $("dev-modal").addEventListener("click",function(e){ if(e.target===this) closeWizard();
    var p=e.target.closest("[data-pair]"); if(p) wizPair(p.dataset.pair);
    if(e.target.id==="wiz-rescan") wizScan(); });
  connect(); requestAnimationFrame(loop);
}
function loop(){ frame++;
  ambientDraw();
  if(target) ease();
  if(currentView==="ops"){ surf.learn.fit(); surf.blind.fit();
    drawLearn(surf.learn); drawBlind(surf.blind);
  }
  // System Health is plain DOM — no canvas to fit or redraw on the animation tick.
  else if(currentView==="devices"){ drawDeviceSparks(); if(selectedDevice) drawDrawerTrace(); }
  if(selectedBed){ surf.dosRisk.fit(); drawRiskTimeline(surf.dosRisk); }
  if((spineTick++ %6)===0) renderSpine();
  requestAnimationFrame(loop);
}
if(document.readyState!=="loading") setup();
else document.addEventListener("DOMContentLoaded",setup);

})();
