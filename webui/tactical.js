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

/* ---------- palette (mirrors tactical.css) ---------- */
var C = {
  ink:"#f0f4ff", muted:"#9aa6c8", dim:"#5a6488",
  cyan:"#5fd0f7", indigo:"#7d8cff", violet:"#b08cff",
  ok:"#41e6a4", watch:"#ffc24d", crit:"#ff6b86",
  grid:"rgba(108,140,255,.06)", gridHi:"rgba(108,140,255,.13)",
};
var TIER_COL = { STABLE:C.cyan, WATCH:C.watch, CRITICAL:C.crit };
var LANE_COL = { safety:C.ok, ml_tracking:C.cyan, render:C.indigo, logging:C.violet };
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
function Surface(id){ this.cv=$(id); this.ctx=this.cv.getContext("2d"); this.w=0; this.h=0; this.dpr=1; }
Surface.prototype.fit=function(){
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
  target=snap;
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
  if(currentView==="roster") renderRoster();
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
    '<div class="bed-vitals"></div>'+
    '<div class="bed-bar"><i></i></div>';
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
    ' mcg/kg/min — <span class="db-z3">Z3 '+p.z3.toUpperCase()+'</span></div>';
}
function openDossier(id){
  selectedBed=id; dosChart=null;
  if(!dosBuilt){ $("dos-comps").innerHTML=COMP_ORDER.map(function(c){
    return '<div class="dos-comp" data-c="'+c+'"><span class="dos-comp-k">'+c+'</span>'+
      '<span class="dos-comp-track"><i></i></span><span class="dos-comp-v">0.00</span></div>';
  }).join(""); dosBuilt=true; }
  $("dossier").classList.add("open");
  $("dossier").setAttribute("aria-hidden","false");
  if(latestBeds[id]) updateDossier(latestBeds[id]);
  fetchChart(id); refreshBundle(id);
  if(dosChartTimer) clearInterval(dosChartTimer);
  dosChartTimer=setInterval(function(){ if(selectedBed){ fetchChart(selectedBed); refreshBundle(selectedBed); } }, 4000);
}
function closeDossier(){ selectedBed=null; dosChart=null;
  if(dosChartTimer){ clearInterval(dosChartTimer); dosChartTimer=null; }
  $("dossier").classList.remove("open");
  $("dossier").setAttribute("aria-hidden","true"); }

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
  COMP_ORDER.forEach(function(c){ var row=$("dos-comps").querySelector('[data-c="'+c+'"]'); if(!row) return;
    var sc=b.components[c]||0, bar=row.querySelector("i");
    bar.style.width=clamp(sc/2.5*100,0,100)+"%"; bar.style.background=COMP_COL[c];
    row.querySelector(".dos-comp-v").textContent=fmt(sc,2); });
  // derived autonomic + perfusion markers (kernel footer; main vitals are in the monitor)
  $("dos-edge").innerHTML=
    kvHTML("Perfusion idx",fmt(e.perfusion,2),"au")+
    kvHTML("Vasomotion",fmt(e.vasomotion,2),"au");
  $("dos-ent").innerHTML=
    kvHTML("ABP variability",fmt(en.abp_var,1),"σ")+
    kvHTML("Lactate clearance",fmt(en.lactate_clearance,0),"%",(en.lactate>2&&en.lactate_clearance<-5)?"warn":"");
  $("dos-drivers").innerHTML=(b.drivers&&b.drivers.length)
    ? b.drivers.map(function(d,i){ return '<div class="drv'+(i<2?" hi":"")+'">'+d+'</div>'; }).join("")
    : '<div class="drv none">All monitored systems within range</div>';
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
    var owner=(V.owner===id), col=owner||nd.tier==="CRITICAL"?C.crit:nd.tier==="WATCH"?C.watch:C.cyan, a=owner?.97:.55;
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
  [0,0.5,1].forEach(function(v){ var y=yAt(v); ctx.strokeStyle=C.grid; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(w-padR,y); ctx.stroke();
    ctx.fillStyle=C.dim; ctx.textAlign="right"; ctx.fillText(v.toFixed(1),padL-5,y); });
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
  var fg=ctx.createLinearGradient(0,plotTop,0,plotBot); fg.addColorStop(0,withAlpha(col,.28)); fg.addColorStop(1,withAlpha(col,.03));
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
  ctx.fillStyle=col; ctx.beginPath(); ctx.arc(xAt(h1),yAt(last),3,0,7); ctx.fill();
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
function renderTerminal(bundle){
  var term=$("terminal");
  if(!bundle){ if(termKey!==null){ term.classList.remove("active");
      $("term-sub").textContent="Layer-4 formal verification · standing by"; $("term-stamp").textContent="";
      $("term-body").innerHTML='<div class="term-idle">// no active sepsis vectors — all beds within range</div>';
      termKey=null; if(termTimer){ clearTimeout(termTimer); termTimer=null; } } return; }
  term.classList.add("active");
  $("term-sub").textContent="active focus · "+bundle.name;
  if(bundle.bed_id===termKey) return;          // already streamed for this bed
  termKey=bundle.bed_id; streamLines(buildTermLines(bundle));
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

/* ============================================================
   WEBSOCKET
   ============================================================ */
var ws=null, connected=false, lastMsg=0;
function clusterWS(){ return window.AEGIS.base().replace(/^http/,"ws")+"/aegis/cluster/stream"; }
function connect(){ try{ ws=new WebSocket(clusterWS()); }catch(e){ setTimeout(connect,1500); return; }
  ws.onopen=function(){ connected=true; setConn(true); };
  ws.onmessage=function(e){ lastMsg=performance.now(); try{ ingest(JSON.parse(e.data)); }catch(err){ console.error(err); } };
  ws.onclose=function(){ connected=false; setConn(false); setTimeout(connect,1500); };
  ws.onerror=function(){ try{ws.close();}catch(e){} };
}
function setConn(live){ $("conn").classList.toggle("live",live); $("conn-txt").textContent=live?"Live":"Reconnecting"; }
setInterval(function(){ if(connected && performance.now()-lastMsg<4000) return;
  fetch(window.AEGIS.http("/aegis/cluster/snapshot")).then(function(r){return r.json();}).then(ingest).catch(function(){}); },2500);

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
  ["ops","val","roster"].forEach(function(v){ $("view-"+v).classList.toggle("active",v===name); });
  [].forEach.call(document.querySelectorAll(".tab"),function(t){ t.classList.toggle("active",t.dataset.view===name); });
  if(name==="val") loadValidation();
  if(name==="roster") renderRoster();
}

/* ============================================================
   VALIDATION VIEW
   ============================================================ */
var valData=null;
function kpiHTML(cls,k,v,unit,s){ return '<div class="kpi'+(cls?" "+cls:"")+'"><div class="k">'+k+
  '</div><div class="v" data-count="'+v+'">0<small>'+unit+'</small></div><div class="s">'+s+'</div></div>'; }
function loadValidation(){ fetch(window.AEGIS.http("/aegis/cohort/validation")).then(function(r){return r.json();})
  .then(function(d){ if(d&&d.summary){ valData=d; renderValidation(d); } }).catch(function(){}); }
function renderValidation(d){ var s=d.summary;
  $("val-kpis").innerHTML=
    kpiHTML("good","Detection rate",Math.round(s.detection_rate*100),"%",s.n_detected+" / "+s.n_septic+" septic stays")+
    kpiHTML("lead","Mean lead time",s.mean_lead_h,"h","before labelled onset")+
    kpiHTML("lead","Max lead time",s.max_lead_h,"h","earliest flagged")+
    kpiHTML("","Cohort size",s.n,"","monitored real stays")+
    kpiHTML("warn","False positive",Math.round(s.fp_rate*100),"%","of stable patients");
  var rows='<div class="vrow hdr"><span>Patient</span><span>Onset</span><span>First alarm</span><span>Result</span></div>';
  d.patients.slice().sort(function(a,b){return (b.lead==null?-99:b.lead)-(a.lead==null?-99:a.lead);}).forEach(function(r){
    var dot=!r.septic?C.dim:(r.first_alarm!=null&&(r.onset_hour==null||r.first_alarm<=r.onset_hour))?C.ok:C.crit;
    var res=!r.septic?(r.first_alarm!=null?'<span class="tag-na">stable · alarmed</span>':'<span class="tag-na">stable</span>')
      :(r.lead!=null&&r.lead>=0?'<span class="tag-pos">+'+r.lead+' h early</span>':'<span class="tag-miss">missed</span>');
    rows+='<div class="vrow"><span class="nm" data-id="'+r.bed_id+'"><i style="background:'+dot+'"></i>'+r.name+'</span>'+
      '<span class="mono">'+(r.onset_hour!=null?'h'+r.onset_hour:'—')+'</span>'+
      '<span class="mono">'+(r.first_alarm!=null?'h'+r.first_alarm:'—')+'</span><span class="mono">'+res+'</span></div>'; });
  $("val-table").innerHTML=rows;
  requestAnimationFrame(animateCounts);
}
/* ============================================================
   PANEL — Per-Patient Baseline Learning
   Each bed's deviation from ITS OWN learned baseline (σ), ranked.
   This is the behavioural model made visible: the system learns every
   patient's personal norm and tracks how far each has drifted from it.
   ============================================================ */
var CH_LABEL={hr:"HR",temp_c:"T°",resp_rate:"RR",spo2:"SpO₂",map:"MAP",wbc:"WBC",lactate:"Lac"};
function devCol(d){ return d>=2.0?C.crit:d>=1.0?C.watch:d>=0.4?C.cyan:C.dim; }
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
  var mx=Math.max.apply(null,counts)||1, padL=14,padR=14,padT=16,padB=26,bw=(w-padL-padR)/bins.length;
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
   MAIN LOOP
   ============================================================ */
var surf={}, spineTick=0;
function setup(){
  ambientInit(); runBoot();
  surf.learn=new Surface("cv-learn"); surf.blind=new Surface("cv-blind");
  surf.dosMem=new Surface("cv-dos-mem"); surf.dosLat=new Surface("cv-dos-lat");
  surf.dosTrend=new Surface("cv-dos-trend"); surf.dosRisk=new Surface("cv-dos-risk");
  surf.leadhist=new Surface("cv-leadhist"); surf.sweep=new Surface("cv-sweep");
  $("opgrid").addEventListener("click",function(e){ var bed=e.target.closest(".bed"); if(bed) openDossier(bed.dataset.id); });
  $("dossier").addEventListener("click",function(e){ var b=e.target.closest(".dos-act");
    if(!b||!selectedBed) return; chaos(selectedBed,b.dataset.act); b.animate([{opacity:.45},{opacity:1}],{duration:280}); });
  $("dos-close").addEventListener("click",closeDossier);
  document.addEventListener("keydown",function(e){ if(e.key==="Escape") closeDossier(); });
  $("tabs").addEventListener("click",function(e){ var t=e.target.closest(".tab"); if(t) setView(t.dataset.view); });
  $("roster-list").addEventListener("click",function(e){ var c=e.target.closest(".pcard"); if(c) openDossier(c.dataset.id); });
  $("view-val").addEventListener("click",function(e){ var n=e.target.closest(".nm"); if(n&&n.dataset.id) openDossier(n.dataset.id); });
  $("roster-search").addEventListener("input",function(e){ rosterQuery=e.target.value; renderRoster(); });
  $("roster-filters").addEventListener("click",function(e){ var b=e.target.closest(".rf"); if(!b) return;
    rosterFilter=b.dataset.f; [].forEach.call(this.children,function(x){x.classList.toggle("active",x===b);}); renderRoster(); });
  window.addEventListener("resize",ambientFit);
  connect(); requestAnimationFrame(loop);
}
function loop(){ frame++;
  ambientDraw();
  if(target) ease();
  if(currentView==="ops"){ surf.learn.fit(); surf.blind.fit();
    drawLearn(surf.learn); drawBlind(surf.blind);
  } else if(currentView==="val"){ surf.leadhist.fit(); drawLeadHist(surf.leadhist);
    surf.sweep.fit(); drawSweep(surf.sweep); }
  if(selectedBed){ surf.dosMem.fit(); surf.dosLat.fit(); surf.dosTrend.fit(); surf.dosRisk.fit();
    drawTrends(surf.dosTrend); drawRiskTimeline(surf.dosRisk);
    drawBedMemory(surf.dosMem,selectedBed); drawLatency(surf.dosLat); }
  if((spineTick++ %6)===0) renderSpine();
  requestAnimationFrame(loop);
}
if(document.readyState!=="loading") setup();
else document.addEventListener("DOMContentLoaded",setup);

})();
