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
  ink:"#eef2f9", muted:"#97a2bb", dim:"#5d6884",
  cyan:"#63cdf2", indigo:"#7e8cf8", violet:"#a98cf5",
  ok:"#4ade9e", watch:"#f2c14e", crit:"#fb6f7d",
  grid:"rgba(126,140,248,.06)", gridHi:"rgba(126,140,248,.12)",
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
var latBuf=new Float32Array(900), latBuf95=new Float32Array(900), latHead=0, latFill=0;
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
  latestBeds={}; (snap.beds||[]).forEach(function(b){ latestBeds[b.bed_id]=b; });
  if(snap.beds){ var sp=snap.beds.filter(function(b){return b.will_be_septic;}).length;
    var real=snap.data_mode && snap.data_mode.indexOf("real")===0;
    $("a-meta").textContent=real ? (snap.beds.length+" real ICU patients · "+sp+" septic")
                                 : (snap.beds.length+" patients · synthetic wing"); }
  if(snap.lane_map){ $("c-meta").textContent="budget "+snap.lane_map.budget; }
  renderGrid(snap.beds);
  renderSpine();          // refresh metrics + ticker the instant data arrives
  if(selectedBed && latestBeds[selectedBed]) updateDossier(latestBeds[selectedBed]);
  renderTerminal(snap.bundle);   // Module C — stream the certified Hour-1 bundle
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
  latBuf[latHead]=V.lat.p50; latBuf95[latHead]=V.lat.p95;
  latHead=(latHead+1)%latBuf.length; latFill=Math.min(latFill+1,latBuf.length);
}

/* ============================================================
   MODULE A — ICU Operations Grid (DOM; FLIP reorder)
   A dense bed matrix fed by the fused per-bed ontology. A bed that
   crosses critical expands, flashes crimson and floats to top-left.
   ============================================================ */
var gridOrder=[];
function bedTag(id){ var m=id.match(/(\d+)/); var i=m?(+m[1]+1):0; return "BED "+(i<10?"0"+i:i); }
function tierOf(p){ return p>=0.70?"critical":p>=0.40?"watch":"stable"; }
function vital(label,val,dec){ return "<span><i>"+label+"</i>"+fmt(val,dec)+"</span>"; }

function bedCellHTML(b){
  return '<div class="bed-top"><span class="bed-tag">'+bedTag(b.bed_id)+'</span>'+
    '<span class="bed-name">'+b.name+'</span>'+
    '<span class="bed-risk">P <b>0.00</b></span></div>'+
    '<div class="bed-sys">—</div>'+
    '<div class="bed-driver"></div>'+
    '<div class="bed-vitals"></div>'+
    '<div class="bed-bar"><i></i></div>';
}
function updateBed(el,b){
  var tier=tierOf(b.fused_risk), en=b.enterprise, ed=b.edge;
  if(el.dataset.tier!==tier){
    if(tier==="critical" && el.dataset.tier && el.dataset.tier!=="critical"){
      el.classList.add("just-crit");
      setTimeout(function(){ el.classList.remove("just-crit"); }, 2700);
    }
    el.dataset.tier=tier;
    el.className="bed "+tier+(el.classList.contains("just-crit")?" just-crit":"");
  }
  el.querySelector(".bed-risk b").textContent=fmt(b.fused_risk,2);
  el.querySelector(".bed-sys").textContent=
    (b.primary_system==="none"?"nominal":b.primary_system+" axis")+
    (b.icu_hour!=null?" · ICU h"+b.icu_hour:"");
  el.querySelector(".bed-driver").textContent=b.drivers&&b.drivers.length?("▸ "+b.drivers[0]):"";
  var compact=vital("HR",ed.hr,0)+vital("MAP",en.map,0)+vital("T°",en.temp_c,1)+vital("Lac",en.lactate,1);
  var full=compact+vital("SpO₂",en.spo2,0)+vital("RR",en.resp_rate,0)+
    vital("WBC",en.wbc,1)+vital("HRV",ed.hrv_ms,0);
  el.querySelector(".bed-vitals").innerHTML=(tier==="critical")?full:compact;
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
  [].forEach.call(host.children,function(el){ if(byId[el.dataset.id]) updateBed(el,byId[el.dataset.id]); });
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
function openDossier(id){
  selectedBed=id; dosChart=null;
  if(!dosBuilt){ $("dos-comps").innerHTML=COMP_ORDER.map(function(c){
    return '<div class="dos-comp" data-c="'+c+'"><span class="dos-comp-k">'+c+'</span>'+
      '<span class="dos-comp-track"><i></i></span><span class="dos-comp-v">0.00</span></div>';
  }).join(""); dosBuilt=true; }
  $("dossier").classList.add("open"); $("dos-scrim").classList.add("open");
  $("dossier").setAttribute("aria-hidden","false");
  if(latestBeds[id]) updateDossier(latestBeds[id]);
  fetchChart(id);
  if(dosChartTimer) clearInterval(dosChartTimer);
  dosChartTimer=setInterval(function(){ if(selectedBed) fetchChart(selectedBed); }, 4000);
}
function closeDossier(){ selectedBed=null; dosChart=null;
  if(dosChartTimer){ clearInterval(dosChartTimer); dosChartTimer=null; }
  $("dossier").classList.remove("open"); $("dos-scrim").classList.remove("open");
  $("dossier").setAttribute("aria-hidden","true"); }

function kvHTML(label,val,unit,cls){
  return '<div class="kv'+(cls?" "+cls:"")+'"><span class="k">'+label+'</span>'+
    '<span class="v">'+val+(unit?'<small>'+unit+'</small>':'')+'</span></div>';
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
  COMP_ORDER.forEach(function(c){ var row=$("dos-comps").querySelector('[data-c="'+c+'"]'); if(!row) return;
    var sc=b.components[c]||0, bar=row.querySelector("i");
    bar.style.width=clamp(sc/2.5*100,0,100)+"%"; bar.style.background=COMP_COL[c];
    row.querySelector(".dos-comp-v").textContent=fmt(sc,2); });
  $("dos-edge").innerHTML=
    kvHTML("Heart rate",fmt(e.hr,0),"bpm", e.hr>100?"alert":(e.hr>90?"warn":""))+
    kvHTML("HRV (RMSSD)",fmt(e.hrv_ms,0),"ms")+
    kvHTML("Perfusion idx",fmt(e.perfusion,2),"au")+
    kvHTML("Vasomotion",fmt(e.vasomotion,2),"au");
  $("dos-ent").innerHTML=
    kvHTML("Temperature",fmt(en.temp_c,1),"°C",(en.temp_c>38||en.temp_c<36)?"alert":"")+
    kvHTML("Resp rate",fmt(en.resp_rate,0),"/min",en.resp_rate>22?"warn":"")+
    kvHTML("SpO₂",fmt(en.spo2,0),"%",en.spo2<92?"alert":"")+
    kvHTML("MAP",fmt(en.map,0),"mmHg",en.map<65?"alert":"")+
    kvHTML("ABP variability",fmt(en.abp_var,1),"σ")+
    kvHTML("WBC",fmt(en.wbc,1),"×10³",(en.wbc>12||en.wbc<4)?"alert":"")+
    kvHTML("Lactate",fmt(en.lactate,1),"mmol/L",en.lactate>=4?"alert":(en.lactate>2?"warn":""))+
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
  var ids=nodeOrder, n=ids.length||1;
  var padT=26, padB=12, rowH0=(h-padT-padB)/n;
  // compact mode when the panel is short (e.g. while the terminal is expanded):
  // drop per-row labels/counts so the bars never overlap.
  var compact=rowH0<17;
  if(compact){ padT=16; padB=8; }
  var rowH=(h-padT-padB)/n, barL=compact?12:104, barR=w-(compact?12:52), barW=barR-barL;
  ctx.textBaseline="middle"; ctx.textAlign="left";
  // header: under preempt show the crisis callout; otherwise the class legend
  if(V.mode==="PREEMPT"){
    ctx.font=FS(10.5,600); ctx.fillStyle=C.crit;
    ctx.fillText("⚡ Hard preempt → "+(V.owner||""), 10, 13);
  } else {
    var lx=10;
    LANE_ORDER.forEach(function(c){ ctx.fillStyle=LANE_COL[c]; ctx.fillRect(lx,8,7,7);
      ctx.font=FS(9.5,500); ctx.fillStyle=C.dim; var lbl=LANE_LABEL[c];
      ctx.fillText(lbl,lx+11,13); lx+=11+ctx.measureText(lbl).width+15; });
  }
  // budget guides
  for(var g=1;g<4;g++){ var gxx=barL+barW*g/4;
    ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(gxx+.5,padT-3); ctx.lineTo(gxx+.5,h-padB+3); ctx.stroke(); }

  ids.forEach(function(id,r){ var nd=V.nodes[id], L=V.lanes[id]; if(!nd) return;
    var cyy=padT+rowH*r+rowH/2, bh=Math.min(16,rowH*(compact?0.62:0.46)), by=cyy-bh/2, owner=(V.owner===id);
    if(!compact){ ctx.textAlign="left"; ctx.font=FS(11,owner?600:500);
      ctx.fillStyle=owner?C.crit:(nd.tier==="WATCH"?C.watch:C.ink);
      ctx.fillText((nd.name||id).slice(0,13), 8, cyy); }
    // track
    ctx.fillStyle="rgba(255,255,255,.03)"; roundRect(ctx,barL,by,barW,bh,bh/2); ctx.fill();
    // segments (rounded, tiny gaps)
    var x=barL;
    LANE_ORDER.forEach(function(c){ var seg=clamp((L[c]||0)/V.budget,0,1)*barW; if(seg<=1) return;
      var col=LANE_COL[c], gr=ctx.createLinearGradient(x,0,x+seg,0);
      gr.addColorStop(0,withAlpha(col,.95)); gr.addColorStop(1,withAlpha(col,.62));
      ctx.fillStyle=gr; roundRect(ctx,x,by,Math.max(seg-1.5,1.5),bh,Math.min(bh/2,4)); ctx.fill();
      x+=seg; });
    if(owner){ ctx.strokeStyle=withAlpha(C.crit,.7); ctx.lineWidth=1.2; ctx.shadowColor=withAlpha(C.crit,.55); ctx.shadowBlur=10;
      roundRect(ctx,barL,by,barW,bh,bh/2); ctx.stroke(); ctx.shadowBlur=0; }
    // lane count chip (omitted in compact mode)
    if(!compact){ ctx.textAlign="right"; ctx.font=FM(10.5,500); ctx.fillStyle=C.muted;
      ctx.fillText(Math.round(nd.lanes), w-12, cyy); }
  });
}

/* ============================================================
   PANEL — Latency Envelope
   ============================================================ */
function logY(v,h,pt,pb){ var lv=Math.log10(Math.max(v,1)); return (h-pb)-(lv/5)*(h-pt-pb); }
function drawLatency(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var padT=14, padB=22, x0=42, x1=w-12, span=x1-x0;
  ctx.textBaseline="middle"; ctx.font=FM(9,500);
  for(var e=0;e<=5;e++){ var v=Math.pow(10,e), y=logY(v,h,padT,padB);
    ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x0,y+.5); ctx.lineTo(x1,y+.5); ctx.stroke();
    ctx.fillStyle=C.dim; ctx.textAlign="right"; ctx.fillText(v>=1000?(v/1000)+"k":v+"", x0-6,y); }
  // deadline
  var dy=logY(DEADLINE_US,h,padT,padB);
  ctx.strokeStyle=withAlpha(C.crit,.8); ctx.lineWidth=1.3; ctx.setLineDash([5,4]);
  ctx.beginPath(); ctx.moveTo(x0,dy+.5); ctx.lineTo(x1,dy+.5); ctx.stroke(); ctx.setLineDash([]);
  ctx.textAlign="left"; ctx.font=FS(9.5,600); ctx.fillStyle=C.crit; ctx.fillText("HARD DEADLINE · 50,000 µs", x0+6, dy-8);

  var n=latFill, step=span/(latBuf.length-1);
  function idx(i){ return (latHead-latFill+i+latBuf.length)%latBuf.length; }
  // p95 fill band -> baseline
  ctx.beginPath();
  for(var i=0;i<n;i++){ var xx=x0+i*step, yy=logY(latBuf95[idx(i)],h,padT,padB); i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy); }
  if(n){ ctx.lineTo(x0+(n-1)*step,h-padB); ctx.lineTo(x0,h-padB); ctx.closePath();
    var fg=ctx.createLinearGradient(0,padT,0,h-padB); fg.addColorStop(0,withAlpha(C.cyan,.16)); fg.addColorStop(1,withAlpha(C.cyan,.01));
    ctx.fillStyle=fg; ctx.fill(); }
  // p50 line
  ctx.lineJoin="round"; ctx.lineWidth=1.7; ctx.shadowColor=withAlpha(C.cyan,.6); ctx.shadowBlur=8; ctx.strokeStyle=C.cyan;
  ctx.beginPath(); for(i=0;i<n;i++){ var x=x0+i*step, y=logY(latBuf[idx(i)],h,padT,padB); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
  ctx.stroke(); ctx.shadowBlur=0;
  if(n){ var lx=x0+(n-1)*step, ly=logY(latBuf[idx(n-1)],h,padT,padB);
    ctx.fillStyle=C.cyan; ctx.beginPath(); ctx.arc(lx,ly,2.6,0,7); ctx.fill();
    ctx.strokeStyle=withAlpha(C.cyan,.35); ctx.beginPath(); ctx.arc(lx,ly,5.5,0,7); ctx.stroke(); }

  // floating readout card
  var margin=DEADLINE_US/Math.max(V.lat.p50,.01);
  var rw=196, rx=w-rw-10, ry=10, rh=62;
  ctx.fillStyle="rgba(10,13,20,.82)"; roundRect(ctx,rx,ry,rw,rh,11); ctx.fill();
  ctx.strokeStyle=C.grid; ctx.lineWidth=1; roundRect(ctx,rx,ry,rw,rh,11); ctx.stroke();
  ctx.textAlign="left"; ctx.textBaseline="alphabetic";
  ctx.fillStyle=C.dim; ctx.font=FS(9.5,600); ctx.fillText("p50 conditioning latency", rx+14, ry+17);
  ctx.fillStyle=C.ink; ctx.font=FM(20,600); var vt=fmt(V.lat.p50,2); ctx.fillText(vt, rx+14, ry+40);
  ctx.fillStyle=C.dim; ctx.font=FM(11,500); ctx.fillText("µs", rx+14+ctx.measureText(vt).width+7, ry+40);
  ctx.fillStyle=C.muted; ctx.font=FM(9.5,500); ctx.textAlign="right";
  ctx.fillText("p95 "+fmt(V.lat.p95,1)+"µs", rx+rw-14, ry+40);
  ctx.fillStyle=C.ok; ctx.font=FS(10,600); ctx.textAlign="left";
  ctx.fillText("▼ "+(margin>=1000?(margin/1000).toFixed(1)+"k":Math.round(margin))+"× under deadline", rx+14, ry+55);
  ctx.textBaseline="middle";
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
   MAIN LOOP
   ============================================================ */
var surf={}, spineTick=0;
function setup(){
  surf.mem=new Surface("cv-mem");
  surf.lanes=new Surface("cv-lanes"); surf.lat=new Surface("cv-lat");
  surf.dosMem=new Surface("cv-dos-mem"); surf.dosLat=new Surface("cv-dos-lat");
  surf.dosTrend=new Surface("cv-dos-trend");
  // open the dossier from a bed card
  $("opgrid").addEventListener("click",function(e){ var bed=e.target.closest(".bed");
    if(bed) openDossier(bed.dataset.id); });
  // chaos / replay controls live in the open patient's dossier
  $("dossier").addEventListener("click",function(e){ var b=e.target.closest(".dos-act");
    if(!b||!selectedBed) return; chaos(selectedBed,b.dataset.act);
    b.animate([{opacity:.45},{opacity:1}],{duration:280}); });
  $("dos-close").addEventListener("click",closeDossier);
  $("dos-scrim").addEventListener("click",closeDossier);
  document.addEventListener("keydown",function(e){ if(e.key==="Escape") closeDossier(); });
  connect(); requestAnimationFrame(loop);
}
function loop(){ frame++;
  for(var k in surf) surf[k].fit();
  if(target) ease();
  drawMemory(surf.mem); drawLanes(surf.lanes); drawLatency(surf.lat);
  if(selectedBed){ surf.dosMem.fit(); surf.dosLat.fit(); surf.dosTrend.fit();
    drawTrends(surf.dosTrend); drawBedMemory(surf.dosMem,selectedBed); drawLatency(surf.dosLat); }
  if((spineTick++ %6)===0) renderSpine();
  requestAnimationFrame(loop);
}
if(document.readyState!=="loading") setup();
else document.addEventListener("DOMContentLoaded",setup);

})();
