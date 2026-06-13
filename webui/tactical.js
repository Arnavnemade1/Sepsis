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
  renderDock(snap);
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
   PANEL — Cluster Topology
   ============================================================ */
function drawTopology(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  // ellipse fills the wide panel horizontally; Ry keeps ~50px label room
  var cx=w/2, cy=h/2, Ry=Math.max(54,h/2-56), Rx=Math.min(w*0.30,Ry*1.95);
  var ids=nodeOrder, n=ids.length||1;
  ctx.textAlign="center"; ctx.textBaseline="middle";

  // faint elliptical guide rings
  for(var gi=1;gi<=3;gi++){ ctx.strokeStyle=withAlpha(C.indigo,.05*(4-gi)/3); ctx.lineWidth=1;
    ctx.beginPath(); ctx.ellipse(cx,cy,Rx*gi/3,Ry*gi/3,0,0,7); ctx.stroke(); }

  ids.forEach(function(id,i){ var a=-Math.PI/2+i*2*Math.PI/n, nd=V.nodes[id]; if(!nd) return;
    nd.x=cx+Math.cos(a)*Rx; nd.y=cy+Math.sin(a)*Ry; });

  var totalLanes=0; ids.forEach(function(id){ totalLanes+=V.nodes[id]?V.nodes[id].lanes:0; });

  // edges (gradient hub->node) + flow particles
  ids.forEach(function(id){ var nd=V.nodes[id]; if(!nd) return;
    var share=nd.lanes/Math.max(totalLanes,1), col=TIER_COL[nd.tier]||C.cyan;
    var g=ctx.createLinearGradient(cx,cy,nd.x,nd.y);
    g.addColorStop(0,withAlpha(C.indigo,.22)); g.addColorStop(1,withAlpha(col,.42));
    ctx.strokeStyle=g; ctx.lineWidth=1+share*9; ctx.lineCap="round";
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nd.x,nd.y); ctx.stroke();
    var dx=nd.x-cx,dy=nd.y-cy,len=Math.hypot(dx,dy)||1,ux=dx/len,uy=dy/len, speed=0.5+share*3.0;
    for(var k=0;k<3;k++){ var ph=((frame*speed*0.01)+k/3)%1, px=cx+ux*len*ph, py=cy+uy*len*ph;
      ctx.fillStyle=withAlpha(col,.85*(1-ph*0.55)); ctx.beginPath();
      ctx.arc(px,py,1.2+share*2.4,0,7); ctx.fill(); } });

  // hub
  var hg=ctx.createRadialGradient(cx,cy,2,cx,cy,30);
  hg.addColorStop(0,withAlpha(C.indigo,.30)); hg.addColorStop(1,withAlpha(C.indigo,.04));
  ctx.fillStyle=hg; ctx.beginPath(); ctx.arc(cx,cy,30,0,7); ctx.fill();
  ctx.strokeStyle=withAlpha(C.indigo,.45); ctx.lineWidth=1; ctx.beginPath(); ctx.arc(cx,cy,30,0,7); ctx.stroke();
  ctx.fillStyle=C.dim; ctx.font=FS(8.5,600); ctx.fillText("Σ w·P", cx, cy-9);
  ctx.fillStyle=C.ink; ctx.font=FM(15,600); ctx.fillText(fmt(V.cost,2), cx, cy+8);

  // nodes
  ids.forEach(function(id){ var nd=V.nodes[id]; if(!nd) return;
    var col=TIER_COL[nd.tier]||C.cyan, rad=10+clamp(nd.lanes/V.budget,0,1)*15;
    var crit=nd.tier==="CRITICAL", pulse=crit?1+Math.sin(frame*0.11)*0.10:1, x=nd.x, y=nd.y;
    // halo
    var hgl=ctx.createRadialGradient(x,y,rad*0.5,x,y,rad*2.1*pulse);
    hgl.addColorStop(0,withAlpha(col,crit?.30:.16)); hgl.addColorStop(1,withAlpha(col,0));
    ctx.fillStyle=hgl; ctx.beginPath(); ctx.arc(x,y,rad*2.1*pulse,0,7); ctx.fill();
    // core
    var cg=ctx.createRadialGradient(x-rad*0.3,y-rad*0.3,1,x,y,rad);
    cg.addColorStop(0,withAlpha(col,.34)); cg.addColorStop(1,withAlpha(col,.10));
    ctx.fillStyle=cg; ctx.beginPath(); ctx.arc(x,y,rad*pulse,0,7); ctx.fill();
    // ring
    ctx.strokeStyle=col; ctx.lineWidth=crit?2.2:1.5;
    if(crit){ ctx.shadowColor=col; ctx.shadowBlur=16; }
    ctx.beginPath(); ctx.arc(x,y,rad*pulse,0,7); ctx.stroke(); ctx.shadowBlur=0;
    // labels
    var below=y>=cy, ly1=clamp(y+(below?rad+15:-rad-26),14,h-16), ly2=clamp(y+(below?rad+28:-rad-13),26,h-5);
    ctx.fillStyle=C.ink; ctx.font=FS(11,600); ctx.fillText((nd.name||id).split(",")[0], x, ly1);
    ctx.fillStyle=withAlpha(col,.95); ctx.font=FM(9.5,500); ctx.fillText("p "+fmt(nd.p,2), x, ly2);
  });
}

/* ============================================================
   PANEL — Memory Fabric
   ============================================================ */
function drawMemory(s){
  var ctx=s.ctx,w=s.w,h=s.h; ctx.clearRect(0,0,w,h);
  var pad=14, headH=34;
  ctx.textBaseline="middle"; ctx.textAlign="left";
  ctx.font=FS(9.5,600); ctx.fillStyle=C.dim; ctx.fillText("BASE", pad,14);
  ctx.font=FM(11,500); ctx.fillStyle=C.muted; ctx.fillText(V.mem.base||"0x0", pad+36,14);
  var sx=pad+170;
  ctx.fillStyle=V.mem.stable?C.ok:C.crit; ctx.beginPath(); ctx.arc(sx,14,3.2,0,7); ctx.fill();
  ctx.font=FS(10.5,500); ctx.fillStyle=V.mem.stable?C.ok:C.crit;
  ctx.fillText(V.mem.stable?"Pointer stable":"Pointer moved", sx+9,14);
  ctx.textAlign="right";
  ctx.font=FS(9.5,600); ctx.fillStyle=C.dim; ctx.fillText("UTIL", w-118,14);
  ctx.font=FM(11,500); ctx.fillStyle=C.muted; ctx.fillText(fmt(V.mem.util*100,2)+"%", w-78,14);
  ctx.font=FS(9.5,600); ctx.fillStyle=C.dim; ctx.textAlign="right"; ctx.fillText("PUBLISHED", w-pad,28);
  ctx.font=FM(10.5,500); ctx.fillStyle=C.muted; ctx.fillText((V.mem.published||0).toLocaleString(), w-pad,14);
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
  var padT=26, padB=12, rowH=(h-padT-padB)/n, barL=104, barR=w-52, barW=barR-barL;
  ctx.textBaseline="middle";
  // header: mode + legend
  ctx.textAlign="left"; ctx.font=FS(10.5,600);
  ctx.fillStyle=V.mode==="PREEMPT"?C.crit:C.muted;
  ctx.fillText(V.mode==="PREEMPT"?("Hard preempt → "+(V.owner||"")):"Normal · lanes ∝ risk", barL, 13);
  var lx=barR; ctx.textAlign="right";
  for(var i=LANE_ORDER.length-1;i>=0;i--){ var c=LANE_ORDER[i], lbl=LANE_LABEL[c];
    ctx.font=FS(9.5,500); ctx.fillStyle=C.dim; ctx.fillText(lbl,lx,13);
    var tw=ctx.measureText(lbl).width; ctx.fillStyle=LANE_COL[c];
    ctx.fillRect(lx-tw-12,9,7,7); lx-=tw+24; }
  // budget guides
  for(var g=1;g<4;g++){ var gxx=barL+barW*g/4;
    ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(gxx+.5,padT-3); ctx.lineTo(gxx+.5,h-padB+3); ctx.stroke(); }

  ids.forEach(function(id,r){ var nd=V.nodes[id], L=V.lanes[id]; if(!nd) return;
    var cyy=padT+rowH*r+rowH/2, bh=Math.min(16,rowH*0.46), by=cyy-bh/2, owner=(V.owner===id);
    ctx.textAlign="left"; ctx.font=FS(11,owner?600:500);
    ctx.fillStyle=owner?C.crit:(nd.tier==="WATCH"?C.watch:C.ink);
    ctx.fillText((nd.name||id).slice(0,13), 8, cyy);
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
    // lane count chip
    ctx.textAlign="right"; ctx.font=FM(10.5,500); ctx.fillStyle=C.muted;
    ctx.fillText(Math.round(nd.lanes), w-12, cyy);
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
function renderDock(snap){
  var host=$("dock-nodes");
  if(host.childElementCount!==snap.topology.length){ host.innerHTML="";
    snap.topology.forEach(function(nd){ var el=document.createElement("div"); el.className="patient"; el.dataset.id=nd.node_id;
      el.innerHTML='<div class="p-row"><span class="p-name">'+nd.name+'</span><span class="p-risk">p&nbsp;<b>0.00</b></span></div>'+
        '<div class="p-meta">bed '+nd.node_id.replace("node_","#")+' · w '+nd.weight+'</div>'+
        '<div class="p-actions"><button class="b-crisis" data-act="crisis">Crisis</button>'+
        '<button class="b-clear" data-act="clear">Clear</button>'+
        '<button class="b-noise" data-act="noise">Noise</button></div>';
      host.appendChild(el); }); }
  snap.topology.forEach(function(nd){ var el=host.querySelector('[data-id="'+nd.node_id+'"]'); if(!el) return;
    var t=nd.tier==="CRITICAL"?"crit":nd.tier==="WATCH"?"watch":"";
    el.className="patient"+(t?" "+t:"");
    el.querySelector(".p-risk").innerHTML="p&nbsp;<b>"+fmt(nd.p,2)+"</b>"; });
}
$("dock-nodes").addEventListener("click",function(e){ var b=e.target.closest("button"); if(!b) return;
  var node=b.closest(".patient").dataset.id;
  fetch(window.AEGIS.http("/aegis/cluster/chaos"),{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({node:node,action:b.dataset.act,intensity:1.0})}).catch(function(){});
  b.animate([{opacity:.45},{opacity:1}],{duration:280});
});

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
  surf.topo=new Surface("cv-topo"); surf.mem=new Surface("cv-mem");
  surf.lanes=new Surface("cv-lanes"); surf.lat=new Surface("cv-lat");
  connect(); requestAnimationFrame(loop);
}
function loop(){ frame++;
  for(var k in surf) surf[k].fit();
  if(target) ease();
  drawTopology(surf.topo); drawMemory(surf.mem); drawLanes(surf.lanes); drawLatency(surf.lat);
  if((spineTick++ %6)===0) renderSpine();
  requestAnimationFrame(loop);
}
if(document.readyState!=="loading") setup();
else document.addEventListener("DOMContentLoaded",setup);

})();
