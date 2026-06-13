/* ============================================================
   AEGIS-OS — Global ICU Tactical Sphere (cockpit engine)
   ------------------------------------------------------------
   One WebSocket feed (~1.6 Hz) -> a smoothed view state -> a single
   60 Hz requestAnimationFrame loop that draws every live visual on
   <canvas>. No chart library, no DOM animation on hot data. The UI
   eases between snapshots so the instrument never "steps".
   Every value on screen traces to a real kernel structure streamed
   from backend/aegis/server.py::_cluster_payload.
   ============================================================ */
(function () {
"use strict";

/* ---------- palette (mirrors webui/os.css tokens) ---------- */
var C = {
  bg:"#06080d", panel:"#0a0d15", grid:"rgba(92,202,240,.06)", gridHi:"rgba(92,202,240,.13)",
  ink:"#f3f6fc", muted:"#9fabc6", dim:"#69748e",
  cyan:"#5ccaf0", blue:"#6f8cf5", violet:"#9a8cf0",
  ok:"#39d9a2", stress:"#f5c451", crit:"#f2606e",
};
var TIER_COL = { STABLE:C.cyan, WATCH:C.stress, CRITICAL:C.crit };
var LANE_COL = { safety:C.ok, ml_tracking:C.cyan, render:C.blue, logging:C.violet };
var LANE_ORDER = ["safety","ml_tracking","render","logging"];
var DEADLINE_US = 50000;          // HARD SYSTEM DEADLINE (µs)

/* ---------- helpers ---------- */
function $(id){ return document.getElementById(id); }
function lerp(a,b,t){ return a + (b-a)*t; }
function clamp(v,a,b){ return v<a?a:v>b?b:v; }
function withAlpha(hex,a){
  var n=parseInt(hex.slice(1),16); return "rgba("+(n>>16&255)+","+(n>>8&255)+","+(n&255)+","+a+")";
}
function fmt(n,d){ return (n||0).toFixed(d==null?0:d); }

/* ---------- high-DPI canvas ---------- */
function Surface(id){
  this.cv=$(id); this.ctx=this.cv.getContext("2d"); this.w=0; this.h=0; this.dpr=1;
}
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
   STATE — raw target (latest snapshot) + smoothed view
   ============================================================ */
var target=null;                  // latest snapshot
var nodeOrder=[];                 // stable node id order
var V={                           // smoothed view state
  cost:0, budget:96, mode:"NORMAL", owner:null,
  nodes:{},                       // id -> {p,slope,lanes,tier,ring,name,x,y}
  lanes:{},                       // id -> {safety,ml_tracking,render,logging}
  lat:{p50:3,p95:5,max:8,mean:4},
  mem:{util:0,cap:1,used:0,base:"0x0",stable:true,published:0,layout:[],rings:{},cursors:[]},
  safety:{certified:0,blocked:0,pass_rate:1},
  devbio:{},
};
var latBuf=new Float32Array(900), latBuf95=new Float32Array(900), latHead=0, latFill=0;
var events=[]; var frame=0;

function ingest(snap){
  target=snap;
  if(!nodeOrder.length) nodeOrder=snap.topology.map(function(n){return n.node_id;});
  // discrete fields snap immediately
  V.mode=snap.scheduler.mode; V.owner=snap.scheduler.owner; V.budget=snap.lane_map.budget;
  snap.topology.forEach(function(n){
    var id=n.node_id;
    if(!V.nodes[id]) V.nodes[id]={p:n.p,slope:n.slope,lanes:n.lanes,tier:n.tier,ring:n.ring,name:n.name};
    var nd=V.nodes[id];
    nd.tier=n.tier; nd.ring=n.ring; nd.name=n.name; nd._tp=n.p; nd._tslope=n.slope; nd._tlanes=n.lanes;
    var ln=snap.lane_map.lanes[id]||{};
    if(!V.lanes[id]) V.lanes[id]={safety:0,ml_tracking:0,render:0,logging:0};
    V.lanes[id]._t=ln;
  });
  V.mem._t=snap.memfabric; V.lat._t=snap.devbio.latency; V.safety._t=snap.safety;
  V.devbio=snap.devbio;
  // precompute memory-region maps ONCE per snapshot (never in the render loop)
  var lay=snap.memfabric.layout||[];
  V.mem._sorted=lay.slice().sort(function(a,b){return a.offset-b.offset;});
  V.mem._byName={}; lay.forEach(function(r){ V.mem._byName[r.name]=r; });
  // interrupt events (dedupe by tick)
  (snap.events||[]).forEach(function(e){
    if(!events.length || events[events.length-1].tick!==e.tick) {
      if(!events.some(function(x){return x.tick===e.tick;})) events.push(e);
    }
  });
  if(events.length>40) events=events.slice(-40);
  renderDock(snap);
}

/* per-frame easing of continuous values toward target */
function ease(){
  var k=0.16;
  V.cost=lerp(V.cost, (target?target.scheduler.global_cost:0), k);
  for(var id in V.nodes){
    var nd=V.nodes[id];
    nd.p=lerp(nd.p, nd._tp==null?nd.p:nd._tp, k);
    nd.slope=lerp(nd.slope, nd._tslope==null?nd.slope:nd._tslope, k);
    nd.lanes=lerp(nd.lanes, nd._tlanes==null?nd.lanes:nd._tlanes, k);
    var L=V.lanes[id], t=L._t||{};
    LANE_ORDER.forEach(function(c){ L[c]=lerp(L[c], t[c]||0, k); });
  }
  if(V.lat._t){ V.lat.p50=lerp(V.lat.p50,V.lat._t.p50_us,k); V.lat.p95=lerp(V.lat.p95,V.lat._t.p95_us,k);
    V.lat.max=lerp(V.lat.max,V.lat._t.max_us,0.05); V.lat.mean=lerp(V.lat.mean,V.lat._t.mean_us,k); }
  if(V.mem._t){ var m=V.mem._t.arena;
    V.mem.util=lerp(V.mem.util,m.utilization,k); V.mem.cap=m.capacity_cells; V.mem.used=m.used_cells;
    V.mem.base=m.base_pointer; V.mem.stable=m.pointer_stable; V.mem.published=V.mem._t.published;
    V.mem.layout=V.mem._t.layout; V.mem.rings=V.mem._t.rings; V.mem.cursors=V.mem._t.cursors; }
  if(V.safety._t){ V.safety.certified=V.safety._t.certified; V.safety.blocked=V.safety._t.blocked;
    V.safety.pass_rate=V.safety._t.pass_rate; }
  // push a real latency sample into the scope buffer every frame (continuous trace)
  latBuf[latHead]=V.lat.p50; latBuf95[latHead]=V.lat.p95;
  latHead=(latHead+1)%latBuf.length; latFill=Math.min(latFill+1,latBuf.length);
}

/* ============================================================
   PANEL D — Monotonic Latency Instrument (log-scale scope)
   ============================================================ */
function logY(v,h,pad){ // 1µs..1e5µs -> bottom..top
  var lo=0, hi=5; var lv=Math.log10(Math.max(v,1));
  return (h-pad) - ((lv-lo)/(hi-lo))*(h-2*pad);
}
function drawLatency(s){
  var ctx=s.ctx,w=s.w,h=s.h,pad=22;
  ctx.clearRect(0,0,w,h);
  // decade grid
  ctx.font='9px '+'"JetBrains Mono",monospace'; ctx.textBaseline="middle";
  for(var e=0;e<=5;e++){
    var v=Math.pow(10,e), y=logY(v,h,pad);
    ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.beginPath();
    ctx.moveTo(40,y+0.5); ctx.lineTo(w-10,y+0.5); ctx.stroke();
    ctx.fillStyle=C.dim; ctx.textAlign="right";
    ctx.fillText(v>=1000?(v/1000)+"k":v+"", 36, y);
  }
  // hard deadline
  var dy=logY(DEADLINE_US,h,pad);
  ctx.strokeStyle=withAlpha(C.crit,.85); ctx.lineWidth=1.4; ctx.setLineDash([6,4]);
  ctx.beginPath(); ctx.moveTo(40,dy+0.5); ctx.lineTo(w-10,dy+0.5); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=C.crit; ctx.textAlign="left";
  ctx.fillText("HARD DEADLINE · 50,000µs", 46, dy-7);

  // trace area
  var x0=40, x1=w-10, span=x1-x0, n=latFill, step=span/(latBuf.length-1);
  function idx(i){ return (latHead - latFill + i + latBuf.length)%latBuf.length; }
  // p95 band (faint)
  ctx.beginPath();
  for(var i=0;i<n;i++){ var xx=x0+i*step, yy=logY(latBuf95[idx(i)],h,pad); i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy); }
  ctx.lineTo(x0+(n-1)*step, h-pad); ctx.lineTo(x0, h-pad); ctx.closePath();
  ctx.fillStyle=withAlpha(C.cyan,.05); ctx.fill();
  // p50 line + glow
  ctx.lineJoin="round"; ctx.lineWidth=1.6;
  ctx.shadowColor=withAlpha(C.cyan,.7); ctx.shadowBlur=8;
  ctx.strokeStyle=C.cyan; ctx.beginPath();
  for(i=0;i<n;i++){ var x=x0+i*step, y=logY(latBuf[idx(i)],h,pad); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }
  ctx.stroke(); ctx.shadowBlur=0;
  // leading dot
  if(n){ var lx=x0+(n-1)*step, ly=logY(latBuf[idx(n-1)],h,pad);
    ctx.fillStyle=C.cyan; ctx.beginPath(); ctx.arc(lx,ly,2.6,0,7); ctx.fill();
    ctx.strokeStyle=withAlpha(C.cyan,.4); ctx.beginPath(); ctx.arc(lx,ly,5.5,0,7); ctx.stroke(); }

  // readout — on a backing plate so it stays legible over the deadline rule
  var margin=DEADLINE_US/Math.max(V.lat.p50,0.01);
  var rw=196, rx=w-rw-8, ry=8, rh=50;
  ctx.fillStyle="rgba(6,8,13,.82)"; roundRect(ctx,rx,ry,rw,rh,6); ctx.fill();
  ctx.strokeStyle=C.grid; ctx.lineWidth=1; roundRect(ctx,rx,ry,rw,rh,6); ctx.stroke();
  ctx.textAlign="left"; ctx.font='11px "JetBrains Mono",monospace';
  ctx.fillStyle=C.cyan; ctx.fillText("p50", rx+12, ry+13);
  ctx.fillStyle=C.ink; ctx.fillText(fmt(V.lat.p50,2)+" µs", rx+44, ry+13);
  ctx.fillStyle=C.muted; ctx.font='10px "JetBrains Mono",monospace';
  ctx.fillText("p95 "+fmt(V.lat.p95,2)+"   max "+fmt(V.lat.max,1), rx+12, ry+27);
  ctx.fillStyle=C.ok;
  ctx.fillText("▼ "+(margin>=1000?(margin/1000).toFixed(1)+"k":Math.round(margin))+"× under deadline", rx+12, ry+41);
}

/* ============================================================
   PANEL C — Thread-Lane Preemption Bars
   ============================================================ */
function drawLanes(s){
  var ctx=s.ctx,w=s.w,h=s.h;
  ctx.clearRect(0,0,w,h);
  var ids=nodeOrder, n=ids.length||1;
  var padT=24, padB=10, rowH=(h-padT-padB)/n, barL=92, barR=w-58, barW=barR-barL;
  ctx.font='10px "JetBrains Mono",monospace'; ctx.textBaseline="middle";
  // mode header + legend (top strip)
  ctx.textAlign="left";
  ctx.fillStyle=V.mode==="PREEMPT"?C.crit:C.dim; ctx.font='9px "JetBrains Mono",monospace';
  ctx.fillText(V.mode==="PREEMPT"?("⚡ HARD PREEMPT → "+(V.owner||"")):"NORMAL · lanes ∝ risk", barL, 12);
  var lx=barL+150;
  LANE_ORDER.forEach(function(c){ ctx.fillStyle=LANE_COL[c]; ctx.fillRect(lx,7,7,7);
    ctx.fillStyle=C.dim; ctx.fillText(c.replace("_"," "), lx+11, 12); lx+=11+ctx.measureText(c.replace("_"," ")).width+14; });
  // budget gridlines
  for(var g=0;g<=4;g++){ var gx=barL+barW*g/4;
    ctx.strokeStyle=C.grid; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(gx+.5,padT-2); ctx.lineTo(gx+.5,h-padB+2); ctx.stroke(); }

  ids.forEach(function(id,r){
    var nd=V.nodes[id], L=V.lanes[id]; if(!nd) return;
    var cy=padT+rowH*r+rowH/2, bh=Math.min(18,rowH*0.5), by=cy-bh/2;
    var owner=(V.owner===id);
    // name
    ctx.textAlign="left"; ctx.fillStyle=owner?C.crit:(nd.tier==="WATCH"?C.stress:C.ink);
    ctx.font=(owner?'600 ':'')+'10.5px "JetBrains Mono",monospace';
    ctx.fillText((nd.name||id).slice(0,12), 8, cy);
    // bar track
    ctx.fillStyle="rgba(255,255,255,.025)"; roundRect(ctx,barL,by,barW,bh,3); ctx.fill();
    // segments
    var x=barL;
    LANE_ORDER.forEach(function(c){
      var frac=clamp((L[c]||0)/V.budget,0,1), seg=frac*barW; if(seg<=0.2) return;
      var col=LANE_COL[c];
      var grad=ctx.createLinearGradient(x,0,x+seg,0);
      grad.addColorStop(0,withAlpha(col,.95)); grad.addColorStop(1,withAlpha(col,.6));
      ctx.fillStyle=grad; roundRect(ctx,x,by,seg,bh,2); ctx.fill();
      if(c==="safety"){ // safety floor marker (never collapses)
        ctx.fillStyle=withAlpha(C.ok,.9); ctx.fillRect(x,by,1.5,bh);
      }
      x+=seg;
    });
    // owner glow outline
    if(owner){ ctx.strokeStyle=withAlpha(C.crit,.8); ctx.lineWidth=1.2;
      ctx.shadowColor=withAlpha(C.crit,.6); ctx.shadowBlur=10;
      roundRect(ctx,barL,by,barW,bh,3); ctx.stroke(); ctx.shadowBlur=0; }
    // total lanes
    ctx.textAlign="right"; ctx.fillStyle=C.muted; ctx.font='10px "JetBrains Mono",monospace';
    ctx.fillText(Math.round(nd.lanes)+"L", w-10, cy);
  });
}

/* ============================================================
   PANEL A — Global Node Topology
   ============================================================ */
function drawTopology(s){
  var ctx=s.ctx,w=s.w,h=s.h;
  ctx.clearRect(0,0,w,h);
  var cx=w/2, cy=h/2, R=Math.min(w*0.40, (h-96)*0.5);
  var ids=nodeOrder, n=ids.length||1;
  // position nodes on a ring
  ids.forEach(function(id,i){
    var a=-Math.PI/2 + i*2*Math.PI/n;
    var nd=V.nodes[id]; if(!nd) return;
    nd.x=cx+Math.cos(a)*R; nd.y=cy+Math.sin(a)*R;
  });
  // edges hub -> node (thickness ∝ lane share, animated flow)
  var totalLanes=0; ids.forEach(function(id){ totalLanes+=V.nodes[id]?V.nodes[id].lanes:0; });
  ids.forEach(function(id){
    var nd=V.nodes[id]; if(!nd) return;
    var share=nd.lanes/Math.max(totalLanes,1);
    var col=TIER_COL[nd.tier]||C.cyan, lw=1+share*10;
    ctx.strokeStyle=withAlpha(col,.28); ctx.lineWidth=lw;
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nd.x,nd.y); ctx.stroke();
    // flow particles toward node, speed ∝ share (more compute = faster flow)
    var dx=nd.x-cx, dy=nd.y-cy, len=Math.hypot(dx,dy), ux=dx/len, uy=dy/len;
    var speed=0.4+share*3.2, nDots=3;
    for(var k=0;k<nDots;k++){
      var ph=((frame*speed*0.01)+k/nDots)%1;
      var px=cx+ux*len*ph, py=cy+uy*len*ph;
      ctx.fillStyle=withAlpha(col, .9*(1-ph*0.5));
      ctx.beginPath(); ctx.arc(px,py,1.4+share*2.2,0,7); ctx.fill();
    }
  });
  // hub
  ctx.fillStyle=withAlpha(C.blue,.10); ctx.beginPath(); ctx.arc(cx,cy,26,0,7); ctx.fill();
  ctx.strokeStyle=withAlpha(C.blue,.5); ctx.lineWidth=1; ctx.beginPath(); ctx.arc(cx,cy,26,0,7); ctx.stroke();
  ctx.fillStyle=C.muted; ctx.font='8px "JetBrains Mono",monospace'; ctx.textAlign="center"; ctx.textBaseline="middle";
  ctx.fillText("SCHED", cx, cy-7);
  ctx.fillStyle=C.ink; ctx.font='12px "JetBrains Mono",monospace';
  ctx.fillText("C="+fmt(V.cost,2), cx, cy+7);
  // nodes
  ids.forEach(function(id){
    var nd=V.nodes[id]; if(!nd) return;
    var col=TIER_COL[nd.tier]||C.cyan;
    var rad=9+clamp(nd.lanes/V.budget,0,1)*14;
    var crit=nd.tier==="CRITICAL";
    var pulse=crit?(1+Math.sin(frame*0.12)*0.12):1;
    // halo
    ctx.fillStyle=withAlpha(col, crit?.18:.10);
    ctx.beginPath(); ctx.arc(nd.x,nd.y,rad*1.9*pulse,0,7); ctx.fill();
    // ring
    ctx.strokeStyle=col; ctx.lineWidth=crit?2.4:1.8;
    if(crit){ ctx.shadowColor=col; ctx.shadowBlur=16; }
    ctx.beginPath(); ctx.arc(nd.x,nd.y,rad*pulse,0,7); ctx.stroke(); ctx.shadowBlur=0;
    // core
    ctx.fillStyle=withAlpha(col,.22); ctx.beginPath(); ctx.arc(nd.x,nd.y,rad-4,0,7); ctx.fill();
    // label — kept clear of the ring and clamped inside the panel
    var below=nd.y>=cy;
    var ly1=clamp(nd.y+(below?rad+13:-rad-25), 12, h-14);
    var ly2=clamp(nd.y+(below?rad+25:-rad-13), 24, h-4);
    ctx.fillStyle=C.ink; ctx.font='10px "JetBrains Mono",monospace'; ctx.textBaseline="middle";
    ctx.fillText((nd.name||id).split(",")[0], nd.x, ly1);
    ctx.fillStyle=col; ctx.font='9px "JetBrains Mono",monospace';
    ctx.fillText("p"+fmt(nd.p,2), nd.x, ly2);
  });
}

/* ============================================================
   PANEL B — Kernel Memory Address Matrix
   ============================================================ */
function drawMemory(s){
  var ctx=s.ctx,w=s.w,h=s.h;
  ctx.clearRect(0,0,w,h);
  var headH=30, pad=12;
  // header readout
  ctx.textAlign="left"; ctx.textBaseline="middle"; ctx.font='10px "JetBrains Mono",monospace';
  ctx.fillStyle=C.dim; ctx.fillText("BASE", pad, 12);
  ctx.fillStyle=C.cyan; ctx.fillText(V.mem.base||"0x0", pad+34, 12);
  ctx.fillStyle=V.mem.stable?C.ok:C.crit;
  ctx.fillText(V.mem.stable?"PTR STABLE ✓":"PTR MOVED ✗", pad+150, 12);
  ctx.fillStyle=C.dim; ctx.textAlign="right";
  ctx.fillText("util "+fmt(V.mem.util*100,2)+"%  ·  published "+ (V.mem.published||0).toLocaleString(), w-pad, 12);
  ctx.textAlign="left";

  var sorted=V.mem._sorted, byName=V.mem._byName||{};
  if(!sorted||!sorted.length){ return; }
  var used=Math.max(V.mem.used,1);
  // cell grid
  var gx=pad, gy=headH, gw=w-pad*2, gh=h-headH-pad;
  var cols=Math.max(24,Math.floor(gw/9)), rows=Math.max(10,Math.floor(gh/9));
  var cw=gw/cols, ch=gh/rows, cells=cols*rows, factor=used/cells;
  function regCol(name){
    if(/\.t$/.test(name)) return withAlpha(C.dim,.12);
    var m=name&&name.match(/node_(\d+)/), ni=m?(+m[1]):0;
    return withAlpha([C.cyan,C.blue,C.violet][ni%3],.16);
  }
  // draw cells — regions are sorted & contiguous, so walk a moving pointer
  // (O(cells+regions), zero per-cell allocation) instead of scanning each cell.
  var ri=0, empty="rgba(120,140,180,.05)";
  for(var ci=0;ci<cells;ci++){
    var idx=(ci*factor)|0, c=ci%cols, rr=(ci/cols)|0;
    while(ri<sorted.length && idx>=sorted[ri].offset+sorted[ri].length) ri++;
    var reg=(ri<sorted.length && idx>=sorted[ri].offset)?sorted[ri]:null;
    ctx.fillStyle=reg?regCol(reg.name):empty;
    ctx.fillRect(gx+c*cw+0.5, gy+rr*ch+0.5, cw-1, ch-1);
  }
  function cell(arenaIdx){ var i=clamp((arenaIdx/factor)|0,0,cells-1); return {c:i%cols, r:(i/cols)|0}; }
  // write pointers (bright sweep)
  var rings=V.mem.rings||{};
  ctx.shadowColor=C.cyan; ctx.shadowBlur=7; ctx.fillStyle=C.cyan;
  for(var path in rings){
    var nm=path.replace("/dev/bio/","").replace(/\//g,".")+".v", reg2=byName[nm]; if(!reg2) continue;
    var nf=(reg2.shape&&reg2.shape[1])||1, p=cell(reg2.offset+rings[path].write_ptr*nf);
    ctx.fillRect(gx+p.c*cw, gy+p.r*ch, cw, ch);
  }
  ctx.shadowBlur=0;
  // read cursors (amber)
  ctx.fillStyle=withAlpha(C.stress,.85);
  (V.mem.cursors||[]).forEach(function(cur){
    var reg3=byName[cur.ring+".v"]; if(!reg3) return;
    var nf=(reg3.shape&&reg3.shape[1])||1, p=cell(reg3.offset+cur.read_ptr*nf);
    ctx.fillRect(gx+p.c*cw+cw*0.3, gy+p.r*ch+ch*0.3, cw*0.4, ch*0.4);
  });
  // legend
  ctx.font='8px "JetBrains Mono",monospace'; ctx.textBaseline="middle";
  ctx.fillStyle=C.cyan; ctx.fillRect(gx, h-7, 6,6); ctx.fillStyle=C.muted; ctx.fillText("write ptr", gx+10,h-4);
  ctx.fillStyle=C.stress; ctx.fillRect(gx+70, h-7, 6,6); ctx.fillStyle=C.muted; ctx.fillText("read cursor", gx+80,h-4);
}

/* ---------- rounded rect ---------- */
function roundRect(ctx,x,y,w,h,r){
  r=Math.min(r,h/2,w/2); ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
}

/* ============================================================
   SPINE + DOCK (DOM, updated at low rate)
   ============================================================ */
function renderSpine(){
  $("s-cost").textContent=fmt(V.cost,2);
  var m=$("s-mode"); m.textContent=V.mode; m.className=V.mode==="PREEMPT"?"mode-preempt":"mode-normal";
  $("s-int").textContent=events.length;
  $("s-cage").textContent=V.safety.certified.toLocaleString()+" / "+V.safety.blocked+" · "+Math.round((V.safety.pass_rate||1)*100)+"%";
  $("s-tick").textContent=target?target.scheduler.tick:0;
  $("s-clock").textContent=new Date().toLocaleTimeString("en-GB");
  // ticker
  if(events.length){
    var last=events.slice(-3).reverse().map(function(e){
      return '<span class="hot">⚡ t'+e.tick+' '+e.owner+' ('+e.reason+' p='+fmt(e.p,2)+')</span>';
    }).join("   ·   ");
    $("tk-body").innerHTML=last;
  }
}
function renderDock(snap){
  var host=$("dock-nodes");
  if(host.childNodes.length!==snap.topology.length){
    host.innerHTML="";
    snap.topology.forEach(function(n){
      var el=document.createElement("div"); el.className="cnode"; el.dataset.id=n.node_id;
      el.innerHTML=
        '<div class="cnode-top"><span class="cnode-dot"></span>'+
        '<span class="cnode-name">'+n.name+'</span><span class="cnode-p">p0.00</span></div>'+
        '<div class="cnode-btns">'+
        '<button class="cbtn crisis" data-act="crisis">CRISIS</button>'+
        '<button class="cbtn clear" data-act="clear">CLEAR</button>'+
        '<button class="cbtn noise" data-act="noise">NOISE</button></div>';
      host.appendChild(el);
    });
  }
  snap.topology.forEach(function(n){
    var el=host.querySelector('[data-id="'+n.node_id+'"]'); if(!el) return;
    var t=n.tier.toLowerCase()==="critical"?"crit":n.tier.toLowerCase()==="watch"?"watch":"";
    el.className="cnode"+(t?" "+t:"");
    el.querySelector(".cnode-dot").className="cnode-dot"+(t?" "+t:"");
    el.querySelector(".cnode-p").textContent="p"+fmt(n.p,2);
  });
}
$("dock-nodes").addEventListener("click",function(e){
  var b=e.target.closest(".cbtn"); if(!b) return;
  var node=b.closest(".cnode").dataset.id, act=b.dataset.act;
  fetch(window.AEGIS.http("/aegis/cluster/chaos"),{
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({node:node,action:act,intensity:1.0})
  }).catch(function(){});
  b.animate([{opacity:.4},{opacity:1}],{duration:300});
});

/* ============================================================
   WEBSOCKET — cluster stream w/ reconnect
   ============================================================ */
var ws=null, connected=false, lastMsg=0;
function clusterWS(){
  return window.AEGIS.base().replace(/^http/,"ws")+"/aegis/cluster/stream";
}
function connect(){
  try{ ws=new WebSocket(clusterWS()); }catch(e){ setTimeout(connect,1500); return; }
  ws.onopen=function(){ connected=true; setConn(true); };
  ws.onmessage=function(e){ lastMsg=performance.now();
    try{ ingest(JSON.parse(e.data)); }catch(err){ console.error(err); } };
  ws.onclose=function(){ connected=false; setConn(false); setTimeout(connect,1500); };
  ws.onerror=function(){ try{ws.close();}catch(e){} };
}
function setConn(live){
  $("conn-dot").className="conn-dot"+(live?" live":"");
  $("conn-txt").textContent=live?"LIVE · /aegis/cluster/stream":"RECONNECTING";
}
// REST fallback poll if WS silent
setInterval(function(){
  if(connected && performance.now()-lastMsg<4000) return;
  fetch(window.AEGIS.http("/aegis/cluster/snapshot")).then(function(r){return r.json();})
    .then(ingest).catch(function(){});
}, 2500);

/* ============================================================
   MAIN LOOP — single 60 Hz rAF
   ============================================================ */
var surf={};
function setup(){
  surf.topo=new Surface("cv-topo"); surf.mem=new Surface("cv-mem");
  surf.lanes=new Surface("cv-lanes"); surf.lat=new Surface("cv-lat");
  connect(); requestAnimationFrame(loop);
}
var spineTick=0;
function loop(){
  frame++;
  for(var k in surf) surf[k].fit();
  if(target) ease();
  drawTopology(surf.topo);
  drawMemory(surf.mem);
  drawLanes(surf.lanes);
  drawLatency(surf.lat);
  if((spineTick++ %6)===0) renderSpine();
  requestAnimationFrame(loop);
}
if(document.readyState!=="loading") setup();
else document.addEventListener("DOMContentLoaded",setup);

})();
