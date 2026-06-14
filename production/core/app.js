// GRADIS Control — 실시간. EventSource로 서버가 밀어주는 거 받아서 표에 박는다.
// 프레임워크 없음. 빌드 스텝 없음. 그냥 돈다.

const $ = (id) => document.getElementById(id);
const fleet = {};   // node_id -> row
let firstInc = true;

function ts(s){ const d = new Date(s*1000); return d.toLocaleTimeString(); }
function ago(s){ const k = Math.max(0, Math.floor(Date.now()/1000 - s));
  return k<60 ? k+"s" : Math.floor(k/60)+"m"; }

function riskCls(r){ return r>=0.8 ? "risk-hi" : r>=0.5 ? "risk-md" : ""; }

function incRow(x){
  const clip = x.clip
    ? `<a href="${x.clip}" target="_blank"><img class="clip" src="${x.clip}"></a>`
    : '<span class="muted">discarded</span>';
  const ackBtn = x.status === "ACK"
    ? '<span class="muted">✓ dispatched</span>'
    : `<button class="ack" onclick="ack(${x.id})">DISPATCH</button>`;
  return `<tr class="inc ${x.status}" id="inc-${x.id}">
    <td>${x.id}</td>
    <td>${ts(x.ts_utc)}</td>
    <td class="kind">${x.kind}</td>
    <td class="${riskCls(x.risk)}">${(x.risk*100|0)}%</td>
    <td>${(x.confidence*100|0)}%</td>
    <td>${x.location||""}</td>
    <td>${x.drone_id}</td>
    <td>${clip}</td>
    <td>${x.status}</td>
    <td>${ackBtn}</td></tr>`;
}

function addIncident(x, flash){
  if(firstInc){ $("incbody").innerHTML=""; firstInc=false; }
  const existing = $("inc-"+x.id);
  if(existing){ existing.outerHTML = incRow(x); return; }
  $("incbody").insertAdjacentHTML("afterbegin", incRow(x));
  if(flash){
    const el = $("inc-"+x.id);
    el.classList.add("flash");
    setTimeout(()=>el && el.classList.remove("flash"), 1500);
    try{ navigator.vibrate && navigator.vibrate(60); }catch(e){}
  }
}

window.ack = function(id){
  fetch(`/api/incidents/${id}/ack`, {method:"POST"});
};

function renderFleet(){
  const ids = Object.keys(fleet).sort();
  $("fleetcount").textContent = ids.length ? `${ids.length} online` : "";
  if(!ids.length) return;
  $("fleetbody").innerHTML = ids.map(id=>{
    const f = fleet[id];
    const b = f.battery==null ? "—" : (f.battery|0)+"%";
    const bc = (f.battery!=null && f.battery<25) ? "batt b-lo" : "batt";
    const ll = (f.lat!=null) ? `${f.lat.toFixed(4)}, ${f.lon.toFixed(4)}` : "—";
    return `<tr><td><b>${id}</b></td><td>${f.node_type}</td><td>${f.state}</td>
      <td class="${bc}">${b}</td><td>${ll}</td><td>${ago(f.last_seen)} ago</td></tr>`;
  }).join("");
}

// 초기 적재
fetch("/api/incidents?limit=100").then(r=>r.json()).then(rows=>{
  rows.reverse().forEach(x=>addIncident(x,false));
});
fetch("/api/fleet").then(r=>r.json()).then(rows=>{
  rows.forEach(f=>fleet[f.node_id]=f); renderFleet();
});

// 라이브 스트림
const es = new EventSource("/api/events");
es.onopen  = ()=> $("conn").textContent = "● live";
es.onerror = ()=> $("conn").textContent = "○ reconnecting…";
es.addEventListener("incident", e=> addIncident(JSON.parse(e.data), true));
es.addEventListener("update",   e=> addIncident(JSON.parse(e.data), false));
es.addEventListener("fleet",    e=>{ const f=JSON.parse(e.data); fleet[f.node_id]=f; renderFleet(); });

setInterval(renderFleet, 5000); // "last seen" 갱신
