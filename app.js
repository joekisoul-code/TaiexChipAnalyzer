/* 台股籌碼分析 PWA 前端 */
const $ = (s) => document.querySelector(s);
const fmt = (v, d = 2, sign = false) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (sign && v > 0 ? "+" : "") + Number(v).toLocaleString("zh-TW", { minimumFractionDigits: d, maximumFractionDigits: d });
const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "muted");
const pct = (v, d = 2) => (v === null || v === undefined) ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d) + "%";
let charts = {};

// ---------------- 導覽
document.querySelectorAll("nav.tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("nav.tabs button").forEach((x) => x.classList.remove("active"));
  document.querySelectorAll(".page").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  $("#page-" + b.dataset.page).classList.add("active");
  load(b.dataset.page);
  window.scrollTo(0, 0);
}));
$("#btn-refresh").addEventListener("click", () => loadAll(true));

// 兩種模式：api = 本機/區網 FastAPI；static = GitHub Pages 靜態 JSON (無伺服器，由 GitHub Actions 排程更新)
const STATIC = window.API_MODE === "static";
const ENDPOINT = { "/api/realtime": "data/realtime.json", "/api/market": "data/market.json", "/api/forecast": "data/forecast.json",
                   "/api/watchlist": "data/watchlist.json", "/api/global": "data/global.json", "/api/status": "data/status.json" };
async function api(path) {
  const url = STATIC ? ENDPOINT[path] + "?t=" + Math.floor(Date.now() / 60000) : path;
  const r = await fetch(url, { cache: "no-store" });
  if (r.status === 202) return { _pending: true, ...(await r.json()) };
  if (r.status === 404 && STATIC) return { _pending: true, message: "資料尚未產生 (GitHub Actions 尚未執行)" };
  if (!r.ok) throw new Error(path + " " + r.status);
  return r.json();
}

function card(label, value, delta, deltaCls = "") {
  return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div><div class="delta ${deltaCls}">${delta || ""}</div></div>`;
}
function bar(name, score, max = 2, text = "") {
  const w = Math.min(100, Math.abs(score) / max * 50);
  const left = score >= 0 ? 50 : 50 - w;
  return `<div class="bar"><div class="name">${name}</div><div class="track"><div class="fill ${score >= 0 ? "" : ""}" style="left:${left}%;width:${w}%;background:${score >= 0 ? "var(--up)" : "var(--down)"}"></div></div><div class="num ${cls(score)}">${fmt(score, 2, true)}</div></div>` + (text ? `<div class="note">${text}</div>` : "");
}
function mkChart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($("#" + id).getContext("2d"), cfg);
}

// ---------------- 即時
async function loadLive() {
  const s = await api("/api/realtime");
  const idx = s.taiex || {}, otc = s.otc || {}, tx = s.tx_night || s.tx || {}, sc = s.score || {};
  $("#hdr-sub").textContent = `${{ open: "盤中", pre: "開盤前", post: "收盤後", night: "夜盤", closed: "休市" }[s.phase] || s.phase}｜${s.ts}${STATIC ? "（靜態資料，排程更新）" : ""}`;
  $("#live-cards").innerHTML =
    card(`加權指數 ${idx.time || ""}`, fmt(idx.last, 2), `${fmt(idx.chg, 2, true)} (${pct(idx.chg_pct)})`, cls(idx.chg)) +
    card("櫃買", fmt(otc.last, 2), pct(otc.chg_pct), cls(otc.chg_pct)) +
    card(`台指期 ${s.tx_night ? "夜盤" : "日盤"}`, fmt(tx.last, 0), `${pct(tx.change_pct)}｜價差 ${fmt(tx.basis, 0, true)}`, cls(tx.change_pct)) +
    card("盤勢即時分", fmt(sc.score, 1, true), sc.label, cls(sc.score)) +
    card("量能推估", (s.vol_pace ? s.vol_pace.toFixed(2) + "x" : "—"), s.amount_projected ? fmt(s.amount_projected, 0) + " 億" : "") +
    (s.vixtwn ? card("台指 VIX", fmt(s.vixtwn.last, 2), s.vixtwn.last > 30 ? "恐慌區" : s.vixtwn.last < 15 ? "自滿區" : "正常", s.vixtwn.last > 30 ? "up" : "muted") : "");
  $("#rt-score").textContent = "";
  $("#rt-combined").textContent = s.combined || "";
  $("#rt-parts").innerHTML = (sc.parts || []).map((p) => bar(p.name, p.score, 2, p.text)).join("");
  const bars = (s.intraday && s.intraday.bars) || [];
  if (bars.length) {
    mkChart("chart-intraday", { type: "line", data: { labels: bars.map((b) => b.time), datasets: [
      { label: "指數", data: bars.map((b) => b.close), borderColor: (idx.chg || 0) >= 0 ? "#ff5b5b" : "#2ecc71", borderWidth: 2, pointRadius: 0, tension: 0.1 },
      { label: "昨收", data: bars.map(() => idx.prev), borderColor: "#666", borderDash: [4, 4], borderWidth: 1, pointRadius: 0 },
      ...(s.ma20 ? [{ label: "MA20", data: bars.map(() => s.ma20), borderColor: "#f2b64b", borderDash: [2, 4], borderWidth: 1, pointRadius: 0 }] : []) ] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { color: "#8b93a7" } } },
        scales: { x: { ticks: { color: "#8b93a7", maxTicksLimit: 8 }, grid: { color: "#262a33" } }, y: { ticks: { color: "#8b93a7" }, grid: { color: "#262a33" } } } } });
  }
  const b = s.breadth;
  $("#rt-breadth").innerHTML = b ? `<div>權值股 ${b.n} 檔：<span class="up">漲 ${b.up}</span>　<span class="down">跌 ${b.down}</span>　均 ${pct(b.avg_chg)}　委買/委賣 ${b.bid_ask_ratio.toFixed(2)}${s.tsmc_chg !== undefined ? `　台積電 ${pct(s.tsmc_chg)}` : ""}</div>` : "";
  $("#rt-global").innerHTML = (s.global || []).map((q) => `<span class="pill"><span class="muted">${q.name}</span> <span class="${cls(q.chg_pct)}">${pct(q.chg_pct)}</span></span>`).join("");
  $("#rt-alerts").innerHTML = (s.alerts_today && s.alerts_today.length) ? s.alerts_today.slice().reverse().map((a) => `<li>🔔 ${a}</li>`).join("") : "<li class='muted'>尚無警示</li>";
}

// ---------------- 判讀
async function loadMarket() {
  const m = await api("/api/market");
  if (m._pending) { $("#mk-head").innerHTML = `<div class="muted">${m.message || "計算中…"}</div>`; return; }
  const A = m.assessment;
  const conf = { 高: "🟢", 中: "🟡", 低: "🔴" }[A.confidence] || "";
  $("#mk-head").innerHTML = `<div class="gauge ${cls(A.composite_smooth)}">${fmt(A.composite_smooth, 1, true)}</div>
    <div><span class="pill">${A.regime}</span><span class="pill">${A.state}</span><span class="pill">信心度 ${conf} ${A.confidence}</span><span class="pill">動能 ${fmt(A.momentum, 0, true)}</span></div>
    <div class="big" style="margin:8px 0 4px">${A.action}</div><div class="note">${A.detail}</div>
    <div style="margin-top:6px">建議持股水位 <b>${A.position}</b>　資料日 ${A.date}　收盤 ${fmt(A.close, 2)}</div>
    ${A.turning ? `<div class="pill warn">🔀 ${A.turning}</div>` : ""}
    ${A.bottom_signals.length ? `<div style="margin-top:6px">底部訊號：${A.bottom_signals.map((x) => `<span class="pill buy">${x}</span>`).join("")}</div>` : ""}
    ${A.top_risks.length ? `<div style="margin-top:6px">高檔風險：${A.top_risks.map((x) => `<span class="pill sell">${x}</span>`).join("")}</div>` : ""}`;
  $("#mk-reasons").innerHTML = `<h3>多方理由</h3><ul class="list">${(A.reasons_pos || []).map((f) => `<li>🔺 <b>${f.name}</b>：${f.comment}</li>`).join("") || "<li class='muted'>無</li>"}</ul>
    <h3>空方理由</h3><ul class="list">${(A.reasons_neg || []).map((f) => `<li>🔻 <b>${f.name}</b>：${f.comment}</li>`).join("") || "<li class='muted'>無</li>"}</ul>`;
  const sg = m.signals || {};
  const c = sg.current || {};
  $("#mk-signals").innerHTML = `<div class="big">${c.label || "—"}</div><div class="note">買點強度 ${c.buy_strength}　賣點強度 ${c.sell_strength}（近 3 日）</div>
    ${(c.buy_signals || []).map((x) => `<div class="pill buy">🟥 ${x.name}（${x.days.slice(-1)}，歷史 10 日超額 ${pct(x.excess10)}，${x.valid}）</div>`).join("")}
    ${(c.sell_signals || []).map((x) => `<div class="pill sell">🟩 ${x.name}（${x.days.slice(-1)}，${pct(x.excess10)}，${x.valid}）</div>`).join("")}`;
  const pts = sg.points || [];
  if (pts.length) {
    mkChart("chart-signals", { type: "line", data: { labels: pts.map((p) => p.date.slice(5)), datasets: [
      { label: "加權指數", data: pts.map((p) => p.close), borderColor: "#e8eaf0", borderWidth: 1.5, pointRadius: 0 },
      { label: "買點", data: pts.map((p) => (p.buy_n > 0 ? p.close * 0.985 : null)), pointStyle: "triangle", pointRadius: 7, pointBackgroundColor: "#ff5b5b", showLine: false, borderColor: "#ff5b5b" },
      { label: "賣點", data: pts.map((p) => (p.sell_n > 0 ? p.close * 1.015 : null)), pointStyle: "triangle", rotation: 180, pointRadius: 7, pointBackgroundColor: "#2ecc71", showLine: false, borderColor: "#2ecc71" } ] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { color: "#8b93a7" } } },
        scales: { x: { ticks: { color: "#8b93a7", maxTicksLimit: 6 }, grid: { color: "#262a33" } }, y: { ticks: { color: "#8b93a7" }, grid: { color: "#262a33" } } } } });
  }
  $("#mk-check").innerHTML = `<div class="note">${A.passed}/${A.total} 通過</div>` + A.checklist.map(([n, ok]) => `<div class="check">${ok === null ? "❔" : ok ? "✅" : "❌"} ${n}</div>`).join("");
  $("#mk-factors").innerHTML = A.factors.map((f) => bar(f.name, f.available ? f.score : 0, 2, `${f.value}<br>→ ${f.comment}`)).join("");
}

// ---------------- 預測
function tone(r) { const d = (r.p_up || 0) - r.base_hit; return d >= 0.04 ? "偏多" : d <= -0.04 ? "偏空" : "中性"; }
async function loadForecast() {
  const f = await api("/api/forecast");
  if (f._pending) { $("#fc-summary").textContent = "計算中…"; return; }
  const fc = f.forecast || {};
  if (fc.error) { $("#fc-summary").textContent = fc.error; return; }
  $("#fc-days").innerHTML = (fc.next_days || []).map((d) => card(`${d.label} ${d.date}｜${tone(d)}`, fmt(d.level, 0), `上漲率 ${Math.round(d.p_up * 100)}%（基準 ${Math.round(d.base_hit * 100)}%）<br><span class="note">區間 ${fmt(d.level_lo, 0)}~${fmt(d.level_hi, 0)}</span>`)).join("");
  $("#fc-summary").textContent = fc.summary || "";
  const h = f.hourly || {};
  if (h.error) { $("#fc-hourly-note").textContent = h.error; $("#fc-hourly").innerHTML = ""; }
  else {
    $("#fc-hourly-note").textContent = `預測日 ${h.day}｜${h.live ? "盤中即時" : "開盤前，以前收為基準"}｜時間點 ${h.mark}｜基準價 ${fmt(h.price, 2)}${h.night_chg_pct !== null && h.night_chg_pct !== undefined ? `｜前晚夜盤 ${pct(h.night_chg_pct)}` : ""}${h.note ? "｜" + h.note : ""}`;
    const rows = Object.entries(h.targets || {});
    $("#fc-hourly").innerHTML = `<tr><th>到</th><th>預估</th><th>區間</th><th>上漲率</th><th>基準</th><th>模型%</th></tr>` +
      rows.map(([t, r]) => `<tr><td>${t}</td><td>${fmt(r.level, 0)}</td><td>${fmt(r.level_lo, 0)}~${fmt(r.level_hi, 0)}</td><td>${r.p_up !== undefined ? Math.round(r.p_up * 100) + "%" : "—"}</td><td>${r.base_hit !== undefined ? Math.round(r.base_hit * 100) + "%" : "—"}</td><td class="${cls(r.pred)}">${pct(r.pred)}</td></tr>`).join("");
  }
  const hz = fc.horizons || {};
  $("#fc-mid").innerHTML = ["5", "10", "20"].filter((k) => hz[k]).map((k) => { const r = hz[k]; return card(`未來 ${k} 日｜${tone(r)}`, `${Math.round(r.p_up * 100)}%`, `基準 ${Math.round(r.base_hit * 100)}%｜同分位平均 ${pct(r.hist_mean)}<br><span class="note">區間 ${pct(r.q20)} ~ ${pct(r.q80)}</span>`); }).join("");
}

// ---------------- 追蹤清單
async function loadWatch() {
  const w = await api("/api/watchlist");
  if (w._pending) { $("#wl-updated").textContent = w.message || "計算中…"; return; }
  $("#wl-updated").textContent = "更新時間 " + w.updated;
  $("#wl-list").innerHTML = Object.entries(w.stocks).map(([sid, a], i) => {
    if (a.error) return `<details class="section stock"><summary><b>${sid}</b><span class="muted">${a.error}</span></summary></details>`;
    const q = a.quote || {};
    const h = a.holders || {};
    const costs = a.costs || [];
    const bins = a.profile_bins || [];
    return `<details class="section stock" ${i === 0 ? "open" : ""}><summary><span><b>${sid} ${a.name || ""}</b> <span class="${cls(q.change_pct)}">${fmt(a.price, 2)} ${pct(q.change_pct)}</span></span><span class="pill ${a.score >= 1.5 ? "buy" : a.score <= -1.5 ? "sell" : ""}">${a.label}</span></summary>
      <ul class="list">${(a.notes || []).map((n) => `<li>• ${n}</li>`).join("")}</ul>
      ${costs.length ? `<h3>各路資金 20 日部位與成本</h3><div class="scroll"><table><tr><th>資金</th><th>20日淨買(張)</th><th>成本</th><th>現價vs成本</th><th>60日淨買</th><th>60日成本</th></tr>${costs.map((c) => `<tr><td>${c["資金"]}</td><td class="${cls(c["20日淨買(張)"])}">${fmt(c["20日淨買(張)"], 0)}</td><td>${fmt(c["20日成本"], 2)}</td><td class="${cls(c["20日現價vs成本%"])}">${pct(c["20日現價vs成本%"])}</td><td class="${cls(c["60日淨買(張)"])}">${fmt(c["60日淨買(張)"], 0)}</td><td>${fmt(c["60日成本"], 2)}</td></tr>`).join("")}</table></div>` : ""}
      ${a.profile && a.profile.poc ? `<h3>60 日分價量</h3><div class="note">密集區 ${a.profile.poc}｜價值區 ${a.profile.va_lo}~${a.profile.va_hi}｜現價上方套牢 ${a.profile.above_pct}% / 下方獲利 ${a.profile.below_pct}%</div><div class="chart"><canvas id="vp-${sid}"></canvas></div>` : ""}
      ${h.big400 !== undefined && h.big400 !== null ? `<h3>大戶 / 散戶</h3><div>大戶 >400 張 ${fmt(h.big400, 2)}%${h.big400_chg4w !== null && h.big400_chg4w !== undefined ? `（4 週 ${pct(h.big400_chg4w)}）` : ""}　散戶 ${fmt(h.retail20, 2)}%${h.foreign_hold ? `　外資持股 ${fmt(h.foreign_hold, 2)}%` : ""}</div>` : ""}
      ${(a.brokers || []).length ? `<h3>券商買賣均價（${a.broker_date}）</h3><div class="scroll"><table><tr>${Object.keys(a.brokers[0]).map((k) => `<th>${k}</th>`).join("")}</tr>${a.brokers.map((r) => `<tr>${Object.values(r).map((v) => `<td>${typeof v === "number" ? fmt(v, 2) : v}</td>`).join("")}</tr>`).join("")}</table></div>` : ""}
    </details>`;
  }).join("");
  Object.entries(w.stocks).forEach(([sid, a]) => {
    const bins = a.profile_bins || [];
    if (!bins.length || !$("#vp-" + sid)) return;
    const last = a.profile.last;
    mkChart("vp-" + sid, { type: "bar", data: { labels: bins.map((b) => b.mid.toFixed(2)), datasets: [{ label: "成交量 %", data: bins.map((b) => b.pct), backgroundColor: bins.map((b) => (b.mid > last ? "#ff5b5b" : "#2ecc71")) }] },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } },
        scales: { x: { ticks: { color: "#8b93a7" }, grid: { color: "#262a33" } }, y: { ticks: { color: "#8b93a7", maxTicksLimit: 12 }, grid: { display: false } } } } });
  });
}

// ---------------- 國際
async function loadGlobal() {
  const g = await api("/api/global");
  const cr = g.cross || {}, gr = g.global || {};
  $("#gl-summary").innerHTML = (cr.summary || []).map((l) => `<li>${l}</li>`).join("") || "<li class='muted'>尚未執行 python cli.py cross</li>";
  const roll = gr.rolling || [];
  $("#gl-rolling").innerHTML = `<tr><th>市場</th><th>近60日相關</th><th>長期平均</th><th>最新%</th></tr>` + roll.map((r) => `<tr><td>${r["市場"]}</td><td>${fmt(r["近60日相關"], 3)}</td><td>${fmt(r["長期平均"], 3)}</td><td class="${cls(r["最新值"])}">${fmt(r["最新值"], 2, true)}</td></tr>`).join("");
}

const LOADERS = { live: loadLive, market: loadMarket, forecast: loadForecast, watch: loadWatch, global: loadGlobal };
async function load(page) { try { await LOADERS[page](); } catch (e) { console.error(e); $("#hdr-sub").textContent = "連線失敗：" + e.message; } }
async function loadAll(force) { const active = document.querySelector("nav.tabs button.active").dataset.page; await load(active); }
loadAll();
setInterval(() => { if (document.querySelector("nav.tabs button.active").dataset.page === "live") loadLive().catch(() => {}); }, 20000);
setInterval(() => loadAll(), 300000);
if ("serviceWorker" in navigator) navigator.serviceWorker.register(STATIC ? "sw.js" : "/sw.js").catch(() => {});
