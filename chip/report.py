"""文字報告 (CLI / 通知用)。"""
from __future__ import annotations

from .analysis.common import Factor


def _bar(score: float, width: int = 8) -> str:
    n = int(round(abs(score) / 2 * width))
    return ("▲" * n).ljust(width) if score > 0 else ("▼" * n).ljust(width) if score < 0 else "·".ljust(width)


def factor_lines(factors: list[Factor]) -> list[str]:
    out = []
    for f in factors:
        if not f.available:
            out.append(f"  {f.name:<18} {'N/A':>6}  {'':8}  {f.comment}")
            continue
        out.append(f"  {f.name:<18} {f.score:+5.1f}  {_bar(f.score)}  {f.value}\n{'':36}→ {f.comment}")
    return out


def market_report(a: dict) -> str:
    lines = [
        f"═══ 台股大盤籌碼判讀  {a['date']}  收盤 {a['close']:,.2f} ({a['ret1']:+.2f}%) ═══" if a.get("ret1") is not None
        else f"═══ 台股大盤籌碼判讀  {a['date']}  收盤 {a['close']:,.2f} ═══",
        f"市場狀態：{a.get('state', '')}｜綜合籌碼分：{a['composite']:+.1f} (3日平滑 {a.get('composite_smooth', a['composite']):+.1f}，5日動能 {a.get('momentum', 0):+.1f})  【{a['regime']}】",
        f"信心度：{a.get('confidence', '')} (因子一致 {a.get('agree_ratio', 0):.0%}，資料完整 {a.get('coverage', 0):.0%})"
        + (f"｜轉折：{a['turning']}" if a.get("turning") else ""),
        f"進場建議：{a['action']}",
        f"  {a['detail']}",
        f"建議持股水位：{a['position']}",
        "",
        "── 主要多方理由 ──",
        *([f"  + {f.name}：{f.comment}" for f in a.get("reasons_pos", [])] or ["  (無)"]),
        "── 主要空方理由 ──",
        *([f"  - {f.name}：{f.comment}" for f in a.get("reasons_neg", [])] or ["  (無)"]),
        "",
        "── 因子明細 (分數 -2~+2) ──",
        *factor_lines(a["factors"]),
        "",
        f"── 進場檢查表 ({a['passed']}/{a['total']} 通過) ──",
    ]
    for name, ok in a["checklist"]:
        lines.append(f"  [{'✓' if ok else ('?' if ok is None else '✗')}] {name}")
    if a["bottom_signals"]:
        lines += ["", "── 底部/逆勢訊號 ──"] + [f"  • {s}" for s in a["bottom_signals"]]
    if a["top_risks"]:
        lines += ["", "── 高檔風險 ──"] + [f"  • {s}" for s in a["top_risks"]]
    return "\n".join(lines)


def stock_report(a: dict) -> str:
    q = a.get("quote")
    rt = f"  即時 {q['last']:,.2f} ({q.get('change_pct', 0):+.2f}%) {q['time']}" if q else ""
    lines = [
        f"═══ {a['name']} ({a['stock_id']})  {a['date']}  收盤 {a['close']:,.2f}{rt} ═══",
        f"個股籌碼分數：{a['composite_raw']:+.1f}  大盤環境調整 {a['market_adj']:+.0f} → {a['composite']:+.1f} 【{a['regime']}】",
        f"進場建議：{a['action']}",
        f"標籤：{'、'.join(a['tags']) if a['tags'] else '無'}",
        "",
        "── 因子明細 ──",
        *factor_lines(a["factors"]),
    ]
    return "\n".join(lines)
