"""Mengisi laporan-latex/main.tex.tpl dengan angka dari laporan-latex/values.json
(hasil scripts/analisis.py) dan membuat tabel LaTeX.

Menghasilkan:
  laporan-latex/main.tex          -> dikompilasi dengan IEEEtran (Overleaf / TeX Live lengkap)
  laporan-latex/main_preview.tex  -> pratinjau dengan kelas article dua kolom (tanpa IEEEtran)

Setiap klaim arah ("turun"/"naik") diperiksa terhadap data lewat assert sebelum berkas ditulis.
Jalankan setelah analisis.py:  python scripts/bangun_laporan.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAP = ROOT / "laporan-latex"
V = json.loads((LAP / "values.json").read_text())
MODELS = ["ANN", "RNN", "LSTM", "GRU"]
REPOURL = "https://github.com/tobaylegendsavaga-create/dlw-tugas-ra-124400075"  # GANTI dengan URL repo GitHub publik


def idc(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def thou(n):
    return f"{int(round(n)):,}".replace(",", ".")


def pct(a, b):
    return (b - a) / a * 100


raw = lambda k: V[k]
ab = pd.read_csv(ROOT / "hasil/A_B_ringkasan_val.csv", dtype={"nilai": str})

# --- param dengan pemisah ribuan
for k in list(V):
    if k.endswith(".param.m"):
        V[k] = thou(float(V[k.replace(".m", ".raw")]))

# --- tanda eksplisit untuk perubahan persen
for grp in ["dC", "dD"]:
    for mod in MODELS:
        for kol in ["val_MAE", "test_MAE", "natal_MAE"]:
            ref_key, new_key = ("ref", "C") if grp == "dC" else ("Da", "Dt")
            d = pct(V[f"{ref_key}.{mod}.{kol}.raw"], V[f"{new_key}.{mod}.{kol}.raw"])
            V[f"{grp}.{mod}.{kol}.pcts"] = ("+" if d >= 0 else "$-$") + idc(abs(d), 1)

# --- arah klaim yang tertulis di naskah (assert terhadap data)
def neg(k): assert V[k.replace(".pctabs", ".pct")].startswith("-"), f"harus turun: {k}"
def pos(k): assert not V[k.replace(".pctabs", ".pct")].startswith("-"), f"harus naik: {k}"
neg("dC.ANN.val_MAE.pctabs")
for m in ["RNN", "LSTM", "GRU"]:
    pos(f"dC.{m}.val_MAE.pctabs")
for m in MODELS:
    neg(f"dC.{m}.test_MAE.pctabs")
    neg(f"dC.{m}.natal_MAE.pctabs")
neg("dD.ANN.test_MAE.pctabs")
for m in ["GRU", "LSTM", "RNN"]:
    pos(f"dD.{m}.test_MAE.pctabs")
for m in ["ANN", "GRU", "LSTM"]:
    assert V[f"Dt.{m}.val_MAE.raw"] < V[f"Da.{m}.val_MAE.raw"], f"val D harus turun: {m}"
# urutan acuan Duomo dan Linate
for pref in ["ref", "Da"]:
    t = {m: V[f"{pref}.{m}.test_MAE.raw"] for m in MODELS}
    assert t["RNN"] < t["LSTM"] < t["ANN"] and t["GRU"] < t["LSTM"], f"urutan {pref}"
    assert abs(t["RNN"] - t["GRU"]) < max(V[f"{pref}.GRU.test_MAE.sraw"], V[f"{pref}.RNN.test_MAE.sraw"]) + 1e-9
# urutan terbaik Duomo, dan perubahan di Linate
t = {m: V[f"C.{m}.test_MAE.raw"] for m in MODELS}
assert t["RNN"] < t["LSTM"] < t["ANN"] and t["GRU"] < t["LSTM"]
t = {m: V[f"Dt.{m}.test_MAE.raw"] for m in MODELS}
assert t["ANN"] < t["LSTM"] and t["ANN"] < t["RNN"]
assert V["Dt.LSTM.test_MAE.m"] == V["Dt.RNN.test_MAE.m"]
# hanya RNN di bawah baseline nilai terakhir pada Natal (acuan Duomo)
nb = V["naif.duomo.best.natal.raw"]
assert V["ref.RNN.natal_MAE.raw"] < nb and all(V[f"ref.{m}.natal_MAE.raw"] > nb for m in ["ANN", "LSTM", "GRU"])
assert V["C.RNN.natal_MAE.raw"] < V["ref.RNN.natal_MAE.raw"]
# semua model Linate kalah dari naif nilai terakhir pada Natal; menang pada test
for pre in ["Da", "Dt"]:
    assert all(V[f"{pre}.{m}.natal_MAE.raw"] > V["naif.linate.best.natal.raw"] for m in MODELS)
    assert all(V[f"{pre}.{m}.test_MAE.raw"] < V["naif.linate.best.test.raw"] for m in MODELS)
assert all(V[f"ref.{m}.test_MAE.raw"] < V["naif.duomo.best.test.raw"] for m in MODELS)
# selisih RNN-GRU & LSTM/ANN lebih besar dari std untuk klaim "melampaui simpangan baku" pada C
def diff_gt_std(m):
    d = abs(V[f"ref.{m}.test_MAE.raw"] - V[f"C.{m}.test_MAE.raw"])
    return d > max(V[f"ref.{m}.test_MAE.sraw"], V[f"C.{m}.test_MAE.sraw"])
assert diff_gt_std("ANN") and diff_gt_std("LSTM") and not diff_gt_std("GRU") and not diff_gt_std("RNN")
def diff_gt_std_D(m):
    d = abs(V[f"Da.{m}.test_MAE.raw"] - V[f"Dt.{m}.test_MAE.raw"])
    return d > max(V[f"Da.{m}.test_MAE.sraw"], V[f"Dt.{m}.test_MAE.sraw"])
assert not diff_gt_std_D("GRU") and not diff_gt_std_D("RNN") and diff_gt_std_D("LSTM")
# val-test gap terbesar pada ANN (konfigurasi terbaik)
gap = {m: V[f"C.{m}.test_MAE.raw"] - V[f"C.{m}.val_MAE.raw"] for m in MODELS}
assert max(gap, key=gap.get) == "ANN"

# --- ablasi
def a(b, v, m, kol="val.raw"):
    kol = "val.sraw" if kol == "sraw" else kol
    return V[f"ab.{b}.{v}.{m}.{kol}"]


# A1 ANN monoton naik
vals = [a("A1", w, "ANN") for w in ["6", "12", "24", "48", "168"]]
assert vals == sorted(vals)
# A1 rekuren datar (rentang kecil dibanding ANN)
assert max(a("A1", w, "RNN") for w in ["6", "12", "24", "48", "168"]) - min(a("A1", w, "RNN") for w in ["6", "12", "24", "48", "168"]) < 20
# A1 pada W=6, ANN setara GRU/RNN (selisih < 5)
assert abs(a("A1", "6", "ANN") - a("A1", "6", "GRU")) < 5 and abs(a("A1", "6", "ANN") - a("A1", "6", "RNN")) < 5
# A2 membaik sampai 128 (ANN, LSTM, GRU) ; RNN jenuh
for m in ["ANN", "LSTM", "GRU"]:
    xs = [a("A2", h, m) for h in ["16", "32", "64", "128"]]
    assert xs == sorted(xs, reverse=True), m
assert abs(a("A2", "64", "RNN") - a("A2", "128", "RNN")) < 2
assert a("A2", "128", "GRU") == min(a("A2", h, m) for h in ["16", "32", "64", "128"] for m in MODELS)
# A3
assert a("A3", "3", "LSTM") == max(a("A3", l, m) for l in ["1", "2", "3"] for m in MODELS if m == "LSTM")
assert a("A3", "3", "RNN") > a("A3", "1", "RNN")
assert a("A3", "2", "GRU") < a("A3", "1", "GRU")
assert abs(a("A3", "2", "GRU") - a("A3", "1", "GRU")) < max(a("A3", "2", "GRU", "sraw"), a("A3", "1", "GRU", "sraw")) + 1e-9
assert float(V["ab.A3.3.RNN.epoch"]) < float(V["ab.A3.1.RNN.epoch"])
# A4: lr 1e-4 mentok 300 pada semua model dan seed
assert all(V[f"ab.A4.0.0001.{m}.mentok"] == "3" for m in MODELS)
assert all(a("A4", "0.0001", m) > a("A4", "0.001", m) for m in MODELS)
assert abs(a("A4", "0.01", "GRU") - a("A4", "0.001", "GRU")) < 2
assert a("A4", "0.01", "LSTM") < a("A4", "0.001", "LSTM")
assert a("A4", "0.01", "RNN") > a("A4", "0.001", "RNN") and a("A4", "0.01", "ANN") > a("A4", "0.001", "ANN")
assert all(a("A4", "0.01", m, "sraw") > a("A4", "0.001", m, "sraw") for m in MODELS)
# A5: net terburuk di semua model; ranking manfaat
for m in MODELS:
    assert a("A5", "net", m) == max(a("A5", f, m) for f in ["net", "net+call+sms", "net+waktu", "semua"]), m
for m in ["ANN", "LSTM", "GRU"]:
    assert a("A5", "net+waktu", m) < a("A5", "net+call+sms", m), m
assert abs(a("A5", "net+waktu", "RNN") - a("A5", "net+call+sms", "RNN")) < 3
assert a("A5", "semua", "RNN") < a("A5", "net+waktu", "RNN") - 10
gain = {m: -pct(a("A5", "net", m), a("A5", "semua", m)) for m in MODELS}
assert gain["ANN"] < min(gain[m] for m in ["RNN", "LSTM", "GRU"])
# B1
for m in MODELS:
    assert a("B1", "256", m) > a("B1", "64", m)
assert V["ab.B1.256.GRU.mentok"] == "3"
assert a("B1", "16", "GRU") < a("B1", "64", "GRU") and a("B1", "16", "LSTM") < a("B1", "64", "LSTM")
b16 = {m: a("B1", "16", m) for m in MODELS}
assert b16["RNN"] < b16["LSTM"] < b16["ANN"] and b16["GRU"] < b16["LSTM"]
assert a("B1", "256", "LSTM") > a("B1", "256", "ANN")
for m in ["ANN", "RNN"]:
    assert abs(a("B1", "16", m) - a("B1", "64", m)) < max(a("B1", "16", m, "sraw"), a("B1", "64", m, "sraw"))
for m in ["LSTM", "GRU"]:
    assert abs(a("B1", "16", m) - a("B1", "64", m)) > max(a("B1", "16", m, "sraw"), a("B1", "64", m, "sraw")) - 1e-9
# B2
assert a("B2", "0.2", "ANN") > a("B2", "0.0", "ANN") and a("B2", "0.2", "RNN") > a("B2", "0.0", "RNN")
for m in ["LSTM", "GRU"]:
    assert abs(a("B2", "0.2", m) - a("B2", "0.0", m)) < max(a("B2", "0.2", m, "sraw"), a("B2", "0.0", m, "sraw")) + 1e-9
for m in MODELS:
    assert a("B2", "0.5", m) > a("B2", "0.0", m) and a("B2", "0.5", m) > a("B2", "0.2", m) - 1e-9
E = lambda m, b: V[f"eff.{m}.{b}.raw"]
for m in MODELS:
    assert E(m, "B2") > E(m, "A1")
assert E("ANN", "B2") > E("ANN", "A5") and E("RNN", "B2") > E("RNN", "A5")
assert E("LSTM", "B2") < E("LSTM", "A5") and E("GRU", "B2") < E("GRU", "A5")
# faktor paling berpengaruh (A4) pada model rekuren; ANN: B2 > A1 > A4
for m in ["RNN", "LSTM", "GRU"]:
    assert E(m, "A4") == max(E(m, b) for b in ["A1", "A2", "A3", "A4", "A5", "B1", "B2"]), m
assert E("ANN", "B2") > E("ANN", "A1") > E("ANN", "A4") and E("ANN", "B2") == max(E("ANN", b) for b in ["A1", "A2", "A3", "A4", "A5", "B1", "B2"])
assert E("LSTM", "A3") == sorted([E("LSTM", b) for b in ["A1", "A2", "A3", "A4", "A5", "B1", "B2"]])[-2]
assert E("GRU", "A5") == sorted([E("GRU", b) for b in ["A1", "A2", "A3", "A4", "A5", "B1", "B2"]])[-2]
assert E("GRU", "A1") == min(E("GRU", b) for b in ["A1", "A2", "A3", "A4", "A5", "B1", "B2"])

# --- nilai turunan untuk naskah
V["c0.rnn_vs_naif"] = idc(-pct(V["naif.duomo.best.test.raw"], V["ref.RNN.test_MAE.raw"]), 1)
p6, p168 = ab[(ab.bagian == "A1") & (ab.model == "ANN") & (ab.nilai == "6")].param.item(), \
           ab[(ab.bagian == "A1") & (ab.model == "ANN") & (ab.nilai == "168")].param.item()
V["c1.a1_param6"], V["c1.a1_param168"], V["c1.a1_ratio"] = thou(p6), thou(p168), idc(p168 / p6, 1)
V["c1.a1_ann_pct"] = idc(pct(a("A1", "6", "ANN"), a("A1", "168", "ANN")), 1)
r128 = lambda m: ab[(ab.bagian == "A2") & (ab.model == m) & (ab.nilai == "128")].param.item()
V["c1.a2_ratio"] = idc(r128("LSTM") / r128("RNN"), 1)
V["c1.a3_lstm_pct"] = idc(pct(a("A3", "1", "LSTM"), a("A3", "3", "LSTM")), 1)
x = [a("A4", "0.0001", m) for m in MODELS]
V["c1.a4_lo"], V["c1.a4_hi"] = idc(min(x), 1), idc(max(x), 1)
ep = [float(V[f"ab.A4.0.01.{m}.epoch"]) for m in MODELS]
V["c1.a4_ep_lo"], V["c1.a4_ep_hi"] = idc(min(ep), 0), idc(max(ep), 0)
n = [a("A5", "net", m) for m in MODELS]
V["c1.a5_lo"], V["c1.a5_hi"] = idc(min(n), 1), idc(max(n), 1)
for m in MODELS:
    V[f"c1.a5_{m.lower()}"] = idc(gain[m], 1)
V["c1.fit_ann_diff"] = idc(a("A5", "semua", "ANN") - a("A5", "net+waktu", "ANN"), 1)
V["meta.torch"] = "2.11.0+cu130"  # dari keluaran sel pengaturan notebook
V["LOKASI"] = "Bandara Linate"
V["REPOURL"] = REPOURL

# --- tabel
def bold_min(vals):
    mn = min(vals)
    return [v == mn for v in vals]


def cell(m, s, bold=False):
    t = f"{idc(m)} $\\pm$ {idc(s)}"
    return f"\\textbf{{{t}}}" if bold else t


# Tabel 1: baseline dan acuan
rows = []
for nm, key in [("Naif: nilai terakhir", "last"), ("Naif: kemarin", "day"), ("Naif: minggu lalu", "week")]:
    rows.append(f"{nm} & --- & {V[f'naif.duomo.{key}.val']} & {V[f'naif.duomo.{key}.test']} & {V[f'naif.duomo.{key}.natal']} \\\\")
rows.append("\\midrule")
for kol in ["val_MAE", "test_MAE", "natal_MAE"]:
    pass
bt = bold_min([V[f"ref.{m}.test_MAE.raw"] for m in MODELS])
for i, m in enumerate(MODELS):
    rows.append(f"{m} & {V[f'ref.{m}.param.m']} & {V[f'ref.{m}.val_MAE.ms']} & "
                f"{'\\textbf{' + V[f'ref.{m}.test_MAE.ms'] + '}' if bt[i] else V[f'ref.{m}.test_MAE.ms']} & {V[f'ref.{m}.natal_MAE.ms']} \\\\")
TAB_BASE = """\\begin{table}[t]
  \\centering
  \\caption{Baseline naif dan konfigurasi acuan di Duomo, horizon 1 jam. MAE rata-rata $\\pm$ simpangan baku (3 \\textit{seed}); baseline naif tanpa simpangan baku.}
  \\label{tab:base}
  \\footnotesize
  \\setlength{\\tabcolsep}{3pt}
  \\begin{tabular}{@{}lrccc@{}}
    \\toprule
    Model & Param & val MAE & test MAE & natal MAE \\\\
    \\midrule
    """ + "\n    ".join(rows) + """
    \\bottomrule
  \\end{tabular}
\\end{table}"""

# Tabel 2: hipotesis
H = lambda m, v, b: V[f"ab.{b}.{v}.{m}.val.m"]
hip = [
    ("B1: \\textit{batch} 256 paling buruk di keempat model",
     f"Val 256 vs 64: ANN {H('ANN','256','B1')} vs {H('ANN','64','B1')}; RNN {H('RNN','256','B1')} vs {H('RNN','64','B1')}; "
     f"LSTM {H('LSTM','256','B1')} vs {H('LSTM','64','B1')}; GRU {H('GRU','256','B1')} vs {H('GRU','64','B1')} (GRU 3/3 \\textit{{seed}} berhenti di 300 epoch)",
     "Terbukti"),
    ("B1: \\textit{batch} 16 dan 64 berselisih lebih kecil dari simpangan baku",
     f"Selisih val 16 vs 64: ANN {idc(abs(a('B1','16','ANN')-a('B1','64','ANN')))}, RNN {idc(abs(a('B1','16','RNN')-a('B1','64','RNN')))}, "
     f"GRU {idc(abs(a('B1','16','GRU')-a('B1','64','GRU')))}, LSTM {idc(abs(a('B1','16','LSTM')-a('B1','64','LSTM')))} (LSTM <<LSTM16>> vs <<LSTM64>>)".replace("<<LSTM16>>", H('LSTM','16','B1')).replace("<<LSTM64>>", H('LSTM','64','B1')),
     "Sebagian: ANN, RNN ya; GRU, LSTM tidak"),
    ("B1: urutan model tidak berubah",
     f"Pada 16: RNN {H('RNN','16','B1')} $\\approx$ GRU {H('GRU','16','B1')} $<$ LSTM {H('LSTM','16','B1')} $<$ ANN {H('ANN','16','B1')}. "
     f"Pada 256: LSTM {H('LSTM','256','B1')} lebih buruk dari ANN {H('ANN','256','B1')}",
     "Sebagian: berubah pada 256"),
    ("B2: \\textit{dropout} 0,2 membantu ANN, netral untuk rekuren",
     f"Val 0,2 vs 0: ANN {H('ANN','0.2','B2')} vs {H('ANN','0.0','B2')}; RNN {H('RNN','0.2','B2')} vs {H('RNN','0.0','B2')}; "
     f"LSTM {H('LSTM','0.2','B2')} vs {H('LSTM','0.0','B2')}; GRU {H('GRU','0.2','B2')} vs {H('GRU','0.0','B2')}",
     "Tidak terbukti: ANN dan RNN memburuk"),
    ("B2: \\textit{dropout} 0,5 memperburuk semua model",
     f"Val 0,5 vs 0: ANN {H('ANN','0.5','B2')} vs {H('ANN','0.0','B2')}; RNN {H('RNN','0.5','B2')} vs {H('RNN','0.0','B2')}; "
     f"LSTM {H('LSTM','0.5','B2')} vs {H('LSTM','0.0','B2')}; GRU {H('GRU','0.5','B2')} vs {H('GRU','0.0','B2')}",
     "Terbukti"),
    ("B2: \\textit{dropout} kurang berpengaruh daripada $W$ dan fitur",
     f"Selisih val maks.--min.: \\textit{{dropout}} vs $W$ vs fitur: ANN {V['eff.ANN.B2']}/{V['eff.ANN.A1']}/{V['eff.ANN.A5']}; "
     f"RNN {V['eff.RNN.B2']}/{V['eff.RNN.A1']}/{V['eff.RNN.A5']}; LSTM {V['eff.LSTM.B2']}/{V['eff.LSTM.A1']}/{V['eff.LSTM.A5']}; "
     f"GRU {V['eff.GRU.B2']}/{V['eff.GRU.A1']}/{V['eff.GRU.A5']}",
     "Tidak terbukti: lebih berpengaruh dari $W$ di semua model, dari fitur di ANN dan RNN"),
]
rows = [f"{h} & {b} & {s} \\\\" for h, b, s in hip]
TAB_HIP = """\\begin{table*}[t]
  \\centering
  \\caption{Hipotesis Bagian B (ditulis sebelum eksperimen) dan hasilnya terhadap val MAE di Duomo.}
  \\label{tab:hip}
  \\footnotesize
  \\renewcommand{\\arraystretch}{1.15}
  \\begin{tabular}{@{}p{0.27\\textwidth}p{0.45\\textwidth}p{0.22\\textwidth}@{}}
    \\toprule
    Hipotesis & Bukti & Status \\\\
    \\midrule
    """ + "\n    ".join(rows) + """
    \\bottomrule
  \\end{tabular}
\\end{table*}"""

# Tabel 3: C dan D
CFG_TXT = {
    "ANN": "$W{=}6$, hid 128, 1 lap., \\texttt{net+waktu}, bs 16",
    "RNN": "$W{=}48$, bs 16",
    "LSTM": "$W{=}6$, hid 128, lr $10^{-2}$, bs 16",
    "GRU": "$W{=}48$, hid 128, 2 lap., bs 16",
}
rows = []


def blok(nama, pa, pt, lok):
    out = [f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{nama}}}}} \\\\"]
    best = {}
    for kol in ["val_MAE", "test_MAE", "natal_MAE"]:
        for pre in (pa, pt):
            pass
    for m in MODELS:
        c = []
        for kol in ["val_MAE", "test_MAE", "natal_MAE"]:
            for pre in (pa, pt):
                c.append(V[f"{pre}.{m}.{kol}.ms"])
        out.append(f"{m} & {CFG_TXT[m]} & " + " & ".join(c) + " \\\\")
    out.append(f"Naif terbaik & & \\multicolumn{{2}}{{c}}{{{V[f'naif.{lok}.best.val']} ({V[f'naif.{lok}.best.val.nama']})}} & "
               f"\\multicolumn{{2}}{{c}}{{{V[f'naif.{lok}.best.test']} ({V[f'naif.{lok}.best.test.nama']})}} & "
               f"\\multicolumn{{2}}{{c}}{{{V[f'naif.{lok}.best.natal']} ({V[f'naif.{lok}.best.natal.nama']})}} \\\\")
    return out


rows += blok("Duomo (kotak 5060)", "ref", "C", "duomo")
rows.append("\\midrule")
rows += blok("Bandara Linate (kotak 4489)", "Da", "Dt", "linate")
TAB_CD = """\\begin{table*}[t]
  \\centering
  \\caption{Konfigurasi acuan (a) dan terbaik (t) pada Duomo (Bagian C) dan Bandara Linate (Bagian D, konfigurasi dari Duomo tanpa diubah). MAE rata-rata $\\pm$ simpangan baku, 3 \\textit{seed}. ``hid'' = \\textit{hidden}, ``lap.'' = lapisan, bs = \\textit{batch size}. Parameter konfigurasi terbaik: ANN <<C.ANN.param.m>>, RNN <<C.RNN.param.m>>, LSTM <<C.LSTM.param.m>>, GRU <<C.GRU.param.m>>.}
  \\label{tab:cd}
  \\footnotesize
  \\setlength{\\tabcolsep}{3pt}
  \\begin{tabular}{@{}ll cc cc cc@{}}
    \\toprule
    & Perubahan dari acuan & \\multicolumn{2}{c}{val MAE} & \\multicolumn{2}{c}{test MAE} & \\multicolumn{2}{c}{natal MAE} \\\\
    \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}
    Model & (konfigurasi t) & a & t & a & t & a & t \\\\
    \\midrule
    """ + "\n    ".join(rows) + """
    \\bottomrule
  \\end{tabular}
\\end{table*}"""

PRE_IEEE = r"""% =====================================================================
% Laporan Tugas: Studi Ablasi RNN, LSTM, GRU vs ANN (Workshop Deep Learning, ITERA)
% FILE INI DIHASILKAN oleh scripts/bangun_laporan.py dari main.tex.tpl.
% Compile: latexmk -pdf main.tex   (atau unggah folder ini ke Overleaf)
% =====================================================================
\documentclass[conference,a4paper]{IEEEtran}
\IEEEoverridecommandlockouts

\usepackage[indonesian]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\graphicspath{{gambar/}}"""

PRE_PREVIEW = r"""% PRATINJAU saja (tanpa IEEEtran): panjang halaman hanya perkiraan. Versi final: main.tex dengan IEEEtran.
\documentclass[10pt,twocolumn,a4paper]{article}
\usepackage[margin=0.62in,top=0.75in,bottom=0.95in,columnsep=0.25in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage[font=footnotesize,labelsep=period]{caption}
\usepackage{titlesec}
\titleformat{\section}{\centering\normalfont\small\scshape}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalfont\small\itshape}{\thesubsection}{0.5em}{}
\titleformat{\subsubsection}{\normalfont\small\itshape}{}{0em}{}
\titlespacing*{\section}{0pt}{8pt}{3pt}
\titlespacing*{\subsection}{0pt}{5pt}{2pt}
\graphicspath{{gambar/}}
\renewcommand{\figurename}{Gambar}\renewcommand{\tablename}{Tabel}\renewcommand{\abstractname}{Abstrak}
\setlength{\parskip}{0pt}
\renewcommand{\baselinestretch}{0.97}
\newcommand{\IEEEauthorblockN}[1]{{\large #1}\par}
\newcommand{\IEEEauthorblockA}[1]{{\small #1}\par}
\newenvironment{IEEEkeywords}{\par\noindent\small\textbf{\textit{Kata kunci}}---}{\par}
\makeatletter
\renewcommand{\maketitle}{\twocolumn[\begin{center}{\LARGE\bfseries \@title\par}\vspace{6pt}\@author\vspace{8pt}\end{center}]}
\makeatother
"""

tpl = (LAP / "main.tpl_dummy").exists() if False else (LAP / "main.tex.tpl").read_text()
TABLES = {"TABEL_BASE": TAB_BASE, "TABEL_HIP": TAB_HIP, "TABEL_CD": TAB_CD}


def isi(teks):
    for k, v in TABLES.items():
        teks = teks.replace(f"<<{k}>>", v)

    def sub(m):
        key = m.group(1)
        if key not in V:
            raise KeyError(f"penanda tidak ditemukan: {key}")
        return str(V[key])

    return re.sub(r"<<([^<>]+)>>", sub, teks)


for nama, pre in [("main.tex", PRE_IEEE), ("main_preview.tex", PRE_PREVIEW)]:
    out = isi(tpl.replace("<<PREAMBLE>>", pre))
    if nama == "main_preview.tex":  # IEEEtran.bst tidak tersedia di pratinjau
        out = out.replace("\\bibliographystyle{IEEEtran}", "\\bibliographystyle{unsrt}")
    (LAP / nama).write_text(out)
print("OK: main.tex dan main_preview.tex ditulis; semua assert lolos")


# ---------------------------------------------------------------- README repo
README = """# Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler Milan

Tugas Workshop Deep Learning, Teknik Telekomunikasi ITERA.

## Kelompok

| Peran | Nama | NIM |
|---|---|---|
| Ketua (kelompok satu orang) | Tobagus Ahmad Yogi | 124400075 |

- **Kelas:** RA
- **Lokasi kelompok:** 5, Bandara Linate (kotak 4489), dari digit terakhir NIM terkecil (124400075)
- **Laporan:** [`laporan-latex/main.pdf`](laporan-latex/main.pdf) (sumber: `laporan-latex/main.tex`)

## Cara Menjalankan

Dengan [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab
```

Buka `tugas_ablasi.ipynb`, isi `LOKASI_KELOMPOK = 5`, `CEPAT = False`, lalu jalankan semua sel.
Semua run tercatat di `hasil/ablasi.csv`. Tanpa instalasi lokal, notebook yang sama dapat diunggah ke
Google Colab bersama folder `data/`.

Perangkat yang kami pakai: Google Colab (GPU NVIDIA T4, PyTorch <<meta.torch>>). Total waktu training:
<<meta.menit>> menit untuk <<meta.n_unik>> pelatihan unik (<<meta.n_baris>> baris di `ablasi.csv`; <<meta.n_salin>>
baris disalin karena konfigurasinya identik dengan run sebelumnya).

Tabel, gambar, dan angka laporan dibuat ulang dari `hasil/ablasi.csv` (tanpa PyTorch):

```bash
python scripts/analisis.py          # tabel (hasil/*.csv), gambar, laporan-latex/values.json
python scripts/bangun_laporan.py    # mengisi laporan-latex/main.tex dari values.json
cd laporan-latex && latexmk -pdf main.tex
```

## Ringkasan Hasil

Di Duomo (satu jam ke depan), konfigurasi acuan menghasilkan test MAE RNN <<ref.RNN.test_MAE.ms>>, GRU <<ref.GRU.test_MAE.ms>>,
LSTM <<ref.LSTM.test_MAE.ms>>, dan ANN <<ref.ANN.test_MAE.ms>>, semuanya di bawah baseline naif terbaik (<<naif.duomo.best.test>>).
Konfigurasi terbaik per model (dipilih dengan val MAE) memberi test MAE RNN <<C.RNN.test_MAE.ms>>, GRU <<C.GRU.test_MAE.ms>>,
LSTM <<C.LSTM.test_MAE.ms>>, dan ANN <<C.ANN.test_MAE.ms>>; namun val MAE hanya turun untuk ANN dan justru naik untuk RNN, LSTM, dan GRU
(interaksi antar faktor). Di Bandara Linate urutan model acuan sama, tetapi konfigurasi terbaik hanya membantu ANN
(test MAE <<Da.ANN.test_MAE.ms>> menjadi <<Dt.ANN.test_MAE.ms>>), dan pada minggu Natal baseline nilai terakhir (<<naif.linate.best.natal>>)
mengalahkan semua model.

## Struktur

```
tugas_ablasi.ipynb      notebook eksperimen (di-commit beserta output)
data/                   data trafik per jam (Duomo + 10 lokasi tugas)
hasil/
  ablasi.csv            seluruh run (Bagian 0, A, B, C, D), 3 seed
  baseline_naif_duomo.csv, baseline_naif_lokasi.csv
  D_generalisasi.csv    keluaran notebook untuk Bagian D
  A_B_ringkasan_val.csv, C_acuan_vs_terbaik_lengkap.csv, D_generalisasi_lengkap.csv, karakter_data.csv
                        dibuat scripts/analisis.py dari ablasi.csv
  fig_*.png             gambar laporan
scripts/                analisis.py (tabel dan gambar), bangun_laporan.py (mengisi laporan dari angka)
laporan-latex/          main.tex (IEEEtran), references.bib, gambar/, main.tex.tpl, values.json
```

Catatan: grafik per faktor buatan notebook (`A1_W.png` dan seterusnya) dapat disalin dari folder `hasil/` Colab.
Berkas `C_terbaik_vs_acuan.csv` dari notebook hanya memuat baris konfigurasi terbaik, sehingga versi lengkap
(acuan dan terbaik) tersedia di `C_acuan_vs_terbaik_lengkap.csv`.

## Penggunaan AI

Claude (Anthropic) membantu memandu pengaturan Colab, menyusun draf hipotesis B1 dan B2 sebelum eksperimen,
memilih konfigurasi terbaik secara mekanis dari `ablasi.csv`, menulis skrip analisis, dan menyusun draf laporan.
Pelatihan model dijalankan penulis di Colab. Rinciannya ada di bagian Pernyataan Penggunaan AI pada laporan.

## Data

Telecom Italia Big Data Challenge (Barlacchi dkk., 2015, *Scientific Data*, lisensi ODbL), versi per jam dari
Zhang dkk. (INFOCOM 2021). Disiapkan oleh pengajar di repo [mctosima/dl-workshop](https://github.com/mctosima/dl-workshop).
"""
(ROOT / "README.md").write_text(isi(README))
print("OK: README.md ditulis")
