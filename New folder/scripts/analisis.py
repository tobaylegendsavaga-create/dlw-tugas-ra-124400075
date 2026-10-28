"""Analisis hasil ablasi: membaca hasil/ablasi.csv dan data/, lalu menulis
tabel (hasil/*.csv), gambar (hasil/*.png) dan laporan-latex/values.json.

Semua angka di laporan berasal dari berkas ini, bukan diketik tangan.
Jalankan dari akar repo:  python scripts/analisis.py
Hanya butuh numpy, pandas, matplotlib (tanpa PyTorch).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HASIL = ROOT / "hasil"
LAP = ROOT / "laporan-latex"
(LAP / "gambar").mkdir(exist_ok=True)

MODELS = ["ANN", "RNN", "LSTM", "GRU"]
WARNA = {"ANN": "#B8860B", "RNN": "#C8102E", "LSTM": "#0B1F3A", "GRU": "#1E8C8C"}
MARK = {"ANN": "s", "RNN": "o", "LSTM": "^", "GRU": "D"}
LOKASI_KODE = 5
LOKASI_NAMA = "Bandara Linate"

df = pd.read_csv(HASIL / "ablasi.csv", dtype={"nilai": str, "kunci": str})
V = {}  # nilai-nilai untuk laporan


def idc(x, d=1):
    """Format angka berkoma Indonesia."""
    return f"{x:.{d}f}".replace(".", ",")


def ms(m, s, d=1):
    return f"{idc(m, d)} $\\pm$ {idc(s, d)}"


def agg(sub, kol):
    g = sub.groupby("model")[kol]
    return g.mean(), g.std()


# ---------------------------------------------------------------- ringkasan per bagian
def ringkas_bagian(nama, kunci):
    sub = df[df.bagian == nama]
    for kol, d in [("val_MAE", 1), ("test_MAE", 1), ("natal_MAE", 1), ("test_MAPE", 1), ("param", 0), ("epoch", 0),
                   ("val_RMSE", 1), ("test_RMSE", 1)]:
        m, s = agg(sub, kol)
        for mod in MODELS:
            V[f"{kunci}.{mod}.{kol}.m"] = idc(m[mod], d)
            V[f"{kunci}.{mod}.{kol}.s"] = idc(s[mod], d)
            V[f"{kunci}.{mod}.{kol}.ms"] = ms(m[mod], s[mod], d)
            V[f"{kunci}.{mod}.{kol}.raw"] = float(m[mod])
            V[f"{kunci}.{mod}.{kol}.sraw"] = float(s[mod])
    return sub


ref = ringkas_bagian("0", "ref")
cc = ringkas_bagian("C", "C")
da = ringkas_bagian("D-acuan", "Da")
dt = ringkas_bagian("D-terbaik", "Dt")

for a, b, k in [("ref", "C", "dC"), ("Da", "Dt", "dD")]:
    for mod in MODELS:
        for kol in ["val_MAE", "test_MAE", "natal_MAE"]:
            x0, x1 = V[f"{a}.{mod}.{kol}.raw"], V[f"{b}.{mod}.{kol}.raw"]
            V[f"{k}.{mod}.{kol}.pct"] = idc((x1 - x0) / x0 * 100, 1)
            V[f"{k}.{mod}.{kol}.abs"] = idc(x1 - x0, 1)
            V[f"{k}.{mod}.{kol}.pctabs"] = idc(abs((x1 - x0) / x0 * 100), 1)

# ---------------------------------------------------------------- baseline naif (dihitung ulang dari data)
def muat(lokasi):
    if lokasi == "duomo":
        d = pd.read_csv(ROOT / "data/milano_duomo_5060.csv", parse_dates=["waktu"])
    else:
        d = pd.read_csv(ROOT / "data/milano_tugas_10_lokasi.csv", parse_dates=["waktu"])
        d = d[d.kode == lokasi]
    return d.set_index("waktu")


def split_masks(t):
    return {"train": (t >= "2013-11-11") & (t < "2013-12-09"), "val": (t >= "2013-12-09") & (t < "2013-12-16"),
            "test": (t >= "2013-12-16") & (t < "2013-12-23"), "natal": (t >= "2013-12-23")}


def naif(lokasi, H=1):
    d = muat(lokasi)
    net, sp, rows = d.net.values, split_masks(d.index), []
    for s in ["val", "test", "natal"]:
        aw = np.arange(168, len(net) - H + 1)
        aw = aw[sp[s][aw + H - 1]]
        a = net[aw[:, None] + np.arange(H)]
        tebak = {"Naif: nilai terakhir": np.repeat(net[aw - 1][:, None], H, 1),
                 "Naif: kemarin": net[aw[:, None] + np.arange(H) - 24],
                 "Naif: minggu lalu": net[aw[:, None] + np.arange(H) - 168]}
        for n, p in tebak.items():
            e = p - a
            rows.append({"lokasi": "Duomo" if lokasi == "duomo" else LOKASI_NAMA, "horizon": H, "model": n, "split": s,
                         "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "MAPE": (np.abs(e) / a).mean() * 100})
    return pd.DataFrame(rows)


nd, nl = naif("duomo", 1), naif(LOKASI_KODE, 1)
nl.to_csv(HASIL / "baseline_naif_lokasi.csv", index=False)
# verifikasi terhadap berkas notebook (Duomo, horizon 1)
nb_duomo = pd.read_csv(HASIL / "baseline_naif_duomo.csv")
nb_duomo = nb_duomo[nb_duomo.horizon == 1]
assert np.allclose(nb_duomo.sort_values(["split", "model"]).MAE.values, nd.sort_values(["split", "model"]).MAE.values), \
    "baseline Duomo tidak cocok dengan notebook"
NAMA = {"Naif: nilai terakhir": "last", "Naif: kemarin": "day", "Naif: minggu lalu": "week"}
for lok, t in [("duomo", nd), ("linate", nl)]:
    for _, r in t.iterrows():
        V[f"naif.{lok}.{NAMA[r['model']]}.{r['split']}"] = idc(r["MAE"], 1)
        V[f"naif.{lok}.{NAMA[r['model']]}.{r['split']}.raw"] = float(r["MAE"])
    for sp in ["val", "test", "natal"]:
        mn = t[t.split == sp].MAE.min()
        V[f"naif.{lok}.best.{sp}"] = idc(mn, 1)
        V[f"naif.{lok}.best.{sp}.raw"] = float(mn)
        V[f"naif.{lok}.best.{sp}.nama"] = t[(t.split == sp) & (t.MAE == mn)].model.item().replace("Naif: ", "")

# ---------------------------------------------------------------- ablasi A/B
FAKTOR = [("A1", "W", "Panjang jendela $W$ (jam)"), ("A2", "hidden", "Ukuran hidden"), ("A3", "layers", "Jumlah lapisan"),
          ("A4", "lr", "Learning rate"), ("A5", "fitur", "Kelompok fitur"), ("B1", "batch", "Batch size"),
          ("B2", "dropout", "Dropout")]
urut = {"A1": ["6", "12", "24", "48", "168"], "A2": ["16", "32", "64", "128"], "A3": ["1", "2", "3"],
        "A4": ["0.0001", "0.001", "0.01"], "A5": ["net", "net+call+sms", "net+waktu", "semua"],
        "B1": ["16", "64", "256"], "B2": ["0.0", "0.2", "0.5"]}
long = []
for b, f, _ in FAKTOR:
    sub = df[df.bagian == b]
    for v in urut[b]:
        for mod in MODELS:
            x = sub[(sub.nilai == v) & (sub.model == mod)]
            if len(x) == 0:  # nilai acuan model ini tidak ada di grid (tidak terjadi untuk faktor di atas)
                continue
            r = dict(bagian=b, faktor=f, nilai=v, model=mod, n_seed=len(x), param=int(x.param.mean()),
                     epoch_mean=x.epoch.mean(), epoch_maks=int(x.epoch.max()), n_mentok_300=int((x.epoch == 300).sum()),
                     val_MAE_mean=x.val_MAE.mean(), val_MAE_std=x.val_MAE.std())
            long.append(r)
            k = f"ab.{b}.{v}.{mod}"
            V[k + ".val.m"] = idc(r["val_MAE_mean"], 1)
            V[k + ".val.s"] = idc(r["val_MAE_std"], 1)
            V[k + ".val.ms"] = ms(r["val_MAE_mean"], r["val_MAE_std"], 1)
            V[k + ".val.raw"] = r["val_MAE_mean"]
            V[k + ".val.sraw"] = r["val_MAE_std"]
            V[k + ".epoch"] = idc(r["epoch_mean"], 0)
            V[k + ".mentok"] = str(r["n_mentok_300"])
            V[k + ".param"] = f"{r['param']:,}".replace(",", ".")
ab = pd.DataFrame(long)
ab.to_csv(HASIL / "A_B_ringkasan_val.csv", index=False)

# besar pengaruh tiap faktor = (maks - min) rata-rata val MAE antar nilai faktor
eff = ab.groupby(["model", "bagian"]).val_MAE_mean.agg(lambda s: s.max() - s.min())
for (mod, b), val in eff.items():
    V[f"eff.{mod}.{b}"] = idc(val, 1)
    V[f"eff.{mod}.{b}.raw"] = float(val)

# ---------------------------------------------------------------- konfigurasi terbaik (kembali diverifikasi dari CSV)
TERBAIK_NB = {"ANN": dict(W=6, hidden=128, layers=1, fitur="net+waktu", batch=16), "RNN": dict(W=48, batch=16),
              "LSTM": dict(W=6, hidden=128, lr=0.01, batch=16), "GRU": dict(W=48, hidden=128, layers=2, batch=16)}
for mod in MODELS:
    x = cc[cc.model == mod].iloc[0]
    for k, v in TERBAIK_NB[mod].items():
        assert str(x[k]) == str(v), (mod, k, x[k], v)
    for k in ["W", "hidden", "layers", "fitur", "lr", "batch", "dropout"]:
        V[f"cfgC.{mod}.{k}"] = str(x[k]).replace(".", ",") if k in ("lr", "dropout") else str(x[k])

# tabel C lengkap (acuan + terbaik) untuk hasil/
rows = []
for nama, sub in [("acuan", ref), ("terbaik", cc)]:
    for mod in MODELS:
        x = sub[sub.model == mod]
        rows.append(dict(model=mod, konfigurasi=nama, param=int(x.param.mean()), epoch=x.epoch.mean(),
                         val_MAE_mean=x.val_MAE.mean(), val_MAE_std=x.val_MAE.std(),
                         test_MAE_mean=x.test_MAE.mean(), test_MAE_std=x.test_MAE.std(),
                         test_MAPE_mean=x.test_MAPE.mean(), test_MAPE_std=x.test_MAPE.std(),
                         natal_MAE_mean=x.natal_MAE.mean(), natal_MAE_std=x.natal_MAE.std()))
pd.DataFrame(rows).round(3).to_csv(HASIL / "C_acuan_vs_terbaik_lengkap.csv", index=False)
rows = []
for nama, sub in [("acuan", da), ("terbaik", dt)]:
    for mod in MODELS:
        x = sub[sub.model == mod]
        rows.append(dict(lokasi=LOKASI_NAMA, model=mod, konfigurasi=nama, epoch=x.epoch.mean(),
                         val_MAE_mean=x.val_MAE.mean(), val_MAE_std=x.val_MAE.std(),
                         test_MAE_mean=x.test_MAE.mean(), test_MAE_std=x.test_MAE.std(),
                         test_MAPE_mean=x.test_MAPE.mean(), test_MAPE_std=x.test_MAPE.std(),
                         natal_MAE_mean=x.natal_MAE.mean(), natal_MAE_std=x.natal_MAE.std()))
pd.DataFrame(rows).round(3).to_csv(HASIL / "D_generalisasi_lengkap.csv", index=False)

# ---------------------------------------------------------------- karakter data
kar = []
for lok in ["duomo", LOKASI_KODE]:
    d = muat(lok)
    sp = split_masks(d.index)
    net = d.net
    tr = net[sp["train"]].values
    ac = lambda k: np.corrcoef(tr[k:], tr[:-k])[0, 1]
    key = "duomo" if lok == "duomo" else "linate"
    r = dict(lokasi=key, mean_train=tr.mean(), mean_test=net[sp["test"]].mean(), mean_natal=net[sp["natal"]].mean(),
             cv_train=tr.std() / tr.mean(), ac1=ac(1), ac24=ac(24), ac168=ac(168),
             rasio_akhir_pekan=net[sp["train"]][net[sp["train"]].index.dayofweek >= 5].mean() /
                               net[sp["train"]][net[sp["train"]].index.dayofweek < 5].mean())
    r["turun_natal_pct"] = (1 - r["mean_natal"] / r["mean_test"]) * 100
    kar.append(r)
    for k, v in r.items():
        if k != "lokasi":
            V[f"kd.{key}.{k}"] = idc(v, 2) if k in ("cv_train", "ac1", "ac24", "ac168", "rasio_akhir_pekan") else idc(v, 1)
pd.DataFrame(kar).round(3).to_csv(HASIL / "karakter_data.csv", index=False)
for key, ref_mod in [("duomo", "RNN"), ("linate", "RNN")]:
    pass
V["kd.duomo.relMAE_RNN_test"] = idc(V["ref.RNN.test_MAE.raw"] / float(kar[0]["mean_test"]) * 100, 1)
V["kd.linate.relMAE_RNN_test"] = idc(V["Da.RNN.test_MAE.raw"] / float(kar[1]["mean_test"]) * 100, 1)

# ---------------------------------------------------------------- meta
u = df.drop_duplicates("kunci")
V["meta.n_baris"] = str(len(df))
V["meta.n_unik"] = str(len(u))
V["meta.n_salin"] = str(len(df) - len(u))
V["meta.menit"] = idc(u.detik.sum() / 60, 1)
V["meta.n_param_ratio_GRU"] = idc(V["C.GRU.param.raw"] / V["ref.GRU.param.raw"], 0)
V["meta.n_param_ratio_LSTM"] = idc(V["C.LSTM.param.raw"] / V["ref.LSTM.param.raw"], 0)
for mod in MODELS:
    for B in ["A1"]:
        for w in urut["A1"]:
            pass

# ---------------------------------------------------------------- gambar 1: ablasi A1-A5, B1-B2
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
                     "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
etiket = {"0.0001": "1e-4", "0.001": "1e-3", "0.01": "1e-2", "net+call+sms": "net+\ncall+sms", "net+waktu": "net+\nwaktu",
          "0.0": "0", "0.2": "0,2", "0.5": "0,5"}
judul = {"A1": "A1  panjang jendela W (jam)", "A2": "A2  ukuran hidden", "A3": "A3  jumlah lapisan",
         "A4": "A4  learning rate", "A5": "A5  kelompok fitur", "B1": "B1  batch size", "B2": "B2  dropout"}
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.5))
for ax, (b, f, _) in zip(axes.ravel(), FAKTOR):
    xs = np.arange(len(urut[b]))
    for mod in MODELS:
        x = ab[(ab.bagian == b) & (ab.model == mod)].set_index("nilai").loc[urut[b]]
        ax.errorbar(xs + (MODELS.index(mod) - 1.5) * 0.06, x.val_MAE_mean, yerr=x.val_MAE_std, color=WARNA[mod],
                    marker=MARK[mod], ms=3.2, lw=1.0, capsize=1.5, elinewidth=0.7, label=mod)
    ax.set_xticks(xs, [etiket.get(v, v) for v in urut[b]])
    ax.set_title(judul[b], loc="left")
    if b in ("A1", "A5"):
        ax.set_ylabel("val MAE")
axes[1, 3].axis("off")
h, l = axes[0, 0].get_legend_handles_labels()
axes[1, 3].legend(h, l, loc="center", frameon=False, title="Model (rata-rata ± std, 3 seed)", title_fontsize=7)
plt.tight_layout()
plt.savefig(LAP / "gambar" / "fig_ablasi.png", dpi=250, bbox_inches="tight")
plt.savefig(HASIL / "fig_ablasi_A_B.png", dpi=250, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- gambar 2: C dan D
fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.9))
for j, (lok, ts_ref, ts_best, nm) in enumerate([("duomo", ref, cc, "Duomo"), ("linate", da, dt, LOKASI_NAMA)]):
    for i, sp in enumerate(["test", "natal"]):
        ax = axes[i, j]
        kol = f"{sp}_MAE"
        x = np.arange(4)
        for jj, (sub, alpha, lab) in enumerate([(ts_ref, 0.45, "acuan"), (ts_best, 1.0, "terbaik")]):
            m = sub.groupby("model")[kol].mean().reindex(MODELS)
            s = sub.groupby("model")[kol].std().reindex(MODELS)
            ax.bar(x + (jj - 0.5) * 0.38, m, 0.38, yerr=s, capsize=2, error_kw=dict(lw=0.7),
                   color=[WARNA[mm] for mm in MODELS], alpha=alpha, label=lab,
                   hatch=None if jj else "//", edgecolor="white", linewidth=0.4)
        bv = V[f"naif.{lok}.best.{sp}.raw"]
        ax.axhline(bv, color="#444", ls="--", lw=0.9)
        ax.set_xticks(x, MODELS)
        ax.set_title(f"{nm}, minggu {'test' if sp == 'test' else 'Natal'} (garis putus: naif terbaik, "
                     f"{V[f'naif.{lok}.best.{sp}.nama']}, {V[f'naif.{lok}.best.{sp}']})", loc="left", fontsize=6.5)
        ax.set_ylabel(f"{sp} MAE")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor="#888", alpha=0.45, hatch="//", edgecolor="white", label="acuan (arsir, pucat)"),
                    Patch(facecolor="#888", label="konfigurasi terbaik (pekat)")] +
           [Patch(facecolor=WARNA[m], label=m) for m in MODELS],
           loc="lower center", ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(LAP / "gambar" / "fig_CD.png", dpi=250, bbox_inches="tight")
plt.savefig(HASIL / "fig_C_D.png", dpi=250, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- gambar 3: pola trafik Duomo vs Linate
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.5), sharex=True)
for ax, lok, nm, warna in [(axes[0], "duomo", "Duomo", "#0B1F3A"), (axes[1], LOKASI_KODE, LOKASI_NAMA, "#1E8C8C")]:
    d = muat(lok)
    ax.plot(d.index, d.net, color=warna, lw=0.5)
    for a, b, c in [("2013-12-09", "2013-12-16", "#D4A017"), ("2013-12-16", "2013-12-23", "#C8102E"),
                    ("2013-12-23", "2013-12-30", "#5A5A5A")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.15, lw=0)
    ax.set_title(nm, loc="left", fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.set_ylabel("net/jam", fontsize=6)
axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))
plt.tight_layout()
plt.savefig(LAP / "gambar" / "fig_pola.png", dpi=250, bbox_inches="tight")
plt.savefig(HASIL / "fig_pola_lokasi.png", dpi=250, bbox_inches="tight")
plt.close()

(LAP / "values.json").write_text(json.dumps(V, indent=1, ensure_ascii=False))
print("values.json:", len(V), "nilai")
print("tulis:", sorted(p.name for p in HASIL.iterdir()))
