# Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler Milan

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

Perangkat yang kami pakai: Google Colab (GPU NVIDIA T4, PyTorch 2.11.0+cu130). Total waktu training:
17,7 menit untuk 273 pelatihan unik (348 baris di `ablasi.csv`; 75
baris disalin karena konfigurasinya identik dengan run sebelumnya).

Tabel, gambar, dan angka laporan dibuat ulang dari `hasil/ablasi.csv` (tanpa PyTorch):

```bash
python scripts/analisis.py          # tabel (hasil/*.csv), gambar, laporan-latex/values.json
python scripts/bangun_laporan.py    # mengisi laporan-latex/main.tex dari values.json
cd laporan-latex && latexmk -pdf main.tex
```

## Ringkasan Hasil

Di Duomo (satu jam ke depan), konfigurasi acuan menghasilkan test MAE RNN 229,6 $\pm$ 5,1, GRU 237,2 $\pm$ 9,3,
LSTM 273,6 $\pm$ 25,4, dan ANN 309,9 $\pm$ 5,9, semuanya di bawah baseline naif terbaik (467,6).
Konfigurasi terbaik per model (dipilih dengan val MAE) memberi test MAE RNN 226,3 $\pm$ 4,9, GRU 230,2 $\pm$ 8,5,
LSTM 247,3 $\pm$ 12,8, dan ANN 279,2 $\pm$ 13,4; namun val MAE hanya turun untuk ANN dan justru naik untuk RNN, LSTM, dan GRU
(interaksi antar faktor). Di Bandara Linate urutan model acuan sama, tetapi konfigurasi terbaik hanya membantu ANN
(test MAE 8,8 $\pm$ 0,7 menjadi 7,9 $\pm$ 0,5), dan pada minggu Natal baseline nilai terakhir (7,1)
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
