<<PREAMBLE>>

\begin{document}

\title{Studi Ablasi RNN, LSTM, dan GRU dibandingkan ANN untuk Prediksi Trafik Seluler Kota Milan}

\author{
\IEEEauthorblockN{Tobagus Ahmad Yogi}
\IEEEauthorblockA{NIM 124400075\\
Teknik Telekomunikasi, Institut Teknologi Sumatera (ITERA)\\
Kelas RA}
}

\maketitle

\begin{abstract}
Prediksi trafik per sel membantu operator mengalokasikan sumber daya jaringan. Laporan ini menguji apakah keunggulan model rekuren atas ANN pada prediksi aktivitas internet satu jam ke depan di kotak Duomo, Milan, bertahan ketika hyperparameter diubah. Dengan data per jam Telecom Italia, kami menjalankan studi ablasi satu faktor per waktu untuk lima faktor wajib (panjang jendela, ukuran \textit{hidden}, jumlah lapisan, \textit{learning rate}, kelompok fitur) dan dua faktor pilihan (\textit{batch size}, \textit{dropout}), masing-masing pada ANN, RNN, LSTM, dan GRU dengan tiga \textit{seed}. Pada konfigurasi acuan, RNN terbaik (test MAE <<ref.RNN.test_MAE.ms>>), disusul GRU, LSTM, dan ANN (<<ref.ANN.test_MAE.ms>>), dan semuanya mengalahkan baseline naif terbaik (<<naif.duomo.best.test>>). Faktor paling berpengaruh berbeda antar model: panjang jendela terutama merugikan ANN, sedangkan \textit{learning rate} $10^{-4}$ membuat semua model belum konvergen dalam 300 epoch. Menggabungkan nilai terbaik tiap faktor menurunkan val MAE ANN sebesar <<dC.ANN.val_MAE.pctabs>>\,\% tetapi menaikkannya pada GRU (<<dC.GRU.val_MAE.pctabs>>\,\%), LSTM, dan RNN. Di Bandara Linate, urutan model acuan sama dengan di Duomo, tetapi konfigurasi terbaik hanya membantu ANN.
\end{abstract}

\begin{IEEEkeywords}
prediksi trafik seluler, RNN, LSTM, GRU, studi ablasi
\end{IEEEkeywords}

% ---------------------------------------------------------------------
\section{Pendahuluan}
Operator seluler meramal trafik per sel untuk mengatur kapasitas dan menghemat energi. Model sekuensial seperti RNN, LSTM \cite{hochreiter1997lstm}, dan GRU \cite{cho2014gru} lazim dipakai karena membaca data secara berurutan dengan jumlah parameter yang tidak bergantung pada panjang jendela, sedangkan ANN harus meratakan seluruh jendela menjadi satu vektor. Di Sesi 2 workshop, keempat model dibandingkan dengan \emph{satu} konfigurasi, dan hasilnya RNN $\approx$ GRU $<$ LSTM $<$ ANN pada test MAE. Pertanyaan laporan ini: apakah kesimpulan itu tetap berlaku bila hyperparameter diubah?

Studi ablasi mengubah satu faktor dalam satu waktu dari konfigurasi acuan dan mengamati dampaknya, sehingga pengaruh tiap faktor dapat dipisahkan. Kontribusi laporan ini: (1) ablasi tujuh faktor pada empat model dengan tiga \textit{seed} di kotak Duomo; (2) pemilihan konfigurasi terbaik per model berdasarkan val MAE, bukan test MAE, dengan pelaporan di minggu test dan minggu Natal; dan (3) uji generalisasi di Bandara Linate.

% ---------------------------------------------------------------------
\section{Data dan Metode}

\subsection{Dataset}
Data berasal dari Telecom Italia Big Data Challenge \cite{barlacchi2015milan} dalam versi agregat per jam dari \cite{zhang2021fedda}, 4 November--29 Desember 2013. Setiap jam memuat aktivitas internet (\texttt{net}, target), panggilan (\texttt{call}), dan SMS (\texttt{sms}); ditambah jam dan hari dalam pekan yang dikodekan sin--cos, sehingga ada tujuh fitur. Lokasi utama adalah kotak Duomo (5060), dan lokasi kelompok adalah kode 5, <<LOKASI>> (kotak 4489), sesuai digit terakhir NIM 124400075. Gambar~\ref{fig:pola} memperlihatkan perbedaan keduanya: rata-rata \texttt{net} per jam pada periode latih adalah <<kd.duomo.mean_train>> di Duomo dan <<kd.linate.mean_train>> di Linate.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{fig_pola.png}
  \caption{Aktivitas internet per jam di Duomo dan Bandara Linate. Bidang berwarna: minggu validasi (kuning), test (merah), dan Natal (abu-abu). Di Linate, trafik minggu Natal turun dan mendatar.}
  \label{fig:pola}
\end{figure}

\subsection{Pembagian Data dan Praproses}
Pembagian murni berdasarkan waktu: latih 11 November--8 Desember (672 jam), validasi 9--15 Desember, test 16--22 Desember, dan Natal 23--29 Desember. Data deret waktu tidak boleh diacak karena jam-jam bertetangga sangat berkorelasi, sehingga pengacakan akan membocorkan masa depan ke data latih. Standardisasi memakai rata-rata dan simpangan baku sebelum minggu validasi. Model menerima jendela $W$ jam terakhir dan meramal \texttt{net} satu jam berikutnya.

\subsection{Model dan Konfigurasi Acuan}
ANN meratakan jendela $[W, 7]$ menjadi satu vektor lalu melewatkannya ke lapisan padat ber-ReLU. RNN, LSTM, dan GRU memproses jendela langkah demi langkah, dan keluaran langkah terakhir masuk ke lapisan linear. Konfigurasi acuan mengikuti Sesi 2: $W=24$, horizon 1 jam, \textit{hidden} ANN 32, RNN 68, LSTM 33, GRU 38 (jumlah parameter sengaja dibuat mirip, <<ref.RNN.param.m>>--<<ref.ANN.param.m>>), ANN 2 lapisan dan model rekuren 1 lapisan, ketujuh fitur, Adam dengan \textit{learning rate} $10^{-3}$ \cite{kingma2015adam}, \textit{batch size} 64, tanpa \textit{dropout}, loss MSE, serta \textit{early stopping} dengan \textit{patience} 30 dan maksimum 300 epoch (bobot dengan val loss terendah dipakai). Implementasi memakai PyTorch \cite{paszke2019pytorch}.

\subsection{Protokol Ablasi dan Evaluasi}
Faktor yang diubah satu per satu: A1 panjang jendela $W\in\{6,12,24,48,168\}$; A2 ukuran \textit{hidden} $\{16,32,64,128\}$; A3 jumlah lapisan $\{1,2,3\}$; A4 \textit{learning rate} $\{10^{-4},10^{-3},10^{-2}\}$; A5 kelompok fitur $\{$\texttt{net}, \texttt{net+call+sms}, \texttt{net+waktu}, semua$\}$; B1 \textit{batch size} $\{16,64,256\}$; B2 \textit{dropout} $\{0;0{,}2;0{,}5\}$. Hipotesis B1 dan B2 dituliskan \emph{sebelum} eksperimen dijalankan. Setiap konfigurasi dilatih dengan \textit{seed} 0, 1, 2, dan hasil dilaporkan sebagai rata-rata $\pm$ simpangan baku sampel. Metrik adalah MAE, RMSE, dan MAPE pada skala asli. Ablasi dinilai dengan \textbf{val MAE}; minggu test dan Natal hanya dipakai untuk acuan, konfigurasi terbaik, dan uji generalisasi. Konfigurasi terbaik tiap model dibentuk dengan mengambil, untuk setiap faktor, nilai dengan val MAE rata-rata terendah, lalu menggabungkannya. Seluruh pelatihan dijalankan di Google Colab (GPU T4, PyTorch <<meta.torch>>): <<meta.n_unik>> pelatihan unik (<<meta.n_baris>> baris pada \texttt{ablasi.csv}, <<meta.n_salin>> di antaranya disalin karena konfigurasinya identik) dengan total waktu <<meta.menit>> menit.

% ---------------------------------------------------------------------
\section{Hasil}

\subsection{Baseline Naif dan Acuan (Bagian 0)}
Tabel~\ref{tab:base} memuat tiga baseline naif dan konfigurasi acuan di Duomo. Pada minggu test, urutan model sama dengan Sesi 2: RNN (<<ref.RNN.test_MAE.ms>>) $\approx$ GRU (<<ref.GRU.test_MAE.ms>>) $<$ LSTM (<<ref.LSTM.test_MAE.ms>>) $<$ ANN (<<ref.ANN.test_MAE.ms>>). Selisih RNN dan GRU lebih kecil daripada simpangan baku GRU, sehingga keduanya tidak dapat dibedakan. Semua model mengalahkan baseline naif terbaik pada minggu test (<<naif.duomo.best.test>>, \emph{<<naif.duomo.best.test.nama>>}); RNN memangkas galat <<c0.rnn_vs_naif>>\,\% dari baseline itu. Pada minggu Natal, baseline terbaik adalah \emph{nilai terakhir} (<<naif.duomo.best.natal>>) dan hanya RNN (<<ref.RNN.natal_MAE.ms>>) yang berada di bawahnya, tipis; ANN (<<ref.ANN.natal_MAE.m>>) dan LSTM (<<ref.LSTM.natal_MAE.m>>) jauh di atasnya. Pola Natal berbeda dari empat minggu latih (rata-rata trafik turun <<kd.duomo.turun_natal_pct>>\,\% dibanding minggu test), sehingga model yang hanya melihat pekan biasa tidak siap menghadapinya, sedangkan baseline nilai terakhir menyesuaikan diri seketika.

<<TABEL_BASE>>

\subsection{Ablasi Wajib (A1--A5)}
Gambar~\ref{fig:abl} merangkum val MAE seluruh faktor.

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.93\textwidth]{fig_ablasi.png}
  \caption{Val MAE (rata-rata $\pm$ simpangan baku, 3 \textit{seed}) terhadap tiap faktor ablasi di Duomo. Panel A1--A5 adalah faktor wajib, B1--B2 faktor pilihan. Sumbu-$y$ berbeda tiap panel.}
  \label{fig:abl}
\end{figure*}

\textbf{A1, panjang jendela.} Val MAE ANN naik monoton dari <<ab.A1.6.ANN.val.m>> ($W=6$) ke <<ab.A1.168.ANN.val.m>> ($W=168$), kenaikan <<c1.a1_ann_pct>>\,\%, sementara parameternya membengkak dari <<c1.a1_param6>> menjadi <<c1.a1_param168>> (<<c1.a1_ratio>>$\times$). Model rekuren hampir datar (RNN <<ab.A1.48.RNN.val.m>>--<<ab.A1.6.RNN.val.m>>, GRU <<ab.A1.48.GRU.val.m>>--<<ab.A1.6.GRU.val.m>>) dengan parameter tetap, 5,3--5,6 ribu. Jendela yang lebih panjang tidak selalu lebih baik; ia terutama merugikan ANN. Pada $W=6$, ANN (<<ab.A1.6.ANN.val.m>>) setara dengan GRU (<<ab.A1.6.GRU.val.m>>) dan RNN (<<ab.A1.6.RNN.val.m>>).

\textbf{A2, ukuran \textit{hidden}.} Model yang lebih besar membantu keempat model sampai 128 tanpa tanda \textit{overfitting} pada val MAE, mungkin karena \textit{early stopping} membatasi kapasitas efektif. GRU turun dari <<ab.A2.16.GRU.val.m>> ke <<ab.A2.128.GRU.val.m>> dan menjadi yang terbaik di grid; RNN jenuh pada 64 (<<ab.A2.64.RNN.val.m>> vs <<ab.A2.128.RNN.val.m>> pada 128). Perbandingan ini tidak setara dari sisi parameter: pada \textit{hidden} 128, LSTM memiliki <<ab.A2.128.LSTM.param>> parameter, <<c1.a2_ratio>>$\times$ RNN (<<ab.A2.128.RNN.param>>).

\textbf{A3, jumlah lapisan.} Menumpuk lapisan umumnya tidak membantu. LSTM paling sensitif: <<ab.A3.1.LSTM.val.m>> (1 lapisan) menjadi <<ab.A3.3.LSTM.val.m>> (3 lapisan), kenaikan <<c1.a3_lstm_pct>>\,\%. RNN naik ke <<ab.A3.3.RNN.val.m>> pada 3 lapisan, dan ANN hampir tidak berubah (<<ab.A3.1.ANN.val.m>>--<<ab.A3.3.ANN.val.m>>). Hanya GRU sedikit membaik pada 2 lapisan (<<ab.A3.2.GRU.val.m>> vs <<ab.A3.1.GRU.val.m>>), selisih yang berada dalam simpangan baku. Model lebih dalam berhenti lebih cepat: epoch RNN turun dari <<ab.A3.1.RNN.epoch>> ke <<ab.A3.3.RNN.epoch>>.

\textbf{A4, \textit{learning rate}.} Pada $10^{-4}$ semua model di semua \textit{seed} berhenti di batas 300 epoch dan val MAE tinggi (<<c1.a4_lo>>--<<c1.a4_hi>>): pelatihan belum konvergen. Pada $10^{-2}$ pelatihan jauh lebih cepat (<<c1.a4_ep_lo>>--<<c1.a4_ep_hi>> epoch) dengan hasil setara $10^{-3}$ untuk GRU (<<ab.A4.0.01.GRU.val.m>> vs <<ab.A4.0.001.GRU.val.m>>), lebih baik untuk LSTM (<<ab.A4.0.01.LSTM.val.m>> vs <<ab.A4.0.001.LSTM.val.m>>), dan lebih buruk untuk RNN (<<ab.A4.0.01.RNN.val.m>> vs <<ab.A4.0.001.RNN.val.m>>) dan ANN (<<ab.A4.0.01.ANN.val.m>> vs <<ab.A4.0.001.ANN.val.m>>); simpangan baku antar \textit{seed} naik pada keempat model.

\textbf{A5, kelompok fitur.} Hanya \texttt{net} adalah yang terburuk di semua model (<<c1.a5_lo>>--<<c1.a5_hi>>). Dari \texttt{net} ke semua fitur, val MAE turun <<c1.a5_rnn>>\,\% (RNN), <<c1.a5_gru>>\,\% (GRU), <<c1.a5_lstm>>\,\% (LSTM), dan hanya <<c1.a5_ann>>\,\% (ANN). Fitur waktu memberi manfaat lebih besar daripada \texttt{call}+\texttt{sms} untuk ANN, LSTM, dan GRU; pada RNN keduanya setara dan saling melengkapi (<<ab.A5.net+waktu.RNN.val.m>> $\rightarrow$ <<ab.A5.semua.RNN.val.m>> bila digabung). Ketiga model rekuren memperoleh manfaat fitur lebih besar daripada ANN.

\subsection{Ablasi Pilihan (B1--B2)}
Tabel~\ref{tab:hip} menguji hipotesis yang ditulis sebelum eksperimen terhadap val MAE Gambar~\ref{fig:abl}.

<<TABEL_HIP>>

\textit{Batch size} 256 terbukti merugikan semua model, dan GRU berhenti di batas 300 epoch pada ketiga \textit{seed} karena hanya sekitar tiga pembaruan per epoch. \textit{Batch} 16 berguna terutama untuk LSTM. \textit{Dropout} tidak membantu model mana pun: ANN dan RNN memburuk sejak 0,2, dan 0,5 memperburuk semua. Dugaan kami, \textit{dropout} di sini dipasang pada keluaran lapisan terakhir (atau tiap lapisan tersembunyi ANN), tempat seluruh informasi untuk meramal satu nilai kontinu dikompresi, sehingga menjatuhkan unit langsung menambah galat ramalan; kami tidak menguji dugaan ini.

\subsection{Konfigurasi Terbaik (Bagian C)}
Hasil seleksi val MAE tertera di Tabel~\ref{tab:cd} (blok Duomo). Perubahan dari acuan: ANN $W=6$, \textit{hidden} 128, 1 lapisan, fitur \texttt{net+waktu}, \textit{batch} 16; RNN $W=48$, \textit{batch} 16; LSTM $W=6$, \textit{hidden} 128, \textit{learning rate} $10^{-2}$, \textit{batch} 16; GRU $W=48$, \textit{hidden} 128, 2 lapisan, \textit{batch} 16. Pada ANN, nilai \texttt{net+waktu} hanya unggul <<c1.fit_ann_diff>> poin dari semua fitur, jauh di bawah simpangan baku; aturan val MAE tetap diterapkan apa adanya.

Penggabungan ini \textbf{tidak menurunkan val MAE untuk tiga dari empat model}: ANN turun <<dC.ANN.val_MAE.pctabs>>\,\% (<<ref.ANN.val_MAE.m>> $\rightarrow$ <<C.ANN.val_MAE.m>>), tetapi RNN naik <<dC.RNN.val_MAE.pctabs>>\,\%, LSTM <<dC.LSTM.val_MAE.pctabs>>\,\%, dan GRU <<dC.GRU.val_MAE.pctabs>>\,\% (<<ref.GRU.val_MAE.m>> $\rightarrow$ <<C.GRU.val_MAE.m>>). Pada minggu test, keempat model lebih rendah dari acuan: ANN <<dC.ANN.test_MAE.pctabs>>\,\%, LSTM <<dC.LSTM.test_MAE.pctabs>>\,\%, GRU <<dC.GRU.test_MAE.pctabs>>\,\%, RNN <<dC.RNN.test_MAE.pctabs>>\,\%, tetapi hanya perbaikan ANN dan LSTM yang melampaui simpangan baku; perbaikan GRU dan RNN berada di dalamnya. Pada minggu Natal, semua model membaik (ANN <<dC.ANN.natal_MAE.pctabs>>\,\%, LSTM <<dC.LSTM.natal_MAE.pctabs>>\,\%, GRU <<dC.GRU.natal_MAE.pctabs>>\,\%, RNN <<dC.RNN.natal_MAE.pctabs>>\,\%), dan RNN (<<C.RNN.natal_MAE.ms>>) semakin jauh di bawah baseline nilai terakhir (<<naif.duomo.best.natal>>). Urutan model pada test tetap RNN <<C.RNN.test_MAE.m>> $\approx$ GRU <<C.GRU.test_MAE.m>> $<$ LSTM <<C.LSTM.test_MAE.m>> $<$ ANN <<C.ANN.test_MAE.m>>.

\subsection{Uji Generalisasi (Bagian D)}
Konfigurasi acuan dan terbaik dari Duomo dilatih ulang tanpa perubahan di <<LOKASI>> (Tabel~\ref{tab:cd}, blok Linate). Urutan model acuan sama dengan di Duomo: RNN (<<Da.RNN.test_MAE.ms>>) $\approx$ GRU (<<Da.GRU.test_MAE.ms>>) $<$ LSTM (<<Da.LSTM.test_MAE.ms>>) $<$ ANN (<<Da.ANN.test_MAE.ms>>), dan semuanya mengalahkan baseline naif terbaik (<<naif.linate.best.test>>, \emph{<<naif.linate.best.test.nama>>}). Konfigurasi terbaik \textbf{tidak ikut membaik} di sini: hanya ANN yang turun (<<dD.ANN.test_MAE.pctabs>>\,\%), sedangkan GRU (<<dD.GRU.test_MAE.pcts>>\,\%), LSTM (<<dD.LSTM.test_MAE.pcts>>\,\%), dan RNN (<<dD.RNN.test_MAE.pcts>>\,\%) bergeser naik; selisih GRU dan RNN masih di dalam simpangan baku (RNN <<Dt.RNN.test_MAE.s>>), sedangkan LSTM sedikit di luarnya. Urutan model pun berubah: ANN (<<Dt.ANN.test_MAE.m>>) kini mengungguli LSTM (<<Dt.LSTM.test_MAE.m>>) dan RNN (<<Dt.RNN.test_MAE.m>>). Val MAE justru turun untuk ANN, GRU, dan LSTM (mis.\ GRU <<Da.GRU.val_MAE.m>> $\rightarrow$ <<Dt.GRU.val_MAE.m>>), jadi perbaikan pada val tidak berpindah ke minggu test. Pada minggu Natal, \emph{semua} model, baik acuan (<<Da.RNN.natal_MAE.m>>--<<Da.ANN.natal_MAE.m>>) maupun terbaik, kalah dari baseline nilai terakhir (<<naif.linate.best.natal>>).

<<TABEL_CD>>

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.86\textwidth]{fig_CD.png}
  \caption{Test MAE dan Natal MAE, konfigurasi acuan (arsir) dan terbaik (pekat), di Duomo dan Bandara Linate (rata-rata $\pm$ simpangan baku, 3 \textit{seed}). Garis putus: baseline naif terbaik pada split yang sama.}
  \label{fig:cd}
\end{figure*}

% ---------------------------------------------------------------------
\section{Pembahasan}
\textbf{Apakah keunggulan model rekuren bertahan?} Sebagian. Urutan RNN $\approx$ GRU $<$ LSTM $<$ ANN bertahan pada konfigurasi acuan di Duomo dan Linate, pada konfigurasi terbaik di Duomo, dan pada \textit{batch} 16 dan 64, tetapi bukan hukum umum: ia berubah pada konfigurasi terbaik di Linate (ANN <<Dt.ANN.test_MAE.m>> di bawah LSTM dan RNN, keduanya <<Dt.LSTM.test_MAE.m>>). Pada $W=6$, ANN setara dengan model rekuren (val MAE <<ab.A1.6.ANN.val.m>> vs <<ab.A1.6.GRU.val.m>> pada GRU); pada \textit{hidden} 128, GRU (<<ab.A2.128.GRU.val.m>>) mengungguli RNN (<<ab.A2.128.RNN.val.m>>); dan pada \textit{batch} 256, LSTM (<<ab.B1.256.LSTM.val.m>>) lebih buruk daripada ANN (<<ab.B1.256.ANN.val.m>>). Keunggulan rekuren dengan demikian terutama berasal dari kemampuannya membaca jendela panjang tanpa menambah parameter: kerugian ANN pada $W$ besar (A1) adalah efek terbesar kedua pada ANN setelah \textit{dropout}. Di antara model rekuren, RNN paling sederhana dan pada sebagian besar konfigurasi tidak kalah dari LSTM dan GRU. Dugaan kami, pada hanya 672 sampel latih dan dependensi yang pendek (pola harian, $W=24$), gerbang LSTM dan GRU belum diperlukan; ini tidak diuji langsung.

\textbf{Faktor paling berpengaruh.} Menurut selisih val MAE terbesar antar nilai faktor, \textit{learning rate} paling dominan pada ketiga model rekuren (RNN <<eff.RNN.A4>> hingga GRU <<eff.GRU.A4>>), tetapi hampir seluruhnya berasal dari nilai $10^{-4}$ yang belum konvergen dalam 300 epoch. Pada ANN, \textit{dropout} (<<eff.ANN.B2>>) dan $W$ (<<eff.ANN.A1>>) lebih berpengaruh daripada \textit{learning rate} (<<eff.ANN.A4>>). Faktor berikutnya bergantung model: jumlah lapisan pada LSTM (<<eff.LSTM.A3>>) dan fitur pada GRU (<<eff.GRU.A5>>). Panjang jendela hampir tidak berpengaruh pada GRU (<<eff.GRU.A1>>). Karena pengaruhnya bergantung model, kesimpulan dari satu konfigurasi acuan tidak dapat dipindahkan begitu saja.

\textbf{Mengapa gabungan terbaik justru bisa lebih buruk?} Ablasi satu faktor mengasumsikan pengaruh yang aditif, padahal faktor saling berinteraksi. Pada GRU, \textit{hidden} 128 dan 2 lapisan bersama-sama menghasilkan <<C.GRU.param.m>> parameter, <<meta.n_param_ratio_GRU>>$\times$ konfigurasi acuan (<<ref.GRU.param.m>>) untuk 672 sampel latih; pelatihan berhenti pada rata-rata <<C.GRU.epoch.m>> epoch, dan val MAE memburuk. LSTM serupa (<<C.LSTM.param.m>> parameter, <<meta.n_param_ratio_LSTM>>$\times$). Di sisi lain, test dan Natal MAE membaik pada kedua model itu. Dengan satu minggu validasi dan satu minggu test, kami tidak dapat memastikan mana yang menggambarkan generalisasi sebenarnya; yang dapat dikatakan adalah bahwa val MAE saja bukan penuntun yang andal untuk konfigurasi gabungan. Seleksi terhadap banyak pilihan pada satu minggu validasi yang sama dengan \textit{early stopping} juga cenderung memberi val MAE yang terlalu optimistis: ANN, yang paling banyak berubah, menunjukkan selisih terbesar antara val (<<C.ANN.val_MAE.m>>) dan test (<<C.ANN.test_MAE.m>>).

\textbf{Mengapa Linate berbeda?} Trafik Linate berukuran jauh lebih kecil (rata-rata <<kd.linate.mean_train>> vs <<kd.duomo.mean_train>>) sehingga MAE tidak dapat dibandingkan langsung (RNN: <<kd.linate.relMAE_RNN_test>>\,\% dari rata-rata minggu test di Linate, <<kd.duomo.relMAE_RNN_test>>\,\% di Duomo). Pola harian Linate lebih lemah dan selisih akhir pekan--hari kerja lebih tajam: autokorelasi lag 24 jam hanya <<kd.linate.ac24>> (Duomo <<kd.duomo.ac24>>), dan rasio rata-rata akhir pekan terhadap hari kerja <<kd.linate.rasio_akhir_pekan>> (Duomo <<kd.duomo.rasio_akhir_pekan>>). Konfigurasi terbaik dari Duomo kemungkinan disesuaikan dengan pola harian yang tajam di sana, sehingga tidak otomatis cocok untuk lokasi yang polanya lebih berisik; kami tidak menguji penjelasan ini. Pada minggu Natal, rata-rata trafik Linate turun <<kd.linate.turun_natal_pct>>\,\% dibanding minggu test dan tampak mendatar (Gambar~\ref{fig:pola}), sehingga baseline nilai terakhir (<<naif.linate.best.natal>>) sulit dikalahkan oleh model yang dilatih pada pekan biasa.

\textbf{Keterbatasan.} Hanya satu pembagian waktu dengan empat minggu latih; validasi dan test masing-masing satu minggu; simpangan baku diambil dari tiga \textit{seed} sehingga kasar dan tidak disertai uji signifikansi; ablasi satu faktor tidak mengukur interaksi; horizon hanya satu jam; dan hanya dua lokasi. Minggu Natal berada di luar distribusi latih, sehingga hasilnya menguji ketahanan, bukan kualitas ramalan biasa.

% ---------------------------------------------------------------------
\section{Kesimpulan}
Pada prediksi trafik satu jam ke depan di Duomo, urutan RNN $\approx$ GRU $<$ LSTM $<$ ANN bertahan di konfigurasi acuan dan terbaik (test MAE terbaik: RNN <<C.RNN.test_MAE.ms>>; ANN <<C.ANN.test_MAE.ms>>), tetapi tidak di setiap konfigurasi: ANN menyusul model rekuren pada jendela pendek ($W=6$). \textit{Learning rate} yang terlalu kecil adalah kesalahan paling mahal, sedangkan \textit{dropout} dan \textit{batch} 256 merugikan. Menggabungkan nilai terbaik tiap faktor hanya menurunkan val MAE ANN (<<dC.ANN.val_MAE.pctabs>>\,\%) dan menaikkannya pada GRU (<<dC.GRU.val_MAE.pctabs>>\,\%), LSTM, dan RNN, meski test dan Natal MAE membaik. Di Bandara Linate, urutan model acuan sama tetapi konfigurasi terbaik hanya menguntungkan ANN, dan pada minggu Natal baseline nilai terakhir mengalahkan semua model. Pesan praktisnya: hasil ablasi perlu divalidasi ulang per lokasi, dan konfigurasi gabungan harus diuji, bukan diasumsikan.

% ---------------------------------------------------------------------
\section*{Kontribusi Anggota}
\begin{tabular}{@{}lp{0.62\linewidth}@{}}
  Tobagus Ahmad Yogi & Seluruh pekerjaan (kelompok satu orang): menjalankan eksperimen di Google Colab, pemilihan konfigurasi, dan penyusunan laporan. \\
\end{tabular}

\section*{Pernyataan Penggunaan AI}
Claude (Anthropic) dipakai untuk membantu memahami petunjuk tugas dan memandu pengaturan Google Colab, menyusun draf hipotesis B1--B2 sebelum eksperimen dijalankan, memilih konfigurasi terbaik secara mekanis dari \texttt{ablasi.csv} berdasarkan val MAE, menulis skrip analisis \texttt{scripts/analisis.py}, serta menyusun draf laporan ini. Seluruh pelatihan model dijalankan oleh penulis dengan \texttt{tugas\_ablasi.ipynb}, dan setiap angka dalam laporan dihasilkan skrip tersebut dari \texttt{hasil/ablasi.csv}.

\section*{Kode dan Data}
Kode, notebook beserta output, dan seluruh hasil run tersedia di \url{<<REPOURL>>}.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
