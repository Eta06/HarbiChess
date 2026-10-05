# UFUK-RESIDUAL: bağımsız denetlenen başarısız sonuç

Sabit1024 update ile eski own-play verisinden öğrenen iki residual aday, kayıtlı strength ekranını geçmedi. Promotion veya gerçek self-learning güç kazanımı iddiası yok. E8, en güçlü bağımsız doğrulanmış checkpoint olarak kalıyor. Yeni own-play verisiyle sonraki deneye geçiliyor; bu başarısızlık silinmiyor.

E8'in tam value logits'ine eklenen, başlangıçta sıfır2520 etkin koordinatlı residual eğitildi. Policy, trunk ve eski value parametreleri sabit kaldı. AdamW2e-5, batch256, dört değil sabit1024 update, own terminal CE+beta1 E8 KL ve önceden kaydedilmiş shrinkage kullanıldı. Seed20262705/06; eski131072 own geçiş/seed yeniden kullanıldı, yeni teacher sorgusu ve yeni own-play yok. Loss azalması güç ölçüsü değildir.

| Seed | residual–E8 | residual–zero additive | residual–SF512 | E8–SF512 | Zero–SF512 |
|---|---:|---:|---:|---:|---:|
|20262705|.09375 (1W/1D/14L)|.09375|.03125 (0W/1D/15L)|0|0|
|20262706|.15625 (2W/1D/13L)|.15625|.03125 (0W/1D/15L)|0|0|

Her hücre, aynı sekiz bilinen açılış ailesinin iki rengiyle16 maçtır. Duplicate E8 ve zero fonksiyonları yeni bağımsız aileler değildir. Toplam160 maç,11568 yasal continuation ply,1540747 gerçek Stockfish node,4383812 neural node ve4008328 NN değerlendirmesi. UNKNOWN cap yok. Aynı aramayla direct>.60, SF gain>.10 ve final SF>=.25 koşulları iki seed'de de başarısız. +.03125 SF farkını başarı saymıyoruz.

Arena ilk1791219945.0279903, son sınır1791227145.0279903: iki saatlik orijinal bütçe korunmuştur. Search Q512/qdepth2/maxdepth8; SF19 requested512, Threads1/Hash16. SF884 hamlede512 üstüne çıktı, maksimum598: actual node bütçesi eşit değildir. Sekiz aile küçük, geliştirmede görülmüş örneklem; rapordaki paired bootstrap aralıkları descriptive, virgin confirmation değildir. Kapalı eski MAX8 qualification muhasebesi değişmedi.

Root tam-history denetimine ek olarak ayrı agent160 legal history, terminal sonuç ve kaynak/model/protokol hash'lerini doğruladı; önceden belirlenen altı NN açılış hamlesinin arama paketlerini yeniden üretip birebir eşleştirdi. Bütün NN hamlelerini yeniden aradığını iddia etmiyor. Bağımsız denetim7.100s, ilk1791225920.413, sabit600 sınırı içinde; CPU bir thread, GPU kullanılmadı.

Final modeller:05 `12e82c58f62a4afb007878da85cdda46e32afd7e1e8d07ec4a6e29a0b203c2ac`,06 `a8e40bba18612e09aff082e9c0f77cab4bec1e05fd195283861625e843d95f17`. Fit source commit `6fcc8b476d25495d1c9c413e55b2c7ba4794013e`; arşivleme commit'iyle yeniden etiketlenmedi. Full model/Adam/global+sampler RNG/input binding checkpoint'leri korunuyor; public lossless native capsule byte readback ayrı kayıtlı. Inference ağırlıkları tek başına training resume değildir.

Kanıtlar: `experiments/ufuk/cpu-residual-value-v1/actual-result/final-fullhistory-result.json` ve `actual-result/independent-final-review/`. Bu aşamada6 CPU test/0skip ve gerçek whole8/pause4/freshresume8/all6 native proof daha önce geçti; final arena adapter5 test geçti. Apple/CUDA donanım denetimi yapılmadı. Başarısız strength sonucu mimari parity veya runtime denetimlerinin yerine kullanılmıyor.
