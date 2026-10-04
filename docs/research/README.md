# HarbiChess araştırma hafızası

Kalıcı kaynak GitHub'dır. Notlar ve kararlar Git'te, büyük model/replay dosyaları commit-linked Releases'ta tutulur. Space okunabilir kopyadır; otomatik senkronizasyon veya zamanlanmış görev kurulmamıştır.

- [MERCEK Laya/karar modeli araştırması](MERCEK-decision-models-20261003.md),
  [kaynaklar](MERCEK-sources-20261003.json) ve
  [ön kayıtlı search bütçesi sonucu](../runs/MERCEK-search-result-20261003.md):
  48 gerçek oyun, 16/64/128 simulation; güç artışı doğrulanmadı, ana öğrenme tanısı öncelikli.
- [PORT Linux sonucu](../runs/PORT-linux-result-20261002.md), [çalıştırma/resume](../PORT-runtime.md)
  ve [teknik kararlar](PORT-decisions-20261002.md): gerçek CPU training/inference ve güç sınırları.
- [Güncel değerlendirme](status-20261004.md): gerçek Linux/MLX CPU, öğretici kazanımı,
  başarısız self-learning, güç/hız ölçümleri ve açık Release yedekleme sorunu.
- [2 Ekim tarihsel değerlendirme](status-20261002.md): o günkü arşiv/audit durumu korunur.
- [UFUK öğretici sonucu](../runs/UFUK-balanced-result-20261003.md),
  [başarısız gerçek self-play](../runs/UFUK-selfplay-result-20261003.md),
  [search sinyali](../runs/UFUK-search-signal-result-20261003.md),
  [raw-selector maçı](../runs/UFUK-policy-search-result-20261003.md) ve
  [CPU derleme ölçümü](../runs/UFUK-compile-result-20261003.md).
- [UFUK derinlik kontrolü sonucu](../runs/UFUK-depth-result-20261003.md):
  aynı başlangıç işlevi/örnekleme akışıyla kapasite eşiği başarısız.
- [Native search hedefi](../runs/UFUK-policy-target-result-20261003.md) ve
  [gerçek depolanmış self-play hedefi](../runs/UFUK-replay-target-result-20261003.md):
  olumlu sınırlı hedef tanıları; maç gücü veya self-learning başarısı değil.
- [Value sabit policy dalı protokolü](../runs/UFUK-policy-adapter-preregistration-20261003.md):
  aynı veride temsil karşılaştırması; dönüşüm weights-only, sonra tam native resume.
- [Policy dalı sonucu](../runs/UFUK-policy-adapter-result-20261003.md): ek temsil eşiği başarısız.
- [Sabit bütçede kök genişliği](../runs/UFUK-root-budget-result-20261003.md) ve
  [gerçek eşlemeli maçlar](../runs/UFUK-root-arena-result-20261003.md): aynı model/16sim
  dar search %87,5 skor; model öğrenmesi değil, sınırlı geliştirme search kazanımı.
- [Dar search ile yeni self-play ön kaydı](../runs/UFUK-narrow-selfplay-preregistration-20261003.md):
  final/başlangıç aynı search kullanır; taze model öğrenmesi ayrıca ölçülür.
- [Dar self-learning sonucu](../runs/UFUK-narrow-selfplay-result-20261003.md),
  [own-outcome value sonucu](../runs/UFUK-value-outcome-result-20261003.md):
  loss ilerlemesi oyun gücü başarısı sayılmadı; iki güç kapısı da başarısız.
- [Ölçülen hedef entropisi](../runs/UFUK-target-entropy-result-20261003.md),
  [genişlik sonucu](../runs/UFUK-width-result-20261004.md),
  [Q-range normalizasyonu](../runs/UFUK-q-range-result-20261004.md): yeni hipotezler,
  başarısız learning/mechanism kapıları, gerçek tam resume ve CPU latency.
- [Saklanan offline control güç testi](../runs/UFUK-offline-control-result-20261004.md):
  yeni stage aileleri/aynı search, model güç kapısı başarısız; SF512 referansı zayıflığı gösterdi.
- [Global policy context sonucu](../runs/UFUK-context-result-20261004.md):
  iki kol8.000update,66tam checkpoint, gerçek eğitilmiş MLXCPU parity;
  native öğrenme başarısız, hız kapısı geçti, maç/promotion yok.
- [Taktik leaf kontrolleri](../runs/UFUK-tactical-leaf-result-20261004.md) ve
  [CPU primary-source araştırması](UFUK-cpu-tactical-mechanisms-20261004.md):
  neural/static/quiescent karşılaştırmasında kalite, süre ve checked fallback kapıları başarısız.
- [2.149 dosyalık yerel arşiv](../runs/UFUK-artifact-archive-20261003.json) ve
  [143 dosyalık ek](../runs/UFUK-artifact-supplement-20261004.json): tüm üyeler geri okunup
  [239 dosyalık context/tactical ek](../runs/UFUK-artifact-context-tactical-supplement-20261004.json):
  bütün üyeler geri okunup hashleri doğrulandı; Release upload400 yüzünden
  yeni uzak binary yedek hâlâ eksik.
- [Codeword rehberi](codewords.md): araştırma snapshot'ındaki 54 prefix ve 896 commit.
- [Tarihsel deney raporları](../runs/).
- [Dosya manifesti](archive-20261002-manifest.json): 944 artifact/distribution dosyasının boyutu ve SHA-256 değeri.
- [CodeProjects / HarbiChess](https://chatgpt.com/space/page_bfc28b10daa081919e1650eca18c61c7): 151 tarihsel belge, mimari audit ve katkı kuralları.
- [Space sayfa eşlemesi](space-index.json).
- [Araştırma arşivi](https://github.com/Eta06/HarbiChess/releases/tag/research-archive-20261002).

## 2 Ekim 2026 arşivi

Kaynak snapshot: `3804f197665c131b0d7e0e52de2d18e14ed9c5c2`. Sonraki ARSIV commitleri dokümantasyon ve arşiv indeksidir; eski checkpoint provenance'ını değiştirmez. Bu bir production model sürümü veya promotion değildir. DENGE başarısızlığı korunur.

Release paketleri: 181 replay shard; 281 model/optimizer safetensors; 480 JSON/profiler kaydı; 2 tarihsel wheel/sdist; kaynak snapshot'a kadar tüm main geçmişini taşıyan Git bundle. Eski wheel/sdist güncel build değildir. Yerel kaynaklar silinmedi. `.idea`, `.venv`, `node_modules` ve cache'ler kapsam dışındadır.

Paketleme sırasında 944 dosyanın arşivden açılan baytları kaynak SHA-256 değerleriyle karşılaştırıldı. Git bundle doğrulandı. GitHub asset digest'leri ayrıca yerel paket hash'leriyle karşılaştırılır. Arşiv doğruluğu, checkpoint/resume veya production training doğruluğunu tek başına kanıtlamaz.

## Geri yükleme

Release'in tüm asset'lerini yeni, boş bir dizine indirin. SHA256SUMS ile `shasum -a 256 -c SHA256SUMS` çalıştırın. Bundle'dan `git clone harbichess-source-3804f19.bundle HarbiChess-restored` ile tarihsel checkout oluşturabilirsiniz. Güncel notlar için GitHub main'i kullanın; bundle sonraki ARSIV commitlerini içermez.

Doğrulanmış tar paketlerini checkout kökünde açın: dosyalar `artifacts/` ve `dist/` yollarını korur. Mevcut dosyaları üzerine yazmamak için boş checkout kullanın. Manifestteki tekil hash'leri de doğrulayın. Bağımlılıklar arşivde değildir; ortamı repo talimatlarıyla yeniden kurun. Restorasyon otomatik training başlatma izni değildir.

## Yeni deney kaydı şablonu

Her aşama için `docs/runs/<keyword>-<deney>.md` kullanın. Repo kökündeki AGENTS.md katkı kurallarını uygulayın.

1. Soru ve hipotez; neden bu deney?
2. Kaynak commit, model/veri hash'leri, schema, seed/opening ve split provenance.
3. Kontrol ve değişen tek mekanizma; frozen koşullar.
4. Önceden sabitlenen compute/wall-clock bütçesi, gate'ler, belirsizlik hesabı ve durma kuralı.
5. Komutlar, ortam/donanım, gerçek süre/throughput ve testler.
6. Sonuçlar: passed/failed/incomplete, bağımsız kanıt ve sınırlamalar.
7. Karar, sonraki hipotez, Release bağlantısı ve dosya envanteri.

Başarısız kayıtlar korunur. Sonuç görüldükten sonra eşik değiştirmek yeni deney gerektirir. AI katkıcılarının Space erişimi olmayabilir; bu durumda Git kaydı tamamlanmalı ve Space senkronizasyon açığı açıkça belirtilmelidir.
