# HarbiChess araştırma hafızası

Kalıcı kaynak GitHub'dır. Notlar ve kararlar Git'te, büyük model/replay dosyaları commit-linked Releases'ta tutulur. Space okunabilir kopyadır; otomatik senkronizasyon veya zamanlanmış görev kurulmamıştır.

- [Güncel değerlendirme](status-20261002.md): başarı, başarısızlık ve doğrulama açıkları.
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
