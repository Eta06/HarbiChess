# HarbiChess çalışma kuralları

## Araştırma kaydı

- Önce docs/research/README.md, docs/research/status-20261002.md ve ilgili docs/runs raporlarını okuyun. Tarihsel raporlar o zamanki sonucu anlatır; güncel gerçekliği tek başına kanıtlamaz.
- Her anlamlı deneyden önce hipotez, kontrol, veri/seed kaynağı, compute veya wall-clock tavanı, başarı/başarısızlık kriterleri ve durma kuralını yazın.
- Deneyden sonra gerçek sonuçları, komutları, kaynak commitini, veri/model SHA-256 değerlerini, testleri, maliyeti ve sınırlamaları docs/runs altında kaydedin. Başarısız deneyi silmeyin veya geriye dönük başarılı saymayın.
- Yeni aşamanın codeword açıklamasını docs/research/codewords.md dosyasına, kalıcı kararları ilgili rapora ekleyin. Codeword başarı göstergesi değildir.
- CodeProjects içindeki HarbiChess sayfası okunabilir araştırma günlüğüdür. Kullanıcının kapsamı içinde güncellerken doğrulanmış GitHub kanıtlarına bağlantı verin; sohbet veya tahmini gerçekleşmiş iş gibi yazmayın. Space erişimi yoksa Git kayıtlarını tamamlayıp senkronizasyonun eksik olduğunu belirtin.
- Otomasyon veya yeni training/generation/promotion için bu dosyayı tek başına yetki saymayın. Mevcut kullanıcı talebini esas alın. Production readiness henüz kanıtlanmış değildir.

## Commit ve yedekleme

- Varsayılan: bir anlamlı dosya = bir atomik commit. Aynı aşamadaki commitler aynı kısa prefix ile başlar. Yalnız gerçekten ayrılmaz teknik değişiklikler birlikte commitlenebilir.
- Açık ve açıklayıcı commit mesajları kullanın; co-author eklemeyin. Kullanıcının yapılandırılmış Git kimliğini değiştirmeyin.
- İlgili testleri çalıştırdıktan sonra dosyaları ayrı commit edin ve yetkilendirilmiş çalışma kapsamında push edin. Sonunda hash/dosya eşleşmesini, test sonuçlarını ve git status durumunu raporlayın.
- Kullanıcının ilgisiz değişikliklerini koruyun. .idea, node_modules, .venv ve cache'leri yüklemeyin.
- Büyük üretilmiş artifact'ler Git geçmişine değil commit-linked GitHub Releases'a gider. Dosya envanteri, SHA-256 ve geri yükleme yönergesi ekleyin.
- Kaynak araştırma snapshot'ı ile arşivleme commitini ayırın. Eski checkpoint provenance alanlarını yeni commit ile değiştirmeyin.
- Arşiv yayımlamak model promotion değildir. Deneylerin failed/incomplete durumlarını koruyun.
- Yüklemeyi ve uzak asset bütünlüğünü doğrulamadan yedekleme tamamlandı demeyin. Yerel dosyaları kullanıcı istemedikçe silmeyin.
- Notion kullanmayın. Bu proje için kalıcı kaynak GitHub, okunabilir günlüğün hedefi CodeProjects/HarbiChess'tir.
