# UFUK-DEVAM: öğretmen başlangıcından gerçek kendi-search öğrenmesine

7 Ekim 2026, sonuçlar görülmeden kayıt. Başlangıç: `9e1ecaedf9f13621816eef3f9b0ae8784da52a55`. Kullanıcı çalışmaya yeniden devam edilmesini istedi. Önceki durdurma, başarısız deneyler, kayıtlar ve dolmuş deadline'lar korunur. Yeni teknik CPU tahsis penceresi `1791362516.685839`–`1791448916.685839`; bu bir kullanıcı deadline'ı veya geçmiş sürenin sıfırlanması değildir. GPU, SSH ve ücretli compute kullanılmaz.

## Hipotez ve değişen şey

Bir kere Stockfish etiketleriyle başlatılmış küçük king-bucket NNUE değer modelinin, bundan sonra yalnız kendi sabit başlangıç modelinin daha derin search deneyiminden öğrenmesi, aynı ucuz maç-search bütçesinde oyun gücünü artırabilir. Öğretmen eğitimi tamamlandı; tekrarlanmaz. Başarı ancak yeni ağırlıkların bağımsız oyunlarda aynı donmuş başlangıçtan ve E8'den güçlü olmasıdır. Search kodu ve maç bütçesi iki tarafta aynı kalır. Bu standart search distillation hipotezidir; NNUE veya distillation için özgünlük iddiası yoktur.

İki ayrı seed: `20262905`, `20262906`. Aynı seed'in öğretmen-256 `whole/candidate.pt` başlangıcı SHA'ları sırasıyla `52e029bba0e895740f975109f4c52358c7a59fe22082cb1fb528821e3fa2b02d` ve `4ff008426e2e8a09eeef5cc232195e015a1c6a498dd0ffb768e905e897eff4c8`. Yeni own phase ağırlık başlangıcıdır: optimizer ve RNG yeniden oluşturulur. Öğretmen eğitiminden tam resume denmez. Own phase içindeki pause/resume ise bütün ağırlıklar, baseline, Adam, Python/Torch/private-sampler RNG state'leriyle tam resume olmak zorundadır.

## Veri ve eğitim

Her seed'in doğrulanmış 4096 TRAIN kökü korunur. Önceden sabitlenmiş SHA sırası, outcome/score seçimi olmadan en çok 128 başlangıcı belirler; her başlangıç en çok 16 ply. Başlangıç modeli iki rengi de oynar. Aynı orijinal `BudgetSearch` (`de53c147…`), 8192 node, qdepth2, maxdepth8; tam hamle geçmişi, legal hamleler, claim-draw ve terminal kuralları kullanılır. Tam 1024 uygun mover-perspective own-search Q satırı gerekir. MSE hedefi `clip(Q,-1,1)`; bu kalibre WDL olasılığı değildir. Stockfish çağrısı veya yeni dış teacher hedefi yoktur. Cap ve açık episode sonları UNKNOWN; beraberlik etiketi yapılmaz.

VAL, UNKNOWN ve 152 kapalı known8 maç dosyasının mevcut board-placement/mirror eşdeğerlikleri korunur. Her gerçek statik-model girişinin alias'ı kaydedilir; korunan alias'a ulaşan episode bütün olarak eğitimden çıkarılır. Atılacak son search hamlesi oynanmaz. Kök havuzu yetmezse koşu FAIL olur; VAL ile doldurulmaz, 1024 azaltılmaz. Metadata-only yeniden kurma mevcut kaynak ve geçmiş SHA'larını doğruladı: 146/135 TRAIN trajectory, 31/40 VAL trajectory, 104064/104798 korunan alias. Hiç inference, Stockfish veya SGD çağrısı yapılmadı.

Model/feature/native matematiği korunur: 196625 float64 parametre, 16 hidden unit, sabit prior + NNUE residual. Own eğitim: 64 Adam update, batch256, lr0.001, betas0.9/0.999, eps1e-8, weight_decay0, gradient clip5, MSE. Sonuçtan sonra endpoint seçimi yapılmaz. Önce gerçek veri üzerinde whole8 / pause4 / yeni process'te resume8; tüm tensor/state/RNG bitleri eşit ve seed başına altı fresh strict-native load gerekir. Sonra öğretmen ağırlığından yeni optimizer/RNG ile ayrı fresh64 fit ve seed başına initial/final iki fresh strict load. Synthetic entegrasyon bu gerçek veri kanıtlarının yerine geçmez.

## Kaynak, süre ve kabul

cgroup4 CPU, 16GiB RAM; phase guard15GiB, disk floor256MiB. Seed worker'ları CPU1/3 ve tek thread. Veri/model çıktıları RAM'de; ham sidecar chunk en çok8MiB, collection toplamı en çok128MiB. Her aşamanın ilk gözlenen saatinden ayrı, değişmez süre: collection7200s, convert600s, resume proof600s, fresh fit1800s. PID/starttick ile yalnız kayıtlı process'leri durduran ayrı guardian kullanılır. Kaynak/admission/arithmetic değişirse yeni version ve yeni kanıt gerekir; eski checkpoint değişmez.

Üretilen tam-history/search/alias/source/parent kayıtları doğrulanır; altı sabit aralıklı gerçek search satırı bağımsız tekrar edilir. Fit sonrasında actual-trained C/Torch parity, 48 TRAIN pozisyonu ve 48 eşleşmiş 512-node arama ile hız kontrolü gerekir. Adayın same-parent ve E8'e göre wall-clock oranı en çok1.10.

## Güç kapısı

Önce açıkça geliştirme amaçlı known8 ekranı: beş kol ×16 renk/açılış eşli oyun ×iki seed =160 oyun. Kollar child–E8, child–parent, child–SF, parent–SF, E8–SF. Search512/q2/max8 bütün HarbiChess kollarında aynı; Stockfish19 nominal512 node, Threads1/Hash16/ClearHash, gerçek node overrun ayrıca raporlanır. Bu pozisyonlar geçmişte kullanıldı; bağımsız kanıt sayılmaz.

Her seed ayrı geçmelidir: child–E8 score>0.60 ve lower confidence bound>0.50; child SF score≥0.25; E8'e göre paired SF score artışı>0.10 ve LCB>0; same-parent direct score>0.60 ve LCB>0.50; parent'a göre paired SF artışı>0 ve LCB>0. İki seed'in birleştirilmesi başarısız seed'i kurtaramaz. Cap oranı≤0.05, hız oranı≤1.10. Eksik maçlar tamamlanmış sonuç yerine kullanılamaz.

Uygun aday donduktan sonra yalnız bir onaylı v2 formal kampanya: doğrulanmış aday eğitim soyundan ayrı, şimdilik seçilmemiş kökler. İki seed ×48 kök ×iki renk ×beş kol =960 oyun. Dört inferential contrast ×iki seed =sekiz LCB; joint alpha≤0.00625, her one-sided bound alpha0.00078125. Tarihsel toplam alpha bütçesi≤0.05625 olarak korunur. Eşikler yukarıdakiyle aynıdır; sonuçtan sonra gevşetilmez. Eski source6/source8 eksikleri ve öğretmenin loglanmamış iç yaprakları açıklanır; projenin bütün geçmişinde görülmemiş pozisyon iddiası yapılmaz. Formal book aday uygun bulunmadan seçilmez.

## Karar ve kalıcı kayıt

Teknik kabul veya güç ekranı FAIL olursa o yöntem başarısız olarak kaydedilir; nedenine göre ayrı hipotez/ablation kaydıyla başka yaklaşım denenir. Mevcut kaynak ve etiketsiz self-learning hedefi korunur. Learned dynamics/rules, planning veya yeni mekanizma ana güç hedefini geciktirmeden ancak ölçülebilir karşılaştırma varsa değerlendirilir; yayınlanabilir yenilik varsayılmaz. Önemli raw veri/native/optimizer/RNG/source/checkpointler yeni Release artifactleriyle ve bağımsız SHA readback ile korunur. Git ve Space yalnız doğrulanmış state'i bildirir; loss, synthetic test veya backup PASS güç başarısı sayılmaz.
