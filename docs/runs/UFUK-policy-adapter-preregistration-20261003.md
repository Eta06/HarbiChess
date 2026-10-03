# UFUK: value sabitken kendi search hedeflerinden policy öğrenme

Kayıt: 2026-10-03, hiçbir adapter güncellemesinden önce. Kaynak başlangıcı
d0540e7; gerçek çalıştırma daha sonraki temiz, sabit commit ile kaydedilecek.

## Hipotez ve gerekçe

96 gerçek replay pozisyonunda depolanmış search hedefi, depolanmış raw policy'ye
göre ortalama +0.0378478 native Stockfish expected score kazandı; 48 aile
bootstrap %95 aralığı [0.0206269, 0.0593419]. Önceden yazılmış mekanizma
eşiği geçti. Bu bir oyun gücü ya da self-learning başarı kanıtı değildir.
Önceki head-only online öğrenme 5/34/9 ile başarısızdı. Aynı başlangıç
fonksiyonunu koruyan ilave policy temsilinin, sabit kendi search verisini
kontrole göre daha iyi öğrenip öğrenmediğini izole edeceğiz.

Orijinal stem/blocks, base value head'i de besler. Bunları eğitmek value'yu
sabit tutmaz. Yeni iki residual blok yalnız _pair_policy'nin yerel trunk
kopyasında çalışacak; value orijinal trunk'tan hesaplanacak. Conv2 sıfır
başlangıcı aynı fonksiyonu verir. Bu bilinen residual adapter mekanizmasıdır;
özgün buluş veya yayın başarısı olarak sunulmaz.

## Kontrol, veri ve eğitim

- Başlangıç: UFUK native bootstrap 18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae.
- Kontrol: 72,497 parametre; yalnız 16,914 pair_* parametresi öğrenir.
- Müdahale: 81,777 parametre; pair_* ve iki policy_adapter_blocks, toplam
  26,194 parametre öğrenir. Her iki kolda orijinal 55,583 parametre donar.
- Optional policy_adapter={schema:1,blocks:2} portable specification. Eski
  ağırlık/replay sürümleri korunur. Dönüşüm weights-only, optimizer reset;
  sonraki native checkpoint tam optimizer/RNG/sampler/step resume saklar.
- Veri: önceki UFUK 96 self-play oyunu, 14,885 kayıt, altı orijinal shard.
  Yeni engine etiketi, öğretmen sorgusu veya yeni self-play üretimi yok.
  Aile = game_index % 48; %4==0 olan 12 aile/24 oyun validation, diğer
  36 aile/72 oyun train. Tüm jenerasyonların train/validation shardları bu
  önceden belirlenmiş aile ayrımıyla yeniden ayrılır; eski shard split'i
  burada deney split'i sayılmaz. Aileler geçmiş native eğitimde görülmüştür.
- Game-balanced replacement sampling; seed 20261012; iki kol aynı sıra,
  batch 64, FP32 CPU1, AdamW LR 5e-5, weight decay 1e-4, clip 5.
  Terminal WDL maskesi korunur; value katmanları donmuş, value öğrenme yok.
- Maksimum 8,000 adım, evaluation her 250, validation policy minimumuna
  göre best; 1,500 adım sabırsızlık. Her kol setup dahil 1,800 s wall.
  Adım 250'de ayrı process tam checkpoint restore; sampler/optimizer/RNG
  uyumu ve frozen parametre/value değişmezliği doğrulanır.
- Native validation gözlemcisi: önceki seed 20261005 ile sabit 20,000
  pozisyon cache'i; training içinde kullanılmaz.

## Başarı ve durma

Yeni model yalnız kendi heldout target CE'si başlangıca göre >=0.05 ve
kontrolün best CE'sine göre >=0.03 iyileşir, native policy CE başlangıca
göre <=0.15 bozulur, frozen value aynı kalırsa arena adayı olur.
Başarısız/incomplete sonuçta arena/promotion/şanslı seed araması yapılmaz.
Gate geçerse ayrıca kaydedilmiş açılış/renk eşlemeli gerçek maç testi gerekir;
CE düşüşü tek başına oyun gücü ya da sürdürülebilir self-learning değildir.
Apple Metal/CUDA testi yok; gerçek MLX CPU forward/loss/gradient/update ve
portable dönüşüm testleri çalıştırılacak. Maksimum mevcut 4 CPU/16 GiB;
yeni ücretli kaynak yok. Nonfinite, kayıt/versiyon/hash veya value eşitliği
bozulursa dur ve son tam checkpoint'i koru. Tüm başarısız kollar saklanır.
