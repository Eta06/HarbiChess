# UFUK-FRESH: yeni veri, native denetimleri ve sabit eğitim

İki gerçek CPU self-play actor8192'şer hamleyi tamamladı. Yeni veri bağımsız full-history/terminal/RNG/protected-game/dedup/split denetiminden geçti. Bu, strength başarısı değildir: MC, own-search-consistency (SC) ve geniş mevcut value head (FULL) aynı yeni veride, sabit son update'te karşılaştırılacak. Henüz yeni adayın bağımsız oyun gücü artışı kanıtlanmış değil.

Actor source `6fcc8b476d25495d1c9c413e55b2c7ba4794013e`, seed20262805/06, frozen teacher-origin E8+Q512/q2/max8, behavior .95selected+.05uniformlegal, ordinary START. Yeni teacher sorgusu yok; başlangıç E8 ve KL anchor teacher kökenlidir. Own terminal WDL ve kayıtlı own root-search Q ayrı tutulur; exploration action ile selected search action aynı sayılmaz.400 toplam ply UNKNOWN sınırı korunur.

Actor ortak ilk1791221180.0397933, dört saatlik son1791235580.0397933; final8192 segmentleri05=1791225500.368684,06=1791225487.464582'de tamamlandı. Eski timer'lar sıfırlanmadı. Segment native'leri actor RNG, bitmemiş history, raw anchors, exploration ve arama kayıtlarını içerir. Raw journal RAM'dedir; yeni public final8192 persistence henüz tamamlandı diye gösterilmez.

| Seed | Raw known/tail | Duplicate rows/groups | Eligible rows/groups | Train/validation rows | Sabit updates |
|---|---:|---:|---:|---:|---:|
|20262805|8141/51|320/4|7821/85|6771/1050|105|
|20262806|8188/4|640/8|7548/74|6247/1301|97|

Protected sekiz canonical FEN4'e değen bütün oyunlar dışlanır; bu final cohort'ta böyle oyun0. UNKNOWN cap0; bitmemiş51/4 satır hiçbir terimde draw yapılmadı. Realized trajectory SHA ile exact dedup ve son SHA byte mod5 bucket0 validation. Internal generated-game split, root/state/source-family independence değildir; her actor aynı START ailesinden gelir.

Bağımsız73.114s denetim, MC/SC dataset, sampler groups, labels, anchors ve FULL104-plane/sparse840/search target row eşleşmesini doğruladı. NN forward veya SGD yapmadı; raw anchor/Q'yu neural olarak tekrar üretmiş sayılmaz. Readiness SHA `90162446656105007ca713c17b01f4e0a762d694a179a5be815e0ba6174bae51`.

MC: sıfır additive2520 etkin residual, CE+beta1 E8 KL+aynı shrinkage. SC: aynı residual/data/split/sampler/optimizer; .5CE+.5 own root-Q scalar MSE+beta1 KL+aynı shrinkage. FULL: mevcut E8 value-only31290 parametre; aynı .5CE+.5MSE+beta1KL, shared trunk/policy bitwise frozen; sparse regularizer yok, bu saf capacity ablation değildir. AdamW2e-5, decay0, batch256/clip5 ve toplam4*N_train presentation slots, `min(1024,floor(4*N_train/256))`; qualification8 update üretim fit'ine taşınmaz. Son105/97 checkpoint önceden sabittir; validation'a göre endpoint seçilmez.

Gerçek MC2048 whole8/pause4/freshresume8 ve tüm6 strict load/seed149.081s/orijinal600'de geçti. Gerçek SC2048 aynı native kontrolü iki seed'de geçti; orijinal ortak ilk1791224926.6862378, son1791225526.6862378. FULL'de üç başarısızlık korunuyor: ilk flat2MiB protocol cap eksikliği; V2'de frozen hash'in parametre sırası farklı olduğundan false final rejection (her update bitwise snapshot PASS); V3'te JSON list/native contract tuple names eşitsizliği, resume öncesi rejection. Kaynak ve log'lar saklandı. V4 names list contract ve canonical hash ordering ile V3'ün1791225858.5434084→1791226458.5434084 clock'unda iki seed whole8/pause4/freshresume8 ve6 strict load/seed geçti; tüm cohort364.277s. Hiçbir strength gate veya kaynak cap gevşetilmedi. Eski v1/v2 clock expired kalıyor.

V4 full native model value tensors/Adam/global+sampler RNG/input contract storage eşleşmesi doğrulandı; offline native resume, actor resume değildir. Her native/candidate FULL2MiB, MC/SC512KiB sınırı, cgroup15GiB/OOM/shmem/dirty/writeback/physical reserve ve disk256MiB sınırı korunur. Yeni değişmiş fullcritic CLI testleri8PASS/0skip ve Ruff; gerçek native proof bunların ötesinde gerçekleşti. Apple/GPU testi yok.

Production MC/SC ayrı yeni ortak1800s bütçe1791226264.0797942→1791228064.0797942, FULL ayrı1800s1791226381.4458697→1791228181.4458697. Her fit'ten sonra checkpoint0/final fresh-process strict native opens zorunlu. Yalnız kayıtlı son aday; search/e8/SF controls, unchanged strength thresholds ve bağımsız confirmation daha sonra. Eski MAX8 kapalı; yeni nominal confidence geçmiş testlerden bağımsızmış gibi yeniden başlatılamaz.

Kaynaklar ve tam makine kayıtları ilgili `cpu-fresh-learning-v1`, `cpu-fresh-sc-v1`, `cpu-fresh-fullcritic-v1/actual-result/` altında. Yeni modeller public artifact readback olmadan kalıcı yedeklendi sayılmayacak.
