# UFUK-CPUFIX: CPU altyapısı geçti; öğrenme güç kontrolü başarısız

GPU erişimi kullanıcı tarafından kaldırıldı. CPU4 çekirdek/16GiB,
PyTorch2.14.1+cpu ile çalışıldı; SSH/Colab/GPU ve yeni ücretli kaynak kullanılmadı.
Main40 ve eski CPUformalINCOMPLETE aynen korunur. En güçlü bağımsız
doğrulanmış model hâlâ teacher-origin e8; bu sonuç self-learning başarısı değildir.

Kayıpsız feature cache’i512 gerçekTRAINpozisyonunda13,631,488B→614,912B
(%4.51) oldu; float32 bitleri aynı. Bu bütünprocessRSS veya hız ölçümü değildir.
Opt-in inactive-file-v1 yalnız uygun inactive dosya cache’ini çalışma-belleği
tahmininden çıkarır; physical ceiling, shmem/dirty/writeback ve OOM guard korunur.
Eski format/default legacy-total-v1 ve eski producer checkout’ları değişmedi.
Eski CPU memoryFAIL’in yalnızcache kaynaklı olduğu kanıtlanmadı.

Hedefli test28PASS/2CUDA-SKIP/24.18s; CUDA yok, iki mevcut CUDA testinin
çalışmadığı açıkça kaydedildi. AppleMetal ve eğitilmişCUDA yolu test edilmedi.
Ruff/compile/CLIhelp geçti. [Kaynak ön kaydı](UFUK-CPUFIX-registration-20261005.md)
ve [atomik source4cae085](https://github.com/Eta06/HarbiChess/commit/4cae08522ac940746cf9127ca8ce4a6b63a47408).
Gerçek CLIwhole2/pause1/freshresume2 kontrolü287.92s/600s:20artifact byte
eşit ve6native fresh-process strictload, iki koşuda18acceptedupdate.
[CLIreceipt](../../experiments/ufuk/cpu-fix-v1/actual-result/CLI-result.json).

## Kendi replay’inden kontrollü eğitim

[Ön kayıt](../../experiments/ufuk/cpu-own-replay-v1/protocol.json),
[çalıştırılan trainer](../../experiments/ufuk/cpu-own-replay-v1/train.py),
[gerçek8update full training restart proof](../../experiments/ufuk/cpu-own-replay-v1/actual-result/restart-proof.json).
İlk decoder-schema ve batch-field hataları önce başarısız receipt’lerle
kaydedildi, sonra düzeltildi. İki başarısızlık gizlenmeden aynı orijinal600s
içinde295.87s’de proof geçti: model/Adam/global+samplerRNG/replay contract
birebir. Bu yeni offline training-native-v1’dir; onlineactorresume iddiası yok.

İki seed kendi bağımsız denetlenmişE1 journal’ını kullanır. Dış teacher sorgusu
ve yeni etiketi yok; başlangıç e8teacherbootstrap, baseanchors bu ağdan gelir.
Eski replay yeniden kullanıldı; yeniunique selfplay toplandı denmedi.
Sabit80update value-only her iki seed’de80/80 retained: frozen rawpolicy/trunk
parametreleri birebir korundu. İç TRAIN-alias NLL2.980835→1.298584 ve
2.253086→0.871234. Bu optimal-play kalitesi veya bağımsız güç kanıtı değildir.
Jointkol her iki seed’de ilk update’in KL>.02 nedeniyle0/1 retained kaldı,
model/Adam/bütünRNG geri alındı. Freshprocess bütünparametrelerin e8 ile
aynılığını doğruladı. [Tam integrity ve provenance özeti](../../experiments/ufuk/cpu-own-replay-v1/actual-result/pilot-summary.json).

## Gerçek maç sonucu: FAIL

[Maç ön kaydı](../../experiments/ufuk/cpu-own-replay-v1/development-protocol.json)
ve [fullhistory bağımsız denetim](../../experiments/ufuk/cpu-own-replay-v1/actual-result/development-fullhistory-result.json).
Bilinen8geliştirme açılışı/16renkeşlemeli oyun/arm, sabit16sim/max4/G0/T1,
400ply, Stockfish512node/Threads1/Hash16MiB. BaşlangıçSF kontrolü aynı
açılışlar/renkler; tüm96oyun,7,495continuationhamlesi ve987,902gerçekSFnode
bağımsız olarak denetlendi. Eski sealedheldout sonuçları açılmadı.

| Seed | e8’e karşı W/D/L ve skor | finalSF W/D/L ve skor | e8SF başlangıç | SFgain |
|---|---|---|---|---|
|20262005|4/5/7, .40625|0/2/14, .0625|0/0/16,0|+.0625|
|20262006|8/4/4, .625|0/0/16,0|0/0/16,0|0|

Unknowncap0. Direct95%pairedbootstrap [.21875,.625]/[.46875,.78125];
SFgain küçük ve iki seed tutarlı değil. Her iki seed ön kayıtlı screenFAIL;
formal bağımsız strengthgate’e aday yok.8geliştirme ailesi genelElo vermez;
0skorda degeneratebootstrap gerçekpopülasyonolasılığı0 kanıtı değildir.
Joint exact0update-noop nedeniyle64duplicateE8oyunu yapılmadı:160plan→96
gerçekoyun bir futilitysapmasıdır, skip ile başarı yaratma değildir; jointFAIL.
Eğitim toplam160acceptedupdate ve2rejectedfirststep,4finalfullnative:
Adamstep/mask ve native-portableağırlık bütünlüğü freshprocess geçti.

## Devam

[UFUK-DENSITY ön kaydı](../../experiments/ufuk/cpu-search-density-v1/protocol.json):
sparse16/all16/all64 ile self-play’de search davranış oranı ve search hedef
bütçesini iki seed’de karşılaştır. Distributionmismatch hipotezi nedensel
teşhis değildir. Her hamlede search standart expertiteration fikridir,
özgünlük varsayılmaz. Mevcut eşikler korunur, önce gerçekall-searchCLI/resume
ve ölçülmüşdisk admission, sonra fixedendpoint ve eşlemeli geliştirme maçları.

Checkpoint/optimizer/RNG/replay yerelde korunuyor. Git küçük kanıtları içerir;
CPUmodel/Replay publicRelease yedeği hâlâPENDING. Eski128publicasset’in doğrulaması
ayrı tutulur. Backup yapılmış veya çalışma tamamlanmış denmez.
