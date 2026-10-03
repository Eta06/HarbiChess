# MERCEK: Laya, hamle seçimi ve özgün araştırma yolu

3 Ekim 2026. Hedef güçlü, hızlı ve self-play ile ilerleyen HarbiChess'tir.
Bu not doğrulanmış kaynakları, henüz sınanmamış hipotezleri ve sonraki kararları
ayırır. Özgünlük, öğretmeni aşma veya makale kabulü şu anda kanıtlanmış değildir.

## Laya bulundu: kullanıcının dış referansı

Verilen blog URL'si 2 ve 3 Ekim'de 404 döndü. Kullanıcının yeni “Jev alternative
Laya” ipucuyla [geliştiricinin model kartı](https://huggingface.co/convaiinnovations/laya)
ve [Jev uyumlu açık kaynak deposu](https://github.com/NandhaKishorM/laya) bulundu.
Kimlik artık yalnız bozuk blog slug'ından çıkarılmış bir tahmin değildir:
kart doğrudan TypeSafe Jev uyumlu HTTP sunucusunu ve karşılaştırmasını açıklıyor.
Laya dış modeldir; HarbiChess'in geliştireceği karar ağının adı değildir.

HF revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851; geliştirici GitHub commit'i
fa9a2a7070b1789912a49ae24603bbfb1a78b001. Model kartı, inference/model kodu,
config ve fine-tuning notebook'u okundu. Laya ağırlıkları indirilmedi, çalıştırılmadı
ve satranç gücü/latency'si yerel olarak test edilmedi.
[Kaynak envanteri](MERCEK-sources-20261003.json) URL, erişim hatası ve dosya hash'lerini tutar.

### Mekanizma

ModernBERT-large encoder + iki transformer head katmanı + seçenek başına skor:
her seçenek bir MASK marker'ında temsil ediliyor, skorlar o sorunun seçenekleri
üzerinde softmax oluyor. Ayrıca act/escalate head var. Hazır İngilizce model
kartına göre toplam 421M parametre; multilingual sürüm 322M. HarbiChess'in bu
tanıdaki MIHVER ağı 1.256.355 parametre. Laya metin üreten bir autoregressive LLM
değil; metin/state ve istenen seçenek şemasını okuyan karar sınıflandırıcısı.

RLCD öğretici dağılıma karşı log + spherical scoring, ordinal score sorularında
ek ranked probability score kullanıyor. Notebook'ta noisy-logit dağılımlarından
GRPO benzeri grup baseline'ı ile REINFORCE ve ayrıca doğrudan soft-target CE var.
Kaynak kodunda log score floor'u ve post-hoc temperature scaling mevcut.
“Proper scoring rule” ideali, doğru hedef dağılımı/population optimum koşullarında
olasılığı dürüst raporlamayı teşvik eder. Sonlu veri, optimizer, clipping,
yanlış teacher ve yeni satranç dağılımı altında kalibrasyon garantisi değildir.

Kartın kendi somut sınırlamaları:

- CPU latency 193–464 ms, T4 üzerinde tek soru 32,8–39,5 ms **yazar ölçümü**;
  bizim CPU/Metal ölçümümüz değildir. Bunlar HarbiChess ile eşit iş/eşit donanım
  benchmark'ı sayılmaz.
- İngilizce model bazı non-Latin görevlerde yüksek güvenle yanlış cevap veriyor.
- [Typed-decisions kartı](https://huggingface.co/convaiinnovations/laya-typed-decisions)
  ECE=0,213, bazı sıcaklıkların training verisinde fit edildiğini ve bu checkpoint
  yeniden kalibre edilene kadar confidence'a güvenilmemesini açıkça bildiriyor.
- Typed choice için yaklaşık 20 seçeneğin altında kalma önerisi var; sınırlı
  token/head bütçesi geniş label uzayını zayıflatıyor. Satrançta 20'den fazla
  legal hamle olağandır. Top-k filtre çözümü iyi hamleyi dışarıda bırakabilir;
  bunu ayrıca ölçmek gerekir.
- Jev typed benchmark rakamları bu projede API ile yeniden ölçülmemiş;
  üçüncü taraf yayımlanmış ölçümlerde sample/prompt farkları var. “Teacher
  ceiling'i geçti” ifadesi satrançta teacher engine'i aşma kanıtı değildir.

### HarbiChess kararı

Fikir araştırmaya değer. Hazır 421M modeli ana search'in her node'unda veya her
hamlede kullanmak şu anda gerekçelendirilmiş değil. Öncelik küçük, satranç state'i
ve history'sini doğrudan kullanan politika/değerlendirme ağıdır. Laya'nın
dinamik seçenek skorlaması ve maliyetli kararı erteleme yaklaşımı, küçük bir
“16 mı 64/128 mi; durmalı mıyım?” controller için referans olabilir.

İlk controller seçenekleri birkaç compute kararı olur; bütün legal hamleleri
dar bir metin head'ine sıkıştırmayız. Girdi: mevcut ağın features'ı, legal policy
entropy/gap, search/value disagreement, legal move count, kalan bütçe ve history
özetleri. “Güvenim yüksek” ile “hamlem iyi” aynı olay değildir.
Controller öğreticisi: daha derin bağımsız referansa göre ek search'ün marjinal
hamle kalitesi kazancı ve ölçülmüş wall maliyeti. Başlangıç kontrolü ucuz
entropy/gap kuralı, sabit budget ve KataGo benzeri cap randomization olmalıdır.
Controller overhead'i dahil eşit toplam süreyle kazanım aranır.

## “Token yerine hamle seçtirmek” düşüncesi

Temel matematik doğru: state'ten seçenek logitleri, legal softmax ve kaliteli
hedefe CE ile bir hamle politikası öğrenebiliriz. Mevcut HarbiChess policy head'i
zaten 4.672 sabit action'ın legal altkümesinde bunu yapıyor. Değişken seçenekli
shared action scorer da bir adaydır; metin backbone'u veya dil üretimi şart değil.

Hamleler keyfi sıralı sınıflardır: Laya'nın ordinal RPS hedefini UCI/action index
sırasına uygulamak anlamsız bir uzaklık icat eder. Gerçek kalite için karşı
oyuncunun cevapları, uzun vadeli sonuç, tekrar/rok/en-passant ve value perspektifi
gereklidir. Tek hamle imitation kaybının azalması oyun gücü artışı değildir.
Tam teacher dağılımı eldeyse önce doğrudan CE/Brier gibi differentiable hedefler
kontrol olur; ek noisy-logit REINFORCE maliyeti ve varyansı ayrı ablation gerektirir.

## Okunan çalışmalar ve teknik sonuçları

Aşağıdaki sonuçlar yazarların raporudur; HarbiChess üzerinde yeniden üretim değildir.
AlphaZero/Mctx/Lc0/KataGo kararları 2 Ekim PORT kaynak incelemesini kullanır;
yeni Laya, ChessBench ve dil modeli kaynakları 3 Ekim snapshot'larında tutulur.

| Birincil kaynak | İlgili bulgu | HarbiChess kararı |
|---|---|---|
| [AlphaZero](https://arxiv.org/abs/1712.01815) | Search policy ile gerçekleşmiş outcome farklı supervision; iteratif self-play | Fresh politika döngüsü, ayrı outcome-known mask, kalıcı learner; champion ayrı |
| [Full Gumbel/Mctx](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py) | Completed Q/sequential halving; improvement doğru action value varsayımına bağlı | Yanlış value'ya daha fazla simulation eklemek iyileşme sayılmaz; frozen budget tanısı |
| [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) | Playout cap randomization ve auxiliary supervision ile compute/data verimi | Sabit toplam compute altında kalite/miktar kontrolü; Go hedefleri satranca kör aktarılmaz |
| [Lc0 trainer](https://github.com/LeelaChessZero/lczero-training/blob/master/docs/README.md) | Sliding pool ve data/update oranı ayrı; optimizer migration ayrı | Generated unique rows, sampled rows ve learner step ayrı sayaç |
| [Amortized Planning / ChessBench, NeurIPS 2024](https://arxiv.org/abs/2402.04494) | 270M transformer, 10M oyun, yaklaşık 15B Stockfish action-value etiketi; searchless güçlü oyun | Kaliteli search bilgisi ağa sıkıştırılabilir; 1,1 TB/çok büyük training ölçeği CPU tahsisimize kopyalanmaz |
| [Decision Transformer, 2021](https://arxiv.org/abs/2106.01345) | Return/state/action koşullu offline sequence modeling | Laya ile aynı model değildir; güçlü data olmadan return conditioning başarı sağlamış sayılmaz |
| [Grounded Chess Reasoning / C1, 2026 preprint](https://arxiv.org/abs/2603.20510) | SFT+Stockfish-verified RLVR, 900-puzzle testte 4B model 48,1%, Gemini teacher 40,8%; 4×H100 | Ek güçlü verification ile dil teacher'ını geçmek mümkün; Stockfish'ten güçlü tam-oyun motor sonucu değildir |
| [Reasoning Through Chess, 2026 preprint](https://arxiv.org/abs/2604.05134) | Best-move SFT+RL kalite artırırken reasoning unfaithful olabilir; multi-move data daha sadık/stabil | Tek-hamle/multi-ply hedefleri ayrı; legality, move-quality, açıklama sadakati farklı ölçüm |
| [ChessGPT, 2023](https://arxiv.org/abs/2306.09200) | Oyun policy kayıtları ile dil açıklamalarını birleştirme | LLM verisi için arama kaydının yanında doğrulanabilir açıklama/variation gerekir |
| [Knapsack RL, 2025](https://arxiv.org/abs/2509.25849) | Exploration bütçesini örneklere adaptif ayırma | “Öğrenilmiş compute allocation” tek başına özgün fikir diye sunulamaz |
| [Stockfish NNUE trainer](https://github.com/official-stockfish/nnue-pytorch) | Sparse data loader, native Apple MPS training yolu ve netleri gerçek oyunlarla karşılaştırma | CPU için incremental evaluator araştırma adayı; trainer Docker kurulumu 30–60 GB diye aynen yüklenmez |

ChessBench'in 2895 Lichess blitz sonucu insan havuzuna ait; eşit zamanlı motor
Elo'su değildir. Paper açıkça hızının compute-limited turnuvalarda pratik olmadığını,
Lc0 searchless ağlarının daha verimli/güçlü olduğunu ve teacher açığını kapatamadığını
söylüyor. FEN-only model repetition geçmişini göremiyor, bazı kazanılmış pozisyonlarda
Stockfish top-move fallback var. HarbiChess history-aware kurallarını korur.

Gumbel OpenReview sayfası erişildi fakat PDF HTTP 403 verdi; o makalenin tam PDF'sini
bu turda okudum denmez. Uygulama/mechanism incelemesi Mctx ve tarihsel ESAS audit'ine
dayanır. HF connector paper/dataset search UNAVAILABLE verdi; public HTTP API ve
paper markdown fallback başarılı oldu. Kaynak hataları envanterde korunur.

## Dataset ve öğretmeni aşma

Kaliteli veri seti üretilebilir. PORT'un 3.089 pozisyonu çalışırlık kanıtıdır;
value holdout kötüleşmesi ve zayıf Stockfish sonucu bu replay'e kalite sertifikası
vermez. Önce küçük, doğrulanmış bir development dataset'i ve bağımsız test gerekir.

Gerekli kayıtlar: başlangıç FEN + bütün oyun history'si, game/opening family id,
legal action set, raw ve search policy, actor checkpoint hash'i, search budget/
wall, gerçekleşmiş WDL ve known mask, teacher kaynak/bütçe/uncertainty, schema ve
data split. State/action Q ve WDL taraf-to-move perspektifi açık olur.
Stockfish WDL tahmini ve gerçekleşmiş final WDL ayrı alan/hedef olur; cap sahte
draw sayılmaz. Aynı position/history veya opening family train/test'e sızmaz;
seed değişimi bağımsız test oluşturmaz. Faz/taktik/endgame coverage ayrıca raporlanır.

Student yalnız aynı teacher'ın hamlesini taklit ederse teacher'ı geçmesi otomatik
değildir. Kayda değer yollar: teacher ağından güçlü/deeper search'ü distill etmek,
fresh self-play/search policy iteration ile yeni outcome geri bildirimi,
ayrı doğrulayıcı/teacher ensemble veya daha iyi action-value supervision.
Student raw ağ, student+search ve veriyi üreten teacher+search aynı compute
bütçelerinde ayrı karşılaştırılır. Ağdan güçlü olmak teacher sisteminden güçlü
olmakla karıştırılmaz. C1'in dil teacher'ını geçmesi ek Stockfish verification/RL
bağlamında okunur; imitation'ın genel bir güç garantisi değildir.

Aynı kayıtlar ileride LLM SFT/RLVR için kullanılabilir: pozisyon + legal hamleler,
tercih edilen hamle + doğrulanmış principal variation, sonuç/evaluation kaynakları.
Arama ağacı doğal dil açıklaması değildir; üretilecek açıklamalar ayrı doğrulanır.
Unconstrained legal oranı, constrained çıktı, gerçek eşleştirilmiş oyun gücü,
value kalibrasyonu ve açıklama sadakati ayrı ölçülür. Bu tur LLM veya Laya fine-tuning'i
başlatılmadı; mevcut CPU'da 4B/7B post-training için pratik hız iddiası yok.

Kullanıcının bu turdaki açıklaması: Laya'yı HarbiChess'e katmak şart değildir.
İleride ayrı geliştirilecek karar modeli için HarbiChess'in kaliteli state/options/
choice/feedback kayıtları bir veri kaynağı olabilir. Bu ayrı model ile motorun
içindeki olası küçük search-budget controller aynı proje/rol sayılmaz.
Satranç verisiyle uzman chess-choice modeli öğrenmek ile genel karar yeteneği
kazanmak ayrı iddialardır; diğer karar alanlarında transfer, seçenek sırasına
dayanıklılık, kalibrasyon ve maliyet bağımsız heldout'larla ölçülmelidir.
Bu ihtimali kayıt şemasında koruruz; şu anda ayrı büyük karar modeli projesi açmayız.

Laya kod/model kartı Apache-2.0 bildiriyor. ChessBench software Apache-2.0,
weights CC-BY-4.0; dataset'in bir kısmı Lichess CC0, kalanı CC-BY-4.0.
Lichess standart oyunları CC0; broadcast verisi CC-BY-SA-4.0. Dış veri/weights
kullanılırsa doğru lisans/attribution, oracle etiketi ve split provenance tutulur.

## Sonraki ana çalışma ve makale eşiği

Araştırma mevcut CNN/Full Gumbel hattına kilitli değildir. Kontrollü aday havuzu:

| Alan | Adaylar | İlk eleme ölçütü |
|---|---|---|
| Temsil/ağ | Küçük CNN; chess-aware transformer; history-aware action scorer; CPU için incremental NNUE benzeri evaluator | Aynı veri/compute ve legal/rule kapsamıyla kalite, batch-1 latency, memory |
| Search | PUCT, Full Gumbel; uygun evaluator ile alpha-beta/hybrid; tree reuse/caching | Aynı toplam zaman ve açılış/renk bütçesinde güç; yanlış value/tactical error |
| Öğrenme | Policy+observed WDL, action-value distillation, auxiliary targets, SFT→fresh self-play | Ayrı oyunlarda calibration/strength ve kalıcı optimizer ilerlemesi |
| Veri | On-policy self-play, daha güçlü teacher ile reanalysis, phase/tactic-balanced sampling | Unique/known rows, reference stability, contamination ve üretim maliyeti |
| Compute | Process actors, ortak batching, encoding/rules profili, adaptive caps/controller | Gerçek oyun/veri throughput ve güç-zaman Pareto; controller overhead dahil |

Bu tablo tamamlanmış denemeler veya hepsinin faydalı olacağı iddiası değildir.
Yakın literatür ve küçük pilotlarla seçenekleri daraltırız. CPU'ya uygun evaluator/
search değişimi veya daha yalın mimari gerekirse yapılabilir; Apple MLX hattı,
version'lı eski checkpoint erişimi ve satranç semantiği korunarak doğrulanır.

[Ön kayıtlı 48-oyun search tanısı](../runs/MERCEK-search-result-20261003.md):
128 simulation, 16'ya göre 9,38× arena zamanı kullanıp aynı 0,125 score aldı.
Primary paired fark 0; küçük suite'in belirsizliği geniş. Bu evidence
“search asla işe yaramaz” demez; daha pahalı search'ü otomatik kaliteli teacher
ilan etmemeyi destekler. Yeni long training/promotion yapılmadı.

1. Frozen pozisyon panelinde teacher kalite tanısı: oyun/aile düzeyinde ayrım,
   taktik/endgame/phase coverage, daha güçlü sabit Stockfish referansına göre
   hamle regret/rank stability, raw/search policy farkı ve WDL calibration.
   Önce referansın budget artışında stabil olup olmadığı ölçülür. Bu, yeni eğitim
   verisi ve nihai bağımsız arena'dan ayrı development panelidir.
2. Öğrenme tanısı: aynı başlangıçtan fresh policy+known terminal WDL; LR ve
   generated/sample ratio az sayıda ön kayıtlı arm ile sınanır. Aynı frozen
   büyük whole-game holdout'ta policy, WDL CE/Brier ve shared-trunk value drift'i
   ölçülür. Sadece value head'ini dondurmak shared trunk drift'ini engellemez.
   Eski DENGE trust/gate mekanizmaları zorunlu yol haritası sayılmaz.
3. Profiling'e göre hız: encoder/rules/search/GIL ve batch bekleme wall süreleri;
   gerekirse güvenli spawn process actors, cache veya farklı temsil. Aynı sonuç
   ve toplam süre altında games/terminal-known rows per second ölçülür; sadece
   forward batch throughput/utilization optimize edilmiş sayılmaz.
4. Baseline gerçekten öğrenince küçük budget/selection controller; sabit search,
   cheap gap/entropy, random caps ve controller-siz aynı ağ kontrolleri. Compute
   overhead'i dahil eşit toplam wall/oyun saatinde güç ve veri verimi ablation'ı.

Özgün katkı adayı: history/value reliability ile ek search'ün ölçülmüş faydasını
birleştirerek compute'u dağıtan ve self-play data kalitesini koruyan küçük mekanizma.
Bu bir **hipotez**; adaptif search/active learning/metareasoning literatürü
geniş olduğundan “ilk biz yaptık” iddiası yok. Standart AlphaZero/Gumbel
mekanizmaları kaynak gösterilerek kullanılabilir. Yeni isim veya katman tek başına
özgünlük sağlamaz. Çözüm gerektiğinde mimariyi sadeleştirebilir.

Makale için: yakın çalışmalardan açık fark, güçlü/ucuz baseline'lar, component
ablations, birden çok training seed'i, bağımsız opening families, renk eşlemesi,
önceden belirlenmiş test/gate, güç-zaman Pareto, güven aralıkları ve tüm başarısız
arm'ların kayıtları gerekir. Validation üzerinde seçilen arm, aynı set üstünde
final sonuç diye sunulmaz. Hayali Elo/garanti/sonuç veya geçmiş başarısızlık silme yok.

Uygulanabilir deney hattına güven yüksek; Laya'nın hazır chess faydası, controller'ın
güç kazancı, teacher'ı aşma, dünya çapında güç ve yayın özgünlüğü henüz açık sorular.
Çalışma kontrollü turlarla ilerler; bu not sürekli çalışan background job değildir.
