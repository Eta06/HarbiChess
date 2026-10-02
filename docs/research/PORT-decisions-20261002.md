# PORT: teknik kararlar ve sonraki araştırma soruları

2 Ekim 2026. Ana hedef güçlü, hızlı, güvenilir self-learning satranç modelidir.
Bu not Linux taşıma kararlarını açıklar; DENGE failed sonucunu veya yeni güç kazanımını varsaymaz.

## Bu turdaki kararlar

- Tahsis: Linux x86_64, AMD EPYC 9V74, cgroup CPU quota=4, memory.max=16 GiB;
  PyTorch `cuda.is_available()=False`, NVIDIA aygıtı yok. GPU kiralanmadı.
- PyTorch FP32 CPU portu; Linux extra'sı opt-in. CUDA cihazı varsa aynı port kullanılabilir,
  fakat CUDA doğrulaması bu turda yapılmadı. Apple MLX dependency ve public API korunur.
- Ortak `NetworkConfig`, `LearnerConfig`, encoder/action/replay/rules/search. Eski model
  çeşitlerinden base, invariant, decoupled MIHVER ve plastic DENGE üretim forward yolları taşındı.
  Spatial-policy/action-value araştırma başlıkları bu dönüştürücüde desteklenmez; strict key/shape
  reddi vardır, bunlar eski MLX hattında kalır. Eski dosyalar ve schema'lar değiştirilmez.
- NHWC flatten korunur, yalnız conv OHWI↔OIHW dönüşür. Aynı policy action index ve STM
  win/draw/loss logits; search value=P(win)-P(loss), backup her ply'da işaret değiştirir.
- Linux learner aynı legal-masked CE ve terminal WDL CE'yi kullanır; max-ply value_weight=0.
  PyTorch AdamW beta=(0.9,0.999), eps=1e-8, bias correction=True; mevcut MLX 0.32
  AdamW default bias_correction=False. Loss aynı olsa da optimizer update dinamiği eşdeğer değildir.
  Yeni CPU pilotu bias-corrected AdamW kullanır; bu kontrollü Linux başlangıcıdır, MLX
  eski training deneyi yeniden üretilmiş sayılmaz.
  Model conversion version=1 weights-only warm start'tır. MLX Adam state→Torch Adam state
  aktarımı yapılmaz. Frameworkler arası bitwise training devamlılığı iddia edilmez.
- PyTorch checkpoint version=1 model + AdamW + torch/CUDA RNG + sampler RNG + rolling replay
  hash'leri + cursor + config/runtime içerir. Aynı runtime koşulları korunmadan exact resume reddedilir.
  Tam run dizini taşınabilir; referanslar göreli ve run_id dizin adından bağımsız saklanır. Generation içi kesinti son tamamlanmış generation'a
  döner. Çok worker'lı batch zamanlaması bitwise self-play garantisi vermez; farklı yeniden üretilmiş
  partial replay üzerine yazılmaz, uyumsuzluk reddedilir. Testteki tam CLI exactlik tek worker/CPU'dadır.
- Tek snapshot üzerinde thread actors ve bir inference worker: fork edilmiş ML framework state'i,
  eşzamanlı learner/inference mutation ve Apple spawn farklarını bu yeni Linux yolunda önler.
  Eski MLX araştırma CLI'ları halen MLX gerektirir; Linux'un yeni giriş noktası torch-loop'tur.
- Frozen qualified MIHVER ağırlıklarıyla başla; DENGE başarısız son checkpoint'i başlangıç yapma.
  Plain rolling fresh search policy + observed terminal WDL, all-parameter AdamW. Her generation
  yeni actor snapshot'ı, optimizer devamlı, üç generation replay window, oyun dengeli sampler.
  Son oyun validation; bütün oyun train/validation ayrımı korunur. Aynı açılış dağılımından küçük
  held-out ölçüm genelleme kanıtı değildir; sonraki güç gate'i ayrı opening families gerektirir.
- Inference sırasında learner'ın dtype/trainability/mode'u değişmez. FP32 snapshot; eski arena'nın
  BF16 in-place dönüşüm riski bu Linux yoluna taşınmaz. Eski arena düzeltilmiş sayılmaz.

## Referans mekanizmalar → kontrollü karar

2 Ekim'de birincil bağlantılar yeniden okundu. İndirilen snapshot hash'leri run evidence'ta saklanır.

1. [AlphaZero](https://arxiv.org/abs/1712.01815): sürekli policy iteration, search policy ve final
   outcome ayrı supervision. Karar: failed arena latest learner'ı sıfırlamasın; publication/champion
   ayrı kalsın. Tarihsel 800 simulations/dev batch ölçeği dört CPU'ya kopyalanmaz.
2. [Mctx Full Gumbel](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py):
   root sequential halving, interior allocation, mixed-value completed Q ve tüm root hareketleri
   için action_weights. Karar: mevcut ortak Full Gumbel'i kullan; self-play'de Gumbel=1, değerlendirmede
   Gumbel=0. 16 simulations yalnız execution smoke; formal improvement accurate action values
   varsayımına bağlıdır. 64/128/256 simulation wall-time/strength ablation sonraki araştırmadır.
3. [Lc0 güncel trainer](https://github.com/LeelaChessZero/lczero-training/blob/master/docs/README.md):
   2025-11-30 tarihli yeni JAX/C++ pipeline dokümanı sliding chunk pool ve nonzero chunks_per_network
   ile incoming data/update oranını ayırıyor; checkpoint migration ayrı komut. Karar: rolling fresh
   veri + kalıcı optimizer, sampled update rows ve generated unique rows ayrı sayaçlar. JAX/C++
   yeniden yazımı dört CPU preflight için gerekli değil; daha sonra profiling kanıtıyla düşünülür.
4. [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md):
   global pooling, playout-cap randomization, exploration pruning, uncertainty weighting ve soft
   auxiliary policy veri/compute verimini hedefler. Karar: önce baseline döngü ölçülsün; bu teknikler
   ancak kontrollü ablation ile eklensin. Go ownership/score hedefleri satranç WDL yerine konmasın.
   MIHVER explicit counts + nonlinear global head korunur; daha fazla heuristic bu turda eklenmez.

Bir sonraki ölçüm: fixed wall-clock 16/64/128 simulations ve actor batch wait/worker count karşılaştırması,
terminal-observed veri oranı, held-out CE/calibration ve paired Stockfish/frozen baseline. Yüksek batch
throughput tek başına güç veya veri kalitesi gate'i değildir. Yanlış WDL hedefleriyle uzun koşu yapılmaz.

## Sonraya bırakılan sorular

Kullanıcının Laya referansı:
https://huggingface.co/blog/sora-2/laya-ai-model-how-it-works-run-it-locally-and-eva

Bu tam URL 2 Ekim'de HTTP 404 verdi. Model kimliği/özellikleri doğrulanamadı; başka model tahmin edilmedi.
Dış referanstır, HarbiChess karar modelinin adı değildir. Erişilebilir kaynak gelince küçük değerlendirme:
hamle/search'e katkı, legal actions, win-rate delta, latency ve toplam compute. Büyük ayrı proje başlatılmaz.

Kaliteli dataset mümkünse her kayıt position/history, legal action set, behavior/search policy,
WDL/result-known mask, model/data schema ve provenance, search budget ve opening/game family taşımalı.
Unknown caps terminal draw diye etiketlenmez; train/test split aile/oyun düzeyinde tutulur. Dataset
oluşturma bu smoke replay'ini otomatik kaliteli veri ilan etmek değildir.

Öğretmeni aşan öğrenci yalnız imitation ile garanti değildir. Yeni search policy iteration,
on-policy outcome/RL feedback, daha fazla/correct search, kapasite veya auxiliary supervision
mekanizmaları ayrı hipotez/ablation gerektirir; aynı örneklerde düşük CE yeterli güç kanıtı değildir.

LLM çalışması sonra gelir: legal move oranı (unconstrained ve constrained ayrı), gerçek paired chess
strength ve position evaluation calibration ayrı ölçülür; açıklama tutarlılığı satranç gücü yerine
geçmez. Aynı dataset'e erişen LLM ve chess student için opening-family leakage kontrolü zorunludur.
