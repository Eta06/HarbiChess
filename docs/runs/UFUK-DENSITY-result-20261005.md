# UFUK-DENSITY: üç yöntem de strength screen FAIL

GPU erişimi kaldırılmıştır; bu aşama yalnız CPU4core/16GiB üzerinde çalıştı. Sonuçlar başarısızdır, self-learning strength kazanımı veya model promotion yoktur. Başlangıç teacher-origin e8 en güçlü bağımsız doğrulanmış model olarak kalır.

## Sabit protokol ve gerçek compute

[Ön kayıt](../../experiments/ufuk/cpu-search-density-v1/protocol.json), [maç protokolü](../../experiments/ufuk/cpu-search-density-v1/arena-protocol.json). Kaynak `4cae08522ac940746cf9127ca8ce4a6b63a47408`; model, book, journal, optimizer/RNG ve helper SHA değerleri [training receipt](../../experiments/ufuk/cpu-search-density-v1/actual-result/training-summary.json) ve altı fresh audit kaydında bulunur. Teacher-origin e8 bootstrap/frozen anchor; yeni Stockfish training etiketi yoktur.

Her sparse16/all16/all64 kolu iki seed, üç epoch ve 6.144 yeni kendi hamlesi: toplam36.864. Sparse16/16sim her8hamledenbirinde; all16 ve all64 herhamlede search. Final değerlendirme hepsinde16sim/max4/G0/T1. Density kolu aynı pass sayısında daha çok update üretir; compute eşlemeli saf nedensellik iddiası yoktur.

Sparse16 toplam64/64, all16 128/224, all64 96/192 accepted/attempted optimizer adımı; toplam288/480. Herseed sparse16~96s, all16~312–328s, all64~961–1004s: uzun training search bu deneyde strength sağlamadı. Gerçek all64 CLIwhole2/pause1/freshresume2 PASS378.75s/600s;20payload+journalbyteeşit,6strictnativeaçılışı. Full native0 ve3; eski native/deney saatleri sıfırlanmadı.

Altı fresh-process veri/native denetimi PASS: model/Adam/RNG/actor/replay bağları, üç journal zinciri, bütün hamle/terminal/mover-WDL/UNKNOWN kayıtları, schedule/actual-mu/certificates. Bu shared-validator denetimi bağımsız chronological NN-search rerun değildir; o kapsam açık kalır ve formal confirmation öncesi gereklidir. CUDA/Apple donanımı denenmedi.

## Tamamlanan oyunların bağımsız kurallar denetimi

[Bağımsız fullhistory sonuç](../../experiments/ufuk/cpu-search-density-v1/actual-result/development-fullhistory-result.json):14turnuva,224oyun,16.598legal continuationhamlesi,1.908.510gerçekSFnodes. Tüm oyunlar başlangıçtan bağımsız python-chess ile tekrarlandı; kurallar/terminal/mover score/açılış-renk/bütçe/WDL/hamle zamanı ve artifact SHA denetimleri PASS. Evaluation-only SF512/Threads1/Hash16MiB; tümcaps0.

|Kol|Seed|e8 W/D/L|Directscore|SF W/D/L|FinalSFscore / pairedgain|
|---|---|---|---|---|---|
|sparse16|20262105|6/3/7|0.46875|0/0/16|0.0 / +0.0|
|sparse16|20262106|3/4/9|0.3125|0/3/13|0.09375 / +0.09375|
|all16|20262105|8/2/6|0.5625|0/2/14|0.0625 / +0.0625|
|all16|20262106|8/5/3|0.65625|0/1/15|0.03125 / +0.03125|
|all64|20262105|3/8/5|0.4375|0/2/14|0.0625 / +0.0625|
|all64|20262106|9/2/5|0.625|0/0/16|0.0 / +0.0|

İki yeni matchseed için e8SF kontrolü 0/0/16 ve0skor. Sabit iki-seed screen: direct>.60, pairedSFgain>.10, finalSF≥.25, cap≤.05. Üçkolunhepsi FAIL; yalnız bir seedde.65625 veya.625 direct skor görev başarısı değildir. Formal güç/istatistik/latency gate yapılmış/geçilmiş değildir. Bu8bilinen geliştirme ailesi virginheldout veya genelElo değildir; seedler aynırootları kullanır;95%pair bootstrap/Hoeffding yalnız betimseldir,0ampirikskor populationprobability0 kanıtı değildir.

## Sonuç ve başka hipotez

Sadece density ve search bütçesini artırmak burada yeterli olmadı. ClosedE1 all16 actual hamlesi deterministic selected hamleyle~61%eşleşiyordu. Bu ölçülmüş davranış farkı terminal-value hedefini finalselected politikasının optimal/minimax değeri saymamayı gerektirir; strength kaybının kanıtlanmış nedeni değildir. Sonraki kontrollü hipotez evaluation selection’a daha yakın, fullsupport greedy-mixture acting; raw network pi, search supervision ve gerçek acting mu ayrı kayıtlanmalı, yeni ledger sürümü/strict replay ve aynıstrength eşikleri gerekir. Genel policy-value alignment/expert-iteration literatürde var; özgünlük iddiası yok.

Eski Main40FAIL, CPUformalINCOMPLETE ve kapalıMAX8 aile kayıtları korunur. Altı yeni fullnative0/3 ve hamreplay/gameartifactler yerelde korunmuştur. Public GitHubReleasebackup PENDING; eski128verifiedasset ayrı. Git/Space küçükkanıtlar senkronize edilir, checkpointler uzaktaymış gibi sunulmaz.
