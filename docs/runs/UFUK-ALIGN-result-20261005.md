# UFUK-ALIGN: davranış uyumu gerçekleşti, strength screen FAIL

GPU kullanımı yoktur. İki seed fixed3epoch, herseed6144yeni kendi selfplayhamlesi, toplam12288. İkisinde64/112 accepted/attemptedupdate. Teacher-origin e8 weights-only bootstrap/yeniAdam ve frozenanchors; yeni teachertrainingetiketi yok. Model promotion veya kanıtlanmış güç kazanımı yok.

[Ön kayıt](../../experiments/ufuk/cpu-aligned-acting-v1/protocol.json), [CLI proof](../../experiments/ufuk/cpu-aligned-acting-v1/actual-result/CLI-result.json), [ham dataset/model/native/optimizer/RNG provenance](../../experiments/ufuk/cpu-aligned-acting-v1/actual-result/training-summary.json). Core source `1e26d663b9d748e2c6afed5b74914a9dd88a21fe`; ayrı helper/protocol/hash bağları kayıtlıdır. GerçekCLI whole2/pause1/freshresume2 PASS285.46s/600s;20artifact byteeşit,6fullnativestrictload;512unique+512duplicate qualificationtransitions.17CPUunitPASS/2existingCUDA-SKIP (16testönce +1zero-mixturetest); CUDA/Applehardwareuntested. Sıfır karışımın legacycollection/training/modelbitleri aynı.

Gerçekmu=.9selectedpointmass+.1shieldedsearchpolicy; rawpi ve CEsearchtarget ayrı. Yeni v5ledger, aynı model/action/encoder/portableformats. Eski sixfieldconfig/defaultv4 değiştirilmedi; eski native producer4cae ile fullresume, weights-only fullresume değildir.

## Denetim hatası ve versioned onarım

İlk audithelper v1 kopyalanan v4schemaassert nedeniyle heriki yeni v5ledgeri reddetti. [İlk hata/saatler](../../experiments/ufuk/cpu-aligned-acting-v1/actual-result/audit-v1-failure.json) korunur. audit_v2 gerçekv5şema gerektirir; mu, kurallar, certificates, bütünnative/Adam/RNG ve3journalzincirinin denetimini kaldırmadı. İkifresh-process v2audit orijinal900sdenetimdeadline içindePASS; eğitim tekrar başlatılmadı, oyunlar ancakdenetimden sonra çalıştı. Sharedvalidator bağımsızchronologicalNNsearchrerun değildir; o formalconfirmationöncesi açık kapsamdır.

Searchselected/actual eşleşme tüm3epochs: seed05 5910/6144=.961914; seed06 5924/6144=.964193. Önceki closedall16E1~.607/.615 idi. Hedeflenen acting alignment değişti; bu tekbaşına strength değildir. Aynıbook/seed/6kmoves/3epoch/all16kontrolü eski auditedendpoints olarak tekrar kullanıldı. Yeni kontrolselfplay üretilmedi, duplicateddata yeni kayıt veya blindedconfirmation sayılmadı.

## 96 tamamlanan gerçek maç

[Tam bağımsız kurallar denetimi](../../experiments/ufuk/cpu-aligned-acting-v1/actual-result/development-fullhistory-result.json):96/96oyun,7241legalcontinuationplies,909972gerçekSFnodes, bütüncaps0. Aynı8bilinenroot/colorpairs,S16/max4/G0/T1/400ply;SF512/1thread/Hash16. FreshE8SF controls0/0.

|Seed|greedy90/e8 W/D/L ve score|greedy90/SF W/D/L ve score / gain|
|---|---|---|
|20262105|5/6/5, 0.5|0/0/16, 0.0 / +0.0|
|20262106|7/5/4, 0.59375|0/2/14, 0.0625 / +0.0625|

İki seed de değişmez direct>.60/SFgain>.10/finalSF≥.25/cap≤.05 screenFAIL. Formalistatistik/latency gate geçilmedi.8bilinenfamily virginheldout/genelElo değildir; betimsel95%pairbootstrap/Hoeffdingpopulationzero kanıtı vermez. %96alignment bu setting’de tekbaşına yeterli olmadı; genelalignmentmekanizması etkisiz veya tümbaşarısızlığınnedeniydi iddiası yapılmaz.

## Başka öğrenme hipotezi

Sonraki aday: eski immutableMain40own-outcome replay’den sade20invariantWDLbaşlığı öğrenmek, policy/trunk sabit ve kötüeski değerlogitlerinin yenidenbaşlatılması explicit. Öğrenmeden gelen etkiyi ayırmak için aynı yenidenbaşlatılmış/eğitimsiz değer başlığı kontrolü gerekir; yalnız e8deniyi olmak başlığınsıfırlanmasını self-learning saydırmaz. Eski Main40sonuçlarıFAILkalır, eski oyunları yeniselfplay diye saymayız. Bu bir hipotezdir; özgünlük/strengthvarsayımı yok.

Core/source ve atomikcommit kimlikleri korunur; tümcheckpoint/replay/model/Adam/RNGstateyerelde. PublicCPUReleasebackupPENDING; eski128verifiedasset ayrı. En güçlü bağımsızdoğrulanmış teacherorigine8; görevbitmedi.
