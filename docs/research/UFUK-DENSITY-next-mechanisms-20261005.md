# UFUK-DENSITY: mekanizma adayları, doğrulanmış başarı değil

CPU4core/16GiB; GPU kullanıcı tarafından kaldırıldı. Aktif sparse16/all16/all64
kontrolü tamamlanmadan yeni büyük deney veya öğrenci promotion başlatılmadı.
[Kapalı TRAIN-E1 tanısı](../../experiments/ufuk/cpu-search-density-v1/closed-E1-acting-diagnostic.json):
all16’da gerçek hamle deterministic search-selected hamleyle1243/2048 ve
1259/2048 kez aynı. Her hamlede search kullanmak aynı davranış politikasını
kullanmak değildir: stochasticT1value hedefleri, finalG0selected politikasının
minimax değeri sayılmaz. Nedensel strength teşhisi henüz yok.

## Birincil literatür

[Trudeau/Bowling2023, Go-Exploit](https://arxiv.org/html/2302.12359v2),
özellikle3.2/6.1/6.2: oynanan veya search’te görülen durumlardan episode
başlatmak value eğitimi dağılımını genişletir; daha kısa episode, daha çok
bağımsız terminal hedefi sağlayabilir. ConnectFour/9x9Go sonuçları doğrudan
HarbiChess veya satranç kanıtı değildir. Paper searchstates için daha
exploitative continuation ile ayrıca value hata ölçer. Kendi terminal-start
müfredatımız ve leaf-restart fikri bu literatüre yakındır, genel özgünlük iddiası yok.

[Grupen/Lee/Selman2023, Policy-Value Alignment](https://arxiv.org/html/2301.11857v2),
4.1/4.2: VIS search-policy ve value-informed selection’ı karıştırır; VISA
simetrik augmentation ile state generalization’ı inceler. Bizde critic/policy
bağımsız öğrenmesi aynı şeyi sağlıyor sayılmaz. Aynalanmış satranç verisinde
kare dönüşümü, pawn yönü, castling, en-passant, geçmiş/repetition ve mover-WDL
tam korunmadan Go simetrileri kopyalanamaz. Önce legal-transition ve terminal
perspective testleri gerekir. Model geometry/legality auxiliary learning ayrı
ölçülebilir hipotezdir; strength kazanımı veya rule discovery varsayılmaz.

[Tsai vd2026, Regret-Guided Search Control](https://arxiv.org/html/2602.20809v1),
3.1–3.3: gerçekleşmiş terminal outcome ile seçilen-action değerlerinin farkından
regret kurup, ranking/value ağlarıyla restart buffer’ı önceliklendirir. Go/Othello/Hex
sonuçları chess garantisi değildir. Yalnız yüksek-loss pozisyonu tekrar eğitmek
satrançta unbiased counterfactual-value sağlamaz; yeni continuation’ın kendi
outcome’u gerekir. Regret-restart/ranking genel fikri yayınlanmış, bizim icadımız değil.

## Ölçümden sonraki karar

Density3kolları aynı final16simde kontrollü maçlara girecek. İki seed screen’i
geçmezse başarısız kalır; daha çok search utilization veya loss iyi diye
uzun kopya koşu açılmaz. Sonraki adaylar: annealed/greedier acting ile
evaluation-policy’ye daha yakın own terminal hedefleri; search-leaf restart
ile gerçek yeni terminal continuation’ları; rule-preserving augmentation veya
observed legality/transition auxiliary loss ile temsil öğrenmesi. Bu fikirler
henüz implement/train edildi denmez; yeni frozenprotocol/seed/budget/ablation
ve aynı strength eşikleri olmadan promotion veya özgünlük iddiası yapılmaz.

Özgünlük standardı: genel yöntemlerin kombinasyonu otomatik novelty değildir.
Yakın literatür ve güçlü eşlemeli baseline/ablation/virgin independent
confirmation yoksa paper katkısı kurulmuş sayılmaz. Öğrenilmiş world model,
legality discovery veya Laya/decision-model entegrasyonu ana hedefin yerine geçmez.
