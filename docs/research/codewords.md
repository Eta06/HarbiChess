# HarbiChess codeword ve commit rehberi

Snapshot: `3804f19`; 896 commit, 54 farklı prefix. Codeword bir geliştirme/deney aşamasının kısa adıdır, başarı rozeti veya takvim ayı değildir. DENGE ve KANIT gibi prefix'ler farklı deneylerde yeniden kullanılmıştır; kesin ayrım tarih, commit ve run kimliğiyle yapılır. ARSIV bu dökümden sonraki yedekleme ve dokümantasyon aşamasıdır.

## KIVILCIM

Proje iskeleti, deterministik satranç kuralları, taşınabilir state/backend sözleşmeleri ve checkpoint manifesti.

17 commit; 2026-08-24–2026-08-24.

- [2bee9fc](https://github.com/Eta06/HarbiChess/commit/2bee9fc1c76e10a108cf4e35cd2331ad33a604aa) KIVILCIM: exclude generated training artifacts
- [fa54283](https://github.com/Eta06/HarbiChess/commit/fa54283510af915f8e5244a9964cd04df9715f2d) KIVILCIM: document project architecture and setup
- [6913a45](https://github.com/Eta06/HarbiChess/commit/6913a4530e711cc1a07a7a75e08c5e76fc234751) KIVILCIM: define Python project and MLX tooling
- [8711fc0](https://github.com/Eta06/HarbiChess/commit/8711fc0d376dc32f4f2ca6af2a6880f9e41292c2) KIVILCIM: lock development dependencies
- [370ff06](https://github.com/Eta06/HarbiChess/commit/370ff0639c3234846b066b791745e1108fdc1c57) KIVILCIM: add package version metadata
- [370ba9c](https://github.com/Eta06/HarbiChess/commit/370ba9c43b918bba0896a467791a610a35784f8f) KIVILCIM: add portable chess state contracts
- [0bcb031](https://github.com/Eta06/HarbiChess/commit/0bcb031090a1102d61552257e7ebd791fdfdf291) KIVILCIM: add neural backend protocol
- [120def4](https://github.com/Eta06/HarbiChess/commit/120def43bbbde23e9be0654b3e4365a3876cbb68) KIVILCIM: add deterministic rules adapter
- [5c25cc3](https://github.com/Eta06/HarbiChess/commit/5c25cc3170535523d7cc7d60f581a69d5c4a335b) KIVILCIM: add MLX device smoke benchmark
- [a1e4e6b](https://github.com/Eta06/HarbiChess/commit/a1e4e6ba8cb275755f7fcefd7845d08e3e767301) KIVILCIM: scope artifact ignores to repository root
- [9c7875b](https://github.com/Eta06/HarbiChess/commit/9c7875ba778283d2d87986a6a13d74924086ad22) KIVILCIM: add checkpoint manifest contract
- [327c684](https://github.com/Eta06/HarbiChess/commit/327c68435565fd50dd8be9b0ba0b0a55e87f28b5) KIVILCIM: add commit-linked checkpoint publisher
- [6e4bb23](https://github.com/Eta06/HarbiChess/commit/6e4bb23e769bddad4f052f8cf67316b2fee464a2) KIVILCIM: test chess state contracts
- [b2e8c98](https://github.com/Eta06/HarbiChess/commit/b2e8c98c3446cae50827efe3b16562f983af8d39) KIVILCIM: test backend position validation
- [60a7fbd](https://github.com/Eta06/HarbiChess/commit/60a7fbd4f506656cc377299f8409f3e6eb8e1f15) KIVILCIM: test rules and terminal detection
- [12db967](https://github.com/Eta06/HarbiChess/commit/12db96728bce09c7251c2ac06569eeb80b764796) KIVILCIM: test checkpoint manifest round trips
- [cb4ab17](https://github.com/Eta06/HarbiChess/commit/cb4ab171d0c07d9700f6925943d7db8fd936bfd8) KIVILCIM: test release asset validation

## KOR

Hamle sözlüğü, history-aware board encoder ve ilk MLX policy/WDL ağı.

10 commit; 2026-08-24–2026-08-24.

- [93113b7](https://github.com/Eta06/HarbiChess/commit/93113b7cf5782d4549c1f26625591d57200e54a7) KOR: add fixed chess action vocabulary
- [68b5261](https://github.com/Eta06/HarbiChess/commit/68b5261632a6002e911b189adee35be08577b9c2) KOR: add history-aware board encoder
- [38d903d](https://github.com/Eta06/HarbiChess/commit/38d903da3e748e8ef3703af41ca0c3793977d7dc) KOR: add MLX policy WDL network
- [6e74414](https://github.com/Eta06/HarbiChess/commit/6e74414685704cc8d881fd8f4ec4151451a8f00f) KOR: add real network batch benchmark
- [b33d032](https://github.com/Eta06/HarbiChess/commit/b33d0325aa3a350e81948b23a5eb2869d05eb6a3) KOR: register network benchmark command
- [6e78f30](https://github.com/Eta06/HarbiChess/commit/6e78f30e9245f7604978e9812b7832e122888b8e) KOR: document network benchmark workflow
- [3e344da](https://github.com/Eta06/HarbiChess/commit/3e344da0934c8e7f27ed85fa865de2ae1b8e5c65) KOR: test policy action encoding
- [d0bd2ec](https://github.com/Eta06/HarbiChess/commit/d0bd2ecba9dabe18a4b7ff78df1aa93635174424) KOR: test board history encoding
- [a9546a4](https://github.com/Eta06/HarbiChess/commit/a9546a4f4f8d4da44da5afbd8210b74b76f4b7b2) KOR: test MLX network outputs
- [e14d4b0](https://github.com/Eta06/HarbiChess/commit/e14d4b02c4055f67e94541972b4f465a62a26a81) KOR: test network benchmark reporting

## NABIZ

M4 Max benchmark, dashboard telemetry, kalite grafikleri ve checkpoint durum kaydı.

24 commit; 2026-08-24–2026-08-24.

- [092a3a3](https://github.com/Eta06/HarbiChess/commit/092a3a3055f150efb917a9965a06f61db3fdd22c) NABIZ: document M4 Max network benchmark
- [f8bd2c1](https://github.com/Eta06/HarbiChess/commit/f8bd2c1c29e556abdd8a621290f9eb3dbdb57ca2) NABIZ: add durable training resume state
- [8f1a799](https://github.com/Eta06/HarbiChess/commit/8f1a7998286742f23beb4a5e100df9178cbfc6f8) NABIZ: test durable training resume state
- [709e0a4](https://github.com/Eta06/HarbiChess/commit/709e0a46bb348e956578a845867a94b45c0bfaf4) NABIZ: add dashboard telemetry snapshot
- [be6e2c7](https://github.com/Eta06/HarbiChess/commit/be6e2c7ca79cbda58874023c1e93e55e0b347788) NABIZ: test dashboard telemetry snapshots
- [e896f7a](https://github.com/Eta06/HarbiChess/commit/e896f7ad77ca5c36bc0e7cadaca7565745406dc5) NABIZ: add training dashboard structure
- [2f7507a](https://github.com/Eta06/HarbiChess/commit/2f7507a65e155f770be3d7ee3f4d4446bf56c1be) NABIZ: style responsive training dashboard
- [8d35143](https://github.com/Eta06/HarbiChess/commit/8d3514333f7a1cec1d023c80c08685c34c841f56) NABIZ: render live dashboard telemetry
- [135c36a](https://github.com/Eta06/HarbiChess/commit/135c36a1619703a086e7bc50af0663b5f340361c) NABIZ: serve dashboard and telemetry stream
- [027d5f4](https://github.com/Eta06/HarbiChess/commit/027d5f471e42f08b9cc13c82739af0a42e08637e) NABIZ: test dashboard HTTP service
- [9176f5a](https://github.com/Eta06/HarbiChess/commit/9176f5a8e78f1606914472a62e2047b8d0803438) NABIZ: register dashboard command
- [4f439e6](https://github.com/Eta06/HarbiChess/commit/4f439e6246a2bcde817e5c46c62a2f63f53eb19e) NABIZ: document dashboard and resume workflow
- [3668f7d](https://github.com/Eta06/HarbiChess/commit/3668f7df241972f2388e775ad2437983bd0d7a48) NABIZ: silence routine dashboard disconnects
- [48bfed3](https://github.com/Eta06/HarbiChess/commit/48bfed380089e084970bfd9a4f41a90b01839398) NABIZ: test dashboard disconnect handling
- [1cc1817](https://github.com/Eta06/HarbiChess/commit/1cc1817074660d92d75459c7daa886d3de68b089) NABIZ: estimate arena Elo confidence
- [aa26c8b](https://github.com/Eta06/HarbiChess/commit/aa26c8bd29ca0b7ba33db6ec89c33610704669d8) NABIZ: test arena quality estimates
- [2fd8a48](https://github.com/Eta06/HarbiChess/commit/2fd8a48c4b8695388a47765acbdaec37834efe2a) NABIZ: track bounded quality history
- [9740ca3](https://github.com/Eta06/HarbiChess/commit/9740ca36eb2dcdfdd993aba83223103ea042807b) NABIZ: test dashboard quality history
- [67f1b2c](https://github.com/Eta06/HarbiChess/commit/67f1b2c19bdd003c44eed9151b864051662bc92a) NABIZ: add model quality panels
- [40b5ccf](https://github.com/Eta06/HarbiChess/commit/40b5ccf6582aeec4b77bdd455bfe7be2419dad25) NABIZ: style training quality charts
- [cb19455](https://github.com/Eta06/HarbiChess/commit/cb19455435e1b16f6c7ceeaac2a81f927ebcb191) NABIZ: render quality trajectory charts
- [f613d3b](https://github.com/Eta06/HarbiChess/commit/f613d3b42612315dc5542c0d90dc8fbf872a02f2) NABIZ: cover dashboard service workflows
- [585e577](https://github.com/Eta06/HarbiChess/commit/585e577a24f0fc2af69dccb27a802cc4dea38481) NABIZ: document model quality measurement
- [9c7c857](https://github.com/Eta06/HarbiChess/commit/9c7c857a28612af6848abcb9c8312495a997437b) NABIZ: link model quality workflow

## VITRIN

Kullanıcının HeroUI/React arayüzünün Python dashboard sunucusuna entegrasyonu.

22 commit; 2026-08-24–2026-08-24.

- [f962f93](https://github.com/Eta06/HarbiChess/commit/f962f93163a8b1809a8d0c5db0f9c25ca47f3907) VITRIN: ignore frontend build caches
- [6643261](https://github.com/Eta06/HarbiChess/commit/66432613488c7712140521ca42a88c579b899ed3) VITRIN: configure HeroUI dashboard package
- [9740e6b](https://github.com/Eta06/HarbiChess/commit/9740e6b47a4e904d75fec17ce6b8d6a5778d6b02) VITRIN: lock dashboard dependencies
- [26178a8](https://github.com/Eta06/HarbiChess/commit/26178a8a0d7fc0f91e1d52b957f22e2226db93a2) VITRIN: configure dashboard TypeScript project
- [358f51f](https://github.com/Eta06/HarbiChess/commit/358f51f3c289b1a90d01da72c5c6eae44b711b6c) VITRIN: configure dashboard application types
- [5bd397c](https://github.com/Eta06/HarbiChess/commit/5bd397cfe27fdf9eefdc309da791bb116c30875b) VITRIN: configure dashboard build types
- [a4dc27a](https://github.com/Eta06/HarbiChess/commit/a4dc27a0e650d3f89ae50c8cb8a5b82c8ddafbc6) VITRIN: configure HeroUI styles
- [fb2a7c1](https://github.com/Eta06/HarbiChess/commit/fb2a7c1aa7b5fed80ee8c95af5379194fa1282fb) VITRIN: build dashboard into Python assets
- [ef072e1](https://github.com/Eta06/HarbiChess/commit/ef072e1bc95659b7b2e637a15327c6dca8012c68) VITRIN: add React dashboard entry page
- [4c6a093](https://github.com/Eta06/HarbiChess/commit/4c6a09330f6ebfab15ff2a23191a82ba6d11b5e4) VITRIN: type dashboard telemetry
- [8ae2dcf](https://github.com/Eta06/HarbiChess/commit/8ae2dcf91b7584e198502aef61f9e7edeeb49033) VITRIN: stream dashboard telemetry
- [621c4a6](https://github.com/Eta06/HarbiChess/commit/621c4a6026c0a6ed916d1fc39d46a5b3538958cb) VITRIN: mount React dashboard
- [b1e444b](https://github.com/Eta06/HarbiChess/commit/b1e444b7ca118f48b68eaf83b4e9236cd6cef788) VITRIN: add HeroUI training dashboard
- [8c9efed](https://github.com/Eta06/HarbiChess/commit/8c9efed20adea4ae7145a0551c7b954f46af23fd) VITRIN: style HeroUI training dashboard
- [27051c4](https://github.com/Eta06/HarbiChess/commit/27051c4b949360cc5ca87d47e467ca7175658ef7) VITRIN: serve compiled dashboard assets
- [f5cbfb2](https://github.com/Eta06/HarbiChess/commit/f5cbfb20fb4323626c046b08f6c8a251bb8f78d3) VITRIN: test compiled dashboard assets
- [927e7c4](https://github.com/Eta06/HarbiChess/commit/927e7c46a7b7d8a3708d269c97c84addbc8771d0) VITRIN: publish dashboard entry asset
- [0bfb663](https://github.com/Eta06/HarbiChess/commit/0bfb663e8e02a959926f92bb893858c046cb9976) VITRIN: publish dashboard style bundle
- [b9715f2](https://github.com/Eta06/HarbiChess/commit/b9715f2e032dabefbd9a2e91434877eed99b7769) VITRIN: publish dashboard script bundle
- [31c7d4f](https://github.com/Eta06/HarbiChess/commit/31c7d4f2ffe2d94cd1e8ccdcf377ffe4c25a9bb7) VITRIN: remove legacy dashboard script
- [86ef8d4](https://github.com/Eta06/HarbiChess/commit/86ef8d4704457c832afb41d115526960126b9702) VITRIN: remove legacy dashboard styles
- [a3ea9b0](https://github.com/Eta06/HarbiChess/commit/a3ea9b0c14fc7b1a07995367606b51841da3cc4f) VITRIN: document dashboard build workflow

## AKIN

PUCT/MCTS, paralel self-play ve ortak MLX inference batching kuyruğu.

17 commit; 2026-08-24–2026-08-24.

- [238a788](https://github.com/Eta06/HarbiChess/commit/238a7881bfee1681587ad313f6f2aa9525cf24bd) AKIN: add shared inference batch queue
- [54e68d8](https://github.com/Eta06/HarbiChess/commit/54e68d879c89644a4e256d3ee4bde8320df90b02) AKIN: test shared inference batching
- [2b33ecf](https://github.com/Eta06/HarbiChess/commit/2b33ecff503b06143eed969983064030e1c681d3) AKIN: add legal neural search evaluator
- [e460989](https://github.com/Eta06/HarbiChess/commit/e460989f53a6d9831910b660c47c3f6be225d400) AKIN: test legal policy evaluation
- [720bf2d](https://github.com/Eta06/HarbiChess/commit/720bf2dd77b230289a3a2e5f6a60accf8eb7c187) AKIN: add deterministic PUCT search
- [d28e2a2](https://github.com/Eta06/HarbiChess/commit/d28e2a2efa68e9bdd9db5c7df1ee908ecf3c885b) AKIN: test PUCT search behavior
- [61ff175](https://github.com/Eta06/HarbiChess/commit/61ff175f15de501b4b2f60790aa3edd6e2b2c67e) AKIN: adapt MLX network for batched search
- [b1f9424](https://github.com/Eta06/HarbiChess/commit/b1f9424a9a9a4ea6abffdce05353df8428b2b208) AKIN: test MLX search backend
- [40c2196](https://github.com/Eta06/HarbiChess/commit/40c219630f4bfe64f501d403a5a0a2e3a4ab49f7) AKIN: add parallel root search coordinator
- [8dec7ba](https://github.com/Eta06/HarbiChess/commit/8dec7bad22fcc1b39ace768f5c1134230996d86a) AKIN: test independent parallel searches
- [51e000c](https://github.com/Eta06/HarbiChess/commit/51e000caa436d6cd36c18dc171a02d6c51410603) AKIN: add complete self-play game generation
- [cb183c8](https://github.com/Eta06/HarbiChess/commit/cb183c8351e0a1562a143af85dd760dec640be99) AKIN: test self-play game targets
- [41e1c64](https://github.com/Eta06/HarbiChess/commit/41e1c64f85017ff68f324eb0850ea813a85ef63d) AKIN: benchmark parallel MLX searches
- [d2efef7](https://github.com/Eta06/HarbiChess/commit/d2efef7c57d5a71a320255ce8500a75ecbf930ac) AKIN: test parallel search benchmark
- [b22f680](https://github.com/Eta06/HarbiChess/commit/b22f68051fcdf3111bcbe7071dad823b5a89a4c8) AKIN: register parallel search benchmark
- [c9aac4f](https://github.com/Eta06/HarbiChess/commit/c9aac4f319d1cb586d15efd07b1650d86e20a016) AKIN: record M4 Max search benchmark
- [3e815c1](https://github.com/Eta06/HarbiChess/commit/3e815c1f612cd57608102f35cd495f47cdc4b5a9) AKIN: document search and self-play pipeline

## OCAK

Versioned replay, oyun bazlı veri ayrımı, diversity, guarded learner ve ilk sanity pilotu.

48 commit; 2026-08-24–2026-08-24.

- [05621e4](https://github.com/Eta06/HarbiChess/commit/05621e48a23c592de445093aa191fc8c36a5829e) OCAK: version the policy action schema
- [78dbc06](https://github.com/Eta06/HarbiChess/commit/78dbc0602cfd6e2e05c7f03c31646be40ddd49f9) OCAK: record selected self-play moves
- [4fcc290](https://github.com/Eta06/HarbiChess/commit/4fcc2909ce918bbde13cd84f4d92ad00f4b80ac1) OCAK: test selected self-play move recording
- [06acf1e](https://github.com/Eta06/HarbiChess/commit/06acf1e8c6e4da046693817d122742d19653f6d8) OCAK: test value perspective invariants
- [9f5a4a4](https://github.com/Eta06/HarbiChess/commit/9f5a4a49e9dd5b702b39a93521c548acad2d8bce) OCAK: add versioned replay records
- [9fab737](https://github.com/Eta06/HarbiChess/commit/9fab737c465eadc97c4b4eff64ba5313d58017c4) OCAK: test replay record validation
- [cf5fe86](https://github.com/Eta06/HarbiChess/commit/cf5fe8669c0de84cca6f6316264aa28e39c459de) OCAK: isolate replay splits by game
- [a5573e0](https://github.com/Eta06/HarbiChess/commit/a5573e0373e28dc00d0002747341b765adcf33d7) OCAK: test game-level replay splitting
- [54487b6](https://github.com/Eta06/HarbiChess/commit/54487b6be5bf960ef31a6e7a6159f4f7a3a4f0e1) OCAK: add atomic checksummed replay shards
- [f5f6e8d](https://github.com/Eta06/HarbiChess/commit/f5f6e8d967abc518990a4fa7b292eb374f864e03) OCAK: test replay shard integrity
- [2b45484](https://github.com/Eta06/HarbiChess/commit/2b45484fb4cbc412fbf1531a1ba03eb78dcb0a6f) OCAK: measure self-play diversity
- [28beb65](https://github.com/Eta06/HarbiChess/commit/28beb65ee710b407272feaff925bb73f03c9f3d8) OCAK: test self-play diversity metrics
- [8b1f82b](https://github.com/Eta06/HarbiChess/commit/8b1f82b7c0d8d0fefca1579cfbccd002c75b49a0) OCAK: build game-balanced training batches
- [beedf0f](https://github.com/Eta06/HarbiChess/commit/beedf0fd7778fba5c0806a95874b0210e69cf6bd) OCAK: test training batch targets
- [cb517e6](https://github.com/Eta06/HarbiChess/commit/cb517e646f957945e52d6fc902bf91f7bdae4548) OCAK: add guarded MLX learner
- [fb6227e](https://github.com/Eta06/HarbiChess/commit/fb6227ea90731661809bc5b437e20c77e75627ae) OCAK: test finite learner progress
- [bad76bb](https://github.com/Eta06/HarbiChess/commit/bad76bb87ca6f99a5e6d84ff6b9fc3086a1a8dfe) OCAK: gate training with a sanity pilot
- [499a2f3](https://github.com/Eta06/HarbiChess/commit/499a2f3f51282ab3e5565707aa30de69dd4c25cd) OCAK: test sanity pilot guardrails
- [ef43591](https://github.com/Eta06/HarbiChess/commit/ef435919c0e34662bad1654f75066daa16516712) OCAK: checksum resume artifacts
- [ef9a625](https://github.com/Eta06/HarbiChess/commit/ef9a625bf4c84613bf049145924eb0f049b16a00) OCAK: save exact atomic training checkpoints
- [6ab645a](https://github.com/Eta06/HarbiChess/commit/6ab645a155627b1e9ab1bc484fdd768f5f94b5fa) OCAK: test exact checkpoint resume
- [3c3c45e](https://github.com/Eta06/HarbiChess/commit/3c3c45e293000cc2a570055b86710fbc6db0270b) OCAK: document training guardrails
- [0b1c678](https://github.com/Eta06/HarbiChess/commit/0b1c678d00b5be2b1f3cdb205c9f6adb80c9775c) OCAK: link learner foundation documentation
- [38637bc](https://github.com/Eta06/HarbiChess/commit/38637bc309f96509ada3e55cce273bf2dcf2b3ae) OCAK: expose sanity telemetry state
- [a7049fb](https://github.com/Eta06/HarbiChess/commit/a7049fbac2dcdd1bc2b92653da75772e2be39bc7) OCAK: test sanity telemetry snapshots
- [6b0e7c5](https://github.com/Eta06/HarbiChess/commit/6b0e7c51379cf77a1d4a09dbb576f19adcb71074) OCAK: report completed self-play games
- [88eb160](https://github.com/Eta06/HarbiChess/commit/88eb16056a1cd3fd2b7fcc05c13124d564d8029c) OCAK: test self-play completion telemetry
- [6186b34](https://github.com/Eta06/HarbiChess/commit/6186b3422935ca44a1a56d60e92c6611b3c470fa) OCAK: stream learner pilot progress
- [ed19cc5](https://github.com/Eta06/HarbiChess/commit/ed19cc5487fe8e99dd621170e06e31129d1a801e) OCAK: test learner pilot progress events
- [b4829fc](https://github.com/Eta06/HarbiChess/commit/b4829fcbacf97708069452fd9567e44fcf8e6e05) OCAK: orchestrate the low-budget sanity run
- [266211b](https://github.com/Eta06/HarbiChess/commit/266211b52703a4c21ed02a5339708837988513a6) OCAK: test the complete sanity pipeline
- [0551eca](https://github.com/Eta06/HarbiChess/commit/0551eca68ab14485ff6d88fe35dc00702eb3c1fa) OCAK: register the sanity runner
- [b65db1c](https://github.com/Eta06/HarbiChess/commit/b65db1c01d68ef8dddda3458182ff624df8aff28) OCAK: type sanity dashboard telemetry
- [3f54ff6](https://github.com/Eta06/HarbiChess/commit/3f54ff665d3deb728e11a2b1c080e966db8b2746) OCAK: accept sanity telemetry schema
- [0bb310c](https://github.com/Eta06/HarbiChess/commit/0bb310c91ea8e37b2be7b736dd4dfc64ea2648e4) OCAK: display sanity run guardrails
- [e8500b4](https://github.com/Eta06/HarbiChess/commit/e8500b492a76ca81c20ff57bb214c490c800d2c1) OCAK: style sanity dashboard panels
- [cef7338](https://github.com/Eta06/HarbiChess/commit/cef733882848ddb021305a4f12dd1c0d68d6cc7e) OCAK: build sanity dashboard styles
- [619dec6](https://github.com/Eta06/HarbiChess/commit/619dec696774dd96043ac3406d8032a3409bb8bd) OCAK: build sanity dashboard client
- [b9ddd17](https://github.com/Eta06/HarbiChess/commit/b9ddd175dbec1ea5151142251086dc02af4219bc) OCAK: document the sanity runner
- [d0e19e3](https://github.com/Eta06/HarbiChess/commit/d0e19e3533e50c806a100fbe9d4e8e1b08b43db7) OCAK: measure terminal outcome coverage
- [6c02523](https://github.com/Eta06/HarbiChess/commit/6c02523994110ab098c11c5b64bf6d24c2e6e325) OCAK: test max-ply outcome metrics
- [987b0aa](https://github.com/Eta06/HarbiChess/commit/987b0aa8ae45b6293f411a6361f3b67c49d9a445) OCAK: expose terminal outcome telemetry
- [0ff45c4](https://github.com/Eta06/HarbiChess/commit/0ff45c47f92e574baf4cb553790507b3b1b57444) OCAK: type terminal outcome telemetry
- [65ab457](https://github.com/Eta06/HarbiChess/commit/65ab457061f38470638cd9902a5ec51cc46cefbf) OCAK: display decisive game coverage
- [712c5a9](https://github.com/Eta06/HarbiChess/commit/712c5a9959c2cbb6f6a4244e2bbafef6782f79ca) OCAK: build terminal coverage dashboard
- [0d97e86](https://github.com/Eta06/HarbiChess/commit/0d97e864866d02de4bc4d8110b7dfba6d570ebe2) OCAK: gate draw-only sanity runs
- [e56b536](https://github.com/Eta06/HarbiChess/commit/e56b536c4cda43d8edfd6c90ddeda76984f0e6bf) OCAK: test terminal outcome guardrails
- [8c4dbce](https://github.com/Eta06/HarbiChess/commit/8c4dbce54685d57df13acb21f4688243ea92e72d) OCAK: record sanity run findings

## DEVIR

Renk dengeli candidate/champion arena ve değerlendirme hattı.

4 commit; 2026-08-24–2026-08-24.

- [f747486](https://github.com/Eta06/HarbiChess/commit/f747486df939659c4fa18355603ae939f4fbbcfa) DEVIR: add color-balanced candidate arena
- [47d7477](https://github.com/Eta06/HarbiChess/commit/47d74774978765a3f995881d0c32e8fb92c5d5c3) DEVIR: test arena color and promotion safety
- [bffa678](https://github.com/Eta06/HarbiChess/commit/bffa6782071f0e1417c7ce00a7e1e1cf1d5447f6) DEVIR: register the arena runner
- [73a84bb](https://github.com/Eta06/HarbiChess/commit/73a84bbac46ce51f7c91851b1e8b9416132592ed) DEVIR: record micro-arena findings

## SUBAT

Daha geniş pilot, terminal dağılımı ve repetition gözlemi.

14 commit; 2026-08-24–2026-08-24.

- [c7d5a43](https://github.com/Eta06/HarbiChess/commit/c7d5a4302912cbbb94875361a15decc4e131e7f4) SUBAT: measure self-play termination distribution
- [babb94a](https://github.com/Eta06/HarbiChess/commit/babb94a43ab552e44d473a32c219f347e6be599a) SUBAT: test termination distribution metrics
- [f2e88d4](https://github.com/Eta06/HarbiChess/commit/f2e88d4273e8166b3a9509b8ab26ba564229116e) SUBAT: expose termination telemetry
- [a439b2f](https://github.com/Eta06/HarbiChess/commit/a439b2f24d99607d6274aae032ab34676076fe1a) SUBAT: test termination telemetry snapshots
- [45ba3eb](https://github.com/Eta06/HarbiChess/commit/45ba3eb7868af893b183512e9111b7cb1858fa3a) SUBAT: preserve baseline and stream terminations
- [a1c5d7e](https://github.com/Eta06/HarbiChess/commit/a1c5d7ef881040a6244c815fc5855c833ce9c358) SUBAT: test preserved baseline telemetry
- [81dabef](https://github.com/Eta06/HarbiChess/commit/81dabef6b0223f56effa5f54e934aa0d767c3bbb) SUBAT: track arena draw terminations
- [ce3f90d](https://github.com/Eta06/HarbiChess/commit/ce3f90d313a8b27e530c6d397636f31dc40146f3) SUBAT: test arena termination telemetry
- [d92789d](https://github.com/Eta06/HarbiChess/commit/d92789d54abf3469d6c607c2b791575a78655149) SUBAT: type termination telemetry
- [37f420e](https://github.com/Eta06/HarbiChess/commit/37f420ebac84b2d285a1cdc46882d19e597c611d) SUBAT: display termination behavior
- [bd98eb3](https://github.com/Eta06/HarbiChess/commit/bd98eb3d0ff7fba2ac0dc58383898b78fd5905b3) SUBAT: style termination metrics
- [d9e0097](https://github.com/Eta06/HarbiChess/commit/d9e009760701f0eee47c3581ec9b5944aa7aada1) SUBAT: build termination dashboard styles
- [cc0e369](https://github.com/Eta06/HarbiChess/commit/cc0e36985fc79f294fc312777b0ca8d0aa7e8cf3) SUBAT: build termination dashboard client
- [2f96b39](https://github.com/Eta06/HarbiChess/commit/2f96b399c9f90f30825570d16760ad22e3a3b579) SUBAT: record second pilot and arena findings

## MART

Board cache, hazırlanmış replay, training/performance ölçümü ve checkpoint seçimi.

50 commit; 2026-08-24–2026-08-25.

- [badc5da](https://github.com/Eta06/HarbiChess/commit/badc5dacc2e1b86078b102f6619be2350fa73c87) MART: cache incremental chess boards
- [864492e](https://github.com/Eta06/HarbiChess/commit/864492e2911e86a6ebd9cb32f0849bbe26828d23) MART: test board cache isolation
- [06e81d3](https://github.com/Eta06/HarbiChess/commit/06e81d3d8ff97390f228551075dcf781d7cd4311) MART: encode borrowed board history
- [e12cfd6](https://github.com/Eta06/HarbiChess/commit/e12cfd68876810e2c36e3fdf13b4b096d90152e3) MART: test prebuilt board encoding
- [978236e](https://github.com/Eta06/HarbiChess/commit/978236e1716fbd80fa6fa9a02ce706cebbef2ca1) MART: share board across neural evaluation
- [c1ad7cf](https://github.com/Eta06/HarbiChess/commit/c1ad7cf1fff23b5414141596e14eaaefd8c040a8) MART: fuse gradient finiteness reduction
- [1a4aae5](https://github.com/Eta06/HarbiChess/commit/1a4aae5683f4491151987a0f5e0e6bdb766f89ff) MART: test fused gradient validation
- [1a6a6c8](https://github.com/Eta06/HarbiChess/commit/1a6a6c8cccd166d9c54f571cefb723cabeedb626) MART: reuse prepared replay batches
- [7fbd8a5](https://github.com/Eta06/HarbiChess/commit/7fbd8a53ea68b4038d0641123d5d49da79f59fd8) MART: test prepared replay selection
- [06b99a3](https://github.com/Eta06/HarbiChess/commit/06b99a3e484dedd3b19bd42931ffa1cf43b26830) MART: train from prepared evaluation data
- [75df3d5](https://github.com/Eta06/HarbiChess/commit/75df3d5b9c640322f0a27ab97fa94417882969e5) MART: test prepared pilot evaluation
- [b038e90](https://github.com/Eta06/HarbiChess/commit/b038e906dc50df555997165a78a125be7acf8639) MART: reuse validation tensors
- [9ebab07](https://github.com/Eta06/HarbiChess/commit/9ebab07e5ca5c48493006c6816e9101750420b8d) MART: snapshot exact learner state
- [b503ec1](https://github.com/Eta06/HarbiChess/commit/b503ec11c1352c71b2d40d8628a799334f24a558) MART: test learner state restoration
- [af39cad](https://github.com/Eta06/HarbiChess/commit/af39cad1e9ed3b93abbd506e80c280ee4829559b) MART: restore best validation step
- [559f396](https://github.com/Eta06/HarbiChess/commit/559f39697c260fb9e0bcf38d3aee34356a3ce55f) MART: test validation early stopping
- [141a98a](https://github.com/Eta06/HarbiChess/commit/141a98aa0e2af2d9b3ccb131378e2958ed43a15d) MART: configure best checkpoint selection
- [e803bb3](https://github.com/Eta06/HarbiChess/commit/e803bb318d511624da3652e04d5f0815b750538f) MART: expose best checkpoint telemetry
- [f03b4e6](https://github.com/Eta06/HarbiChess/commit/f03b4e68a529af1304c41c5934077d03046d1277) MART: test early-stop dashboard state
- [b81facb](https://github.com/Eta06/HarbiChess/commit/b81facb94d7cf9f7ea838424bba4eb56b80b0726) MART: publish best checkpoint telemetry
- [0334ecc](https://github.com/Eta06/HarbiChess/commit/0334eccf527be293cdcc508bab61b2bd1c3598d7) MART: type early-stop telemetry
- [5bfe9e7](https://github.com/Eta06/HarbiChess/commit/5bfe9e791f037426fe7a361ce2599edb53d8a82e) MART: display best validation checkpoint
- [0d19bae](https://github.com/Eta06/HarbiChess/commit/0d19baed52783514e894db13b30b989ccca9eb47) MART: build early-stop dashboard client
- [1f7af99](https://github.com/Eta06/HarbiChess/commit/1f7af99174104b1cd638b31921b001cdf0889edd) MART: classify avoidable arena repetitions
- [f9a848e](https://github.com/Eta06/HarbiChess/commit/f9a848ed615a2b2c8a2ed02995f9604bc6904816) MART: test avoidable repetition telemetry
- [5dba5e6](https://github.com/Eta06/HarbiChess/commit/5dba5e6f784fcc68aa95b1b9e8c53c587729d60e) MART: expose avoidable repetition count
- [6231e25](https://github.com/Eta06/HarbiChess/commit/6231e25fe352251564200217f9b25f452843aeed) MART: test repetition dashboard state
- [91bc683](https://github.com/Eta06/HarbiChess/commit/91bc68319931df33c404f73c85a785a93bb825a7) MART: guard self-play repetition collapse
- [2fc8cdc](https://github.com/Eta06/HarbiChess/commit/2fc8cdc6a8eb312d16b93b0c9aa38a04604ecb23) MART: type avoidable repetition telemetry
- [cf786f4](https://github.com/Eta06/HarbiChess/commit/cf786f49dfacff25bc97e74eaf5feed7e18def47) MART: display avoidable repetitions
- [1c2238a](https://github.com/Eta06/HarbiChess/commit/1c2238a1127361023b0287c47cc32f258f24eb88) MART: build repetition dashboard client
- [9b94105](https://github.com/Eta06/HarbiChess/commit/9b94105725c16dd9cb215caf8a6217531f4ff2c2) MART: define masked policy outputs
- [944b3b2](https://github.com/Eta06/HarbiChess/commit/944b3b2bec666119bb1220df576498a355466610) MART: gather legal logits on MLX
- [9754643](https://github.com/Eta06/HarbiChess/commit/97546431e915b0d83c602d50b8085e2f1744c67d) MART: test MLX masked policy gathering
- [2cb05f5](https://github.com/Eta06/HarbiChess/commit/2cb05f558ca879c753ccf0caea4ac8fe66f8509e) MART: batch masked inference requests
- [75314a3](https://github.com/Eta06/HarbiChess/commit/75314a36ed1604ab1978648a386127b9059c68e1) MART: test masked batching fallback
- [104e670](https://github.com/Eta06/HarbiChess/commit/104e6709ebbf8ee67da0a94e73a386e187d78e7c) MART: request only legal policy logits
- [2303312](https://github.com/Eta06/HarbiChess/commit/2303312f9302a009281b619baae198bbebf22a03) MART: test masked neural evaluation
- [53e26fd](https://github.com/Eta06/HarbiChess/commit/53e26fd93e9565154f7a32b81c0bc4ce72dcaacb) MART: cache immutable board encodings
- [edd3f73](https://github.com/Eta06/HarbiChess/commit/edd3f73aa52c61d6845a9b9510adcf0e231f91f2) MART: test encoding cache reuse
- [c0baecd](https://github.com/Eta06/HarbiChess/commit/c0baecda697df3530655d44df407c34092088eba) MART: reuse cached search encodings
- [4e49393](https://github.com/Eta06/HarbiChess/commit/4e493935f9499c1a5062cd92b8b448b494e72b6a) MART: align restored training telemetry
- [949ba3a](https://github.com/Eta06/HarbiChess/commit/949ba3a72530e1778c3c3c4893537aa157085e76) MART: test restored training step telemetry
- [9454aa4](https://github.com/Eta06/HarbiChess/commit/9454aa4defaed53b3b99b5ab0e804f429497fa12) MART: index encoder planes directly
- [8cc1f39](https://github.com/Eta06/HarbiChess/commit/8cc1f39ae8c7b8242b3d76a2d54ce7fbe9605072) MART: expose profiler comparisons
- [db9eff6](https://github.com/Eta06/HarbiChess/commit/db9eff62f14106d2d3282b9355e7bfa471f53bb1) MART: test profiler dashboard state
- [03b072c](https://github.com/Eta06/HarbiChess/commit/03b072c1d0aa6fc8d94e7206c29c9f864a769a18) MART: type profiler comparisons
- [eb106c7](https://github.com/Eta06/HarbiChess/commit/eb106c7b50f5cdaecc102905ec5341239e547aea) MART: display profiler comparisons
- [107e1cf](https://github.com/Eta06/HarbiChess/commit/107e1cfb640e6adfb1c9c755b199545c05d0daf0) MART: build profiler dashboard client
- [e87ec28](https://github.com/Eta06/HarbiChess/commit/e87ec28ef06810e31e26d86da666ac009b1d0403) MART: record profiler and candidate findings

## NISAN

Repetition-aware target yönlendirmesi ve birden fazla validation checkpoint karşılaştırması.

16 commit; 2026-08-25–2026-08-25.

- [7b76cdd](https://github.com/Eta06/HarbiChess/commit/7b76cdd13b7ac7cd5dba1477a5da74a8f9b396a8) NISAN: redirect comparable repetition continuations
- [aa74cf0](https://github.com/Eta06/HarbiChess/commit/aa74cf0127ac9c58c4c02ca29c42441ed5b7b762) NISAN: test repetition continuation targets
- [18be3ce](https://github.com/Eta06/HarbiChess/commit/18be3ce59e4227354a02459380011fb236c741ee) NISAN: version repetition-aware replay targets
- [43b1e10](https://github.com/Eta06/HarbiChess/commit/43b1e1030fdb3b2e39503bbc1f1aec339904ebb4) NISAN: measure repetition target redirects
- [67c3d70](https://github.com/Eta06/HarbiChess/commit/67c3d70a067636b7d637fe41366d2ad624125146) NISAN: retain spaced validation candidates
- [d9e589a](https://github.com/Eta06/HarbiChess/commit/d9e589a54abc9cf8cbff0993adc6a57c7f2aa4d8) NISAN: test validation candidate retention
- [42b8e99](https://github.com/Eta06/HarbiChess/commit/42b8e9927a7fa385f842ee99bbfd068850086957) NISAN: persist exact validation candidates
- [a4bb431](https://github.com/Eta06/HarbiChess/commit/a4bb4313bfca338377860b2a18c254d864e59927) NISAN: select candidate checkpoint for arena
- [db81d21](https://github.com/Eta06/HarbiChess/commit/db81d21f8f526caa258f44cdef106b188698c02d) NISAN: test arena checkpoint selection
- [05a0c32](https://github.com/Eta06/HarbiChess/commit/05a0c32cab52a8b0d4ef61898aedbb9a133e3d1b) NISAN: expose policy iteration telemetry
- [f25d3a3](https://github.com/Eta06/HarbiChess/commit/f25d3a341cb9fa32d8b5e28d997d7969c65289c4) NISAN: type policy iteration telemetry
- [d60448d](https://github.com/Eta06/HarbiChess/commit/d60448d8cfa4f05082c4823a0718cad1ac6251bf) NISAN: display replay redirect metrics
- [16bfb34](https://github.com/Eta06/HarbiChess/commit/16bfb3453f659a0440713a4bb34b531aa910609c) NISAN: build policy iteration dashboard
- [f24f8a7](https://github.com/Eta06/HarbiChess/commit/f24f8a75b5a23df714debaabf5933fcd6dc2a824) NISAN: publish selected arena checkpoint
- [99ae535](https://github.com/Eta06/HarbiChess/commit/99ae535343b942a1c0e90409e83453e3db9018b3) NISAN: test selected checkpoint telemetry
- [4f733c1](https://github.com/Eta06/HarbiChess/commit/4f733c13bc69785967bc3bd1a21b345ef0ba2c30) NISAN: record policy iteration findings

## MAYIS

Continuation replay ve yalnız seçilen hamle yerine anlamlı repeat policy mass dönüşümü.

16 commit; 2026-08-25–2026-08-25.

- [e058f9a](https://github.com/Eta06/HarbiChess/commit/e058f9adb37d23edb0e6673658574384266b60e0) MAYIS: add repetition-aware target transformer
- [e48171c](https://github.com/Eta06/HarbiChess/commit/e48171c97ac40b708c05c59ad6de1e53cd0fa728) MAYIS: transform meaningful repeat policy mass
- [0f521d1](https://github.com/Eta06/HarbiChess/commit/0f521d183136561299641a28a89b2c0f82982a8b) MAYIS: test policy-mass repetition targets
- [aac323f](https://github.com/Eta06/HarbiChess/commit/aac323fe078420e2ed4d59bac65a319109c4fd07) MAYIS: version continuation target semantics
- [0e585ed](https://github.com/Eta06/HarbiChess/commit/0e585ed443d823d6c4a43b5936b7b4d432f8c090) MAYIS: balance continuation replay sampling
- [a6c6911](https://github.com/Eta06/HarbiChess/commit/a6c69119e8569e872cc3d7d17a1c404560d992ed) MAYIS: test continuation replay mixture
- [c14abbb](https://github.com/Eta06/HarbiChess/commit/c14abbb45f857bc4ac4e8e67f9192c1be3d19a1d) MAYIS: configure continuation replay mixture
- [005ba24](https://github.com/Eta06/HarbiChess/commit/005ba242c2bd8ed9e3c5fcf23d10983b1c1719ab) MAYIS: train with versioned continuation shards
- [f4c0540](https://github.com/Eta06/HarbiChess/commit/f4c0540d9697459a19443c30996335db2869d012) MAYIS: mine avoidable repetition roots
- [43e189e](https://github.com/Eta06/HarbiChess/commit/43e189efeff1ee0e929b57e4e3edb88a10682481) MAYIS: test arena continuation replay
- [2692398](https://github.com/Eta06/HarbiChess/commit/269239819ce1d4494681e005c810c2d5e55c5bfc) MAYIS: expose continuation replay telemetry
- [fcf7730](https://github.com/Eta06/HarbiChess/commit/fcf7730e6013f96cbeb4ab15ac66da7e660974d4) MAYIS: type continuation replay telemetry
- [468fc2f](https://github.com/Eta06/HarbiChess/commit/468fc2f533e38442217fa5f5cac44e62f9439637) MAYIS: display continuation replay metrics
- [71df928](https://github.com/Eta06/HarbiChess/commit/71df928e32dabd236985b6bfe3021c639fe7a8cb) MAYIS: build continuation replay dashboard
- [b5a1979](https://github.com/Eta06/HarbiChess/commit/b5a19795cb6aa6fe3621999c2077ed8d5d2c2fe2) MAYIS: initialize candidate from champion
- [03b5ad5](https://github.com/Eta06/HarbiChess/commit/03b5ad59d2fc81e65749c0a177003d6dafe41863) MAYIS: record continuation replay findings

## HAZIRAN

Generation/recency replay karışımı, erken durma nedenleri ve profiler tabanlı optimizasyon.

22 commit; 2026-08-25–2026-08-25.

- [9b93ed1](https://github.com/Eta06/HarbiChess/commit/9b93ed1ad7328864b20d280395ffa2a5a74d392e) HAZIRAN: scan claimable repetition moves efficiently
- [5b5974b](https://github.com/Eta06/HarbiChess/commit/5b5974b70b84458ecc6200b75828fffae755ea70) HAZIRAN: test isolated repetition move scanning
- [a27e398](https://github.com/Eta06/HarbiChess/commit/a27e398dda3deda87b1631c22b509562160a4bdc) HAZIRAN: limit continuation repetition probes
- [042d5d9](https://github.com/Eta06/HarbiChess/commit/042d5d982eecd33f43fbb254a189d67d6c7702ec) HAZIRAN: measure inference queue bottlenecks
- [08694e6](https://github.com/Eta06/HarbiChess/commit/08694e6cf385b47aabb47b9e4f2cfc6043528e6e) HAZIRAN: report search queue profile metrics
- [ee864a3](https://github.com/Eta06/HarbiChess/commit/ee864a3cc7585df38b2bfd356d79e6ad53d08fed) HAZIRAN: prepare replay batches on MLX device
- [98ad2e0](https://github.com/Eta06/HarbiChess/commit/98ad2e06146a347813217b76ba2d38fb510f7f27) HAZIRAN: test prepared MLX batch selection
- [5ebddee](https://github.com/Eta06/HarbiChess/commit/5ebddee54a1c1290e0ea6fdea26ff070aef07a03) HAZIRAN: separate validation cadence and stop cause
- [241ec00](https://github.com/Eta06/HarbiChess/commit/241ec00d9897bebff367ff14595816ba071c32a4) HAZIRAN: test explicit pilot stop telemetry
- [95b50a2](https://github.com/Eta06/HarbiChess/commit/95b50a26040566ae5e9d8955ade5cbd0eec8bf40) HAZIRAN: merge continuation generations by recency
- [ad130c8](https://github.com/Eta06/HarbiChess/commit/ad130c8aba7ff9636c1f9c134560abe94c0eb7c6) HAZIRAN: test continuation generation merge
- [67fa0f2](https://github.com/Eta06/HarbiChess/commit/67fa0f2c9bc369fedb0cf5f3c5a532106916ec77) HAZIRAN: weight continuation sampling by recency
- [17445fc](https://github.com/Eta06/HarbiChess/commit/17445fc59d6e7e77e19f006389fd91bc899335c9) HAZIRAN: test recency weighted replay sampling
- [677e9bb](https://github.com/Eta06/HarbiChess/commit/677e9bb0bee993d366f3fa5149ac1e5170b1a40c) HAZIRAN: expose training stop diagnostics
- [7e7be14](https://github.com/Eta06/HarbiChess/commit/7e7be14e675d26e7631eda827a3a3d69a574d0b0) HAZIRAN: integrate profiled continuation pilot
- [3696f79](https://github.com/Eta06/HarbiChess/commit/3696f79215baf72d0befe56f5af93a59d833924f) HAZIRAN: profile batched arena inference
- [ab8c7b9](https://github.com/Eta06/HarbiChess/commit/ab8c7b936e95f21ee5ab2675936c37e059f49b22) HAZIRAN: type training stop diagnostics
- [f9a8ef8](https://github.com/Eta06/HarbiChess/commit/f9a8ef8b126c3c53086b5f479c2c68b2d01fabfd) HAZIRAN: display training and arena stop causes
- [c8dfbd1](https://github.com/Eta06/HarbiChess/commit/c8dfbd1fe1396855698d2e6b69260df3dd9918e4) HAZIRAN: build stop diagnostics dashboard
- [215e0cf](https://github.com/Eta06/HarbiChess/commit/215e0cf62b4c12a671a17f33ac8ab2b7b550fcd1) HAZIRAN: label profiled MCTS throughput
- [6516043](https://github.com/Eta06/HarbiChess/commit/651604398c52dabd0bdd0e2e26d050030e332fdb) HAZIRAN: build profiled throughput dashboard
- [fdad6f3](https://github.com/Eta06/HarbiChess/commit/fdad6f375c3cdd5d61c6cc9fa30aadf1ad414f69) HAZIRAN: record recency replay findings

## TEMMUZ

Continuation açık/kapalı/filtreli sabit-compute ablation ve champion hizası audit'i.

7 commit; 2026-08-25–2026-08-25.

- [e770afb](https://github.com/Eta06/HarbiChess/commit/e770afb2b35760b146549ee9896a40ae372d7249) TEMMUZ: audit continuation targets against champion
- [7870d10](https://github.com/Eta06/HarbiChess/commit/7870d10cf890d4f1e0e21ba2f5aec0a7ecdd26ba) TEMMUZ: test continuation quality verdicts
- [d065def](https://github.com/Eta06/HarbiChess/commit/d065defee404d4a9a2f592636f679a8f3cb2a68b) TEMMUZ: classify champion aligned continuation targets
- [402c4da](https://github.com/Eta06/HarbiChess/commit/402c4da5093d00ea7bbcce2c1326c0ea00481445) TEMMUZ: test search aligned audit verdicts
- [998f217](https://github.com/Eta06/HarbiChess/commit/998f2171affc90455cdd57b0eb0992538e40e610) TEMMUZ: train fixed compute replay ablations
- [6dc83f3](https://github.com/Eta06/HarbiChess/commit/6dc83f3d9e7828cfdaf19f845ceeac45aa9d4297) TEMMUZ: test matched ablation exposure
- [fd17015](https://github.com/Eta06/HarbiChess/commit/fd170157b4cbc88d9796c66ed86df682e69978a6) TEMMUZ: record continuation ablation findings

## AGUSTOS

Branch-level confidence, tekrar etmeyen devam için pratik avantaj alt sınırı.

13 commit; 2026-08-25–2026-08-25.

- [3ca5f0a](https://github.com/Eta06/HarbiChess/commit/3ca5f0a4585e391b99fc55b9306f30a1ae34e7b0) AGUSTOS: version branch confidence targets
- [9b76b6e](https://github.com/Eta06/HarbiChess/commit/9b76b6e8e79568807436689b1e60e583d44cb243) AGUSTOS: test branch evidence schema
- [0b86a2f](https://github.com/Eta06/HarbiChess/commit/0b86a2f72e569b1dcfecfcc800bb6778e2dfd5f6) AGUSTOS: read legacy target shards
- [6ca6e66](https://github.com/Eta06/HarbiChess/commit/6ca6e66ae5b030d63fd5d98785fbf521064d2575) AGUSTOS: test legacy target compatibility
- [ea523cb](https://github.com/Eta06/HarbiChess/commit/ea523cb2c265dd2ded50b432db2a4306bac60820) AGUSTOS: generate confidence gated branches
- [3e43596](https://github.com/Eta06/HarbiChess/commit/3e435963e4edf89b3423c5d3bc866f056d3599c4) AGUSTOS: test branch confidence targets
- [7ebc98f](https://github.com/Eta06/HarbiChess/commit/7ebc98fce9ed071008a30af262763160584ac0b9) AGUSTOS: store practical confidence margin
- [c20411d](https://github.com/Eta06/HarbiChess/commit/c20411df2f03693e997431c2610d238a6f107d3d) AGUSTOS: test confidence margin schema
- [db76ae6](https://github.com/Eta06/HarbiChess/commit/db76ae6c29c23f5948bbce9be624ce75708d4cf7) AGUSTOS: gate branches on practical advantage
- [86e70bf](https://github.com/Eta06/HarbiChess/commit/86e70bff4f24808021cd243889e6dca012800464) AGUSTOS: test practical branch gate
- [560ffbb](https://github.com/Eta06/HarbiChess/commit/560ffbb1cc5134a90b8479e56e37d75255951975) AGUSTOS: identify confidence gated ablations
- [399487e](https://github.com/Eta06/HarbiChess/commit/399487e99d842700623a97e710c8dd5a18c54815) AGUSTOS: test confidence gated treatment
- [3722cd3](https://github.com/Eta06/HarbiChess/commit/3722cd34f31ff9726343629a905b1f5c8b4987ca) AGUSTOS: record branch confidence findings

## EYLUL

V4 paired strength ve repetition davranış gate'leri.

3 commit; 2026-08-25–2026-08-25.

- [5f3fd13](https://github.com/Eta06/HarbiChess/commit/5f3fd136df014c8698422ef3d31eb5672ded6569) EYLUL: gate paired strength and behavior
- [699ae53](https://github.com/Eta06/HarbiChess/commit/699ae53439f6c6751584f7c33806cf874e0081ab) EYLUL: test paired promotion guardrails
- [85e8d38](https://github.com/Eta06/HarbiChess/commit/85e8d38da3cabec5e50633e119750edddb71b54b) EYLUL: record v4 evidence gate findings

## EKIM

V5 kısa horizon çok hamleli repetition risk modeli.

9 commit; 2026-08-25–2026-08-25.

- [e2ca31f](https://github.com/Eta06/HarbiChess/commit/e2ca31f7e03d2597a73830992c74531b0906a095) EKIM: version multi-ply repetition risk targets
- [621fbba](https://github.com/Eta06/HarbiChess/commit/621fbba6cc7f0456e3fc3acb25aee1600d804564) EKIM: weight branch targets by repetition risk
- [2112136](https://github.com/Eta06/HarbiChess/commit/2112136f692e5ef4804a4a199ee2daf339d6fea9) EKIM: test multi-ply repetition risk targets
- [7b5c91d](https://github.com/Eta06/HarbiChess/commit/7b5c91df90c405131c22177775d5262578d455e2) EKIM: preserve legacy target schema coverage
- [07a72ef](https://github.com/Eta06/HarbiChess/commit/07a72ef78688f25db92b11023268a0b13d4fe772) EKIM: gate continuation branches by multi-ply risk
- [715894e](https://github.com/Eta06/HarbiChess/commit/715894eb0f7591b76c03e13a4dad4b79e9df405c) EKIM: test short-horizon repetition risk gate
- [f713aa4](https://github.com/Eta06/HarbiChess/commit/f713aa4ca834ee697070a8b60490a63c1749e688) EKIM: add repetition risk ablation treatment
- [bc71bba](https://github.com/Eta06/HarbiChess/commit/bc71bba4a07c399848e53db623944193d6359b3a) EKIM: test repetition risk ablation treatment
- [8a3aba6](https://github.com/Eta06/HarbiChess/commit/8a3aba6cf741af6dba42db6c4d984a088ac4fae9) EKIM: record multi-ply repetition risk findings

## KASIM

V6 loop olasılığını beklenen loop değeriyle birlikte değerlendirme.

13 commit; 2026-08-26–2026-08-26.

- [0735982](https://github.com/Eta06/HarbiChess/commit/0735982883e7b5212120152ecba462513d441f7e) KASIM: version value-aware repetition targets
- [3fe6de8](https://github.com/Eta06/HarbiChess/commit/3fe6de8d4da004c8f609cf42e0672c324fb2e327) KASIM: weight targets by expected loop value
- [ba0e6d8](https://github.com/Eta06/HarbiChess/commit/ba0e6d878147cbf8133613f83a68de3688abfcf4) KASIM: generate value-aware continuation replay
- [d0631fe](https://github.com/Eta06/HarbiChess/commit/d0631fedf790a8ac4ad465d871eb5340ac29b904) KASIM: test value-aware repetition risk
- [f7a6386](https://github.com/Eta06/HarbiChess/commit/f7a63868b8eb56b6870224f7eb1ba92c81cca111) KASIM: test legacy value-risk decoding
- [2824587](https://github.com/Eta06/HarbiChess/commit/282458762a29975ebe5e86dc29d0934de67da951) KASIM: test target schema five compatibility
- [3059dc8](https://github.com/Eta06/HarbiChess/commit/3059dc8df81c1f2da14b8965f5d3f03a83923130) KASIM: add value-aware ablation treatment
- [b17116d](https://github.com/Eta06/HarbiChess/commit/b17116dbb5bbbc492ac79bc32abf52f4c4e9ebe9) KASIM: test value-aware ablation treatment
- [facd3c8](https://github.com/Eta06/HarbiChess/commit/facd3c860e590db7cb8a68896fd1cfdc1a340336) KASIM: mark exact loop value samples
- [d34bb75](https://github.com/Eta06/HarbiChess/commit/d34bb75cfc55e955e1a4aed7aa2e4df314ba2f4e) KASIM: preserve exact claimable draw values
- [2d0643c](https://github.com/Eta06/HarbiChess/commit/2d0643c1d93146795bed305ac9b33367a6a97af2) KASIM: test exact repetition draw values
- [4e81592](https://github.com/Eta06/HarbiChess/commit/4e815925875cf29fc1a1ab92052a2c0d11d6b726) KASIM: test legacy exact-value defaults
- [9ee5c25](https://github.com/Eta06/HarbiChess/commit/9ee5c25112b8b56f2ab70a1fa5d85889884f5077) KASIM: record value-aware repetition findings

## ARALIK

V7 binary redirect yerine sürekli value-regret policy karışımı.

9 commit; 2026-08-26–2026-08-26.

- [cc2461d](https://github.com/Eta06/HarbiChess/commit/cc2461dad4187d391725a09edda6f12558548fcd) ARALIK: version continuous policy regret targets
- [f978621](https://github.com/Eta06/HarbiChess/commit/f9786212174eeed90320d8a245b8e03c6088e401) ARALIK: blend continuation policies by value regret
- [36cc932](https://github.com/Eta06/HarbiChess/commit/36cc932ff83930115728f7953e641d5a579ba9ba) ARALIK: test continuous value regret blending
- [0465ae4](https://github.com/Eta06/HarbiChess/commit/0465ae41f2fd0b7893518ba124e969fec346f8c5) ARALIK: test policy regret target schema
- [d32e9bd](https://github.com/Eta06/HarbiChess/commit/d32e9bd91e596d8d5254311d570d8a20c88319d6) ARALIK: add value regret ablation treatment
- [fd73d98](https://github.com/Eta06/HarbiChess/commit/fd73d984fb8a3ab55ce3ecb3bfd0f4d3dcb0857c) ARALIK: test value regret ablation treatment
- [23f24f4](https://github.com/Eta06/HarbiChess/commit/23f24f4a314b11315d24a2067ec2a19ae19403f1) ARALIK: allow zero-regret policy preservation
- [7e6a66c](https://github.com/Eta06/HarbiChess/commit/7e6a66c5b5424e8bc958c5fff8bb96e8b6125748) ARALIK: test zero-regret redirect metadata
- [c5bc07c](https://github.com/Eta06/HarbiChess/commit/c5bc07cdaaf9d8ee48426fb4e6829bbb3c14c5a5) ARALIK: record continuous value regret findings

## KANIT

Birden fazla kullanımı var: frozen V7 arena kanıtı ve daha sonra fresh pairwise teacher doğrulaması.

10 commit; 2026-08-26–2026-08-28.

- [c8e267e](https://github.com/Eta06/HarbiChess/commit/c8e267e059c6cfaf8c5f70254ed028d329fc4cc4) KANIT: preregister frozen v7 arena evidence
- [62c5762](https://github.com/Eta06/HarbiChess/commit/62c5762562fe7a7ff6c9b9b3427f733adfb73fdb) KANIT: record frozen v7 evidence findings
- [ee95612](https://github.com/Eta06/HarbiChess/commit/ee95612f7ea7a129cd8fdb63af1abb199d418ad6) KANIT: preregister fresh pairwise validation
- [3cbdef6](https://github.com/Eta06/HarbiChess/commit/3cbdef60c2a4b28849a55b05c5747e09c19324bb) KANIT: expose pairwise bootstrap seed
- [4165d31](https://github.com/Eta06/HarbiChess/commit/4165d31a27b433841bdbd23fd4e7f353100df1b1) KANIT: accept qualified pairwise teacher
- [9300ac0](https://github.com/Eta06/HarbiChess/commit/9300ac0687884d66202b146797a673351845c43c) KANIT: test pairwise teacher evidence
- [938c423](https://github.com/Eta06/HarbiChess/commit/938c423dec578d666b102689913c9b2914d13614) KANIT: freeze candidate evaluation seed
- [c250f34](https://github.com/Eta06/HarbiChess/commit/c250f3415f389b04ae9ec9cba8b571d1d738ae85) KANIT: add fresh candidate validation
- [9c66ece](https://github.com/Eta06/HarbiChess/commit/9c66ece51670402443bba47d714f35c7bb87212a) KANIT: test fresh validation contract
- [02aed41](https://github.com/Eta06/HarbiChess/commit/02aed41092e20c1ea4786537dbe27c9d81bb217f) KANIT: record fresh learner failure

## ATAK

Value-improved policy target ile daha anlamlı strength kazanımı denemesi.

7 commit; 2026-08-26–2026-08-26.

- [ab72584](https://github.com/Eta06/HarbiChess/commit/ab725841b3eb2c12feb9b475050ef7d04437ac16) ATAK: add value-improved search policy target
- [51bf575](https://github.com/Eta06/HarbiChess/commit/51bf5752ba8fc12af6d8dc403a258b02199fc5cf) ATAK: integrate value-improved self-play targets
- [e67225c](https://github.com/Eta06/HarbiChess/commit/e67225c19ebce43bb0251afda7b2d3ed956142ed) ATAK: expose value-improved target run controls
- [bbd4973](https://github.com/Eta06/HarbiChess/commit/bbd4973592ee7e358d499282eb2a959e47807a68) ATAK: version value-improved replay targets
- [7fb5b30](https://github.com/Eta06/HarbiChess/commit/7fb5b3002d8917b48b2355aace4c578d1eeb9bfe) ATAK: test value-improved search policy target
- [ff52477](https://github.com/Eta06/HarbiChess/commit/ff5247798e1461336e924a2281c4dad6f79d7ec0) ATAK: preregister value-improved target evidence
- [d84cb58](https://github.com/Eta06/HarbiChess/commit/d84cb5878a0cd8d0afca21c428cd4f91651e56f2) ATAK: record value-improved target findings

## KILIC

Sabit root bütçesinde sequential halving ve yüksek güvenli top-action ayrımı.

11 commit; 2026-08-26–2026-08-26.

- [60777f4](https://github.com/Eta06/HarbiChess/commit/60777f4213c482f435e1af812befe2c586c7600e) KILIC: add fixed-budget root sequential halving
- [0883749](https://github.com/Eta06/HarbiChess/commit/088374900c2260f35a64617ff5d03b9712857d3a) KILIC: integrate confidence-gated root search
- [98ba18e](https://github.com/Eta06/HarbiChess/commit/98ba18e3d790452ffb0089c942f43528ee1b06ad) KILIC: version root-search replay evidence
- [592e45f](https://github.com/Eta06/HarbiChess/commit/592e45fd29cf6707d86177f32643c81fe10ac398) KILIC: measure high-confidence root adjustments
- [eedc068](https://github.com/Eta06/HarbiChess/commit/eedc068a960884c7e5cd9228bb41c428f9350f14) KILIC: expose fixed-budget root search runs
- [e5519fe](https://github.com/Eta06/HarbiChess/commit/e5519fe9e3d93e61132709a44226f5fbfd3f3b8e) KILIC: test fixed-budget root sequential halving
- [5c39d43](https://github.com/Eta06/HarbiChess/commit/5c39d43e795d63a8e2816ec8dc8c33949356031a) KILIC: clean root search imports
- [85dbf64](https://github.com/Eta06/HarbiChess/commit/85dbf64564aa642513d1f5d136c68def7c08a8f0) KILIC: preregister fixed-budget root evidence
- [7e1bdb2](https://github.com/Eta06/HarbiChess/commit/7e1bdb26d3074adcfd11786fffd2a0932d4d0d02) KILIC: test root-search replay confidence
- [0b9639f](https://github.com/Eta06/HarbiChess/commit/0b9639fcfdb438bb111e47ed6f1459af2e7d602d) KILIC: test root adjustment coverage metrics
- [cf437fd](https://github.com/Eta06/HarbiChess/commit/cf437fd593b44901bc1c86a126eff4ffd6c1e431) KILIC: record root halving findings

## OMURGA

Legacy heuristics opt-in, max-ply unknown value maskesi, teacher qualification, tactical/sign/allocation ve oracle teşhisleri.

76 commit; 2026-08-26–2026-08-28.

- [645828b](https://github.com/Eta06/HarbiChess/commit/645828bc088141f4b9598647c60d2bd45f9d9e6d) OMURGA: make legacy target transforms opt in
- [472aac6](https://github.com/Eta06/HarbiChess/commit/472aac6c6b2468fba1034f65d446cbc3ebd0d66f) OMURGA: test opt-in self-play targets
- [6f5a957](https://github.com/Eta06/HarbiChess/commit/6f5a9573911cecc2e07b8b7b544ffdf3ae655856) OMURGA: version unknown value targets
- [211fb90](https://github.com/Eta06/HarbiChess/commit/211fb90ca195d072f0f2fe73465ea87e34878b7a) OMURGA: test unknown replay outcomes
- [84e21e9](https://github.com/Eta06/HarbiChess/commit/84e21e9c65c68fb29c7f8f731ba72120c08de72e) OMURGA: mask unknown replay values
- [8e94b7e](https://github.com/Eta06/HarbiChess/commit/8e94b7edee0202c8fbb4f2b39726ae8e4a66c9d0) OMURGA: test truncated value masks
- [fca770e](https://github.com/Eta06/HarbiChess/commit/fca770ec16a54fd97485b95181dbd078aa52b553) OMURGA: exclude unknown outcomes from value loss
- [6b2c534](https://github.com/Eta06/HarbiChess/commit/6b2c534cc7a71f2f90f44d73ddf055baf6e3a343) OMURGA: test masked learner values
- [8ccb459](https://github.com/Eta06/HarbiChess/commit/8ccb459781c0195a41f9e02b82cd061a7275fc06) OMURGA: preserve unknown arena truncations
- [024fda2](https://github.com/Eta06/HarbiChess/commit/024fda25f7f3fed494c22589168aa459043ad7b8) OMURGA: test truncated arena replay
- [ffb57c0](https://github.com/Eta06/HarbiChess/commit/ffb57c0c7732feb85e0ddebd1b98130e2cca3ddc) OMURGA: expose legacy target compatibility flag
- [781a52e](https://github.com/Eta06/HarbiChess/commit/781a52e6a7cae672f33551ec08c6760899aa555f) OMURGA: test clean learning defaults
- [b61a756](https://github.com/Eta06/HarbiChess/commit/b61a7562aced566b66207c170f117027e38fe866) OMURGA: add exploration-aware policy targets
- [52cdc33](https://github.com/Eta06/HarbiChess/commit/52cdc33311b5feef0b66c3911daa3d62d5d2e72f) OMURGA: test policy target pruning
- [062bbaa](https://github.com/Eta06/HarbiChess/commit/062bbaa0feb7054bb676a6a4eae8602a588440f2) OMURGA: add Gumbel root sequential halving
- [cf6e068](https://github.com/Eta06/HarbiChess/commit/cf6e0684b5c4657866064195030e3554641772d8) OMURGA: test Gumbel search budget
- [6bd55c1](https://github.com/Eta06/HarbiChess/commit/6bd55c1181b11fe31da14a3881310529e2f31249) OMURGA: add stratified teacher qualification
- [f5a13ff](https://github.com/Eta06/HarbiChess/commit/f5a13ff72246b8b242c01eb2e25eee7546d45dec) OMURGA: test teacher qualification metrics
- [72aec79](https://github.com/Eta06/HarbiChess/commit/72aec79b1f7f8c069cc31a3d4083a77ad394d6a5) OMURGA: expose teacher qualification command
- [cf527d6](https://github.com/Eta06/HarbiChess/commit/cf527d6ce01712f340aa86508c7eb85c03da889a) OMURGA: track teacher qualification telemetry
- [c2d3ce7](https://github.com/Eta06/HarbiChess/commit/c2d3ce7e2043b8cbad2abe923285d1d4ad5432ea) OMURGA: test qualification dashboard defaults
- [2067186](https://github.com/Eta06/HarbiChess/commit/2067186a86c4c1e910f2a8e6177b55172fd50963) OMURGA: publish teacher gate telemetry
- [cf0b0d2](https://github.com/Eta06/HarbiChess/commit/cf0b0d2af73928f08225ecd50c8b07ce936dc6d1) OMURGA: test blocked teacher telemetry
- [15d772f](https://github.com/Eta06/HarbiChess/commit/15d772f82b6bac831dce7473670187427d988bf3) OMURGA: type teacher qualification telemetry
- [bb02496](https://github.com/Eta06/HarbiChess/commit/bb02496cefd4ef9d40fd2d6f2c5c79964915f0e3) OMURGA: show teacher qualification gate
- [cffacdd](https://github.com/Eta06/HarbiChess/commit/cffacdd8a006aede9bcc58b240cd5d8f2791905d) OMURGA: style teacher qualification panel
- [a8489e2](https://github.com/Eta06/HarbiChess/commit/a8489e205c80b545120c4d11fd7cc0d402686b31) OMURGA: build teacher qualification dashboard
- [f43a088](https://github.com/Eta06/HarbiChess/commit/f43a0883eabfabc0047622ef80c0397da9441886) OMURGA: report teacher qualification findings
- [db91ffb](https://github.com/Eta06/HarbiChess/commit/db91ffba0ac32f04e58a50220942e70e3b069cc6) OMURGA: add tactical search diagnostics
- [ee851a8](https://github.com/Eta06/HarbiChess/commit/ee851a8534ad6d5194c9cba35299fe19edc4d446) OMURGA: test tactical diagnostic oracles
- [ac41e3f](https://github.com/Eta06/HarbiChess/commit/ac41e3f52b8e05b56562f7a644df5ce5533d9aee) OMURGA: add champion search audit runner
- [4a035bb](https://github.com/Eta06/HarbiChess/commit/4a035bb74d4a92aeeb85915183ae8a07a819a700) OMURGA: test search convention audit
- [c218f5d](https://github.com/Eta06/HarbiChess/commit/c218f5da6726fc71fe059b6a6b5f6725bbbcd54e) OMURGA: expose search diagnostic command
- [6721140](https://github.com/Eta06/HarbiChess/commit/672114044adb460f252120fea0756512ee80e334) OMURGA: compare serial and batched search
- [34a282f](https://github.com/Eta06/HarbiChess/commit/34a282fe2e9fe37538a6db3a56a137d8b710662a) OMURGA: measure replay value calibration
- [394d5ad](https://github.com/Eta06/HarbiChess/commit/394d5ad019f4c1bf5cfb154cc16f762fe4de923b) OMURGA: test verifier value perspective
- [dd7917d](https://github.com/Eta06/HarbiChess/commit/dd7917dcd20bfd887edc3a20cdf307c9810fef92) OMURGA: report search root cause
- [d806232](https://github.com/Eta06/HarbiChess/commit/d8062326e02006e6d83b8d52a33b73b3b9cc866c) OMURGA: add deterministic leaf value oracle
- [27dbeb2](https://github.com/Eta06/HarbiChess/commit/27dbeb2c0185f3fe33921628392d515d55bd0408) OMURGA: test isolated oracle values
- [912ebee](https://github.com/Eta06/HarbiChess/commit/912ebeefce261a546214fa7a2ef27981f65faa31) OMURGA: add frozen value oracle comparison
- [d141f5c](https://github.com/Eta06/HarbiChess/commit/d141f5c1d3a4f0bfeb0a045708eb8f55e4f1dddd) OMURGA: test value oracle diagnostic guardrails
- [0020b55](https://github.com/Eta06/HarbiChess/commit/0020b55e53ba7e01a2bf8f481c64701c26b90979) OMURGA: expose value oracle diagnostic
- [f935b61](https://github.com/Eta06/HarbiChess/commit/f935b6135ab95f15aca73f22116857ece5767945) OMURGA: calibrate oracle material scale
- [fa2a697](https://github.com/Eta06/HarbiChess/commit/fa2a697ab3fb09f129d2a10e8ea6c243da204bfe) OMURGA: freeze oracle material scale
- [6f05ea5](https://github.com/Eta06/HarbiChess/commit/6f05ea5c3a70554c052e560bee4c1b93bf1be435) OMURGA: add value pipeline diagnostic
- [92636a4](https://github.com/Eta06/HarbiChess/commit/92636a480bdd549d950808f166b74f2cff3c94a6) OMURGA: test legacy value target detection
- [f201e99](https://github.com/Eta06/HarbiChess/commit/f201e99568c8798709ed3ba0b4e0424b6699feef) OMURGA: expose value pipeline diagnostic
- [35812af](https://github.com/Eta06/HarbiChess/commit/35812aff8bd0ce22eb63d2af5877f8c3c7d02072) OMURGA: read value audit shard header
- [2135de3](https://github.com/Eta06/HarbiChess/commit/2135de38ffa292dfb7f6d5d64d4f026d3b0b8e4e) OMURGA: stabilize diagnostic log probabilities
- [7e5aaaa](https://github.com/Eta06/HarbiChess/commit/7e5aaaa19cb0e99b1aaaeeaf86132d8af55d996a) OMURGA: guard checkpoints against value regression
- [cf7562a](https://github.com/Eta06/HarbiChess/commit/cf7562ab740b0c08691436fa4f46b1e944c26e65) OMURGA: test value-safe checkpoint selection
- [3adbc7e](https://github.com/Eta06/HarbiChess/commit/3adbc7e0b712c96aad8baa2817e1c564bc1ac85b) OMURGA: report value-safe candidate losses
- [1836cd3](https://github.com/Eta06/HarbiChess/commit/1836cd32de3a3a5dfc9efd850ec2800898f1cfda) OMURGA: add frozen value bootstrap diagnostic
- [cb7c587](https://github.com/Eta06/HarbiChess/commit/cb7c5878cbe1c225293689a4e14c12f686913ad1) OMURGA: test frozen value bootstrap scope
- [ed0e425](https://github.com/Eta06/HarbiChess/commit/ed0e425f361218914b1df0765d92d1336413c120) OMURGA: expose frozen value bootstrap
- [1fb8ea5](https://github.com/Eta06/HarbiChess/commit/1fb8ea50409bcd6b31b8a478469a36071bea9718) OMURGA: qualify diagnostic model overrides
- [a66e32b](https://github.com/Eta06/HarbiChess/commit/a66e32b83c8bab86e6ecf94e8866552eac6da652) OMURGA: integrate qualified bootstrap teacher
- [c6331fb](https://github.com/Eta06/HarbiChess/commit/c6331fbda4576c5e7ed9ac05592169bceafc5c30) OMURGA: test bootstrap teacher opt in
- [234cd11](https://github.com/Eta06/HarbiChess/commit/234cd118e5bcf901196edf8ec946e383c3bb01c3) OMURGA: lower joint learner step size
- [f1dd9d1](https://github.com/Eta06/HarbiChess/commit/f1dd9d118f99b5d38782674b4e9690065b23fbde) OMURGA: test safer learner rate default
- [c55bef7](https://github.com/Eta06/HarbiChess/commit/c55bef7f0f875cee0e60aabee3c885e48b784248) OMURGA: publish bootstrap teacher qualification
- [2895642](https://github.com/Eta06/HarbiChess/commit/28956426575b8d0707d52e53679f227ecbc29f46) OMURGA: test teacher dashboard safety
- [c4d3dab](https://github.com/Eta06/HarbiChess/commit/c4d3dab88f84d2bd4464cc64b78f1042e94ccc7b) OMURGA: reuse boards in tactical oracle
- [2527948](https://github.com/Eta06/HarbiChess/commit/2527948d34a251ad2e21f7132c219a66bbdc3784) OMURGA: require known value validation targets
- [c158e5f](https://github.com/Eta06/HarbiChess/commit/c158e5f214aaef7b4bd7c165f076c764dc57b44d) OMURGA: test empty value target rejection
- [e41b6aa](https://github.com/Eta06/HarbiChess/commit/e41b6aa7834f9176a5c8452c2a99aeaaa4d7c242) OMURGA: report known value sample counts
- [766b924](https://github.com/Eta06/HarbiChess/commit/766b924f55b9df520354005fea249b4b12e19c73) OMURGA: test value-empty run rejection
- [e20cf76](https://github.com/Eta06/HarbiChess/commit/e20cf76fbac8d55cd354f98a9439b49c8ef53827) OMURGA: parallelize tactical oracle processes
- [cb41b45](https://github.com/Eta06/HarbiChess/commit/cb41b452cb6592ab2e2a395e8a38764783079b9d) OMURGA: test process oracle equivalence
- [7085b28](https://github.com/Eta06/HarbiChess/commit/7085b28ad7470d87076caa7e5be049f7f00c250d) OMURGA: route teacher oracle across processes
- [834b8b2](https://github.com/Eta06/HarbiChess/commit/834b8b2014b503979d51e51f7c9cd393dd01429b) OMURGA: test process teacher configuration
- [4bba5c2](https://github.com/Eta06/HarbiChess/commit/4bba5c24ecf18b350a8db76f48a2ad7cc6d33e06) OMURGA: gate candidates on tactical retention
- [bced6ca](https://github.com/Eta06/HarbiChess/commit/bced6ca4bcfdfd83726432f420c7c1a4af940f99) OMURGA: isolate run tests from tactical gate
- [d682700](https://github.com/Eta06/HarbiChess/commit/d682700047c676fd3d29e39569daf81176c51d99) OMURGA: preserve teacher dashboard evidence
- [146a145](https://github.com/Eta06/HarbiChess/commit/146a145648f04afca6575eda1cb6e452ce68680d) OMURGA: test teacher evidence persistence
- [475289a](https://github.com/Eta06/HarbiChess/commit/475289a3d07869e450c612b0f221cde71d794a89) OMURGA: document teacher recovery evidence

## KOPRU

Qualified teacher replay aktarımı, capacity/representation matrisi ve ayrı frozen performans optimizasyonları.

88 commit; 2026-08-28–2026-08-28.

- [db1a543](https://github.com/Eta06/HarbiChess/commit/db1a5436d0d09a8142042b37c17a2f0a45113ead) KOPRU: expose clean network priors from search
- [9c8c84b](https://github.com/Eta06/HarbiChess/commit/9c8c84b1ecd3240f17ab325c7d578e9dcc3a9c22) KOPRU: preserve network priors through root halving
- [2bf03f7](https://github.com/Eta06/HarbiChess/commit/2bf03f76eed3f3ab038dc418a68253768dee819f) KOPRU: preserve network priors through continuation filtering
- [f79746c](https://github.com/Eta06/HarbiChess/commit/f79746cfb2ede90564d6c3bbc2d40b8dcd7f65af) KOPRU: record per-position teacher policy telemetry
- [670dc68](https://github.com/Eta06/HarbiChess/commit/670dc68179aa4dde71910be365fafbc7c39148e0) KOPRU: version replay teacher evidence fields
- [efb0eba](https://github.com/Eta06/HarbiChess/commit/efb0eba823a7940604c6c16ae8ddefa88ae1128c) KOPRU: preserve schema ten max-ply semantics
- [47a73f9](https://github.com/Eta06/HarbiChess/commit/47a73f97f477ab201fd0536f31a4544ff0a7d395) KOPRU: add replay coverage qualification
- [712ab4d](https://github.com/Eta06/HarbiChess/commit/712ab4dfa8b73a719ec21bb85089512d4cbc1b82) KOPRU: add teacher-gated replay-only generation
- [c8fe99b](https://github.com/Eta06/HarbiChess/commit/c8fe99babe4086c1f427f4cb37ff83ba1f5b58bc) KOPRU: test teacher policy telemetry capture
- [14876b7](https://github.com/Eta06/HarbiChess/commit/14876b73744becbbd15320bae5074b2520dfcb04) KOPRU: test replay teacher evidence validation
- [5e72864](https://github.com/Eta06/HarbiChess/commit/5e7286426bed1146dd795ecdbf47457ecf10d7b8) KOPRU: test replay coverage qualification
- [40b10d4](https://github.com/Eta06/HarbiChess/commit/40b10d432fa69e6cddbdf8ed75ef19d6b80ffae0) KOPRU: test replay-only generation gate
- [97ba7e4](https://github.com/Eta06/HarbiChess/commit/97ba7e41e1d0e3c55b56127fd7f95556b3e9af90) KOPRU: preregister replay and learner transfer gates
- [6131fa7](https://github.com/Eta06/HarbiChess/commit/6131fa7a2ff39686731617f2755609130ab28e74) KOPRU: add policy and calibration quality metrics
- [d87c557](https://github.com/Eta06/HarbiChess/commit/d87c5577a2259916fc2c43804240c3ffe445ce48) KOPRU: gate learner transfer before arena
- [5ef3416](https://github.com/Eta06/HarbiChess/commit/5ef3416d06e8935e317d6a1538287f81719c5954) KOPRU: test model quality metrics
- [f0d161b](https://github.com/Eta06/HarbiChess/commit/f0d161b78d227dec40d9671872c9f2653341d1d6) KOPRU: test learner transfer guardrails
- [5e4899a](https://github.com/Eta06/HarbiChess/commit/5e4899a97e4b1166c8c6d2d24edf737806a483b2) KOPRU: handle forced moves in teacher diagnostics
- [7bf7de7](https://github.com/Eta06/HarbiChess/commit/7bf7de788fc97f0d4cd52964a89b2fa24069890a) KOPRU: test forced-move teacher diagnostics
- [4cc4525](https://github.com/Eta06/HarbiChess/commit/4cc45256895600d79de0927115bd38426c3568c9) KOPRU: encode legal policy masks in training batches
- [1b133b3](https://github.com/Eta06/HarbiChess/commit/1b133b39360f948aa379952426ffc4a9d9af8b89) KOPRU: normalize policy loss over legal actions
- [1625323](https://github.com/Eta06/HarbiChess/commit/16253235fb1a9a0df96802d5b7e8b6bcb8986162) KOPRU: report applied and unclipped gradient norms
- [5c3fbc9](https://github.com/Eta06/HarbiChess/commit/5c3fbc992abcdb4b756899945c69f5bac0e4b242) KOPRU: persist unclipped pilot gradient telemetry
- [d4e76f1](https://github.com/Eta06/HarbiChess/commit/d4e76f1bafebdf871bb4010cbd629c6e6b3228f7) KOPRU: persist transfer gradient clipping telemetry
- [3899d37](https://github.com/Eta06/HarbiChess/commit/3899d3716d9c5c54c9b50d81399ae95363b4c1da) KOPRU: separate legal and global policy imitation
- [63b0b16](https://github.com/Eta06/HarbiChess/commit/63b0b162b1589a409ac9ab7b3fcbb8e655af9aed) KOPRU: test legal masks in training batches
- [bfcbee4](https://github.com/Eta06/HarbiChess/commit/bfcbee490669a8eec3ffb9137376c2a700ef4c32) KOPRU: test applied gradient clipping telemetry
- [396904b](https://github.com/Eta06/HarbiChess/commit/396904bb8a3058b337f68921ae501cae0ca05adb) KOPRU: update pilot fixtures for gradient telemetry
- [bd0eb55](https://github.com/Eta06/HarbiChess/commit/bd0eb552242c70747aff79a898a6f2bbb607ea56) KOPRU: test legal policy imitation metric
- [5f625d0](https://github.com/Eta06/HarbiChess/commit/5f625d008ea77b9481aaabd799d91be0459ea121) KOPRU: update transfer quality fixture
- [b2ea3e2](https://github.com/Eta06/HarbiChess/commit/b2ea3e2e3f5ef25594492dca7ac44b8de0e5a1f8) KOPRU: preregister legal policy loss ablation
- [6f7b13b](https://github.com/Eta06/HarbiChess/commit/6f7b13bd247776e9f7aa689d48ba655489e02bbd) KOPRU: evaluate metric-independent validation snapshots
- [016f20c](https://github.com/Eta06/HarbiChess/commit/016f20cf467268fc3e20fcd1302df4e4fc2084d8) KOPRU: test metric-blind checkpoint selection
- [4d247d7](https://github.com/Eta06/HarbiChess/commit/4d247d7b4ac36f7335740457bb25956044568c01) KOPRU: audit stored targets against clean teacher
- [dcf7d2c](https://github.com/Eta06/HarbiChess/commit/dcf7d2c96fe205025dc3f20cc0de0544ea541513) KOPRU: test replay teacher policy distances
- [d906cc5](https://github.com/Eta06/HarbiChess/commit/d906cc5c67dd78501a8fd613bace4cda8f067ce1) KOPRU: compare pruned and noisy teacher targets
- [d2ce2ac](https://github.com/Eta06/HarbiChess/commit/d2ce2ac555652def148236c812542b60d8b31a57) KOPRU: isolate noisy target search configuration
- [172a0f3](https://github.com/Eta06/HarbiChess/commit/172a0f33351de71e98551ecdc9b543dc75b96e02) KOPRU: separate exploration noise from policy targets
- [a4873c8](https://github.com/Eta06/HarbiChess/commit/a4873c8f1bde0c454dc772afbd6e54e0f7b521c5) KOPRU: configure clean-target self-play generation
- [812ae55](https://github.com/Eta06/HarbiChess/commit/812ae55a9584ca03322c6bb88ceb2cf65a1280eb) KOPRU: test clean teacher target selection noise
- [581eb5f](https://github.com/Eta06/HarbiChess/commit/581eb5f6d18677ddc0ab942197b5efae44fdbcd2) KOPRU: test mutually exclusive root noise modes
- [3067ba1](https://github.com/Eta06/HarbiChess/commit/3067ba172e9c039df71cf8c66b897e978c8299a1) KOPRU: preregister clean target sanity replay
- [4b3b84c](https://github.com/Eta06/HarbiChess/commit/4b3b84c331b3c71982ed4b584123feccea7fc065) KOPRU: decouple noisy behavior from clean search targets
- [cf75d80](https://github.com/Eta06/HarbiChess/commit/cf75d8028a6d5519a9e755293689ab4868e3d774) KOPRU: version off-target behavior actions
- [a3eda09](https://github.com/Eta06/HarbiChess/commit/a3eda0994d4cf57726281e5b8428e15539c9bb6a) KOPRU: configure dual-root replay generation
- [36e4714](https://github.com/Eta06/HarbiChess/commit/36e471442b5f9453a8ae4ad9afa6a68a1feed4b4) KOPRU: test dual-root self-play targets
- [1af1a29](https://github.com/Eta06/HarbiChess/commit/1af1a2937337572ebc8e3e89b3df8a85452f6ee6) KOPRU: test decoupled behavior replay records
- [9417f30](https://github.com/Eta06/HarbiChess/commit/9417f309b3bad5e9a2a76921a635e4ef8fd72dca) KOPRU: test dual-root generation configuration
- [725ae1d](https://github.com/Eta06/HarbiChess/commit/725ae1d4d2be8ac2dfefadbcc217538ed9280ee7) KOPRU: preregister dual-search replay sanity
- [ea81eff](https://github.com/Eta06/HarbiChess/commit/ea81eff96771cb59ae17a1db19c8d22f75953a4b) KOPRU: freeze replay teacher alignment gate
- [af4edc8](https://github.com/Eta06/HarbiChess/commit/af4edc87938ff795968da024032e46c34564d5e7) KOPRU: test frozen replay teacher gate
- [705d67b](https://github.com/Eta06/HarbiChess/commit/705d67b0939f9e4fd6836a3bbdffcbf1b8389669) KOPRU: audit alignment after replay gate failure
- [5e96258](https://github.com/Eta06/HarbiChess/commit/5e96258f7e1cd1f92a03f113283b03f1ddfbcb74) KOPRU: test failed replay alignment audit
- [652ebc3](https://github.com/Eta06/HarbiChess/commit/652ebc35d05fa224f1743059f0d6057a8beb73f2) KOPRU: separate replay coverage from teacher strength
- [a070154](https://github.com/Eta06/HarbiChess/commit/a070154263cba516756108d399b8408175dfa6fe) KOPRU: test telemetry-only replay coverage
- [28d9c15](https://github.com/Eta06/HarbiChess/commit/28d9c15b507819510ce3bf09df35bb7f121c693f) KOPRU: update replay coverage run fixture
- [c6fb6e0](https://github.com/Eta06/HarbiChess/commit/c6fb6e0a5f72323082f631ff6010c90b56d3187a) KOPRU: preregister qualified replay transfer
- [479bc25](https://github.com/Eta06/HarbiChess/commit/479bc258daa35642d9fa555bb90bdb1f2732c9d0) KOPRU: require fresh replay alignment before learning
- [ea829e3](https://github.com/Eta06/HarbiChess/commit/ea829e3d71f6d323fb71180b1af3566b6c7cff50) KOPRU: test replay alignment learner gate
- [7c13ee0](https://github.com/Eta06/HarbiChess/commit/7c13ee00b48702fd7e8d0d5dc4bdcc2887aa21e4) KOPRU: serialize learner alignment provenance
- [84484e8](https://github.com/Eta06/HarbiChess/commit/84484e8a682fc796fecfaeae431ffc4bf7ce722c) KOPRU: test learner config serialization
- [d5339f8](https://github.com/Eta06/HarbiChess/commit/d5339f81e73896bf1a2baa6937dbb654f2691500) KOPRU: gate teacher top-action imitation
- [27c306f](https://github.com/Eta06/HarbiChess/commit/27c306fd5f60f88884a837ecd142ac797f5bf473) KOPRU: test teacher top-action transfer gate
- [9ce3e1e](https://github.com/Eta06/HarbiChess/commit/9ce3e1ee410edb15d6ec0fe601b29d4ff2ebc729) KOPRU: benchmark oracle search worker allocation
- [ed44a1a](https://github.com/Eta06/HarbiChess/commit/ed44a1a386406d1cf4493b76023bc71391fd6144) KOPRU: test oracle worker benchmark configuration
- [86bed28](https://github.com/Eta06/HarbiChess/commit/86bed28fb12a8e23acd87682feb9131e92435be5) KOPRU: cache model quality validation encoding
- [4860e90](https://github.com/Eta06/HarbiChess/commit/4860e903574682efb3897e31d78397ea951df134) KOPRU: test cached model quality equivalence
- [17f35ed](https://github.com/Eta06/HarbiChess/commit/17f35ed0d86236c747cbcdb7eb80c167aea15471) KOPRU: reuse validation encoding across learner gates
- [c4e708a](https://github.com/Eta06/HarbiChess/commit/c4e708ab1c43ccd73db12fc1887b509412eed40b) KOPRU: preregister confidence-aware policy transfer
- [0c865bb](https://github.com/Eta06/HarbiChess/commit/0c865bb8e8c8d072c8caf553378cf9fc5df98b4c) KOPRU: add confidence-aware top-action loss
- [a242a61](https://github.com/Eta06/HarbiChess/commit/a242a61294ee8ab7b37bd0c389df592cf83ea09a) KOPRU: test confidence-aware policy loss
- [e3e1ee9](https://github.com/Eta06/HarbiChess/commit/e3e1ee946abe44199ee215c7e2fe0f3c2b0edeb7) KOPRU: validate replay against reconstructed board
- [4caa98f](https://github.com/Eta06/HarbiChess/commit/4caa98ff7e35b018a5a605705af029b3cb227d5d) KOPRU: reuse board during batch preparation
- [437d0be](https://github.com/Eta06/HarbiChess/commit/437d0be360f5dcde8c755319747c4254dddf637a) KOPRU: report performance and transfer bottlenecks
- [cab9747](https://github.com/Eta06/HarbiChess/commit/cab9747d79ef98bf45422bfdbae360a5de2fb465) KOPRU: preregister capacity transfer matrix
- [67eac2e](https://github.com/Eta06/HarbiChess/commit/67eac2ef562ca2b374c3591a284d6d0fe8017a11) KOPRU: preserve logits during network expansion
- [47c4f74](https://github.com/Eta06/HarbiChess/commit/47c4f7462b2dcd8d7975971223a7427577cb0a70) KOPRU: test function preserving expansion
- [cfa280d](https://github.com/Eta06/HarbiChess/commit/cfa280d4dd40b2825e980d4fd1ad1507159a48f8) KOPRU: add frozen capacity transfer matrix
- [494c93b](https://github.com/Eta06/HarbiChess/commit/494c93b7827198b055ceb22e1fa03ef972a30e74) KOPRU: test capacity transfer gates
- [3d11048](https://github.com/Eta06/HarbiChess/commit/3d11048afcaebf7a3c052a9e9fc018c84db9093f) KOPRU: wire tactical rules in capacity matrix
- [f313004](https://github.com/Eta06/HarbiChess/commit/f313004cf8f0b75287eb1f762fb7f26fc915e565) KOPRU: smoke test capacity tactical gate
- [e56e22f](https://github.com/Eta06/HarbiChess/commit/e56e22feb0557c246aca150c16ec737b58a38450) KOPRU: add sparse legal policy head
- [0f10bbe](https://github.com/Eta06/HarbiChess/commit/0f10bbe915320d8c40c0e4a326a0aed44b9a397e) KOPRU: test sparse policy equivalence
- [9bd6fd0](https://github.com/Eta06/HarbiChess/commit/9bd6fd012669db3a706d336df5d98d134f763105) KOPRU: reuse cached search transitions
- [3ea943f](https://github.com/Eta06/HarbiChess/commit/3ea943fc6b3e8b592ef13a266a05521f32d9d6ae) KOPRU: test cached transition reuse
- [feeb4b8](https://github.com/Eta06/HarbiChess/commit/feeb4b8835aa1707a0934f6a414ffcdca5b050f8) KOPRU: expose search scheduling benchmark controls
- [3663079](https://github.com/Eta06/HarbiChess/commit/366307933edda6537a5c7391bd211f8e51b08a0e) KOPRU: test search benchmark scheduling options
- [663b7b8](https://github.com/Eta06/HarbiChess/commit/663b7b83189d72bf65e589f53a8c2b5f5b2b5dc6) KOPRU: report capacity and throughput intersection

## MIHENK

Farklı search budget'larında teacher policy/Q tutarlılığı ve güven sınıfları.

4 commit; 2026-08-28–2026-08-28.

- [a1e7dbb](https://github.com/Eta06/HarbiChess/commit/a1e7dbb4e78b969f17bbfc9e77b88f96c145b96a) MIHENK: preregister teacher consistency audit
- [d04ea32](https://github.com/Eta06/HarbiChess/commit/d04ea32474314d63009fe02c03ac280b6d2f2bd7) MIHENK: add cross budget teacher consistency audit
- [8fa3f6b](https://github.com/Eta06/HarbiChess/commit/8fa3f6b37ad10f63afe03488ec892e8c4666334c) MIHENK: test teacher consistency classification
- [29aacc4](https://github.com/Eta06/HarbiChess/commit/29aacc40278b5db6eb29bc48fa9a3b38b22ca8d4) MIHENK: report teacher consistency gate

## DENGE

Birden fazla kullanımı var: erken uncertainty-preserving consensus target; daha sonra calibration ve fresh cumulative non-inferiority pilotları.

27 commit; 2026-08-28–2026-09-02.

- [cf12a15](https://github.com/Eta06/HarbiChess/commit/cf12a15c903b9562e3ab9dfb47e4145d78d291d0) DENGE: preregister uncertainty preserving target audit
- [f4a15ef](https://github.com/Eta06/HarbiChess/commit/f4a15ef5f67b83217f82f4550b11ee8be7dd02ca) DENGE: add verified consensus target audit
- [6c62f46](https://github.com/Eta06/HarbiChess/commit/6c62f46d5ed507d2f8c4d3a275737300a286d2d4) DENGE: test consensus target qualification
- [1e3b85a](https://github.com/Eta06/HarbiChess/commit/1e3b85acc8a516cf15c84ea99d2438c2a8dc1a43) DENGE: report consensus target gate
- [6b320bd](https://github.com/Eta06/HarbiChess/commit/6b320bd4e098be233d5c93c1ccf1d12b58bd13ba) DENGE: preregister value calibration ablation
- [4a72722](https://github.com/Eta06/HarbiChess/commit/4a72722201f345bfa1a95e4971bd2d4f2c0d433b) DENGE: add value calibration hook
- [8465587](https://github.com/Eta06/HarbiChess/commit/84655876eed9eee58416bbcc9ab98f31e2aa10b7) DENGE: add scalar value temperature
- [7b2643d](https://github.com/Eta06/HarbiChess/commit/7b2643d13077534c009554f0064b4527a37897ec) DENGE: fit game-balanced value temperature
- [76b0367](https://github.com/Eta06/HarbiChess/commit/76b036728cde6e71ef20717d6d48520478fc339e) DENGE: test value temperature isolation
- [c91d9af](https://github.com/Eta06/HarbiChess/commit/c91d9af146ea24746228aa31259492fd1dffefbe) DENGE: test scalar calibration fitting
- [0f51682](https://github.com/Eta06/HarbiChess/commit/0f51682d042d832ab813e5275cd99601fdabeb08) DENGE: add frozen calibration diagnostic
- [f574e12](https://github.com/Eta06/HarbiChess/commit/f574e1297919f30d66c85e3366a4b96064925c75) DENGE: test calibration diagnostic guards
- [1910699](https://github.com/Eta06/HarbiChess/commit/19106996e0eeef3fa716ab1f45ae1b464df86a20) DENGE: preregister constrained calibration
- [9f92ee0](https://github.com/Eta06/HarbiChess/commit/9f92ee0856b305a6b0c16ff3188a2b9447d24d08) DENGE: constrain calibration by old Pearson
- [8471f43](https://github.com/Eta06/HarbiChess/commit/8471f43af8097a4c21a029499891fafc79a717f5) DENGE: add guarded calibration diagnostic
- [2fcd452](https://github.com/Eta06/HarbiChess/commit/2fcd452bfb52a7956bf46c464f3a7228ca7e4a92) DENGE: test Pearson constrained calibration
- [6eca75c](https://github.com/Eta06/HarbiChess/commit/6eca75c052e45e7ddf7c3791e02454fcf355db9f) DENGE: calibrate rolling value updates
- [59b48ca](https://github.com/Eta06/HarbiChess/commit/59b48caf7bcec56cc69de0e1f08bac2f5d7a5391) DENGE: test calibration replay isolation
- [00c43fb](https://github.com/Eta06/HarbiChess/commit/00c43fb46eb2ca241767c5f269159cc880d1e961) DENGE: test frozen temperature fitting
- [bd59756](https://github.com/Eta06/HarbiChess/commit/bd5975637a6f8406623a9bf6b79abd30f78498b6) DENGE: preregister fresh cumulative pilot
- [4779ca2](https://github.com/Eta06/HarbiChess/commit/4779ca2e851b2386afc29bd6b570a36c8e06cee4) DENGE: preregister game-paired continuation gate
- [7f89f56](https://github.com/Eta06/HarbiChess/commit/7f89f5658126629340d483a84471f8bd0c71ec92) DENGE: expose continuation game identity
- [a302954](https://github.com/Eta06/HarbiChess/commit/a302954f6f5704d3342ab54acff25c5227231244) DENGE: enforce game-paired continuation evidence
- [b4839e5](https://github.com/Eta06/HarbiChess/commit/b4839e5f7c8d7f25a173a90a40a13b32048acf30) DENGE: test game-paired continuation gate
- [d72ac70](https://github.com/Eta06/HarbiChess/commit/d72ac7055b8925c588745e056b8ae637243b6f78) DENGE: record failed cumulative pilot
- [e4ca66d](https://github.com/Eta06/HarbiChess/commit/e4ca66d0c6b3cda4ab9c98e10d2e5b68ee6113df) DENGE: add continuation drift audit
- [3804f19](https://github.com/Eta06/HarbiChess/commit/3804f197665c131b0d7e0e52de2d18e14ed9c5c2) DENGE: test continuation drift audit

## UYUM

Ortak yönde hareket eden belirsizlik koruyan target audit'i.

4 commit; 2026-08-28–2026-08-28.

- [483aa82](https://github.com/Eta06/HarbiChess/commit/483aa823e8f5bdba628f00375cec8ddc6507012e) UYUM: preregister common direction target audit
- [015e5df](https://github.com/Eta06/HarbiChess/commit/015e5df665a7eaf4ed420f5901fc69a453edd113) UYUM: add common direction target audit
- [21dee7d](https://github.com/Eta06/HarbiChess/commit/21dee7d4978a45761c56f0a5595e507637f9ce6c) UYUM: test common direction target gate
- [6717902](https://github.com/Eta06/HarbiChess/commit/6717902cb0334385779f3e260e871677fd6afbc9) UYUM: report common direction target gate

## TERAZI

Search Q güvenilirliğini bağımsız verifier ile karşılaştırma.

4 commit; 2026-08-28–2026-08-28.

- [c0deb8f](https://github.com/Eta06/HarbiChess/commit/c0deb8f17f8376cb2e8be92e29aef5935a0fb7ec) TERAZI: preregister search Q reliability audit
- [1b0cac4](https://github.com/Eta06/HarbiChess/commit/1b0cac4bf7d31fddfc4f2a8ae9eb1cd18ad4b698) TERAZI: add search Q reliability audit
- [1494d7b](https://github.com/Eta06/HarbiChess/commit/1494d7be4674ef94f5c64eda67de2877fdda44c6) TERAZI: test search Q reliability gate
- [9224d91](https://github.com/Eta06/HarbiChess/commit/9224d91dd4222321697a786d5ea8efb15782df56) TERAZI: report search Q reliability gate

## ODAK

Tam sabit bütçeli root allocation ve sequential halving qualification.

7 commit; 2026-08-28–2026-08-28.

- [2b051e4](https://github.com/Eta06/HarbiChess/commit/2b051e418dbd6773ba7d38aceb1a5eb910cae8e6) ODAK: preregister fixed budget root allocation
- [719d800](https://github.com/Eta06/HarbiChess/commit/719d800e9da0261b5553b33a918f479ef04af187) ODAK: clarify terminal search budget accounting
- [106a4c3](https://github.com/Eta06/HarbiChess/commit/106a4c31a6c70c9418d80752d5d5a3fbd0e0e96f) ODAK: add exact budget sequential halving
- [23115f2](https://github.com/Eta06/HarbiChess/commit/23115f2e990461b8d947e5913edee04b56ec8c00) ODAK: test sequential halving allocation
- [6559993](https://github.com/Eta06/HarbiChess/commit/65599939db874118bda68bbda1fdf411653e158a) ODAK: add root allocation qualification
- [36826cc](https://github.com/Eta06/HarbiChess/commit/36826cce6fc543e5655765c6abd73c02973f1389) ODAK: test root allocation qualification gate
- [4a89999](https://github.com/Eta06/HarbiChess/commit/4a89999dc2c48d5e01179b47df5a49f2187d5904) ODAK: report root allocation qualification

## DEGER

Policy'yi başlangıçta koruyan action-value head ve transfer.

6 commit; 2026-08-28–2026-08-28.

- [b93b0f2](https://github.com/Eta06/HarbiChess/commit/b93b0f25c0f28377f446482bac0110265aa39a43) DEGER: preregister action value transfer
- [3ad6ef5](https://github.com/Eta06/HarbiChess/commit/3ad6ef5994545e1001c03db76750f393dd5600e1) DEGER: add function preserving action value head
- [6f9f314](https://github.com/Eta06/HarbiChess/commit/6f9f314d333a0f5b4c90286ce69cbf7fdfe0ce34) DEGER: test action value expansion invariance
- [96eb79a](https://github.com/Eta06/HarbiChess/commit/96eb79a29101ef2e7b3619d4221e31f99309a839) DEGER: add frozen action value transfer
- [f83418b](https://github.com/Eta06/HarbiChess/commit/f83418b9dcf79f2de1686a6062bda1d4c5ace474) DEGER: test isolated action value learning
- [3fffa8f](https://github.com/Eta06/HarbiChess/commit/3fffa8f7af39808c24811c759363cdac68eab90d) DEGER: report action value transfer failure

## DOKU

Fresh spatial action-value veri/etiket qualification.

4 commit; 2026-08-28–2026-08-28.

- [abaf60c](https://github.com/Eta06/HarbiChess/commit/abaf60cea85928432ceb9942c7066fd5d8b9ef0a) DOKU: preregister spatial action value transfer
- [4579a46](https://github.com/Eta06/HarbiChess/commit/4579a46fc310ace7e1e9ecbc982ad9f17747e771) DOKU: add fresh action value dataset audit
- [bf56220](https://github.com/Eta06/HarbiChess/commit/bf562209c9d94e506094160a10b149e204b9e979) DOKU: test fresh Q label qualification
- [551ad77](https://github.com/Eta06/HarbiChess/commit/551ad775bfad76864fb4eb4c5b3811a4c577a1df) DOKU: report fresh Q label gate

## OLCEK

Belirsizlik ağırlıklı spatial Q representation ve transfer.

12 commit; 2026-08-28–2026-08-28.

- [6548c23](https://github.com/Eta06/HarbiChess/commit/6548c23d0a4bb7ebf4a5bbb0892e7c8fbc9f2059) OLCEK: preregister uncertainty weighted Q transfer
- [ed9cc9c](https://github.com/Eta06/HarbiChess/commit/ed9cc9c89f9536e951b84d0390a7362937db199c) OLCEK: add spatial action value head
- [7320e0f](https://github.com/Eta06/HarbiChess/commit/7320e0f67ffebd2f9293ab7c63c9d0fc3f1c7852) OLCEK: test spatial action value invariance
- [3d0ab0c](https://github.com/Eta06/HarbiChess/commit/3d0ab0c4885f79c1e6fb44172cce043fbbc751b9) OLCEK: preserve verifier labels across exclusions
- [e374e25](https://github.com/Eta06/HarbiChess/commit/e374e257b0df6b75f74f13da8348a10b6ce008d8) OLCEK: test multiple dataset exclusions
- [d879ace](https://github.com/Eta06/HarbiChess/commit/d879ace42edb56f2e14574297d9b54a481603310) OLCEK: add uncertainty weighted Q labels
- [e160d1f](https://github.com/Eta06/HarbiChess/commit/e160d1fe2803ba3e3a1491580dfa1560e29c8f59) OLCEK: test Q uncertainty qualification
- [a27e478](https://github.com/Eta06/HarbiChess/commit/a27e478aaceef05bbf7b3dd7077a928edcf79907) OLCEK: add spatial Q transfer run
- [502c2ec](https://github.com/Eta06/HarbiChess/commit/502c2ec8146a6911cca3a6d566119abf1ba60748) OLCEK: test spatial Q transfer mapping
- [dc25a81](https://github.com/Eta06/HarbiChess/commit/dc25a819613f203a7ed2c0f0c39c1ef48836cad6) OLCEK: align Q correlation with supervised support
- [438d251](https://github.com/Eta06/HarbiChess/commit/438d2513a41c809a4408509798517f8b395ffbdd) OLCEK: test zero weight label exclusion
- [e49f944](https://github.com/Eta06/HarbiChess/commit/e49f94466beaf4e86f6d115f2bd2ae1498517541) OLCEK: report spatial Q transfer failure

## BAG

Hamlenin başlangıç/hedef kareleriyle koşullanan action-value head.

10 commit; 2026-08-28–2026-08-28.

- [ae7e6c1](https://github.com/Eta06/HarbiChess/commit/ae7e6c18e2fb30dcb66e4dbcc5e84869a657d609) BAG: preregister move conditioned Q transfer
- [e0c5b70](https://github.com/Eta06/HarbiChess/commit/e0c5b70af91369fd7399f583114af33ff00dbf1d) BAG: expose canonical action destinations
- [203507a](https://github.com/Eta06/HarbiChess/commit/203507a53273935b8bf6d5965b61b4a61501ab09) BAG: test action destination geometry
- [11832c8](https://github.com/Eta06/HarbiChess/commit/11832c8b856202210df829f8f21d293a2ad737f5) BAG: add move conditioned action value head
- [2427983](https://github.com/Eta06/HarbiChess/commit/242798360ddf81f2a1057e2aa958cdc4339610a5) BAG: test move conditioned Q invariance
- [244c087](https://github.com/Eta06/HarbiChess/commit/244c08724a4fc888386cdb72b05292fa1e243e6b) BAG: expose fresh dataset seed
- [951034d](https://github.com/Eta06/HarbiChess/commit/951034d7adc295ba330ce9d4c38877f7e95074ff) BAG: expose uncertainty bootstrap seed
- [f163f02](https://github.com/Eta06/HarbiChess/commit/f163f02b37ea72f2755241aa80a1a0e204adf6b5) BAG: run selectable action value transfer
- [039237b](https://github.com/Eta06/HarbiChess/commit/039237b80f3f51fecc34fb33a7f793d5091aa767) BAG: test selectable Q architecture
- [4fc3ad7](https://github.com/Eta06/HarbiChess/commit/4fc3ad7308ca8434b282a138b797a29cb2a3f56a) BAG: record move conditioned Q failure

## AKIS

Uncertainty policy transferi ve train/validation overfit teşhisi.

5 commit; 2026-08-28–2026-08-28.

- [9505532](https://github.com/Eta06/HarbiChess/commit/9505532ad4a47e0eb3185d07af0e2b2c3ac5b468) AKIS: preregister uncertainty policy transfer
- [03befea](https://github.com/Eta06/HarbiChess/commit/03befea199bfb14ba385ae01db91f68febff60d1) AKIS: add uncertainty policy transfer
- [edcdffb](https://github.com/Eta06/HarbiChess/commit/edcdffb5c54826bb6d386b1494ee1d217d41d57e) AKIS: test uncertainty policy transfer
- [18ba761](https://github.com/Eta06/HarbiChess/commit/18ba76135b875772be39fb047a369a53ce0251bd) AKIS: measure train side policy transfer
- [961966c](https://github.com/Eta06/HarbiChess/commit/961966c5f902e57b999d4360e73347642eae2b1b) AKIS: record policy transfer overfit

## VERI

Teacher coverage ve veri ölçeği; target semantics audit'i.

6 commit; 2026-08-28–2026-08-28.

- [5e0742e](https://github.com/Eta06/HarbiChess/commit/5e0742e6ad2d09b2cc9a38718047db1c076efcd3) VERI: preregister teacher coverage transfer
- [bb4f32c](https://github.com/Eta06/HarbiChess/commit/bb4f32c990ad47b7684b227510964c358d71f110) VERI: expose teacher dataset scale
- [1b59ceb](https://github.com/Eta06/HarbiChess/commit/1b59ceb1a678aa74374d74f62a199504dcc92c9f) VERI: quarantine unlabelable Q rows
- [a6f8a8c](https://github.com/Eta06/HarbiChess/commit/a6f8a8c69cf6c93f25fea3142d2bde8ab7a47230) VERI: test uncertainty coverage gate
- [8af17c9](https://github.com/Eta06/HarbiChess/commit/8af17c934455bc3f4c455b5d44d39d1cae9c06f0) VERI: expose policy transfer seed
- [b693770](https://github.com/Eta06/HarbiChess/commit/b693770852c95fc1c773d8b96efc862b4e59f430) VERI: record target semantics failure

## KILAVUZ

Q rehberli, KL kısıtlı policy improvement target.

4 commit; 2026-08-28–2026-08-28.

- [d300243](https://github.com/Eta06/HarbiChess/commit/d300243e9f53d5834a6587c41401253b5d5ee655) KILAVUZ: preregister Q guided policy target
- [6562067](https://github.com/Eta06/HarbiChess/commit/6562067afee7355d29ceb28700dde4fca25243be) KILAVUZ: build KL constrained policy targets
- [a9d2368](https://github.com/Eta06/HarbiChess/commit/a9d23686707391d6176e08d8cd5f6896388d94b6) KILAVUZ: test policy improvement target
- [1bfa956](https://github.com/Eta06/HarbiChess/commit/1bfa9563f61b1f6fd91ef8a02fb8d14615974a1a) KILAVUZ: record policy target rejection

## SIPER

Daha muhafazakâr Q policy target ve learner transfer kontrolü.

6 commit; 2026-08-28–2026-08-28.

- [8c3899d](https://github.com/Eta06/HarbiChess/commit/8c3899d443e13287e03036cfb80711526ea79f41) SIPER: preregister conservative policy target
- [476e88d](https://github.com/Eta06/HarbiChess/commit/476e88d7853114bed5a524b227acdacba798a5fb) SIPER: add conservative Q target mode
- [caab547](https://github.com/Eta06/HarbiChess/commit/caab5471aea467598513bd192f870a8813c2b86a) SIPER: test conservative target mode
- [f22b5f3](https://github.com/Eta06/HarbiChess/commit/f22b5f35e5c6e35e7499ee0fbcad960807c101f0) SIPER: train explicit policy improvement targets
- [6ebf386](https://github.com/Eta06/HarbiChess/commit/6ebf38652fd6c04b445e76ce4031f31443acccc4) SIPER: test explicit policy targets
- [dbecdfe](https://github.com/Eta06/HarbiChess/commit/dbecdfe14514b65d899fda560acb7dc924bc885e) SIPER: record qualified target transfer failure

## AKTARIM

Train-only fit matrisi ve consensus transfer turunun özeti.

5 commit; 2026-08-28–2026-08-28.

- [d8a9bcc](https://github.com/Eta06/HarbiChess/commit/d8a9bccfa7c8e2b933a265bcf53c70d25cd26086) AKTARIM: preregister train fit matrix
- [047316f](https://github.com/Eta06/HarbiChess/commit/047316fd14e6f6952a1a29364ffde3a52600fb0e) AKTARIM: add train only fit diagnostic
- [0b7e5bb](https://github.com/Eta06/HarbiChess/commit/0b7e5bb850b498272f9f509efdb978e83ba35826) AKTARIM: report failed fit diagnostics
- [06fa099](https://github.com/Eta06/HarbiChess/commit/06fa099b61ba342cfd6eff140867a9859abffd7d) AKTARIM: record train fit matrix
- [f05780b](https://github.com/Eta06/HarbiChess/commit/f05780b3d8fde54d3b690dc0f497ae5e31c4bf19) AKTARIM: summarize consensus transfer round

## FREN

Policy delta küçültme/projection ile davranışı koruma deneyi.

6 commit; 2026-08-28–2026-08-28.

- [82263b6](https://github.com/Eta06/HarbiChess/commit/82263b634739a8568748cb26f3c05ec1af7de99d) FREN: preregister policy delta projection
- [74f3a6c](https://github.com/Eta06/HarbiChess/commit/74f3a6cb0d97909fa48cc937defcb8d442f44d31) FREN: support scaled policy deltas
- [bcf04a0](https://github.com/Eta06/HarbiChess/commit/bcf04a0fa83365fba908b12c0d6afc257bd660d4) FREN: test scaled policy deltas
- [e17e334](https://github.com/Eta06/HarbiChess/commit/e17e334833644010be51572030ca86c26ae769ed) FREN: add train safe policy projection
- [a34c5c7](https://github.com/Eta06/HarbiChess/commit/a34c5c74614f4c28cd04b7d9374fbb0b4b40ae92) FREN: test policy projection gates
- [03787f5](https://github.com/Eta06/HarbiChess/commit/03787f570831b4312d460d0dbffb1c4341efb4ee) FREN: record policy projection failure

## YAPI

Spatial policy adapter mimarisi.

7 commit; 2026-08-28–2026-08-28.

- [5557570](https://github.com/Eta06/HarbiChess/commit/555757001d5a599426604e40561d97d53c2763a5) YAPI: preregister spatial policy adapter
- [9f20b34](https://github.com/Eta06/HarbiChess/commit/9f20b3465d1d86f290830b6004fa86dbf8ecc5eb) YAPI: add spatial policy adapter network
- [9e2f9b4](https://github.com/Eta06/HarbiChess/commit/9e2f9b4295c5f79a4d5ed884850639f44bdd6457) YAPI: test spatial policy invariance
- [016f6df](https://github.com/Eta06/HarbiChess/commit/016f6df26b3ff453feef3c9753fc87b769801204) YAPI: expose shared policy quality audit
- [accfdf6](https://github.com/Eta06/HarbiChess/commit/accfdf693c8e9cb087e515599c6e8ef11ae2f90b) YAPI: add spatial policy transfer
- [ac41efd](https://github.com/Eta06/HarbiChess/commit/ac41efde01dd8cc8ca399eba67ebab468d3d136d) YAPI: test spatial policy transfer
- [29745c8](https://github.com/Eta06/HarbiChess/commit/29745c803ee1c5abcc12e7cdafed84e0a60dc904) YAPI: record spatial policy failure

## BAGLANTI

Relational policy adapter ve ortak temsil bağlantısı.

8 commit; 2026-08-28–2026-08-28.

- [4b97a4c](https://github.com/Eta06/HarbiChess/commit/4b97a4c3ff3f3137bc6a3e87231c529fb57b9ebf) BAGLANTI: preregister relational policy adapter
- [d9ad041](https://github.com/Eta06/HarbiChess/commit/d9ad0411cfbaf77e52ca56c9049910b7f78593dd) BAGLANTI: add relational policy adapter
- [c390263](https://github.com/Eta06/HarbiChess/commit/c3902637dd1d3bf3799b30c97c14e933c81b9a30) BAGLANTI: test relational policy invariance
- [c38e807](https://github.com/Eta06/HarbiChess/commit/c38e807ee8e7f1aeea659e72c70ea4c5a235a585) BAGLANTI: run relational policy transfer
- [17bef78](https://github.com/Eta06/HarbiChess/commit/17bef7806acd84d58397d1500c157ad74c5c035a) BAGLANTI: test relational transfer mode
- [e515cc7](https://github.com/Eta06/HarbiChess/commit/e515cc7b452ae348ad089267cb627672080be0f2) BAGLANTI: bind shared policy adapter
- [4368d5c](https://github.com/Eta06/HarbiChess/commit/4368d5c81e9bdfed5e7aeddbc1287960586b7ea4) BAGLANTI: test shared adapter learner
- [2256e96](https://github.com/Eta06/HarbiChess/commit/2256e9643f1b3362ae0c600c593ba280ec4c1d5e) BAGLANTI: record relational policy failure

## YAKINSAMA

Policy fitting/convergence teşhisi; train uyumu genelleme kanıtı sayılmaz.

4 commit; 2026-08-28–2026-08-28.

- [946bf4f](https://github.com/Eta06/HarbiChess/commit/946bf4f52e40cb1935338551b58e54c6cd4d4cc8) YAKINSAMA: preregister policy convergence
- [f64289e](https://github.com/Eta06/HarbiChess/commit/f64289ed4aab3182f66e154a532c17c4b683da23) YAKINSAMA: add policy convergence diagnostic
- [333b2b1](https://github.com/Eta06/HarbiChess/commit/333b2b17bf449b28d8972d5b00f573565d459bce) YAKINSAMA: test convergence checkpoints
- [f1246f9](https://github.com/Eta06/HarbiChess/commit/f1246f9f6392c5fdb80c45e5387a7882fda7ed08) YAKINSAMA: record qualified convergence

## SINAV

Fresh policy validation ve teacher genelleme kontrolü.

2 commit; 2026-08-28–2026-08-28.

- [c849411](https://github.com/Eta06/HarbiChess/commit/c8494116682e9c3f757f9908a720568e52f47ab7) SINAV: preregister fresh policy validation
- [daf0b2a](https://github.com/Eta06/HarbiChess/commit/daf0b2a1fabc14ec405859370cc2b75e032af439) SINAV: record fresh teacher failure

## TESHIS

Teacher instability segmentleri ve düz value tahminleri teşhisi.

3 commit; 2026-08-28–2026-08-28.

- [1ef491c](https://github.com/Eta06/HarbiChess/commit/1ef491c9e6e078d920a22ec0cf10c59aab2a59f7) TESHIS: add teacher instability segments
- [23903ff](https://github.com/Eta06/HarbiChess/commit/23903ffea2d1218f5a4a3cdbbb4933a45f5e35b2) TESHIS: test teacher instability summary
- [14c9bdf](https://github.com/Eta06/HarbiChess/commit/14c9bdf0714234981990a2e017e959b9cbf27e44) TESHIS: record flat value instability

## MARJ

Anlamlı action-value farkı taşıyan çiftlerde teacher gate.

3 commit; 2026-08-28–2026-08-28.

- [96033c1](https://github.com/Eta06/HarbiChess/commit/96033c1cc32d07a1d355d8cfe64241375434b5fc) MARJ: preregister decisive pair teacher gate
- [fb3a5dc](https://github.com/Eta06/HarbiChess/commit/fb3a5dc71599dc1d0fbf69fdc28cf1b755e921ff) MARJ: add decisive pair teacher gate
- [c354d0e](https://github.com/Eta06/HarbiChess/commit/c354d0ed81589d1f9bba5b4848fa8febdc579d48) MARJ: test decisive pair qualification

## CIPA

Eski replay'e anchor uygulanarak transferi koruma.

4 commit; 2026-08-28–2026-08-28.

- [f0c349d](https://github.com/Eta06/HarbiChess/commit/f0c349d96a83672f58382d3f67fa78759604aa4f) CIPA: preregister replay anchored transfer
- [7b6de2a](https://github.com/Eta06/HarbiChess/commit/7b6de2a05c7bc166bf70a9740d63e15cf4ecac86) CIPA: add replay anchored transfer
- [55fb704](https://github.com/Eta06/HarbiChess/commit/55fb704a9be6be54e185017d7ecbc5b9cb4610e9) CIPA: test replay anchor contract
- [63e25a6](https://github.com/Eta06/HarbiChess/commit/63e25a66516ce9d3ed5af72e70d49fa6d6c978b5) CIPA: record anchored transfer failure

## KOK

Policy/value ortak representation transferi.

6 commit; 2026-08-28–2026-08-28.

- [2436ec8](https://github.com/Eta06/HarbiChess/commit/2436ec83e3fc30aa899704941de65274a1d1fc9f) KOK: preregister joint representation transfer
- [d7e3789](https://github.com/Eta06/HarbiChess/commit/d7e3789816448e14c86102865363713201e013e0) KOK: add joint representation transfer
- [99cbab0](https://github.com/Eta06/HarbiChess/commit/99cbab0033c9e8d14dd8049a5360357a5e7bb98f) KOK: test joint transfer contract
- [87e9acf](https://github.com/Eta06/HarbiChess/commit/87e9acfe54e2ffe42566b1498ceedb9101dc90c2) KOK: select joint learner board inputs
- [ee244d1](https://github.com/Eta06/HarbiChess/commit/ee244d15c3595f1bee3019f6ffa5e1a8d1408055) KOK: test joint board input selection
- [c6201af](https://github.com/Eta06/HarbiChess/commit/c6201af9c02ccaf963266b87f157205b81abd9cd) KOK: record joint transfer failure

## HACIM

Genişletilmiş yüksek bütçeli teacher etiketi ve güven kontrolü.

4 commit; 2026-08-28–2026-08-28.

- [7e1e9ba](https://github.com/Eta06/HarbiChess/commit/7e1e9ba010009d21d999a22f85e6bbbecee309b9) HACIM: preregister expanded high budget labels
- [fa6e32e](https://github.com/Eta06/HarbiChess/commit/fa6e32eb1d4d735ae5056b1eb5bad9eacb326353) HACIM: expose label audit on dashboard
- [9b40c48](https://github.com/Eta06/HarbiChess/commit/9b40c4861db519b974124521cd484b8db7787906) HACIM: test label telemetry default
- [9118d81](https://github.com/Eta06/HarbiChess/commit/9118d81b3827fb5fb880c43eb6d04f1c06cfae84) HACIM: record expanded teacher failure

## ESAS

Referans sistemlerle mekanizma audit'i; sistem düzeyinde teacher qualification, FPU ve Full Gumbel.

25 commit; 2026-08-28–2026-08-28.

- [273060f](https://github.com/Eta06/HarbiChess/commit/273060fe81fde4d2a184bc2df06b716d42b63bed) ESAS: audit policy iteration mechanisms
- [8b4fba3](https://github.com/Eta06/HarbiChess/commit/8b4fba391a1f6ed996cba15940c5d72366782792) ESAS: preregister system teacher qualification
- [24fabc6](https://github.com/Eta06/HarbiChess/commit/24fabc666e1fc0592c174be318e4d46c8301673c) ESAS: add system teacher qualification
- [dd3f806](https://github.com/Eta06/HarbiChess/commit/dd3f80610e0508fc1a2584474f83018179d76c6c) ESAS: test system teacher qualification
- [8face73](https://github.com/Eta06/HarbiChess/commit/8face739d8f25b3a950aaeaed751a3e9ba32f5a5) ESAS: record system teacher result
- [8c721c1](https://github.com/Eta06/HarbiChess/commit/8c721c1183cb8926290262036f589744136cf3a7) ESAS: preregister short horizon value transfer
- [afb03dc](https://github.com/Eta06/HarbiChess/commit/afb03dcafd8c2833783e0c0ea89c71928481b352) ESAS: add short horizon value transfer
- [36bfd23](https://github.com/Eta06/HarbiChess/commit/36bfd2380de7c3679be1633ecdd5bc35cb6d9414) ESAS: test short horizon value targets
- [f03d718](https://github.com/Eta06/HarbiChess/commit/f03d71878fa82b0c3dbe72bf4af5d34474a7cf63) ESAS: report clipped auxiliary gradient norm
- [5fdb057](https://github.com/Eta06/HarbiChess/commit/5fdb057b3629da5081418dea62dd62b3b5a0169e) ESAS: preregister value interference diagnostic
- [65d132b](https://github.com/Eta06/HarbiChess/commit/65d132b1fbe9913cc5ea6c5f8bde35cb536705e4) ESAS: record value learning diagnostics
- [d086907](https://github.com/Eta06/HarbiChess/commit/d08690733482608eb76a89d89cc9d0e33734783b) ESAS: preregister parent relative FPU
- [0a1c76c](https://github.com/Eta06/HarbiChess/commit/0a1c76cc87e25a0453d0b58968fd6035e40f7585) ESAS: add parent relative FPU search
- [b39abe4](https://github.com/Eta06/HarbiChess/commit/b39abe4ddc25d6fe97c795dbac7ab5f177956645) ESAS: expose FPU in tactical diagnostics
- [ce0c46b](https://github.com/Eta06/HarbiChess/commit/ce0c46b871ed71a36350fb7d4fb0665988606c87) ESAS: qualify configurable FPU search
- [efaa506](https://github.com/Eta06/HarbiChess/commit/efaa506c4adaf46460eca3217866124f38ddd438) ESAS: test parent relative FPU search
- [bad99ac](https://github.com/Eta06/HarbiChess/commit/bad99ac900ccf43dec46581b8225e101728e0583) ESAS: record parent relative FPU result
- [5450541](https://github.com/Eta06/HarbiChess/commit/545054108160e83b408243b686ddf0b267e326ba) ESAS: preregister full Gumbel search
- [5b01308](https://github.com/Eta06/HarbiChess/commit/5b013082f589b734c27196f213eec295cccd129d) ESAS: add shared tree Full Gumbel search
- [7ad6c2a](https://github.com/Eta06/HarbiChess/commit/7ad6c2a2b5d80ac3151d287994a6d318fac87cb2) ESAS: test Full Gumbel search mechanics
- [3517a5c](https://github.com/Eta06/HarbiChess/commit/3517a5cb13246a2303f8aba31e35582fb26a562f) ESAS: diagnose selectable search allocation
- [4625756](https://github.com/Eta06/HarbiChess/commit/4625756477ec2c8c4307403861019ea5b36f8e86) ESAS: qualify Full Gumbel system teacher
- [1133aa0](https://github.com/Eta06/HarbiChess/commit/1133aa0f195493957d21ce899b214d19aaf6a3c2) ESAS: test Full Gumbel qualification config
- [a581fd7](https://github.com/Eta06/HarbiChess/commit/a581fd7f20cd48a89af8bab9b068140d626bead5) ESAS: test Full Gumbel tactical diagnostics
- [0b1c315](https://github.com/Eta06/HarbiChess/commit/0b1c3156265f9d114bc23e3dcc8652dab70a7a21) ESAS: record Full Gumbel teacher result

## AKTAR

Full Gumbel soft target/provenance, fixed-batch eşdeğerliği ve policy transferi.

27 commit; 2026-08-28–2026-08-28.

- [1b62775](https://github.com/Eta06/HarbiChess/commit/1b62775bdb88992775c71f6fd4cff4128cce1645) AKTAR: preregister Full Gumbel transfer
- [112e032](https://github.com/Eta06/HarbiChess/commit/112e032b66ae590875ab31a7a916bb14615912ba) AKTAR: generate Full Gumbel soft targets
- [45311ef](https://github.com/Eta06/HarbiChess/commit/45311efc48ee8560109b78450d024e2bec08b7ef) AKTAR: test Full Gumbel target provenance
- [6929642](https://github.com/Eta06/HarbiChess/commit/6929642944670c5a9071711fabab841dc57a0297) AKTAR: record target provenance failure
- [db00a98](https://github.com/Eta06/HarbiChess/commit/db00a989362fc999f4c079f032577170871580d5) AKTAR: add fixed shape MLX inference batches
- [2df8b6b](https://github.com/Eta06/HarbiChess/commit/2df8b6b56f1454aec19ce2d0c9d7625361037529) AKTAR: test fixed shape MLX inference batches
- [e6395dc](https://github.com/Eta06/HarbiChess/commit/e6395dc31f5fc3b71c5f1692bc0224010df87126) AKTAR: enforce fixed batch target equivalence
- [54a2730](https://github.com/Eta06/HarbiChess/commit/54a27301fabfe5ed4ab6a19deb6937547f6c93d6) AKTAR: record fixed batch target result
- [b256950](https://github.com/Eta06/HarbiChess/commit/b256950b9df473fcad74cb13ee1431c12f79f0a7) AKTAR: normalize target visit provenance
- [1863531](https://github.com/Eta06/HarbiChess/commit/18635311d273c011ad927f1125f96f8e0bd3fb34) AKTAR: test serialized visit provenance
- [cd5aaa9](https://github.com/Eta06/HarbiChess/commit/cd5aaa92f291eacfa4000616add6b80aa23569ca) AKTAR: benchmark fixed batch wait windows
- [1981d04](https://github.com/Eta06/HarbiChess/commit/1981d045ef9e75c8e5a88d6a842c30dad8ac06fb) AKTAR: test fixed batch benchmark gates
- [bc5333f](https://github.com/Eta06/HarbiChess/commit/bc5333f201ea5ef7abe8d889225f321c0d06dac4) AKTAR: expose deterministic batch wait
- [f06821a](https://github.com/Eta06/HarbiChess/commit/f06821a7ac27a384979d2b845efc6b729f15cd83) AKTAR: preregister fixed inference shapes
- [dba536a](https://github.com/Eta06/HarbiChess/commit/dba536aaf4d3f3e62cb65c6032facd82de92b476) AKTAR: benchmark fixed inference shape matrix
- [bc491db](https://github.com/Eta06/HarbiChess/commit/bc491dbb80bb6664af69cfd6c37a444edbbff5d4) AKTAR: test fixed inference shape matrix
- [3c91346](https://github.com/Eta06/HarbiChess/commit/3c913462fd6a64ea09030d902145586e9e6dc905) AKTAR: expose deterministic inference shape
- [3aa33da](https://github.com/Eta06/HarbiChess/commit/3aa33dae4c981953f9b8f721d154c8b2836337bf) AKTAR: bound queue to fixed inference shape
- [cb069b5](https://github.com/Eta06/HarbiChess/commit/cb069b5b2659e9d16f3c9d7793e3c4bd89baec9b) AKTAR: batch deterministic target audit
- [cf87539](https://github.com/Eta06/HarbiChess/commit/cf875399796395d6273ce1de9d2228d297e6b41a) AKTAR: add guarded Full Gumbel transfer
- [339d8b4](https://github.com/Eta06/HarbiChess/commit/339d8b41653435edbe207791e257e4d0d4b11f82) AKTAR: test guarded Full Gumbel transfer
- [5fb2152](https://github.com/Eta06/HarbiChess/commit/5fb21529251f761f720bb7874c401af5833013ce) AKTAR: record Full Gumbel transfer result
- [1ea76eb](https://github.com/Eta06/HarbiChess/commit/1ea76eb431b51cabedf9f05da86829fd4c9c95d0) AKTAR: preregister policy anchor ablation
- [2a58ba1](https://github.com/Eta06/HarbiChess/commit/2a58ba1ce300459f60fcf48db8c6390ecb067718) AKTAR: add guarded policy anchor ablation
- [4e8cd4a](https://github.com/Eta06/HarbiChess/commit/4e8cd4a0e9a4322ef25ac98d6ca66cc11ffc3974) AKTAR: test policy anchor transfer gates
- [cd1fa7c](https://github.com/Eta06/HarbiChess/commit/cd1fa7cb6976f1d0e8377dbbfb5ca60b03c4d1ca) AKTAR: record policy anchor ablation
- [da829f1](https://github.com/Eta06/HarbiChess/commit/da829f1f84e1e7e2d9c83892c1ab630eb4760cf6) AKTAR: diagnose search transfer regression

## KRITIK

Corrected terminal hedefleriyle joint transfer, value collapse, gradient ve deterministic material probe.

22 commit; 2026-08-30–2026-08-30.

- [852802b](https://github.com/Eta06/HarbiChess/commit/852802b73d459d13e24d0846512130b21ad26cac) KRITIK: preregister joint policy value transfer
- [48ae071](https://github.com/Eta06/HarbiChess/commit/48ae07174b1bdcbaf5c575f4c2b15b0a8a9712af) KRITIK: add guarded joint policy value transfer
- [a4c3492](https://github.com/Eta06/HarbiChess/commit/a4c349246e5706f98b706174af4d09771a5dfbc8) KRITIK: test joint value transfer gates
- [63dc84c](https://github.com/Eta06/HarbiChess/commit/63dc84c14551db33f805a08779441712a35ae0dd) KRITIK: record joint value transfer failure
- [f870bf8](https://github.com/Eta06/HarbiChess/commit/f870bf848692b3b069b4f634b81d9fd6bab15d9b) KRITIK: preregister value signal audit
- [728d4a6](https://github.com/Eta06/HarbiChess/commit/728d4a68b4faf845983a29b2c85a0614245214a8) KRITIK: add matched value signal audit
- [0c92db6](https://github.com/Eta06/HarbiChess/commit/0c92db628879c3870f1d81d72819c811a0f9788b) KRITIK: test value signal audit controls
- [b752d97](https://github.com/Eta06/HarbiChess/commit/b752d97cc082a9629f8a8030935e45854e42453d) KRITIK: classify partial value memorization
- [a84052b](https://github.com/Eta06/HarbiChess/commit/a84052b2f33617e36f6c512cedc4f65e9563c2c7) KRITIK: test partial memorization diagnosis
- [c6786a4](https://github.com/Eta06/HarbiChess/commit/c6786a432d37551977390de85a03d08719b5fbd8) KRITIK: preregister corrected replay scale control
- [e41cb17](https://github.com/Eta06/HarbiChess/commit/e41cb1799c570b06d94ecf3eeca829354e47eeca) KRITIK: add deduplicated corrected replay transfer
- [a95c17f](https://github.com/Eta06/HarbiChess/commit/a95c17f11f53cac2ccf7a67e35c8e28deb55c45c) KRITIK: test corrected replay trajectory isolation
- [f28b1a6](https://github.com/Eta06/HarbiChess/commit/f28b1a699bb968b16398471b423c724479df2999) KRITIK: preregister shared representation transfer
- [d4afed2](https://github.com/Eta06/HarbiChess/commit/d4afed2ea3b914a0a9ea3dfdd723d3ee65a53e23) KRITIK: enable shared representation audit arm
- [3b34e77](https://github.com/Eta06/HarbiChess/commit/3b34e77ccd1a7895f3a86afd06010eca9706dfb9) KRITIK: test representation audit opt in
- [99fea57](https://github.com/Eta06/HarbiChess/commit/99fea578f2a1c5584e5e4f256d7340a2260ad61e) KRITIK: preregister gradient balanced transfer
- [79ba3b8](https://github.com/Eta06/HarbiChess/commit/79ba3b8c0f7b5f4665db61c692e862b85bb437fe) KRITIK: balance joint transfer gradients
- [e4d49c0](https://github.com/Eta06/HarbiChess/commit/e4d49c05019e6f9eb0db606dc328301ff4b2443c) KRITIK: test joint loss weight guards
- [aec2377](https://github.com/Eta06/HarbiChess/commit/aec2377eba54a49607185511fbf908eee017e027) KRITIK: preregister deterministic value probe
- [f20c381](https://github.com/Eta06/HarbiChess/commit/f20c381ad8b4ec49eb3af6c6b0ead69f4c121e36) KRITIK: add deterministic value representation probe
- [916347e](https://github.com/Eta06/HarbiChess/commit/916347eec0e6a5a05084898032c546e45f550a7f) KRITIK: test deterministic value probe
- [efe7107](https://github.com/Eta06/HarbiChess/commit/efe7107740d63a944ac66da6aa4ad1e579fa46a1) KRITIK: record value collapse audit

## MIHVER

Count-scaled global/invariant value representation, ayrı material/WDL head ve nonlinear WDL.

38 commit; 2026-08-30–2026-08-30.

- [a29665c](https://github.com/Eta06/HarbiChess/commit/a29665c9a82c8136b72facf93792c395a6f245ae) MIHVER: preregister invariant value representation
- [f086218](https://github.com/Eta06/HarbiChess/commit/f0862187e85802378b5c87618cb35b8d0470ed92) MIHVER: add invariant value network
- [f0533db](https://github.com/Eta06/HarbiChess/commit/f0533dbf2b7025719afe5c4551694db8fdaf245e) MIHVER: test invariant value preservation
- [e77bccf](https://github.com/Eta06/HarbiChess/commit/e77bccf3895a76d39798ebd2f56dc6c3ebb7603c) MIHVER: add invariant material qualification
- [6523536](https://github.com/Eta06/HarbiChess/commit/65235361521e50f16b2de9e034c2928ca11a93cf) MIHVER: test invariant material selection
- [656ff0f](https://github.com/Eta06/HarbiChess/commit/656ff0f56960739ca87191c6ab6f7d5d207c284a) MIHVER: preregister count scaled value features
- [cd50669](https://github.com/Eta06/HarbiChess/commit/cd5066936e9e903b1c7df7b286cd0e6479cbc5b5) MIHVER: scale current board invariant features
- [1f8d478](https://github.com/Eta06/HarbiChess/commit/1f8d47867031f2a3168fd406f11e58e8e55c61d5) MIHVER: test count scaled invariant input
- [27f5348](https://github.com/Eta06/HarbiChess/commit/27f5348a48a9b4cf3ba9bc6045c94f754670894b) MIHVER: preregister invariant WDL calibration
- [eacf4fa](https://github.com/Eta06/HarbiChess/commit/eacf4fa2b7e16838e9e4f92666ed28b323484758) MIHVER: isolate trainable value tower
- [58a804f](https://github.com/Eta06/HarbiChess/commit/58a804f3049e23198333437d94f65bb9d7ed673b) MIHVER: test value tower isolation
- [d335e0e](https://github.com/Eta06/HarbiChess/commit/d335e0ec28e03ea88c6264d735a57aeca0bab30c) MIHVER: add invariant WDL calibration
- [7158389](https://github.com/Eta06/HarbiChess/commit/71583899dc784b08852a299d773d3090e13d6b95) MIHVER: test invariant WDL gates
- [3aa91ad](https://github.com/Eta06/HarbiChess/commit/3aa91ad1cf60e7169db9160e64c9f9a3f54fd6b0) MIHVER: preregister distributional material targets
- [19cf080](https://github.com/Eta06/HarbiChess/commit/19cf0801723c88bc2c37d16d1a5df69d84ea298b) MIHVER: add distributional material objective
- [9181354](https://github.com/Eta06/HarbiChess/commit/91813546134df5474acf153f720edffce562fb8d) MIHVER: test distributional material targets
- [e62fc64](https://github.com/Eta06/HarbiChess/commit/e62fc649cfcc9563de3ff10c57d5ba9fef7a740f) MIHVER: preregister balanced material objective
- [465ee86](https://github.com/Eta06/HarbiChess/commit/465ee866d45db3631af12984c06277c8284e40cd) MIHVER: balance material value gradients
- [802b073](https://github.com/Eta06/HarbiChess/commit/802b07322b6b49bcf4948c536b189761f622fd86) MIHVER: test balanced material weight guard
- [09276a3](https://github.com/Eta06/HarbiChess/commit/09276a3cf58358443841e6a0dda2baec8cdf187f) MIHVER: preregister decoupled value heads
- [c4d066d](https://github.com/Eta06/HarbiChess/commit/c4d066d4902c62ae915ad7114a3a170fcd21d916) MIHVER: add decoupled auxiliary value head
- [46e832d](https://github.com/Eta06/HarbiChess/commit/46e832d25b59e78ce7f29bd12bfceb590468969c) MIHVER: test decoupled value isolation
- [cce36ee](https://github.com/Eta06/HarbiChess/commit/cce36eeb81f078d14aac117072288daeb3367422) MIHVER: add decoupled value qualification
- [7dc519f](https://github.com/Eta06/HarbiChess/commit/7dc519f65afd33a9a58b298b04d3863b46e64787) MIHVER: test decoupled value gates
- [5571793](https://github.com/Eta06/HarbiChess/commit/5571793c343b2aa41e92ea771cba1fb3370e1305) MIHVER: preregister mixed WDL sampling
- [fd93447](https://github.com/Eta06/HarbiChess/commit/fd934470eb9a16f582143d2200d363a11dadb2ce) MIHVER: mix balanced and natural WDL sampling
- [f0ed8a5](https://github.com/Eta06/HarbiChess/commit/f0ed8a562293a06798b10b7ba874ad3637f338e3) MIHVER: test mixed WDL sampling guard
- [7e7d59d](https://github.com/Eta06/HarbiChess/commit/7e7d59dce711a054e9f575fafd1bb0625f2ed7f5) MIHVER: preregister nonlinear invariant value head
- [4565048](https://github.com/Eta06/HarbiChess/commit/456504863d1da4b6879448a0b9dbf8c8434901fd) MIHVER: add nonlinear invariant value head
- [74b1e48](https://github.com/Eta06/HarbiChess/commit/74b1e48389f8dfc50bd9000d8058f8ea73a698e7) MIHVER: qualify nonlinear invariant parameters
- [0ff222f](https://github.com/Eta06/HarbiChess/commit/0ff222ff6bee2a0eb2dbdcc9087179d5644cd6a4) MIHVER: test nonlinear value isolation
- [7b132ea](https://github.com/Eta06/HarbiChess/commit/7b132eaf7fa6b166f8cdc266856184f96f25847c) MIHVER: preregister value downstream gates
- [4fc3020](https://github.com/Eta06/HarbiChess/commit/4fc3020810b2054658a13933c3227e05f125146c) MIHVER: evaluate complete value representation
- [6d3c8b9](https://github.com/Eta06/HarbiChess/commit/6d3c8b9e86d2b3ae1dff9b2c26aa05c9d7c7b8bf) MIHVER: test complete value evaluation path
- [18be110](https://github.com/Eta06/HarbiChess/commit/18be110e52faeb9c194dc7875d978b180287f21d) MIHVER: add value downstream qualification
- [de8a303](https://github.com/Eta06/HarbiChess/commit/de8a303c9893049e1e330cf0a0f4440610f6a50e) MIHVER: test value downstream guards
- [a006320](https://github.com/Eta06/HarbiChess/commit/a006320fe8b6295e1eb04148dc9555ade2f2b449) MIHVER: use supported qualification status
- [fd7e208](https://github.com/Eta06/HarbiChess/commit/fd7e2089dddb377e30b9b4e8b569317cb10fb509) MIHVER: report qualified value representation

## DEVRIYE

Latest-network rolling replay pilotu, headwise checkpoint seçimi ve dtype izolasyonu.

52 commit; 2026-08-30–2026-08-30.

- [7e990d3](https://github.com/Eta06/HarbiChess/commit/7e990d3183a774e09d7ff71b052cc85cc2ed061f) DEVRIYE: preregister continuous learner pilot
- [0ae32ec](https://github.com/Eta06/HarbiChess/commit/0ae32ec11fc579571f4bcbeef4ade2896515d142) DEVRIYE: isolate continuous learner heads
- [6315062](https://github.com/Eta06/HarbiChess/commit/6315062be900a76e9811747c322884f483207a2d) DEVRIYE: test continuous head isolation
- [a2b0e24](https://github.com/Eta06/HarbiChess/commit/a2b0e24decd3a8b0bd77b8e244ce1e9a9fb56f32) DEVRIYE: add rolling latest network pilot
- [be4c6be](https://github.com/Eta06/HarbiChess/commit/be4c6becf44574cc95521ff6ad9e9d3261302a16) DEVRIYE: test continuous learner gates
- [b8deb27](https://github.com/Eta06/HarbiChess/commit/b8deb278a62565be8c804950bebd2d5446cb1d61) DEVRIYE: audit inference dtype mutation
- [37f2759](https://github.com/Eta06/HarbiChess/commit/37f2759aa573276498ec976c9598ca312ebf243d) DEVRIYE: isolate learner from inference dtype
- [f3d221b](https://github.com/Eta06/HarbiChess/commit/f3d221b78699dc039887959bc5704d84e133c51e) DEVRIYE: test inference dtype isolation
- [7c8f6b9](https://github.com/Eta06/HarbiChess/commit/7c8f6b9a571a5e32a65ec17045f9a9b387163128) DEVRIYE: preregister validation checkpoint selection
- [97f6f2c](https://github.com/Eta06/HarbiChess/commit/97f6f2c96c2ddfc924b3da481f08fa05dba8e4ab) DEVRIYE: select gated validation checkpoints
- [5670d11](https://github.com/Eta06/HarbiChess/commit/5670d1182618e07a75b0cf1413baf4cfb89142dd) DEVRIYE: test gated checkpoint selection
- [41ce2ec](https://github.com/Eta06/HarbiChess/commit/41ce2ec57292750a19d6ddaa6b70214921ef85ce) DEVRIYE: preregister minimum sufficient updates
- [3de84b0](https://github.com/Eta06/HarbiChess/commit/3de84b0e336ac6fe75d9b1978bc0e701a0521eb6) DEVRIYE: stop at earliest qualified update
- [a17f392](https://github.com/Eta06/HarbiChess/commit/a17f39244492fd1a772399f03897feb1f7b5c812) DEVRIYE: test earliest qualified update
- [b04f98e](https://github.com/Eta06/HarbiChess/commit/b04f98e13610af1ceaa0a9164aacd2f2812e0947) DEVRIYE: preregister fresh value replay
- [50ff642](https://github.com/Eta06/HarbiChess/commit/50ff64246ae70ffc3f5f8674f77d420357ce5ad6) DEVRIYE: add latest network replay generation
- [aa7fce0](https://github.com/Eta06/HarbiChess/commit/aa7fce0c352d16afb54b8a0c521dd016e543e7e7) DEVRIYE: test latest replay configuration
- [d8faf65](https://github.com/Eta06/HarbiChess/commit/d8faf657703ba1db2abd6785b005ae88c14c61ba) DEVRIYE: train value on rolling fresh replay
- [fb004d9](https://github.com/Eta06/HarbiChess/commit/fb004d9bcdf0a1acf3cdafeab3dfccede93af567) DEVRIYE: test fresh replay pilot constraints
- [e21ce07](https://github.com/Eta06/HarbiChess/commit/e21ce07845414b9a32346454e77469ce865be67e) DEVRIYE: use Full Gumbel policy targets in replay
- [754fa3e](https://github.com/Eta06/HarbiChess/commit/754fa3e881727d6c36195e42383364844189696b) DEVRIYE: test Full Gumbel replay policy semantics
- [1419798](https://github.com/Eta06/HarbiChess/commit/1419798513d3d146381eab18851b4a2c233a4672) DEVRIYE: preregister stratified continuation replay
- [e8fd9a1](https://github.com/Eta06/HarbiChess/commit/e8fd9a17ed856bb23df5261b1cf4df190277dd09) DEVRIYE: support continuation self-play starts
- [2dfb970](https://github.com/Eta06/HarbiChess/commit/2dfb970a14db9c2cad34f506c21988646497137c) DEVRIYE: test continuation self-play starts
- [b70e72e](https://github.com/Eta06/HarbiChess/commit/b70e72ea0f182af59d10806d354c400a93de3d2c) DEVRIYE: bound continuation rollout length
- [1290a05](https://github.com/Eta06/HarbiChess/commit/1290a0569ca172b5a1991e7248635050900e3ed6) DEVRIYE: test continuation rollout bounds
- [c9cbde6](https://github.com/Eta06/HarbiChess/commit/c9cbde6a85bd46909cf8dd34ae357065f4a9eb2d) DEVRIYE: generate replay from stratified starts
- [d77cdd3](https://github.com/Eta06/HarbiChess/commit/d77cdd3c5875a9d002acac1fbdbdce5f0451d76e) DEVRIYE: stratify rolling replay continuations
- [7d6e111](https://github.com/Eta06/HarbiChess/commit/7d6e111f05e2036e23556eb014a699fb1715bb73) DEVRIYE: test stratified replay selection
- [a7fd852](https://github.com/Eta06/HarbiChess/commit/a7fd852c42655f5ffa3bd7a9a93c3b86e0877951) DEVRIYE: stabilize teacher divergence telemetry
- [cd0186d](https://github.com/Eta06/HarbiChess/commit/cd0186d894b9ef2e967f9c880e974a74118f3b39) DEVRIYE: preregister balanced rolling value replay
- [26690b9](https://github.com/Eta06/HarbiChess/commit/26690b9c95b702e3f3ffaad9254bf6ca90eb9ae0) DEVRIYE: balance fresh value outcomes
- [e05a70c](https://github.com/Eta06/HarbiChess/commit/e05a70cd4ac9c1c270094964e406ce3e9a8de534) DEVRIYE: preregister scaled teacher transfer
- [d9d392a](https://github.com/Eta06/HarbiChess/commit/d9d392ab9d8ced18ef4e9373c010e9d5af8415b6) DEVRIYE: add fixed outcome replay sampler
- [0128f2b](https://github.com/Eta06/HarbiChess/commit/0128f2b3ac8a8e9a762edef3cf1bac0ddb412a29) DEVRIYE: test fixed outcome replay mix
- [e1cfe2c](https://github.com/Eta06/HarbiChess/commit/e1cfe2c24c3db3629cec823faf9c68e146b08ffb) DEVRIYE: scale qualified teacher transfer
- [2c92f5f](https://github.com/Eta06/HarbiChess/commit/2c92f5f4188d658b18f777770959914187fe8ce8) DEVRIYE: test scaled teacher defaults
- [f1d985b](https://github.com/Eta06/HarbiChess/commit/f1d985b57abba501fbcc13194b6487d872828d4d) DEVRIYE: preregister independent outcome scale
- [8920915](https://github.com/Eta06/HarbiChess/commit/89209152bab0cff531b5783566bb05da23bccdf5) DEVRIYE: scale independent rolling outcomes
- [e19c733](https://github.com/Eta06/HarbiChess/commit/e19c733d6cc0e6b09921471ba3c37cf7c0ae1e74) DEVRIYE: test independent outcome defaults
- [d902f34](https://github.com/Eta06/HarbiChess/commit/d902f34342d9e8d0d9d753e0167cfa74b28a2170) DEVRIYE: preregister headwise checkpoints
- [720a661](https://github.com/Eta06/HarbiChess/commit/720a6611e1dfdb49ebbbb8edf78baf87075184af) DEVRIYE: select policy and value checkpoints independently
- [3e76178](https://github.com/Eta06/HarbiChess/commit/3e761782374d8efaa83a333de6a7a1fd09b3d9ab) DEVRIYE: test headwise checkpoint composition
- [30d0edd](https://github.com/Eta06/HarbiChess/commit/30d0eddce7c453550bd5ee385d5d984b7307e131) DEVRIYE: preregister value checkpoint cadence
- [5e65405](https://github.com/Eta06/HarbiChess/commit/5e65405aaa16af19074f82fcaee15f85f6caddb4) DEVRIYE: capture earliest healthy value checkpoint
- [99ef8fe](https://github.com/Eta06/HarbiChess/commit/99ef8fe233fb609432cd21d1191744ffe1ed2abc) DEVRIYE: record cached headwise qualification
- [bf38710](https://github.com/Eta06/HarbiChess/commit/bf3871092b8fda8ad7031d36d8c9a39173477d15) DEVRIYE: preregister value stability ablation
- [c8aaf2e](https://github.com/Eta06/HarbiChess/commit/c8aaf2ebab09dd86d897d99fcdf072cb20487dc2) DEVRIYE: preregister value gradient batches
- [4230606](https://github.com/Eta06/HarbiChess/commit/42306067e389f5a7da2425b2525c93b5bdb0e6fd) DEVRIYE: preregister pooled value reservoir
- [1ce6f25](https://github.com/Eta06/HarbiChess/commit/1ce6f25cf78b83b80d9ec4f92dd899d16e59926f) DEVRIYE: preregister multiobjective value loss
- [cd249f9](https://github.com/Eta06/HarbiChess/commit/cd249f93f98e60dcfd40da4a25eb8760250700eb) DEVRIYE: distinguish chain rejection from rollback
- [db28411](https://github.com/Eta06/HarbiChess/commit/db284115a773be6e2dcbeb7ef1997dfd05f8861b) DEVRIYE: report continuous pilot evidence

## YELKEN

Stable-base + plastic-residual value, constrained gradient ve stability/plasticity ablation.

15 commit; 2026-08-30–2026-08-30.

- [2d32097](https://github.com/Eta06/HarbiChess/commit/2d32097f89399e13e2f4be066cce9fd9a5118aaa) YELKEN: preregister stable plastic value experiment
- [81c98fe](https://github.com/Eta06/HarbiChess/commit/81c98fed77b16c76154269e40fed1b492ce8cb12) YELKEN: add plastic residual value network
- [5b9126a](https://github.com/Eta06/HarbiChess/commit/5b9126a343c59397a260781b4d5184f3058288ed) YELKEN: test plastic value isolation
- [e1e1f13](https://github.com/Eta06/HarbiChess/commit/e1e1f131f298b2e191587afbb67c08ee91033b24) YELKEN: add stable plastic value ablation
- [47643e7](https://github.com/Eta06/HarbiChess/commit/47643e764b915842d484ee62cbdb2a25567681f2) YELKEN: test stable plastic ablation gates
- [eef9556](https://github.com/Eta06/HarbiChess/commit/eef955683386d9883157d3f712edf14f03dceb11) YELKEN: preregister constrained residual gradients
- [aab6417](https://github.com/Eta06/HarbiChess/commit/aab64178edc992d45133ecf486f3edc7fa89e2d9) YELKEN: add Pareto constrained value ablation
- [2a2fc02](https://github.com/Eta06/HarbiChess/commit/2a2fc02f25865703d997bec8d83501e9443b53e1) YELKEN: test constrained gradient combiners
- [d3fdb00](https://github.com/Eta06/HarbiChess/commit/d3fdb0064302db60181e4abb8448239d40645381) YELKEN: add replay value target conflict audit
- [c532e5f](https://github.com/Eta06/HarbiChess/commit/c532e5fb0aff75a36072c7a088a90883dca82b53) YELKEN: test replay target conflict audit
- [5add53a](https://github.com/Eta06/HarbiChess/commit/5add53ab0b3b073c785d376198b0543b4eddd121) YELKEN: preregister uncertainty preserving WDL transfer
- [2fe89db](https://github.com/Eta06/HarbiChess/commit/2fe89db95a504cf8b04cf1c2757ed8645f47e861) YELKEN: freeze multiobjective gradient normalization
- [ec23ccf](https://github.com/Eta06/HarbiChess/commit/ec23ccf993ab451d0dc16c7779fbe0e05dcceea5) YELKEN: add uncertainty preserving WDL ablation
- [0c278b4](https://github.com/Eta06/HarbiChess/commit/0c278b4478f22be02fcce42b24237cf5fea91fdb) YELKEN: test soft WDL target transfer
- [532d30c](https://github.com/Eta06/HarbiChess/commit/532d30c1967adb4a8299b91c9943e3833a038ace) YELKEN: report stable plastic value evidence

## PUSULA

Cumulative non-inferiority, bootstrap/power, holdout ve production-readiness gate tasarımı.

60 commit; 2026-08-30–2026-08-31.

- [798ecdb](https://github.com/Eta06/HarbiChess/commit/798ecdb1503959bf45178a7cdd3ba44a1fc34e05) PUSULA: add cumulative noninferiority statistics
- [612cb3a](https://github.com/Eta06/HarbiChess/commit/612cb3ab6b0f9f5a39d0573e230799548fd6955c) PUSULA: test cumulative bootstrap gates
- [5ab3f0b](https://github.com/Eta06/HarbiChess/commit/5ab3f0b2334864d79b11d1305230c0777ec86e3a) PUSULA: add paired power planner
- [729b1b7](https://github.com/Eta06/HarbiChess/commit/729b1b7ff4b7a824300c1d139459dd1833956df1) PUSULA: test paired power planning
- [a81d222](https://github.com/Eta06/HarbiChess/commit/a81d22253ada2715a165ed49c598cf7158270018) PUSULA: preregister cumulative production gate
- [1333640](https://github.com/Eta06/HarbiChess/commit/133364028bd5545b688327505cfc03a70bfa8fcc) PUSULA: integrate cumulative rolling learner gate
- [7ca331b](https://github.com/Eta06/HarbiChess/commit/7ca331bf6f1e949013d3c069227ecf664db686b2) PUSULA: test cumulative learner integration
- [deec9d2](https://github.com/Eta06/HarbiChess/commit/deec9d2439e387ea907d877333e2cd5cc98be524) PUSULA: isolate final capability holdout
- [47af149](https://github.com/Eta06/HarbiChess/commit/47af149865b26577565a6413a520372d8eafa15c) PUSULA: add continuous checkpoint integrity
- [e54a955](https://github.com/Eta06/HarbiChess/commit/e54a955342e0fe7bb1fd6a3904e06b841df881c1) PUSULA: test continuous checkpoint integrity
- [783c7bf](https://github.com/Eta06/HarbiChess/commit/783c7bf6ae487e443f76efa86c9567912bb00b2d) PUSULA: verify exact update resume
- [8efaeaf](https://github.com/Eta06/HarbiChess/commit/8efaeafc8875db14eb30e1f913d44495fb35174d) PUSULA: test exact update resume
- [7e0e1a6](https://github.com/Eta06/HarbiChess/commit/7e0e1a6efc50dcfe54ed1dc0caa5d4faac1e7bb1) PUSULA: specify deterministic resume boundary
- [6256d42](https://github.com/Eta06/HarbiChess/commit/6256d427ac8f0bdff762d331672286f706e7fe9d) PUSULA: stratify distinct continuation states
- [a61a8d9](https://github.com/Eta06/HarbiChess/commit/a61a8d9f44a615768e0ef6b153cd96ad37ee5e51) PUSULA: test repeated-game continuation starts
- [f791e94](https://github.com/Eta06/HarbiChess/commit/f791e94097d2a233a26a9cc80768b3437e82fb73) PUSULA: clarify continuation start independence
- [c6f3d75](https://github.com/Eta06/HarbiChess/commit/c6f3d7558054a68ce76f54c1039d49cfb968e36d) PUSULA: align local gates with preregistration
- [59535b8](https://github.com/Eta06/HarbiChess/commit/59535b843b26e138d0b833463481189295a493e6) PUSULA: test preregistered local gates
- [bf169f9](https://github.com/Eta06/HarbiChess/commit/bf169f9c8efc7437fe5664e9a01b703e2a3d63c9) PUSULA: preregister fresh replacement run
- [3d86465](https://github.com/Eta06/HarbiChess/commit/3d86465e3900f4f6a312b009eeb0dcbc8676ff8a) PUSULA: select value checkpoints on fresh holdout
- [4dc646c](https://github.com/Eta06/HarbiChess/commit/4dc646c687decbb4a8edddc76f275a767795f411) PUSULA: test fresh value checkpoint selection
- [5dad0ee](https://github.com/Eta06/HarbiChess/commit/5dad0eece59e7958aa62366bc420934666a1cef1) PUSULA: preregister fresh value selection
- [0014ddf](https://github.com/Eta06/HarbiChess/commit/0014ddffa6195228f5efeb84eee10b805499a37a) PUSULA: preserve stable value function
- [b289dd8](https://github.com/Eta06/HarbiChess/commit/b289dd8679ff6edf45831d9ff948101a45ccb093) PUSULA: test cumulative drift constraints
- [bf4d681](https://github.com/Eta06/HarbiChess/commit/bf4d6811acc3159784ad40e2f33e936c2bd3a1f0) PUSULA: preregister stable rehearsal run
- [9ab5d95](https://github.com/Eta06/HarbiChess/commit/9ab5d952659f43affb979bdee7e7292cec0b7693) PUSULA: constrain cumulative value steps
- [e521bd4](https://github.com/Eta06/HarbiChess/commit/e521bd485cb0dc8b1b4872910fe2f73c7cb5bb79) PUSULA: test value trust region
- [cddd10a](https://github.com/Eta06/HarbiChess/commit/cddd10a74518ccd5163b2538bf2a317a6b62ca50) PUSULA: preregister residual trust region
- [57d9bb1](https://github.com/Eta06/HarbiChess/commit/57d9bb1109601a180bd192904f28c73096701200) PUSULA: gate local arena by evidence
- [8eaf308](https://github.com/Eta06/HarbiChess/commit/8eaf308792583ce40dc818aa450d659251245cd9) PUSULA: test local arena confidence gate
- [62ddf8a](https://github.com/Eta06/HarbiChess/commit/62ddf8a30c8cee080881665d34c7aac01b837097) PUSULA: preregister evidence aware arena
- [8a082c3](https://github.com/Eta06/HarbiChess/commit/8a082c3fe745810ccaed3fe4014d5451ecb2243d) PUSULA: gate fresh value calibration
- [d3fb983](https://github.com/Eta06/HarbiChess/commit/d3fb983d6e1e054072ce78087bc23aede2d03415) PUSULA: test fresh calibration gate
- [2708be9](https://github.com/Eta06/HarbiChess/commit/2708be924388426ebd30824862ce92196497af3e) PUSULA: bootstrap local fresh safety
- [332ae48](https://github.com/Eta06/HarbiChess/commit/332ae4833468a57317c9d0026a8f1e75637b31ae) PUSULA: test paired fresh safety
- [75bedd2](https://github.com/Eta06/HarbiChess/commit/75bedd2215dc084aa4aed731ebea3cda09a7eb33) PUSULA: preregister powered calibration run
- [4e62b70](https://github.com/Eta06/HarbiChess/commit/4e62b70747b9ec76460f938313bb7c927ea1d364) PUSULA: bootstrap historical update safety
- [797218c](https://github.com/Eta06/HarbiChess/commit/797218c739760584ec57fe7f48a1724582e2c1b2) PUSULA: test historical confidence gate
- [c007559](https://github.com/Eta06/HarbiChess/commit/c007559323ba65e2741d0a72c3cd7f24e0bca979) PUSULA: preregister paired historical safety
- [6992dff](https://github.com/Eta06/HarbiChess/commit/6992dffab12dbadff25ed6cdd67cd17d4af38912) PUSULA: enforce powered retention gates
- [e1ea0ad](https://github.com/Eta06/HarbiChess/commit/e1ea0adf03539d95e593b6227547d9bb70e079a2) PUSULA: test powered retention gates
- [d01c1d9](https://github.com/Eta06/HarbiChess/commit/d01c1d90a1dee5f3ccb366873fe1be31e92835a9) PUSULA: preregister powered retention run
- [7e8f16a](https://github.com/Eta06/HarbiChess/commit/7e8f16af9b208c3bbeb1f3f7e116b473c19d3e30) PUSULA: scale independent outcome replay
- [68f6e54](https://github.com/Eta06/HarbiChess/commit/68f6e54d5ef6bb64cc6df21099fe2a5a93c44556) PUSULA: test minimum fresh transfer
- [a52a3c7](https://github.com/Eta06/HarbiChess/commit/a52a3c7ead38053cf445cf25750ff5b3731dfeb2) PUSULA: preregister independent game scale
- [9f055c0](https://github.com/Eta06/HarbiChess/commit/9f055c007ee53f03903d0b860462f19180856531) PUSULA: preserve powered old calibration
- [96d8cca](https://github.com/Eta06/HarbiChess/commit/96d8ccae0547ba2d565eb1e3640941904e997b04) PUSULA: test powered old qualification
- [29446d9](https://github.com/Eta06/HarbiChess/commit/29446d97e4f34dedd5bc723e3b0564665f334a13) PUSULA: preregister old calibration power
- [ec1cb4d](https://github.com/Eta06/HarbiChess/commit/ec1cb4d78c1869b04f12ed09ad0ef42b340b59f6) PUSULA: select tactically safe policy step
- [e160ed0](https://github.com/Eta06/HarbiChess/commit/e160ed073608d4efb98f818636772081fb8577d4) PUSULA: test tactical policy selection
- [c219c42](https://github.com/Eta06/HarbiChess/commit/c219c4234bbd0660eada59dfcd8723ec1c0a308b) PUSULA: preregister tactical policy selection
- [27c5714](https://github.com/Eta06/HarbiChess/commit/27c57144f6aece08dd851253da1d1499b2d665e5) PUSULA: select paired-safe value step
- [d896834](https://github.com/Eta06/HarbiChess/commit/d8968340a628a4c6e09564465a0946a3857cde91) PUSULA: test paired-safe value selection
- [9e00ca5](https://github.com/Eta06/HarbiChess/commit/9e00ca5bf26a75deacff93ea53b83d3425762264) PUSULA: preregister paired fresh selection
- [2422269](https://github.com/Eta06/HarbiChess/commit/2422269802d65c0fab069397c7cd28839a32202b) PUSULA: use paired old capability gate
- [5c02a44](https://github.com/Eta06/HarbiChess/commit/5c02a448bc870d0226df23d29d86a0334a9fa29a) PUSULA: test paired-only old screening
- [bd3c379](https://github.com/Eta06/HarbiChess/commit/bd3c37901bc45c3d99ccae7f10acbc9e51965d85) PUSULA: preregister paired cumulative gate
- [69818ab](https://github.com/Eta06/HarbiChess/commit/69818ab91f6de2f8b545fcd99ab582c4b89eb4d4) PUSULA: select continuation-safe value step
- [8e2f6bf](https://github.com/Eta06/HarbiChess/commit/8e2f6bfa2bca566ec522a3b23aba1f1d0cad440c) PUSULA: test paired continuation safety
- [3739f4b](https://github.com/Eta06/HarbiChess/commit/3739f4bf6ae637100eadf6f22c643ab769fc975f) PUSULA: preregister continuation noninferiority

## UFUK — 3 Ekim 2026

Kaynak/rengin etiket örnekleme yanlılığını kaldıran daha geniş teacher curriculum,
aynı veri üzerinde frozen policy-head ve end-to-end kompakt ağ kontrolü, yeni
hamle devamlarıyla güç testi. Başarı veya promotion adı değildir.

- [Ön kayıt](../runs/UFUK-balanced-preregistration-20261003.md)

## AYNA — 3 Ekim 2026

Sabit tam-history pozisyonlarında policy/search/value hatasını ayrı Stockfish
referanslarıyla ölçme; tanıya göre gerçek öğrenme ve bağımsız güç/hız deneyi.
Codeword başarı veya promotion değildir.

- [Teacher tanısı ön kaydı](../runs/AYNA-teacher-preregistration-20261003.md)

## MERCEK — 3 Ekim 2026

Laya/Jev karar mekanizmasının birincil kaynak incelemesi, küçük compute bütçesinde
search kalitesi tanısı ve özgün katkı için ölçülebilir araştırma planı. Başarı veya
model promotion kodu değildir; dış Laya modelinin adı değiştirilmez.

- [Search bütçesi ön kaydı](../runs/MERCEK-search-preregistration-20261003.md)

## PORT — 2 Ekim 2026

Linux CPU/CUDA için PyTorch inference/training, MLX ile ortak schema ve loss sözleşmeleri,
base/invariant/MIHVER/DENGE ağırlık aktarımı, optimizer/RNG/replay içeren version 1 resume,
fresh search policy + terminal WDL rolling self-play. CPU thread/batch, gerçek wall-clock ve
Stockfish/random/frozen-network teşhisleri. Apple/Metal ve CUDA cihaz testi ayrı doğrulama
borcudur; mevcut MLX yolu korunur. Prefix başarı veya promotion anlamına gelmez.

- [Ön kayıt](../runs/PORT-linux-preregistration-20261002.md)
- [Linux sonucu](../runs/PORT-linux-result-20261002.md)


## UFUK-CPUFIX

5 Ekim 2026: GPU kullanmadan kayıpsız epoch feature cache’i ve opt-in Linux aggregate bellek politikası; gerçek CLI restart kontrolü ve kendi oyun kayıtlarından kontrollü öğrenme. Eski failed/incomplete koşular ve native formatlar korunur. Yeni aşama veya loss düşüşü oyun gücü başarısı değildir.

## UFUK-DENSITY

5 Ekim 2026: CPU üzerinde self-play’de search kullanılan hamle oranı ve kendi search hedefinin bütçesi için eşlemeli kontroller. Sparse16/all16/all64, aynı e8 başlangıcı ve iki seed; eski koşular yeniden adlandırılmaz. Search davranış farkı nedensellik veya güç başarısı değildir; bağımsız maç, resume ve veri bütünlüğü ölçülür. Standart expert iteration literatürü temel alınır, özgünlük varsayılmaz.

- **UFUK-ALIGN** (2026-10-05, CPU-only): fullsupport selected-action/search-policy mixture, rawpi/searchsupervision/actualmu ayrı v5ledger, eski auditedall16kontrolü. Hipotez/deney; strength veya özgünlük iddiası değil. Eski FAIL/INCOMPLETE/MAX8 korunur.

- **UFUK-VALUE** (2026-10-05, CPU-only): eski ownterminalreplay’den63parametreli20invariantcritic; policy/trunk sabit, yenidenbaşlatılmış/eğitimsizheadkontrolüyle learningiayır. Hipotez/kontrol, başarı/özgünlükvarsayımı yok; yeniownmoves sayılmaz, eskiMain40FAIL korunur.

- **UFUK-POSITION** (2026-10-05, CPU-only): mevcut sparse-value-v1 32x32 konumsal critic; eski kendi oyunlarının sonuçları, sabit e8 policy/gövde, eğitimsiz aynı-mimari kontrolü. Ağırlık aktarımı fresh Adam ile açıkça ayrılır; sonraki native v2 tam offline resume. Standart piece-square MLP, özgünlük/strength varsayımı yok.

- **UFUK-QSEARCH** (2026-10-05, CPU-only): aynı öğrenilmiş/öğrenilmemiş değer modelleri için bütün yasal kök hamleleri kapsayan512node alpha-beta/quiescence kontrolü. Standart arama; search-only kazancı selflearning değildir. Önce VALUE2048 sabit critic, aynı-search e8/zero kontrolleri ve değişmeyen strength gates.

- **UFUK-SHRINK** (2026-10-05, CPU-only): effective2520-coordinate affine current-piece critic in existing cross-backend sparse-value-v1, spatial residual regularization, absolute-color shortcut masked. Standard regularized linear feature model, no novelty/strength claim; own-outcome only, zero-control/freshAdam/fulloffline-native distinctions kept. Operatorharddeadline6October11İstanbul/08UTC overrides longer experimental caps without resetting originalclocks.

## UFUK-RESIDUAL

5October2026 CPU-only fixed-e8 additive sparsevalue-v2 residual critic. Version1 replacement remainsunchanged;Torch/MLX additiveWDL logits,unsupportedNumpy explicitlyrejects. FixedownMC1024steps/beta1KL anchor, notnewteacherlabel ornoveltyclaim. Allfailures/oldnative/clocks/gates preserved.

## UFUK-BACKUP

5October2026 byteexact offlineCPU native3filecapsules via22serial sealedpublicActions inputs to existingRelease. Parent/model tensorcontentcopies plusliteralremainingbytes; originalAdam/RNG/dataset/native/source retained. Publicationnotpromotion orfullactorresume. IndependentanonymousSHAreadback and strictreadonlyrestore receipts required.

## UFUK-FRESH

- **UFUK-FRESH** (2026-10-05, CPU-only): fixed fresh Q512/q2 ownplay under frozenE8, v2 anchors/actualbehavior search separate, UNKNOWN excluded, protected whole-game exclusions and exact realizedtrajectory-dedup/internal split. Native v2 selfplay is actor/RNG/unfinishedgame complete; trainingnative is separate offline head/Adam/global+samplerRNG/input-bound resume. Baseline is teacher-origin E8, no newteacherqueries. MC, own-search-consistency and fullcritic alternatives use fixedfinal8192 data and unchanged independent strengthgate; standardmechanisms/no noveltyclaim. Actualtwo-seed whole8/pause4/freshresume8/all6native each passed149.081s onoriginal600clock, infrastructureonly.


## UFUK-CLASSICAL

6October deadline continuation, CPU-only: human-authored material/PSQT/pawn/mobility/king prior plus18 learned residual coordinates; no externalteacher queries/E8labels. Fixed16384 own Q512/q2 actions/seed20262905/06, separate actual same-seed native actor qualification, prospective four-literal proof→production transfer. Same-prior trained/untrained control is mandatory: faster or stronger prior alone is not self-learning. Standard value tuning/search distillation, no originality claim; independent unchanged strength gate decides.


## UFUK-PST1 / UFUK-QUIET1 / UFUK-VAULT6 / UFUK-OWNQ1

5–6October CPU continuation: PST1 tests242-feature value learning with exact native resume; QUIET1 tests learned own-search quiet ordering with an unchanged scalar evaluator. OWNQ1 preregisters1024 balanced training roots/seed at8192 own-search nodes for nonlinear residual targets; no external teacher query. VAULT6 preserves22 raw files publicly, VAULT7 prospectively preserves CLASSIC/PST/QUIET rawstates and closedCLASSIC160 evidence. These are phase labels, not success. CLASSIC160 and FRESH224 independently failed unchanged gates; old failed experiments/MAX8 ledger remain closed. Full-state proof/backup does not imply strength. Reports/protocols preserve exact source/data/model hashes and phase clocks.
