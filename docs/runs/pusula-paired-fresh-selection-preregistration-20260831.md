# PUSULA paired-fresh value selection preregistration

`pusula-continuous-pilot-20260831-13` remains failed. Update 1 passed with three
tactically safe policy checkpoints. Update 2 also retained tactical capability,
but its selected value checkpoint showed statistically supported paired fresh
Pearson harm versus the preceding accepted network and was rolled back.
Production remains closed.

The value selector had the same ordering defect previously fixed for policy. It
evaluated all 360 step/alpha candidates against update-0 historical point and
bootstrap rules plus update-0 fresh direction/calibration, then selected minimum
fresh CE. Only that single selected candidate was compared with the previous
accepted network by the paired fresh bootstrap. A safer candidate could not be
chosen after the selected one failed.

PUSULA-14 moves the unchanged 2,000-sample paired fresh bootstrap into candidate
selection. Every value step/alpha candidate is compared with the previous
accepted network on the same game-disjoint rolling fresh tuning games. A
candidate is ineligible if CE, macro CE, Brier, or Pearson has a confidence
interval wholly on the harmful side. Selection among candidates that pass old
point safety, old paired safety, update-0 fresh direction/ECE, and paired fresh
safety remains minimum fresh CE, then macro CE, then earliest step. If none is
eligible, the update fails.

The new run is `pusula-continuous-pilot-20260831-14`, seed `2026091301`. No
threshold, bootstrap size, learner duration, exposure, replay scale, search
allocation, loss weight, architecture, arena, continuation, or final old/fresh
qualification setting changes from PUSULA-13. Its data is not reused. A full
pass may authorize production continuous-loop integration but never automatic
release promotion.
