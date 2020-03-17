---
title: "ZeRO stage 1 with reduced communication"
date:   2020-03-13
excerpt: "Partition-aware ZeRO with up to 2x reduction in communication time!"
---

# ZeRO stage 1 with reduced communication
* Partition-aware approach instead of initial implementation that used a global collective (all-reduce)
* Total communication volume reduction 1.5x -> 1x of data parallelism
* Up to 2x reduction in communication time compared to all-reduce

# Further updates coming soon!
