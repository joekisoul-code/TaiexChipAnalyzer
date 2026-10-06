# radar5 baseline (harness.py sanity.py, 2026-10-07)
annual walk-forward (embargo 40d), production selection rules (pool 40 / cap 6 / cool 30 / q .9 / q+ .97 / gate m_bias20<0; surge pool 80 top 3 cool 28)

fixed170 (train+cand = production's 2026-09 list; has look-ahead for 2022~2025):
  A  n374 hit .535 win21 .695 avg +8.35 q10 -12.6
  A+ n142 hit .648 win21 .796
  B  n4962 hit .381 win21 .505
  S3 n608 surge .326 win20 .553 avg +6.48
  pool base: hit .403 win21 .520 / spool surge .189 win20 .519
PIT170 (monthly membership = top-170 by 120d avg turnover as of prior trading day; train+cand):
  A  n390 hit .444 win21 .592 avg +5.19 q10 -15.5
  A+ n139 hit .583 win21 .755
  B  n5410 hit .374 win21 .487
  S3 n673 surge .318 win20 .517 avg +5.19
  pool base: hit .380 win21 .491 / spool surge .195 win20 .492
earlier (radar_study2, all-market cand, fixed170 train; = production before 10-06 #9): A hit .471 win .617; S3 win .469
