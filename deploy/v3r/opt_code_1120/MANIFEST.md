# opt_code_1120 优化工件包（本地拉回）

> 拉回：2026-11-20 12:09（北京）｜通道：ssh+base64 字节精确｜核验：远端 md5sum ↔ 本地 MD5 逐文件比对
> 配套文档：代码优化点_复用清单_20261120.md（同目录上层）；完整设计推演见 code_fix_1120/DESIGN_fadvise_dontneed_1120.md
> s71 容器三文件 md5 与 s70 完全一致（11-20 11:50 实测），故仅从 s70 拉取即为双机部署态

| 本地路径 | 大小(B) | md5（远端=本地） | 状态 | 远端来源 |
|---|---|---|---|---|
| deployed_container\transfer.py | 20193 | 1abb1b36cddc7a115b17837958486a3d | OK | container:/vllm-workspace/speculators/hs_connectors/src/hs_connectors/transfer.py |
| deployed_container\readonly_views.py | 5291 | 1295b1f3657316ee5b084e900d45fc1e | OK | container:/vllm-workspace/speculators/hs_connectors/src/hs_connectors/readonly_views.py |
| deployed_container\page_cache_drop.py | 2727 | 5b372fb93cd358489228541346aaf8c1 | OK | container:/vllm-workspace/speculators/hs_connectors/src/hs_connectors/page_cache_drop.py |
| scripts\launch_v3r.sh | 4812 | f7bc15fca57bcebd50928863a9face2d | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/scripts/launch_v3r.sh |
| scripts\launch_v3r_views.sh | 4902 | 7515f8863e8cc161542813e34974c602 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/scripts/launch_v3r_views.sh |
| code_fix_1120\DESIGN_fadvise_dontneed_1120.md | 26497 | 2366e3bb6c813e3724bfed61bb08993a | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/DESIGN_fadvise_dontneed_1120.md |
| code_fix_1120\hs_readonly_views.py | 5291 | 1295b1f3657316ee5b084e900d45fc1e | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/hs_readonly_views.py |
| code_fix_1120\page_cache_drop.py | 2727 | 5b372fb93cd358489228541346aaf8c1 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/page_cache_drop.py |
| code_fix_1120\patch_transfer_1120.py | 1549 | 5b5e36656c6ff2ca57dd4829ec46b241 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/patch_transfer_1120.py |
| code_fix_1120\patch_transfer_v2_1120.py | 2406 | 502b72ecd8e0637335a4f41b477168e8 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/patch_transfer_v2_1120.py |
| code_fix_1120\patch_transfer_s71.py | 2184 | 5d0044b83256cf9a3e9e5d333035d39b | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/patch_transfer_s71.py |
| code_fix_1120\test_page_cache_drop.py | 13659 | 57885f9adc42de70816f81d0c77fe3dc | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/test_page_cache_drop.py |
| code_fix_1120\run_ut8_1120.py | 426 | f076bdef6b378d55e7308b51d3480588 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/run_ut8_1120.py |
| code_fix_1120\make_launch_variant.sh | 521 | 0df14ddd9edc2e25f9adb96416734bec | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/make_launch_variant.sh |
| code_fix_1120\sync_s71.sh | 494 | e7529d41052e545f8f50c821cf14bc39 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/sync_s71.sh |
| code_fix_1120\debug_views.py | 1603 | 9ad9912368d6b3eaeaa22884cc38981a | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/debug_views.py |
| code_fix_1120\debug2_views.py | 361 | 1d2fc09a9c692fe82058ccd4dc247777 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/debug2_views.py |
| code_fix_1120\probe_anchor.sh | 588 | cbb1458a44d6b639d1efd22a1b955898 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/probe_anchor.sh |
| code_fix_1120\probe_anchor2.sh | 605 | 4d43be2066c8a05eef6c447a9c0b5e30 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/probe_anchor2.sh |
| code_fix_1120\probe_pool.sh | 915 | a582b621a8d91fb3768e13144baeb247 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/probe_pool.sh |
| code_fix_1120\hs_connectors_backup_1120.tar.gz | 56878 | 8d6be93a5418af25028e3927ff8e5082 | OK | cephfs:/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/code_fix_1120/hs_connectors_backup_1120.tar.gz |
| ops\emergency_restart.sh | 1708 | 5096ffe65394bbdf49807029c50905a0 | OK | devserver:/tmp/emergency_restart.sh |
| ops\views_switch.sh | 2243 | (本地生成) | OK | 本地→devserver 已预置 |

## 目录结构
- deployed_container/ —— 容器内实际运行的三个文件（transfer.py 已接线版 + 两优化模块）
- scripts/ —— launch_v3r.sh（现行）+ launch_v3r_views.sh（views 激活变体，差异仅 3 个 env）
- code_fix_1120/ —— 源交付目录：设计文档/源码/补丁三代/16 用例测试/排障探针/原始包备份 tar.gz（回滚用）
- ops/ —— emergency_restart.sh v2（应急重启）+ views_switch.sh（step700 边界切换，已批准）
