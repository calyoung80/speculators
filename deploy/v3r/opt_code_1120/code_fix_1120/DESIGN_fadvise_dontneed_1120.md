# 设计文档：states 读取后即弃页缓存（FADV_DONTNEED）——v3r 内存结构性超卖的 C 层缓解

日期：2026-11-20 00:30 定稿；01:30 增 §F 社区先例；02:00 增 §I 路 A 实现与调测 ｜作者：监控代理｜状态：**v1+v2 均已实现；容器内 16/16 全过（v1 的 load_file 语义结论已勘误，见 §I.7）**
对象：hs_connectors 文件后端（容器 /vllm-workspace/speculators/hs_connectors/src/hs_connectors/transfer.py）

---

## A. 分析记录（为何做这件事）

### A.1 事故事实（11-19 晚，滑落 #8）

- 22:36:30 libceph: osd29 down → 34 秒内 MemAvailable 19.9→1.5G、Inactive(file) 15→2.9G，s70/s71 双机 sshd 饿死，训练停于 step 392（存档点 300，损 92 步）。
- s70 28 分钟后自愈（23:03），s71 至本文撰写时仍未复活（>2h，TCP:22 超时，ICMP 通）。
- 死亡瞬间分量：AnonPages 807G + file LRU ≈500G ≈ **1.3TB 工作集 > 1TB 物理内存**。

### A.2 根因：双份驻留 + 无界保留

```
states 文件（/mnt/cephfs/glm5_3_hidden，实测 2.3TB，每样本 cmpl-{req}-0-{hash}.safetensors ≈1-2.2GB，共 2441 文件）
  → 读取进入内核页缓存                    【第 1 份：file LRU ~500G，"可回收"】
  → 复制进 worker 进程堆（torch 张量/整理缓冲）【第 2 份：Anon ~807G，不可回收】
```

- 代码路径：`OfflineHiddenStatesSource.get_one → transfer.get_cached(idx) → _load_hs_file → safetensors.load_file`。
- 读过的 states 页**没有任何驱逐机制**，驻留量随训练推进单调增长（上游 #810 原文：“保留量应受活跃窗口约束，而非随训练步数无限增长”——该保证在现版本**不存在**）。
- 正常态页缓存份是“免费”的（内存压力时内核回收）；但 osd 抖动把 file LRU **钉死**（osd19 五次复现，今晚 osd29）→ 1.3TB 穿仓 → 秒死。

### A.3 社区证据（上游 = vllm-project/speculators，即本仓库母体）

| 编号 | 状态 | 内容 |
|---|---|---|
| #810 RFC | closed not-planned（stale bot，11-15） | 有界异步 hidden-state 扇出：窗口租约+驱逐+背压 |
| #811 PR | open draft / needs-rebase（9-22 起无更新） | #810 完整实现（POSIX 文件 artifact 缓存） |
| #1239 Roadmap | open | 维护��承认“大 hidden state 尺寸是现存瓶颈”（ToDo） |

结论：上游确诊同病、方案搁置。我们不等上游，先做 **C 层最小缓解**（不动训练语义）。

### A.4 为什么选 fadvise(DONTNEED)（本设计）

- 每 states 文件每 epoch 只读一次（8 epoch 间隔数小时，页缓存根本撑不到复用）→ 读完即弃**无吞吐代价**；
- 一个系统调用解决 500G 级驻留，不动调度/不动 loss/不动数据语义 → **训练效果零风险**；
- env 门控 opt-in，默认关 → 部署安全（看门狗重拉也不变行为）。

配套（非本代码，另行申请）：jemalloc `dirty_decay_ms:0,muzzy_decay_ms:0`（压 worker 保留页）；PF 2→1（减在途）。

---

## B. 接口设计

### B.1 新模块 `hs_connectors/page_cache_drop.py`

```python
def fadvise_enabled() -> bool:
    """HS_FADVISE_DONTNEED 环境变量为 '1' 时返回 True（每次现读，不缓存）。"""

def drop_page_cache(path: str | os.PathLike, *, min_bytes: int | None = None) -> bool:
    """对 path 整文件发起 posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)，请求内核
    丢弃该文件的干净页缓存。
    - min_bytes：文件小于该值则跳过（默认 64MiB，防小文件刷调用；None=用默认）。
    - 任何失败（文件不存在/不支持/权限）静默吞掉，返回 False；绝不抛异常。
    - 返回 True 仅表示 fadvise 已发出（内核不保证立即全部丢弃）。
    """
```

### B.2 接线点 `transfer.py::_load_hs_file`（唯一咽喉，File/FP8 后端共用）

```python
def _load_hs_file(file_path):
    ...（lock 逻辑不变）
    if file_path.exists():
        sample = load_file(file_path)
        drop_page_cache(file_path)      # ← 新增唯一一行调用（内部自带 env 门控）
        return sample
    return None
```

### B.3 行为契约

1. **默认关闭**：env 未设 = 与原代码逐字节等价的行为（多一次 env 读取，µs 级）。
2. **开启后**：每次 states 加载完成即请求弃页缓存；仍映射中的页（若 load_file 返回 mmap 视图）本次为 no-op，由后续 natural munmap 后的下次调用覆盖（见 E.2 v2 预留）。
3. **永不影响训练**：所有异常路径吞掉；返回值不影响调用方。
4. 门控变量：`HS_FADVISE_DONTNEED`（=1 开）；`HS_FADVISE_MIN_BYTES`（默认 67108864）。

---

## C. 单元测试用例（test_page_cache_drop.py，纯 python3 断言，不依赖 pytest）

| # | 用例 | 断言 |
|---|---|---|
| UT1 | 默认关闭 | monkeypatch os.posix_fadvise 计数；env 未设时调用 drop_page_cache → 计数 0 |
| UT2 | 开启后调用 | env=1 + 存在的临时文件 → 恰好 1 次调用，参数=(fd, 0, 0, POSIX_FADV_DONTNEED) |
| UT3 | 异常吞没 | env=1，posix_fadvise 抛 OSError → drop_page_cache 返回 False、不抛 |
| UT4 | 缺文件安全 | env=1，get_cached(不存在 idx) → None 且无异常 |
| UT5 | 小文件跳过 | env=1，1KB 文件 → fadvise 计数 0（<64MiB 守卫） |
| UT6 | 集成·本地盘 | 256MB 文件写/读热 → mmap+mincore 驻留率≈100% → drop → <50% |
| UT7 | 集成·cephfs（关键） | 同 UT6 但文件写在 /mnt/cephfs —— **验证 cephfs 客户端尊重 DONTNEED**；若不降 >50% 则本方案对 cephfs 无效，升级 v2 |
| UT8 | 集成·load_file 语义 | env=1 经 _load_hs_file 加载真实 safetensors：返回张量正确 + drop 发生 + 用 /proc/self/smaps 判定 load_file 是复制还是 mmap 视图（决定是否需要 v2） |

通过标准：UT1-5 全过 + UT6 过 + **UT7 过**（UT7 不过则方案证伪，代码保留但禁用）。

### C.2 实测结果（11-20 00:35，s70 容器内，宿主空载）

| # | 结果 | 备注 |
|---|---|---|
| UT0 模块导入 | PASS | reload 安全 |
| UT1 默认关 | PASS calls=0 | 行为逐字节等价 |
| UT2 参数 | PASS `(fd,0,0,4)` | 4=POSIX_FADV_DONTNEED |
| UT3 异常吞没 | PASS | 注入 OSError 不外泄 |
| UT4 缺文件 | PASS | |
| UT5 小文件守卫 | PASS | |
| UT6 本地盘 | PASS **dropped=256/256MB（100%）** | |
| UT7 cephfs | PASS **dropped=255/256MB（99.6%）** | **cephfs 尊重 DONTNEED——方案对 2.3TB 池有效** |
| UT8 语义 | PASS **load_file=copy**，mapped_after_gc=0 | **立即 fadvise 即完整机制；v2 不需要**；同时闭合 Anon 807G 之谜（每样本=堆内复制份） |

测试缺陷自查记录：首版测试 UT2/UT3 用 1MB 文件被 64MiB 守卫拦截（未真正走到 fadvise 路径）→ 加 `min_bytes=0` 修正；UT8 漏在 main() 调用 → 单独补跑。两处均自纠，最终 9/9。

---

## D. 实现与部署

- 代码目录：本目录（code_fix_1120/）含 page_cache_drop.py、patch_transfer_1120.py（补丁脚本）、patched transfer.py（容器内已打）、test_page_cache_drop.py、run_ut8_1120.py、备份 tar。
- 备份：`hs_connectors_backup_1120.tar.gz`（原包整目录 tar，md5=8d6be93a5418af25028e3927ff8e5082，容器内外一致）。
- 部署状态：**s70 容器已打补丁**（transfer.py:22/24/142 三处 + page_cache_drop.py，py_compile 过）；**s71 待其复活后同步**（同样的两文件复制即可，补丁脚本可重跑）。
- 回滚：tar 解回原路径即恢复（或删 page_cache_drop.py + 还原 transfer.py 单文件，备份内含原件）。
- 安全性：默认关（env 未设）+ 导入 try/except 兜底——过夜看门狗重拉时行为与原版**完全一致**（UT1 验证）。
- **启用**：用户拍板后 launch 脚本加 `export HS_FADVISE_DONTNEED=1`（建议先在恢复窗口观察一轮 free/file-LRU 曲线；预期 file LRU ~500G→~0，free 底线 2-6G→500G+ 量级，osd 钉死失去杀伤面）。

## E. 边界与后续

- E.1 会计异常（file LRU 500G vs Cached 34G 的差）本方案顺带消解：驻留没了，差值也没了。
- E.2 ~~v2 预留~~：**已由 UT8 排除**（load_file=copy，无 mmap 视图残留）。
- E.3 本方案只压"第 1 份"（页缓存）；第 2 份（Anon 807G）靠 jemalloc decay/PF（部署项，另行申请）。
- E.4 上游路径（#811 窗口租约）为根治方向，本方案与之不冲突（可叠加）。
- E.5 本地镜像：D:\GLM5.3_WORK\dspark\1109_summary\（设计文档+代码+测试全一份）。

## F. 社区先例与设计修正（11-20 00:50 补充调研）

### F.1 先例核验表

| 本方案 | 社区先例 | 状态 | 关键实测 |
|---|---|---|---|
| fadvise 即弃（v1 已实现） | [InstinctFlash reclaim_checkpoint_pagecache.py](https://github.com/LH-and-FPGA/InstinctFlash/blob/main/eval/framework_comparison_2026-09-13/reclaim_checkpoint_pagecache.py) | 工具级使用，无框架在 dataloader 路径集成 | 原语标准；接线点（_load_hs_file）为我们的增量 |
| 路 A mmap 视图 | [sglang #37822](https://github.com/sgl-project/sglang/pull/37822) `readonly_safetensors`（collaborator mickqian） | **关闭未合**（10-06）——技术已验证未进主线 | worker 匿名 12GiB→3.4GiB；带宽 0.12→26.6 GiB/s（222×） |
| 路 B 行级分段读取 | vLLM `SafetensorsSlice`（[lazy TP weight loading commit](https://app.semanticdiff.com/gh/vllm-project/vllm/commit/fadfefcc6e784ba8f734f8eb9e606b369a2b363e)，weight_utils.py） | **已进 vLLM 主线**（权重分片路径） | TP 按行切片读 safetensors，行级寻址可行性实证 |
| 有界保留/窗口租约 | speculators [#810](https://github.com/vllm-project/speculators/issues/810) RFC / [#811](https://github.com/vllm-project/speculators/pull/811) PR | RFC 被 stale bot 关闭；PR 停 draft/needs-rebase | — |
| safetensors 双份驻留（跨平台同病） | [ComfyUI #10896](https://github.com/Comfy-Org/ComfyUI/issues/10896)（37 评论） | 已关闭（用户自 workaround copy=False） | DGX Spark 上 safetensors 加载需双倍内存 |
| **生产者侧**（glm5_3_hidden 池的上游端） | [vLLM #66794](https://github.com/vllm-project/vllm/issues/66794) RFC + [PR #66785](https://github.com/vllm-project/vllm/pull/66785)（11-10 提出，0 维护者评论） | 开放中 | extract_hidden_states 从 HBM 池改流式 CPU 页；GLM 式配置 72KiB/token；NPU 64k 并发 2.41x→4.28x；32 单测+逐元素一致性 |

### F.2 由先例导出的设计修正

1. **路 A 必须用 PROT_READ 只读映射**（sglang #37822 实测）：safetensors 默认私有可写映射（`UntypedStorage.from_file(shared=False)`）在设备复制时被驱动以写意图钉页 → **内核写时复制 → 每个触碰页变匿名内存且压力下永不回落**（1GiB 文件实测 1022MiB 变匿名）；只读映射 0 匿名 + 222× 带宽。states 数据只读，PROT_READ 语义正确。
2. 路 B（行级 pread）寻址可行性由 vLLM TP 权重切片实证；锚点前置的 RNG 语义变化仍是训练效果考量（见 §B 讨论与 00:50 问答复盘）。
3. 元观察：六先例中仅 SafetensorsSlice 进主线——社区实现普遍齐备、review 带宽是瓶颈。**内部 fork 直接摘技术是被 sglang 作者自己走通的路径**。

### F.3 分段加载可行性（11-20 00:50 问答复盘，路 B 的设计依据）

- 现状全量加载的根因：**锚点选择在训练核心**（dflash/core.py:352 `select_anchors` GPU 侧每步随机；:439 锚点×block_size=8 连续行块；dspark/model_definitions.py:52 消费 gather 后的 [N,8,h]），而加载层 `_build_sample` 把全长张量 flatten 进队列——加载器无法预知需要哪些行。
- 数学：每样本 states=32768 行×5 层×4096×bf16≈1.34GB；训练真正消费=1024 锚×8 行=8192 行≈**25%**（~340MB）。75% 是"运了不用"（占堆、占队列 19.2GB/rank、占 PCIe 4×）。
- 三路：A=PROT_READ 视图（零语义）；B=锚点前置+行级 pread（-75%，动 RNG 语义需效果验证）；C=数据侧预压实（**不可行**：锚点每 epoch 随机重选，8 epoch 并集≈全覆盖）。
- 待验证项（路 B 前置）：gather 集合闭合性（markov/prev_emb 递归是否只要锚块内行）；metrics 路径是否消费全长（metrics.py 有 num_blocks=seq_len//block_size 的全序列视图）。

## G. 实现记录（v1，11-20 00:20-00:40 执行）

### G.1 执行顺序（用户指令：分析记录→备份→设计→实现）

①备份（容器内 tar 整包，md5=8d6be93a5418af25028e3927ff8e5082，容器内外一致）→ ②设计文档定稿（接口+UT0-UT8 用例定义）→ ③实现（page_cache_drop.py 新模块 + transfer.py 补丁）→ ④容器内 py_compile → ⑤单测 9/9 过。

### G.2 transfer.py 补丁（两处，s70 容器内已应用）

```diff
--- a/hs_connectors/src/hs_connectors/transfer.py
+++ b/hs_connectors/src/hs_connectors/transfer.py
@@ import 区（mooncake_te_store 导入行之后）
+
+try:  # 1120 fix: drop states page cache after load (opt-in via HS_FADVISE_DONTNEED=1)
+    from hs_connectors.page_cache_drop import drop_page_cache
+except Exception:  # pragma: no cover -- training must never die on this
+    def drop_page_cache(*_args, **_kwargs):  # type: ignore[misc]
+        return False
@@ _load_hs_file（File/FP8 后端共用咽喉）
     if file_path.exists():
-        return load_file(file_path)
+        sample = load_file(file_path)
+        drop_page_cache(file_path)
+        return sample
```

补丁以脚本执行（code_fix_1120/patch_transfer_1120.py）：锚点字符串唯一性断言 → 精确替换 → 写回，防重复打/错打。

### G.3 page_cache_drop.py 核心逻辑（全文=code_fix_1120/page_cache_drop.py，本地镜像 1109_summary/）

`fadvise_enabled()`：现读 `HS_FADVISE_DONTNEED`（=="1" 才开，每次现读不缓存）。
`drop_page_cache(path, *, min_bytes=None)`：门控 → `stat` 尺寸守卫（默认 64MiB，`HS_FADVISE_MIN_BYTES` 可覆写）→ `os.open(O_RDONLY)` → `os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` → 关 fd。**任何异常吞掉返回 False，绝不抛**；非 POSIX 平台自动失效。

### G.4 测试实证（s70 容器内，宿主空载，11-20 00:35）

```
UT0 module_import PASS                UT4 missing_file PASS
UT1 default_off calls=0 PASS          UT5 small_file_guard PASS
UT2 (fd,0,0,4=DONTNEED) PASS          UT6 local_disk dropped=256/256MB（100%）PASS
UT3 exception_swallowed PASS          UT7 cephfs dropped=255/256MB（99.6%）PASS ★
UT8 load_file=copy, mapped_after_gc=0, fadvise_calls=1 PASS ★
```

★ 决定性结论：**cephfs 客户端尊重 DONTNEED（99.6% 释放）——对 2.3TB 池有效**；**load_file 为复制语义——立即 fadvise 即完整机制**（若未来改路 A mmap 视图，则需按 §F.2 用 PROT_READ + 样本释放后延迟弃置）。

### G.5 部署状态与回滚

- **s70 容器**：已打补丁（transfer.py 三处 + page_cache_drop.py），py_compile 过；默认关=行为与原版逐字节等价（UT1 实证）。
- **s71 容器**：待主机复活后同步（重跑 patch_transfer_1120.py，锚点断言防重复）。
- **回滚**：code_fix_1120/hs_connectors_backup_1120.tar.gz 解回原路径即恢复。
- **启用**（待用户拍板）：launch 脚本加 `export HS_FADVISE_DONTNEED=1`；预期 file LRU ~500G→~0、free 底线 2-6G→500G+ 量级、osd 钉死失去主要杀伤面。
- 过夜看门狗重拉不受影响（默认关+导入兜底+py_compile 验证）。

### G.6 v1 局限与下一档

- v1 只压"第 1 份"（页缓存 ~500G）；"第 2 份"（Anon ~807G=load_file 复制+collate 队列 307G+保留页）仍在——下一档见 §E.3 与 §F.2：
  - 路 A（PROT_READ 视图）：双份坍缩为单份，零训练语义变化；
  - 路 B（锚点前置+行级 pread）：样本 1.34G→340MB（25%，1024 锚×8 行块/32768 行），动 RNG 语义需效果验证；
  - 配套部署项：jemalloc `dirty_decay_ms:0,muzzy_decay_ms:0`、PF 2→1。
- 生产者侧根治（vLLM #66794 流式 CPU 页）落地后，states 文件格式/写入模式可能变化，届时本补丁的接线点（_load_hs_file）不变、语义自动适配（读完即弃对新格式同样成立）。

## H. 猜测证伪档��（11-19/20 深夜排障全程，防止下次重猜）

| # | 当时的猜测 | 证伪证据 | 正解 | 下次一眼判别法 |
|---|---|---|---|---|
| H1 | "饱和（worker 750G）是正常稳态，可长期运行"（20:15-22:00 三轮巡逻定性） | 挤压事件 free 底线序列递减（2.3G→2.2G），22:35 死亡 | **边界运行**：恢复余量≈0，每次挤压在挖坟 | 盯**挤压事件 free 底线序列**：递减=死亡螺旋前兆；"能恢复"≠"可持续" |
| H2 | "avail 单点骤降=估计器噪声"（21:25，528G 一跳） | memtrace 同期 free 同降 | 真实挤压事件 | avail 与 free **同向**才可信；只看 avail 单点易误判 |
| H3 | "可持续抖动平衡"（22:07 对挤压-释放周期的定性） | 同 H1 | 边界运行的振荡相位 | 出现挤压-释放周期本身=已在边界上，不需要"平衡"定性 |
| H4 | "第二轮挤压已 V 型恢复"（22:25 巡逻） | memtrace：22:30 free=2.7G | 下降中继 | 恢复判定用 **free 底线**验证，不用 avail 回弹 |
| H5 | "第三形态：纯饱和活锁，无 osd 参与"（22:37 死亡后第一判断） | flap4 实录：osd29 down@22:36:30，34 秒内 Inactive(file) 15→3G 钉死 | **同 #6/#7 机制**（osd 抖动）+饱和放大器；细究：22:35 avail=18.1 早于 flap 90 秒=慢性超卖已临终，flap 补刀 | **dmesg -W 观察器比推断快**——出事先查实录再立假说 |
| H6 | "数据集仅 14G → 可本地化到 58G 本地盘"（下午方案前提） | STATE 池实测 **2.3TB**（glm5_3_hidden；du 当场超时被我放过） | 14G=arrow 索引；states 走 2.3TB 共享池 | **du 超时本身就是"池巨大"的信号**——限时重试或 statfs，不许放行 |
| H7 | "486G 会计异常（file LRU 500G vs Cached 34G）=好奇项存档" | 事后：states 池页的指纹 | 大差值=有大文件池被读/映射 | 会计异常=数据结构指纹，**追 Reader 打开的文件**而不是记档案 |
| H8 | "s71 会像 s70 一样 ~28 分钟自愈"（23:10 隐含预期） | 未证伪也未验证：4h+ 仍死（TCP:22 超时、ICMP 通） | 未知——无带外管理口时的盲区 | 无法判别时**明说无法判别**，别给时间表 |
| H9 | （更早，审计表在案）numpy bypass THP 论、pin_memory=False | 用户挑战后撤回/永久撤回 | 见交接文档 §13 审计行 | 用户质疑=最高优先级复核信号 |
| H10 | v1 UT8 判"load_file=copy"（v1 交付的核心结论之一） | v2 修单位 bug 后实测 **mmap-view**（存活时映射在/死后归零） | v1 即刻 fadvise≈no-op；有效组合=双 env（§I.7） | **跨量纲比较先统一单位**（KB vs B）；关键结论要有第二判据 |
| H11 | "v2 模块对真实文件失败原因未知"（VUT2-4 假失败，一度怀疑 dtype 不支持） | HS_VIEWS_DEBUG 打栈：`__slots__` 漏 `__weakref__` → finalize TypeError 被兜底吞 | 一行修复后 16/16 | **兜底 except 必配调试开关**，否则吞证据 |
| H12 | 池文件命名假设 `hs_*.safetensors`（未先 ls） | 实为 `cmpl-*`，2441 文件 | glob 修正后 VUT 套件才真正跑起来 | **列目录先于写 glob**——同 H6 病根，第二次犯 |

### H.10 由证伪史固化的决策规则（已生效）

1. **free 底线 <5G 持续 2 个采样 → 禁止"等存档"，立即重启或抛给用户**（MemAvail 150 红线在饱和态下滞后于真实死亡信号）。
2. 挤压-恢复周期出现 = 边界运行，自动收紧巡逻至 240s。
3. 风险决策（等存档 vs 立即重启）用户在线必须问，不自决（22:15 教训：78 步 vs 整夜的期望值算错）。
4. 立假说前先拉实录（dmesg -W / memtrace / flap4），杜绝"先猜后找证"。

## I. 路A实现设计：PROT_READ 零拷贝视图加载（v2，11-20 01:30 定稿）

### I.1 目标与门控

目标：消灭"第 2 份"（load_file 堆复制 ~807G 中的样本原始份）——样本张量=只读映射视图，物理页与页缓存**共享单份**；collate 批次副本（队列）保持不变。门控：`HS_READONLY_VIEWS=1`（默认关=与 v1 行为一致）。与 `HS_FADVISE_DONTNEED` 正交可叠加：视图路径自带**延迟弃置**（映射关闭后 fadvise）。

### I.2 接口（新模块 hs_readonly_views.py）

```python
def views_enabled() -> bool
    """HS_READONLY_VIEWS == '1'。"""

def load_file_views(path) -> dict[str, torch.Tensor] | None
    """解析 safetensors 头（8B 长度 + JSON）→ PROT_READ mmap → 逐张量
    np.frombuffer(memoryview 切片) → torch.from_numpy → bf16 用 .view(torch.bfloat16)
    重解释 → .view(shape)。全程零拷贝。
    生命周期：tensor → numpy 数组 → memoryview → mmap 链式保活；另设 _MMapHolder
    （weakref-able）挂到每个张量属性上，weakref.finalize(holder, drop_page_cache, path)
    ——全部张量死亡 → holder 归零 → munmap → fadvise 延迟弃置。
    失败（损坏头/缺文件/不支持 dtype）→ 返回 None（调用方回退 load_file）。
    """
```

接线（transfer.py::_load_hs_file 第二分支）：

```python
if file_path.exists():
    if views_enabled():
        sample = load_file_views(file_path)
        if sample is not None:
            drop_page_cache(file_path)   # 对映射中页为 no-op；兼容重读场景
            return sample
    sample = load_file(file_path)        # 原路径（默认）
    drop_page_cache(file_path)
    return sample
```

### I.3 安全性论证

- **COW 陷阱规避**（§F.2 sglang 教训）：PROT_READ 映射不可写 → 无 COW 可能；本训练 torch 为 CPU 版，视图仅被 collate 读取（CPU memcpy 不触发 COW），NPU 传输发生在 collate 后的堆副本上。
- **只读性**：视图张量源自只读 buffer，误写会抛错而非静默复制。
- **dtype 覆盖**：BF16/F16/F32/F64/I64/I32/I16/I8/U8/BOOL；FP8 等 exotic → 回退 load_file（不炸）。
- **__metadata__ 等非张量头项**：跳过。

### I.4 测试用例（VUT，追加到 test_page_cache_drop.py）

| # | 用例 | 断言 |
|---|---|---|
| VUT1 | 默认关 | env 未设 → 走 load_file 路径（spy load_file_views 未被调用） |
| VUT2 | 值等价 | 真实 cephfs states 文件：views 输出与 load_file 输出逐张量 equal（值/dtype/shape） |
| VUT3 | 零拷贝 | 加载后样本存活期间 /proc/self/smaps 有该文件映射且权限 r--p；进程 RSS 增量 << 文件大小 |
| VUT4 | 生命周期+延迟弃置 | del 样本+gc → smaps 映射消失；fadvise spy 计数+1（finalizer 触发） |
| VUT5 | 只读 | 视图张量 in-place 写 → 抛错（或 numpy 只读标志） |
| VUT6 | 端到端 | _load_hs_file(views=on) 返回正确张量；token_ids 与 arrow 记录 torch.equal 通过（_build_sample 联动） |
| VUT7 | 回退 | 损坏文件（截断头）→ None → 不抛 |

### I.5 配套部署项（不并入本补丁）

jemalloc decay（`MALLOC_CONF=thp:never,dirty_decay_ms:0,muzzy_decay_ms:0`）+ 双 env 导出 → 独立 launch 变体脚本（不动 launch_v3r.sh，看门狗重拉零影响）。

### I.6 调测结果（11-20 02:00，s70 容器，16/16 全过）

```
UT1-8 同 v1（含 UT8 语义勘误，见下）      VUT1 默认关 views_calls=0 ✓
VUT2 值等价 token_ids:True hidden_states:True（真实 2.2GB 池文件）✓
VUT3 零拷贝 rss_delta=0.0MB file=2256MB mapped=4801308KB readonly=True ✓ ★
VUT4 生命周期+延迟弃置 mapped 4801308KB→0KB, fadvise 0→1 ✓ ★
VUT5 写保护 rc=-11 segfault（子进程内，无静默写）✓ ⚠
VUT6 端到端 _load_hs_file(views=on) values_eq=True ✓
VUT7 损坏回退 None ✓
```

★ VUT3：加载 2.2GB states 后进程 RSS 增量 **0.0MB**（页按需缺页，物理单份与页缓存共享）；VUT4：样本死亡→映射归零+finalizer 触发 fadvise（延迟弃置生效）。
⚠ VUT5：PROT_READ 视图被 in-place 写时 **SIGSEGV 而非异常**（torch.from_numpy 不拦截 numpy 只读标志，警告即"undefined behavior"）。**纪律：数据路径对 states 视图只读**——已核对 `_build_sample`（切片/flatten/torch.equal 均读）与 collate（stack 生成新张量）满足；未来改动数据管线时必须保持。

### I.7 实现缺陷自查（含勘误）

1. **勘误：v1 的 UT8 结论"load_file=copy"是单位 bug**——v1 测试的 view_mode 判据 `mapped_kb >= fsize*0.9`（KB 对字节）恒 False。修正判据后实测 **load_file=mmap-view**（样本存活时映射在、死后归零）。连锁修正：①v1 的"即刻 fadvise"对映射中页≈no-op，单开 `HS_FADVISE_DONTNEED` 对本负载基本无效；②**有效组合=双 env（views+delayed drop）**；③Anon 807G 中"load_file 复制份"假设不成立，重定性为 collate 队列+pin+分配器驻留（jemalloc decay 项的权重上调）；④sglang COW 陷阱（私有可写映射）对默认路径是**潜在**风险（当前 collate 先复制、NPU 不触碰映射页，未触发）——PROT_READ 顺带免疫。
2. `_MMapHolder.__slots__` 漏 `__weakref__` → finalize 抛 TypeError 被兜底吞成 None（VUT2-4 假失败）。教训：**兜底 except 会吞证据，调试开关（HS_VIEWS_DEBUG=1 打栈）应一开始就在**。
3. 池命名假设 `hs_*.safetensors` 错误（实为 `cmpl-{req}-0-{hash}`，2441 文件）——又一个"没先 ls 就假设"（同 H6 病根）。

### I.8 交付物清单（v2 后全部在 cephfs code_fix_1120/ 与本地 1109_summary/）

- `hs_readonly_views.py`（容器内部署为 hs_connectors/readonly_views.py，含 HS_VIEWS_DEBUG 调试开关）
- `patch_transfer_v2_1120.py`（v1→v2 补丁脚本，锚点断言可重跑）
- `test_page_cache_drop.py`（UT1-8 + VUT1-7 全套，主入口含全部用例）
- `debug_views.py` / `debug2_views.py`（排障探针，留档）
- `make_launch_variant.sh` → 生成 `scripts/launch_v3r_views.sh`（3 行差异：双 env + jemalloc decay；TAG/日志路径不变=看门狗兼容）
- **启用方式（待用户拍板）**：看门狗/人工改用 launch_v3r_views.sh，或原脚本加同 3 行。s71 复活后须先跑 patch 脚本同步。
