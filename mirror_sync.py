#!/Users/mac/asr_env/bin/python3
# -*- coding: utf-8 -*-
"""rsync 本地镜像同步（2026-10-04）——根治 SMB 挂载掉线对实时管线的影响

【为什么是 python3 而不是 rsync/bash】实测 launchd 环境下 TCC 按"发起系统调用的
进程"做 /Volumes 访问归因：asr_env 的 python3 已获磁盘授权（5008/backfill 同款），
而 /bin/bash、/usr/bin/perl、/bin/ls、rsync 全部 EPERM 被拒。故拉取/推送的 NAS IO
必须由本脚本（asr_env python3）亲自执行。

数据流：手机APK → NAS(上传缓冲) →【pull 每分钟】→ 本地镜像
        本地镜像 → audio_processor/5008 全本地处理归档
        本地 processed/failed →【push 每5分钟】→ NAS 备份（推完删本地）
        本地 audio_segments →【push 增量推】→ NAS 归档（本地保留3天作播放缓存）

用法: mirror_sync.py pull | push
硬约束：
  - 永不触碰 processed/ 之外的历史存量（5~9月哭声补跑仍读 NAS watched）
  - 拉取只限近3天日期目录 + mtime>30s + 双stat尺寸稳定确认（防拉走正在上传的半截文件）
  - 所有 NAS 访问带子线程硬超时，挂载假死绝不吊死（对齐 audio_processor 模式）
  - 本地磁盘 <3GB 时暂停 pull
  - 拉取经 .part 临时名落地、校验尺寸后改名，杜绝半截文件混入实时管线
"""
import os
import sys
import json
import time
import shutil
import threading
import fcntl
from datetime import datetime, timedelta

MIRROR = "/Users/mac/asr_mirror/records"
NAS = "/Volumes/download/records"
LOG = "/Users/mac/asr-server/log/mirror_sync.log"
DEVICES = ["Pixel-6", "Pixel-5"]
PULL_DAYS = 3            # pull 覆盖最近 N 天日期目录
PULL_MIN_AGE_SEC = 30    # mtime 早于该秒数才考虑拉（配合双 stat 稳定确认防半截）
STABILITY_PROBE_SEC = 3  # 双 stat 间隔：两次尺寸一致才认定上传完成，否则留给下轮
KEEP_LOCAL_DAYS = 3      # processed/failed/segments 本地保留天数
WATCHED_STALE_DAYS = 7   # watched 目录滞留文件兜底清理天数
MIN_FREE_GB = 3          # 本地磁盘低于该值暂停 pull
COPY_TIMEOUT = 180       # 单文件复制硬超时（秒）
LIST_TIMEOUT = 10        # 目录列举硬超时（秒）
LOCK_FILE = "/tmp/asr_mirror_sync.lock"
SEG_STATE = os.path.join(MIRROR, ".segments_pushed.json")  # 切片增量推送状态
PROC_STATE = os.path.join(MIRROR, ".proc_pushed.json")     # processed/failed 备份推送状态
FLUSH_EVERY = 200        # 每成功推送 N 个文件即增量落盘状态（push 被卡死/杀也不丢进度）

_log_fh = None


def log(tag, msg):
    global _log_fh
    try:
        if _log_fh is None:
            _log_fh = open(LOG, "a", encoding="utf-8")
        _log_fh.write(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] [{tag}] {msg}\n")
        _log_fh.flush()
        # 日志滚动：超 20MB 保留尾部 5MB
        if os.path.getsize(LOG) > 20 * 1024 * 1024:
            _log_fh.close()
            _log_fh = None
            with open(LOG, "rb") as f:
                f.seek(-5 * 1024 * 1024, 2)
                tail = f.read()
            with open(LOG + ".tmp", "wb") as f:
                f.write(tail)
            os.replace(LOG + ".tmp", LOG)
    except Exception:
        pass


_TIMEOUT = object()  # 哨兵：区分"超时"与"函数返回 None"（shutil.copy2 等返回 None）


def run_with_timeout(fn, timeout, *args, **kwargs):
    """子线程执行 fn，超时返回 _TIMEOUT 哨兵（线程留后台自灭，主流程绝不吊死）"""
    result = {}

    def _do():
        try:
            result["v"] = fn(*args, **kwargs)
        except Exception as e:
            result["err"] = str(e)

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return _TIMEOUT
    if "err" in result:
        raise RuntimeError(result["err"])
    return result.get("v")


def safe_listdir(path, timeout=LIST_TIMEOUT):
    """带硬超时的目录列举；假死返回 None（与成功但空目录区分）"""
    try:
        v = run_with_timeout(lambda: os.listdir(path), timeout)
        return None if v is _TIMEOUT else v
    except Exception:
        return None


def probe_nas():
    """NAS 探活：根目录可列举（带超时）。TCC 环境下本进程已获磁盘授权。"""
    return isinstance(safe_listdir(NAS, 10), list)


def copy_with_verify(src, dst, timeout=COPY_TIMEOUT):
    """复制文件到 dst.part → 校验尺寸 → 原子改名。返回 True=已就位；False=失败/超时。
    失败时清理 .part 残留。"""
    part = dst + ".part"
    try:
        def _do():
            shutil.copy2(src, part)
        if run_with_timeout(_do, timeout) is _TIMEOUT:
            log("COPY", f"⚠️ 复制超时({timeout}s): {src}")
            _silent_rm(part)
            return False
        if os.path.getsize(part) != os.path.getsize(src):
            log("COPY", f"⚠️ 尺寸不一致: {dst}（源{os.path.getsize(src)} != 目标{os.path.getsize(part)}）")
            _silent_rm(part)
            return False
        os.replace(part, dst)
        return True
    except Exception as e:
        log("COPY", f"⚠️ 复制异常 {src} → {dst}: {e}")
        _silent_rm(part)
        return False


def _silent_rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def recent_dates():
    today = datetime.now().date()
    return [(today - timedelta(days=i)).isoformat() for i in range(PULL_DAYS)]


def _stat_fp(fp, timeout=5):
    """带超时的 (mtime, size) 探测；SMB 假死/文件消失返回 None（留给下轮）"""
    def _do():
        st = os.stat(fp)
        return st.st_mtime, st.st_size
    try:
        v = run_with_timeout(_do, timeout)
    except Exception:
        return None
    return None if v is _TIMEOUT else v


def free_gb():
    try:
        return shutil.disk_usage("/").free / (1024 ** 3)
    except Exception:
        return None


# ==================== pull：NAS → 本地镜像 ====================
def do_pull():
    free = free_gb()
    if free is not None and free < MIN_FREE_GB:
        log("PULL", f"⛔ 本地磁盘仅剩 {free:.1f}GB（阈值 {MIN_FREE_GB}GB），暂停拉取")
        return

    if not probe_nas():
        log("PULL", "⛔ NAS 挂载无响应（10s 硬超时），本轮跳过（本地数据不受影响）")
        return

    pulled_total = 0
    for dev in DEVICES:
        for date in recent_dates():
            src_dir = os.path.join(NAS, dev, date)
            entries = safe_listdir(src_dir)
            if not entries:  # None=假死, []=空
                if entries is None:
                    log("PULL", f"⚠️ 列举超时跳过: {src_dir}")
                continue
            dst_dir = os.path.join(MIRROR, dev, date)
            os.makedirs(dst_dir, exist_ok=True)

            names = sorted(n for n in entries if n.endswith(".m4a"))
            # 2026-10-05 性能修复：稳定确认由"逐文件各 sleep(3)"改为"整批 sleep 一次"。
            # 旧实现循环内调用 is_upload_stable()（内含 time.sleep(STABILITY_PROBE_SEC)），
            # 284 个文件需约 852s 才完成一轮，表现为"拉取卡住零产出"。
            # 语义不变：仍是"间隔 STABILITY_PROBE_SEC 两次 stat 尺寸一致"才认定上传完成。
            cand = []
            for n in names:
                st = _stat_fp(os.path.join(src_dir, n))
                if st is None:
                    continue
                mtime, size = st
                if time.time() - mtime > PULL_MIN_AGE_SEC:
                    cand.append((n, size))
            if not cand:
                continue
            time.sleep(STABILITY_PROBE_SEC)
            stale = []
            for n, size in cand:
                st = _stat_fp(os.path.join(src_dir, n))
                if st is not None and st[1] == size:
                    stale.append((n, os.path.join(src_dir, n)))
            if not stale:
                continue

            ok = 0
            for n, fp in stale:
                if copy_with_verify(fp, os.path.join(dst_dir, n)):
                    try:
                        os.remove(fp)  # 校验通过才删 NAS 源（等价 --remove-source-files）
                        ok += 1
                    except OSError as e:
                        log("PULL", f"⚠️ 删 NAS 源失败 {n}: {e}")
            if ok:
                log("PULL", f"✅ {dev}/{date} 拉取 {ok} 个文件 → 本地（NAS 源已删）")
            pulled_total += ok
    log("PULL", f"本轮共拉取 {pulled_total} 个文件")


# ==================== push：本地 → NAS 备份 ====================
def _load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(path, state):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except Exception as e:
        log("PUSH", f"⚠️ 状态保存失败 {path}: {e}")


def _load_seg_state():
    return _load_state(SEG_STATE)


def _save_seg_state(state):
    _save_state(SEG_STATE, state)


def do_push():
    if not probe_nas():
        log("PUSH", "⛔ NAS 挂载无响应，本轮跳过（本地数据继续累积，恢复后自动补推）")
        return

    # 状态只在循环外加载一次、跨设备合并（原实现把 load/save 放在设备循环内，
    # 且 save 只保留当前设备的 alive 键，导致后处理的设备把先处理的整文件抹掉
    # → 两台设备每轮互相清零 → 全量重推。见 do_push 内 seg 段注释。）
    proc_state = _load_state(PROC_STATE)
    seg_state = _load_seg_state()
    proc_dirty = 0   # 自上次落盘以来新增的成功推送数
    seg_dirty = 0

    for dev in DEVICES:
        # 1. processed/failed 推 NAS 备份；本地保留 KEEP_LOCAL_DAYS 天
        # 【2026-10-05】不再"推完即删"：backfill_pixels.py 已改读本地镜像 processed/，
        # 需本地留档，否则补救无源可读。本地过期由 do_cleanup() 按 mtime 兜底清理。
        # 幂等靠本地状态文件（不是 NAS stat）：已备份的直接跳过，稳态下本轮对 NAS
        # 零访问；且有新文件才 makedirs——否则每 5 分钟重复 NAS 调用会推活挂死的
        # SMB 会话（2026-10-05 09:07/09:26 各卡死一个 push 进程）。
        for sub in ("processed", "failed"):
            base = os.path.join(MIRROR, dev, sub)
            if not os.path.isdir(base):
                continue
            for date in sorted(d for d in os.listdir(base) if not d.startswith(".")):
                date_dir = os.path.join(base, date)
                if not os.path.isdir(date_dir):
                    continue
                pending = []
                for n in sorted(os.listdir(date_dir)):
                    if not n.endswith(".m4a"):
                        continue
                    try:
                        size = os.path.getsize(os.path.join(date_dir, n))
                    except OSError:
                        continue
                    rel = f"{dev}/{sub}/{date}/{n}"
                    if proc_state.get(rel) == size:
                        continue  # 已备份（本地记录），跳过——不碰 NAS
                    pending.append((n, rel, size))
                if not pending:
                    continue  # 该日期无新文件 → 本轮完全不访问 NAS
                nas_date_dir = os.path.join(NAS, dev, sub, date)
                os.makedirs(nas_date_dir, exist_ok=True)
                for n, rel, size in pending:
                    if copy_with_verify(os.path.join(date_dir, n), os.path.join(nas_date_dir, n)):
                        proc_state[rel] = size
                        proc_dirty += 1
                        if proc_dirty >= FLUSH_EVERY:
                            _save_state(PROC_STATE, proc_state)
                            proc_dirty = 0

        # 2. audio_segments 增量归档（本地保留 KEEP_LOCAL_DAYS 天作播放缓存）
        # 【2026-10-05 修复】原实现有两处缺陷，导致两台设备的片段记录每轮互相清零 → 全量重推：
        #   (a) 状态键不含设备前缀：rel 相对 MIRROR/<dev>/audio_segments，Pixel-5/6 共用命名空间；
        #   (b) state 的 load/save 都在设备循环内，且 save 只保留"当前设备"的 alive 键
        #       → 后处理的设备把先处理的整文件抹掉（DEVICES=[Pixel-6,Pixel-5] 时尤甚）。
        # 现改为：键带设备前缀 dev/rel；循环外统一 load/save；每 FLUSH_EVERY 个增量落盘。
        seg = os.path.join(MIRROR, dev, "audio_segments")
        if os.path.isdir(seg):
            for date in sorted(d for d in os.listdir(seg) if not d.startswith(".")):
                date_dir = os.path.join(seg, date)
                if not os.path.isdir(date_dir):
                    continue
                # 先本地筛出待推文件；无新增则整目录跳过，不再对 NAS 做任何调用
                # （原实现每轮都 makedirs，稳态下纯属空转，还会推活挂死的 SMB 会话）
                pending = []
                for root, _dirs, files in os.walk(date_dir):
                    rel_root = os.path.relpath(root, seg)
                    for n in files:
                        if n.endswith(".part"):
                            continue
                        src = os.path.join(root, n)
                        rel = os.path.join(rel_root, n).lstrip("./")  # NAS 目标相对路径
                        key = f"{dev}/{rel}"                          # 状态键：带设备前缀
                        try:
                            size = os.path.getsize(src)
                        except OSError:
                            continue
                        if seg_state.get(key) == size:
                            continue  # 已确认推送过，跳过（省掉大量 NAS 操作）
                        pending.append((src, rel, key, size))
                if not pending:
                    continue
                for src, rel, key, size in pending:
                    dst = os.path.join(NAS, dev, "audio_segments", rel)
                    # 对账：NAS 已有同名同尺寸文件 → 直接补记状态、跳过重传
                    # （一次性回收历史积压：如 Pixel-5 那批早已归档却被丢状态的片段）
                    st = _stat_fp(dst)
                    if st is not None and st[1] == size:
                        seg_state[key] = size
                        seg_dirty += 1
                    else:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if copy_with_verify(src, dst):
                            seg_state[key] = size
                            seg_dirty += 1
                    if seg_dirty >= FLUSH_EVERY:
                        _save_seg_state(seg_state)
                        seg_dirty = 0

    # processed/failed 状态：修剪到本地现存文件，防无限膨胀
    alive_proc = set()
    for dev in DEVICES:
        for sub in ("processed", "failed"):
            base = os.path.join(MIRROR, dev, sub)
            if not os.path.isdir(base):
                continue
            for date in os.listdir(base):
                dd = os.path.join(base, date)
                if os.path.isdir(dd):
                    for n in os.listdir(dd):
                        if n.endswith(".m4a"):
                            alive_proc.add(f"{dev}/{sub}/{date}/{n}")

    # audio_segments 状态：同样修剪到本地现存文件（键含设备前缀）
    alive_seg = set()
    for dev in DEVICES:
        seg = os.path.join(MIRROR, dev, "audio_segments")
        if not os.path.isdir(seg):
            continue
        for root, _dirs, files in os.walk(seg):
            rel_root = os.path.relpath(root, seg)
            for n in files:
                if not n.endswith(".part"):
                    alive_seg.add(f"{dev}/{os.path.join(rel_root, n).lstrip('./')}")

    # 末次落盘：修剪 + 补齐未达 FLUSH_EVERY 的尾部增量
    _save_seg_state({k: v for k, v in seg_state.items() if k in alive_seg})
    _save_state(PROC_STATE, {k: v for k, v in proc_state.items() if k in alive_proc})
    log("PUSH", "✔ 推送轮完成")

    do_cleanup()


# ==================== cleanup：本地过期数据清理 ====================
def do_cleanup():
    now = time.time()
    for dev in DEVICES:
        # processed/failed: >KEEP_LOCAL_DAYS 天兜底硬删（正常 push 已删光）
        for sub in ("processed", "failed"):
            base = os.path.join(MIRROR, dev, sub)
            if not os.path.isdir(base):
                continue
            n = 0
            for root, _dirs, files in os.walk(base):
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        if now - os.path.getmtime(fp) > KEEP_LOCAL_DAYS * 86400:
                            os.remove(fp)
                            n += 1
                    except OSError:
                        continue
            # 清空日期目录
            for d in os.listdir(base):
                dp = os.path.join(base, d)
                if os.path.isdir(dp):
                    try:
                        if not os.listdir(dp):
                            os.rmdir(dp)
                    except OSError:
                        pass
            if n:
                log("CLEAN", f"🧹 {dev}/{sub} 兜底清理 {n} 个 >{KEEP_LOCAL_DAYS}天 文件")

        # audio_segments 日期目录: >KEEP_LOCAL_DAYS 天且 NAS 同目录非空 → 删本地（播放回退 NAS）
        seg = os.path.join(MIRROR, dev, "audio_segments")
        if os.path.isdir(seg):
            for date in sorted(os.listdir(seg)):
                dp = os.path.join(seg, date)
                if not os.path.isdir(dp) or date.startswith("."):
                    continue
                try:
                    if now - os.path.getmtime(dp) <= KEEP_LOCAL_DAYS * 86400:
                        continue
                except OSError:
                    continue
                nas_files = safe_listdir(os.path.join(NAS, dev, "audio_segments", date), 10)
                if nas_files:  # 非 None 且非空 = NAS 已有归档
                    shutil.rmtree(dp, ignore_errors=True)
                    log("CLEAN", f"🧹 {dev}/audio_segments/{date} 已归档 NAS，本地清理")

        # watched 目录滞留文件兜底（audio_processor 长期停摆 >7 天才触发）
        n = 0
        for date in recent_dates():
            dd = os.path.join(MIRROR, dev, date)
            if not os.path.isdir(dd):
                continue
            for fn in os.listdir(dd):
                fp = os.path.join(dd, fn)
                if not fn.endswith(".m4a") or not os.path.isfile(fp):
                    continue
                try:
                    if now - os.path.getmtime(fp) > WATCHED_STALE_DAYS * 86400:
                        os.remove(fp)
                        n += 1
                except OSError:
                    continue
        if n:
            log("CLEAN", f"🧹 {dev} watched 滞留清理 {n} 个文件")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("pull", "push"):
        print("用法: mirror_sync.py pull|push")
        sys.exit(1)

    # 互斥锁：防同模式重叠（launchd 周期短于执行耗时时）
    lock_fh = open(LOCK_FILE + f".{mode}", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(0)  # 上一轮还在跑，直接让路

    try:
        do_pull() if mode == "pull" else do_push()
    except Exception as e:
        log(mode.upper(), f"⛔ 未捕获异常: {e}")
    finally:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
        except Exception:
            pass


if __name__ == "__main__":
    main()
