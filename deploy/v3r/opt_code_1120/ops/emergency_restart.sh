#!/bin/bash
# Emergency controlled restart: osd28 flapping + s70 avail 46G < 150 red line.
# Recovery point: epoch0_step400 (verified saved). Abandon steps 400-425.
set -u
LOGD=/mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/log
TS=$(date +%H%M%S)
echo "=== A) backup log ==="
timeout 30 ssh s70 "cp $LOGD/train_v3r_r0.log $LOGD/train_v3r_r0.log.bak_1120_$TS && echo backed_up" 2>/dev/null
echo "=== B) TERM both (bracket pattern) ==="
for H in s70 s71; do
  timeout 60 ssh $H "for p in \$(pgrep -f 'speculators[.]train'); do kill -TERM \$p 2>/dev/null; done; echo termed" 2>/dev/null
done
echo "=== C) wait for clear ==="
sleep 90
for H in s70 s71; do
  P=$(timeout 30 ssh $H "pgrep -fc 'speculators[.]train' 2>/dev/null || echo 0" 2>/dev/null)
  echo "$H procs=$P"
  if [ "$P" != "0" ]; then
    timeout 60 ssh $H "for p in \$(pgrep -f 'speculators[.]train'); do kill -9 \$p 2>/dev/null; done; echo forced" 2>/dev/null
    sleep 10
  fi
done
echo "=== D) memory after clear ==="
for H in s70 s71; do
  timeout 30 ssh $H "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null | xargs echo "$H avail_GiB:"
done
echo "=== E) ensure containers up (13.6 lesson) then relaunch ==="
for H in s70 s71; do
  timeout 60 ssh $H "docker ps --format '{{.Names}}' | grep -q '^train_ddp$' || docker start train_ddp" 2>/dev/null
done
sleep 5
timeout 60 ssh s70 "docker exec -d train_ddp bash -c 'bash /mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/scripts/launch_v3r.sh 0 16 2'" && echo s70_launched
sleep 5
timeout 60 ssh s71 "docker exec -d train_ddp bash -c 'bash /mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/scripts/launch_v3r.sh 1 16 2'" && echo s71_launched
echo "=== done, verify in 5min ==="
