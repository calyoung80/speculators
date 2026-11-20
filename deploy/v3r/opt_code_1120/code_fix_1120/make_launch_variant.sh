cd /mnt/cephfs/hw/yangyunsheng/dspark_fresh32k_v3r/scripts || exit 1
sed -e 's|^export MALLOC_CONF=thp:never$|export MALLOC_CONF=thp:never,dirty_decay_ms:0,muzzy_decay_ms:0|' \
    -e 's|^export NUMPY_MADVISE_HUGEPAGE=0$|export NUMPY_MADVISE_HUGEPAGE=0\nexport HS_READONLY_VIEWS=1\nexport HS_FADVISE_DONTNEED=1|' \
    launch_v3r.sh > launch_v3r_views.sh
chmod +x launch_v3r_views.sh
echo "=== diff vs original ==="
diff launch_v3r.sh launch_v3r_views.sh
echo "=== byte sizes ==="
wc -c launch_v3r.sh launch_v3r_views.sh
