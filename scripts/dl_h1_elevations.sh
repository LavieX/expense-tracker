#!/bin/bash
cd ~/develop/expense-tracker
for m in 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06; do
    echo "=== Elevations $m ==="
    DISPLAY=:0 expense download --month $m --source elevations --verbose
    echo ""
done
echo "DONE"
