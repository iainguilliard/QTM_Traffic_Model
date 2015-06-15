#!/bin/bash
export SGE_CELL=default
export SGE_EXECD_PORT=537
export SGE_QMASTER_PORT=536
export SGE_ROOT=/opt/gridengine

while read args; do
    echo $args
    /opt/gridengine/bin/linux-x64/qsub -pe orte-slots 12 -N grid001 -v ARG1="$args" solve.sh
done
