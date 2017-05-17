#!/bin/bash
date
while read args; do
    echo "qtm_solve.py" $@ $args
    date
    /System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 qtm_solve.py $@ $args
    date
done
date
