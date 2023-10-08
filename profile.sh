#!/bin/bash

VTUNE=vtune
TARGET=infer-openvino
DATE="$(date +%Y%m%d%H%M%S)"
BATCH=(1 8)
CORES=(10)
STEPS=(1)
echo $DATE

for s in ${STEPS[@]}; do
    for b in ${BATCH[@]}; do
        for c in ${CORES[@]}; do
            LOG="results-vtune/${DATE}-${TARGET}-s${s}-b${b}-c${c}"
            ${VTUNE} -collect hotspots -r $LOG python ${TARGET}.py \
                --num-inference-steps $s \
                --num-images-per-prompt $b\
                --num-cores $c
            sleep 5
        done
    done
done

