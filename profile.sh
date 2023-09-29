#!/bin/bash

VTUNE=vtune
TARGET=profile.py


${VTUNE} -collect performance-snapshot python ${TARGET}
