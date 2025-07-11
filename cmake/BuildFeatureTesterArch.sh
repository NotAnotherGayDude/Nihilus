#!/bin/bash
cmake -S ./ -B ./Build -DCMAKE_BUILD_TYPE=Release -DNIHILUS_DETECT_ARCH=TRUE
cmake --build ./Build --config=Release -v