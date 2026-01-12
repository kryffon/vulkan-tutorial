#!/bin/bash

set -e
glslc ./shaders/basic.vert -o ./shaders/bin/basic.vert.spv
glslc ./shaders/basic.frag -o ./shaders/bin/basic.frag.spv
odin run . -debug
