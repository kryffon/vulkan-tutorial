#!/bin/bash

set -e
glslc ./shaders/basic.vert -o ./shaders/bin/basic.vert.spv
glslc ./shaders/basic.frag -o ./shaders/bin/basic.frag.spv
glslc ./shaders/compute.comp -o ./shaders/bin/compute.comp.spv
odin run . -debug
