#!/bin/bash
# build_python_lib.sh - Build shared library for Python interface

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building Bloch Python interface..."
echo ""

# Check if bloch.c exists
if [ ! -f c/bloch.c ]; then
    echo "Error: bloch.c not found in c/ subdirectory"
    exit 1
fi

# OpenMP is optional: if your compiler supports it, the threaded Bloch loop will
# use it; otherwise the library still builds and runs in serial mode.
OPENMP_FLAG=""
if cc -fopenmp -x c - -o /tmp/bloch_omp_check >/dev/null 2>&1 <<'EOF'
#include <omp.h>
int main(void) { return omp_get_num_threads() < 0; }
EOF
then
    OPENMP_FLAG="-fopenmp"
else
    echo "Warning: OpenMP support not detected by cc; continuing in serial mode."
    echo "On macOS, install libomp if you want multi-threaded acceleration."
    echo "On Linux, ensure the OpenMP runtime is present if you want to enable it."
fi

# Detect OS and build
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS, building libbloch.dylib..."
    cc -O3 -fPIC $OPENMP_FLAG -dynamiclib -o python/libbloch.dylib c/bloch.c -lm

    if [ -f python/libbloch.dylib ]; then
        echo "Successfully built python/libbloch.dylib"
        ls -lh python/libbloch.dylib
    else
        echo "Failed to build libbloch.dylib"
        exit 1
    fi

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux, building libbloch.so..."
    cc -O3 -fPIC $OPENMP_FLAG -shared -o python/libbloch.so c/bloch.c -lm

    if [ -f python/libbloch.so ]; then
        echo "Successfully built python/libbloch.so"
        ls -lh python/libbloch.so
    else
        echo "Failed to build libbloch.so"
        exit 1
    fi

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Detected Windows, building bloch.dll..."
    cc -O3 $OPENMP_FLAG -shared -o python/bloch.dll c/bloch.c -lm
    
    if [ $? -eq 0 ] && [ -f python/bloch.dll ]; then
        echo "Successfully built python/bloch.dll"
        ls -lh python/bloch.dll
    else
        echo "Failed to build bloch.dll"
        exit 1
    fi

else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Bloch python library built successfully!"
echo ""
echo "To install the Python package, run:"
echo "  pip install -e ."
echo "=========================================="
