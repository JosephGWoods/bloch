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

# Detect OS and build
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS, building libbloch.dylib..."
    cc -O3 -fPIC -dynamiclib -o python/libbloch.dylib c/bloch.c -lm

    if [ -f python/libbloch.dylib ]; then
        echo "Successfully built python/libbloch.dylib"
        ls -lh python/libbloch.dylib
    else
        echo "Failed to build libbloch.dylib"
        exit 1
    fi

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux, building libbloch.so..."
    cc -O3 -fPIC -shared -o python/libbloch.so c/bloch.c -lm

    if [ -f python/libbloch.so ]; then
        echo "Successfully built python/libbloch.so"
        ls -lh python/libbloch.so
    else
        echo "Failed to build libbloch.so"
        exit 1
    fi

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "Detected Windows, building bloch.dll..."
    cc -O3 -shared -o python/bloch.dll c/bloch.c -lm
    
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
