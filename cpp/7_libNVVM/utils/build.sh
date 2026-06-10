#!/bin/sh

mkdir build
cd build
# JP: libNVVM sample の CMake configure/install prefix をここで固定します。build/test/install の出力先を sample tree 内に限定します。
cmake -DCMAKE_INSTALL_PREFIX=../install ..
# JP: build 後に test を実行し、成功した artifact だけを install します。command 自体は英語のまま保持します。
make && make test && make install
