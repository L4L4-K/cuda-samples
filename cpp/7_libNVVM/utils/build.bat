mkdir build
cd build
REM JP: Windows/NMake 用に libNVVM sample の CMake configure/install prefix をここで固定します。
cmake.exe -DCMAKE_INSTALL_PREFIX=..\install -G "NMake Makefiles" ..
REM JP: build、test、install を順に連結します。失敗した段階で後続 command に進まないことを確認します。
nmake && nmake test && nmake install && cd ..
