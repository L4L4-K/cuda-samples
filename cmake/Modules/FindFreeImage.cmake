# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

find_path(FreeImage_INCLUDE_DIR
  NAMES freeimage.h FreeImage.h
  PATHS /usr/include /usr/local/include
)

find_library(FreeImage_LIBRARY
  NAMES freeimage FreeImage
  PATHS /usr/lib /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FreeImage DEFAULT_MSG FreeImage_LIBRARY FreeImage_INCLUDE_DIR)

if(FreeImage_FOUND)
  set(FreeImage_LIBRARIES ${FreeImage_LIBRARY})
  set(FreeImage_INCLUDE_DIRS ${FreeImage_INCLUDE_DIR})
endif()
