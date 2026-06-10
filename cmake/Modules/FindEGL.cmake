# JP: この build file では CMake target、CUDA architecture、library dependency を確認します。target 名や link 設定は英語のまま保持します。

find_path(EGL_INCLUDE_DIR
  NAMES EGL/egl.h
  PATHS /usr/include /usr/local/include
)

find_library(EGL_LIBRARY
  NAMES EGL
  PATHS /usr/lib /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EGL DEFAULT_MSG EGL_LIBRARY EGL_INCLUDE_DIR)

if(EGL_FOUND)
  set(EGL_LIBRARIES ${EGL_LIBRARY})
  set(EGL_INCLUDE_DIRS ${EGL_INCLUDE_DIR})
endif()
