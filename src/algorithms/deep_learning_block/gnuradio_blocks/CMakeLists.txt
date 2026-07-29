# GNSS-SDR is a Global Navigation Satellite System software-defined receiver.
# This file is part of GNSS-SDR.
#
# SPDX-FileCopyrightText: 2010-2020 C. Fernandez-Prades cfernandez(at)cttc.es
# SPDX-License-Identifier: BSD-3-Clause


set(NEURAL_NETWORK_ROOT ${GNSSSDR_SOURCE_DIR}/../modules/neural_network)

if(WIN32)
    set(ONNXRUNTIME_ROOT_PATH "C:/Users/macie/Studia/magisterka/semestr_4_1/01_praca_magisterska/test/onnxruntime-win-x64-gpu-1.26.0/onnxruntime-win-x64-gpu-1.26.0" CACHE PATH "Path to ONNX Runtime root directory")
elseif(UNIX)
    set(ONNXRUNTIME_ROOT_PATH "/home/maciek/Projects/GNSS-Jamming/02_related/onnxruntime-linux-x64-gpu-1.26.0" CACHE PATH "Path to ONNX Runtime root directory")
else()
    message(FATAL_ERROR "Not supported platform")
endif()

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT_PATH}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

set(DATA_TYPE_GR_BLOCKS_SOURCES
    deeplearningblock.cpp
    ${NEURAL_NETWORK_ROOT}/src/onnx_model.cpp
    ${NEURAL_NETWORK_ROOT}/src/stft_input.cpp
)

set(DATA_TYPE_GR_BLOCKS_HEADERS
    deeplearningblock.h
)

list(SORT DATA_TYPE_GR_BLOCKS_HEADERS)
list(SORT DATA_TYPE_GR_BLOCKS_SOURCES)

if(USE_CMAKE_TARGET_SOURCES)
    add_library(deep_learning_block STATIC)
    target_sources(deep_learning_block
        PRIVATE
            ${DATA_TYPE_GR_BLOCKS_SOURCES}
        PUBLIC
            ${DATA_TYPE_GR_BLOCKS_HEADERS}
    )
else()
    source_group(Headers FILES ${DATA_TYPE_GR_BLOCKS_HEADERS})
    add_library(deep_learning_block
        ${DATA_TYPE_GR_BLOCKS_SOURCES}
        ${DATA_TYPE_GR_BLOCKS_HEADERS}
    )
endif()

target_link_libraries(deep_learning_block
    PUBLIC
        Gnuradio::runtime
        Boost::headers
    PRIVATE
        core_libs
        algorithms_libs
        Volk::volk
        Volkgnsssdr::volkgnsssdr
        ${ONNXRUNTIME_LIB}
)

target_include_directories(deep_learning_block
    PUBLIC
        ${GNSSSDR_SOURCE_DIR}/src/core/interfaces
        ${GNSSSDR_SOURCE_DIR}/src/core/libs
    PRIVATE
        ${NEURAL_NETWORK_ROOT}/include
        ${ONNXRUNTIME_ROOT_PATH}/include
)

if(GNURADIO_USES_STD_POINTERS)
    target_compile_definitions(deep_learning_block
        PUBLIC -DGNURADIO_USES_STD_POINTERS=1
    )
endif()

set_property(TARGET deep_learning_block
    APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
