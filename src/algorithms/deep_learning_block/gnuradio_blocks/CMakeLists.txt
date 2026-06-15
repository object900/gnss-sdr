# GNSS-SDR is a Global Navigation Satellite System software-defined receiver.
# This file is part of GNSS-SDR.
#
# SPDX-FileCopyrightText: 2010-2020 C. Fernandez-Prades cfernandez(at)cttc.es
# SPDX-License-Identifier: BSD-3-Clause


set(DATA_TYPE_GR_BLOCKS_SOURCES
    deeplearningblock.cpp
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
        Volk::volk
        Volkgnsssdr::volkgnsssdr
)

target_include_directories(deep_learning_block
    PUBLIC
        ${GNSSSDR_SOURCE_DIR}/src/core/interfaces
        ${GNSSSDR_SOURCE_DIR}/src/core/libs
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
