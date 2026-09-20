# Spectraflow ESP32 CSI node - ESP-IDF project wrapper.
#
# Build with:
#   idf.py set-target esp32
#   idf.py menuconfig      # optional, the configuration lives in main/csi_collector.c
#   idf.py build flash monitor

cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(esp32_csi_node)
