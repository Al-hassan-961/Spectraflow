# Component registration for the Spectraflow CSI collector.
#
# REQUIRES lists the ESP-IDF components csi_collector.c actually includes:
#   esp_wifi  - esp_wifi.h (CSI capture, promiscuous mode, esp_wifi_80211_tx)
#   esp_event - esp_event.h (Wi-Fi and IP event handlers)
#   esp_netif - esp_netif.h / esp_netif_ip_addr.h (netif glue, esp_ip4_addr_t)
#   esp_timer - esp_timer.h (esp_timer_get_time() capture timestamps)
#   lwip      - lwip/sockets.h (UDP socket and sendto)
#   nvs_flash - nvs_flash.h (Wi-Fi calibration data / NVS bring-up)
#
# freertos, log, esp_system and esp_common are part of the ESP-IDF common
# requirements and are therefore available without being listed here.

idf_component_register(
    SRCS "csi_collector.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi esp_event esp_netif esp_timer lwip nvs_flash
)
