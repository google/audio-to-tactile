/* Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tactile_ble_service_nus.h"  /* NOLINT(build/include) */

#include <string.h>

#include "app_config.h"   /* NOLINT(build/include) */
#include "ble_nus.h"      /* NOLINT(build/include) */
#include "nrf_log.h"      /* NOLINT(build/include) */
#include "nrf_log_ctrl.h" /* NOLINT(build/include) */
#include "nrf_sdh.h"      /* NOLINT(build/include) */
#include "nrf_sdh_ble.h"  /* NOLINT(build/include) */
#include "nrf_sdh_soc.h"  /* NOLINT(build/include) */

static void NusDataHandler(ble_nus_evt_t* p_event);
static ParseNusMessageToProto custom_nus_message_to_proto;
BLE_NUS_DEF(app_nus, NRF_SDH_BLE_TOTAL_LINK_COUNT);

/* Place holder function to echo the message back. */
static void NusDataHandler(ble_nus_evt_t* p_event) {
  if (BLE_NUS_EVT_RX_DATA == p_event->type) {
    NRF_LOG_DEBUG("Received data from BLE NUS:%d",
                  p_event->params.rx_data.length);
    /* NRF_LOG_HEXDUMP_DEBUG(p_event->params.rx_data.p_data,
     *                       p_event->params.rx_data.length);
     */

    if (custom_nus_message_to_proto) {
      custom_nus_message_to_proto((uint8_t*)p_event->params.rx_data.p_data,
                          p_event->params.rx_data.length);
    }
  }
}

void SendNusData(uint8_t* buffer, uint16_t length) {
  uint32_t err_code =
      ble_nus_data_send(&app_nus, buffer, &length, g_app_connection_handle);
  if ((err_code != NRF_ERROR_INVALID_STATE) &&
      (err_code != NRF_ERROR_RESOURCES) && (err_code != NRF_ERROR_NOT_FOUND)) {
    APP_ERROR_CHECK(err_code);
  }
}

void TactileBleServiceNusInit(ParseNusMessageToProto nus_message_to_proto) {
  ble_nus_init_t nus_params = {0};
  nus_params.data_handler = NusDataHandler;
  APP_ERROR_CHECK(ble_nus_init(&app_nus, &nus_params));
  if (nus_message_to_proto) {
    custom_nus_message_to_proto = nus_message_to_proto;
  }
}
