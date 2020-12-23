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

#include "tactile_ble_gatt.h"  /* NOLINT(build/include) */

#include <stdint.h>

#include "app_config.h"             /* NOLINT(build/include) */
#include "nrf_ble_gatt.h"           /* NOLINT(build/include) */
#include "nrf_log.h"                /* NOLINT(build/include) */
#include "nrf_log_ctrl.h"           /* NOLINT(build/include) */
#include "nrf_sdh.h"                /* NOLINT(build/include) */
#include "nrf_sdh_ble.h"            /* NOLINT(build/include) */
#include "nrf_sdh_soc.h"            /* NOLINT(build/include) */
#include "tactile_ble_connection.h" /* NOLINT(build/include) */

NRF_BLE_GATT_DEF(app_gatt);
uint16_t mtu_size = BLE_GATT_ATT_MTU_DEFAULT;

static void GattEventHandler(nrf_ble_gatt_t* p_gatt,
                             nrf_ble_gatt_evt_t const* p_event) {
  if ((g_app_connection_handle == p_event->conn_handle) &&
      (p_event->evt_id == NRF_BLE_GATT_EVT_ATT_MTU_UPDATED)) {
    mtu_size = p_event->params.att_mtu_effective;
  }
  NRF_LOG_DEBUG("ATT MTU exchange completed. central 0x%x peripheral 0x%x",
                p_gatt->att_mtu_desired_central,
                p_gatt->att_mtu_desired_periph);
}

void GattInit() {
  uint32_t err_code = nrf_ble_gatt_init(&app_gatt, GattEventHandler);
  APP_ERROR_CHECK(err_code);

  err_code =
      nrf_ble_gatt_att_mtu_periph_set(&app_gatt, NRF_SDH_BLE_GATT_MAX_MTU_SIZE);
  APP_ERROR_CHECK(err_code);
}
