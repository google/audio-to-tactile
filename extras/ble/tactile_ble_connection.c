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

#include "tactile_ble_connection.h"  /* NOLINT(build/include) */

#include "app_config.h"      /* NOLINT(build/include) */
#include "app_timer.h"       /* NOLINT(build/include) */
#include "ble_conn_params.h" /* NOLINT(build/include) */
#include "nrf_log.h"         /* NOLINT(build/include) */
#include "nrf_log_ctrl.h"    /* NOLINT(build/include) */
#include "nrf_sdh.h"         /* NOLINT(build/include) */
#include "nrf_sdh_ble.h"     /* NOLINT(build/include) */
#include "nrf_sdh_soc.h"     /* NOLINT(build/include) */

/* APP_TIMER_TICKS converts time in ms to ticks of clock. */
#define kFirstConnectionParamsUpdateDelayTick APP_TIMER_TICKS(5000)
#define kNextConnectionParamsUpdateDelayTick APP_TIMER_TICKS(30000)
#define kMaxConnectionParamsUpdateCount 3

uint16_t g_app_connection_handle = BLE_CONN_HANDLE_INVALID;

static void OnConnectionParamsEvent(ble_conn_params_evt_t* p_event) {
  if (BLE_CONN_PARAMS_EVT_FAILED == p_event->evt_type) {
    uint32_t err_code = sd_ble_gap_disconnect(
        g_app_connection_handle, BLE_HCI_CONN_INTERVAL_UNACCEPTABLE);
    APP_ERROR_CHECK(err_code);
  }
}

static void ConnectionParamsErrorHandler(uint32_t nrf_error) {
  APP_ERROR_HANDLER(nrf_error);
}

void ConnectionParamsInit() {
  ble_conn_params_init_t params = {0};
  params.p_conn_params = NULL;
  params.first_conn_params_update_delay = kFirstConnectionParamsUpdateDelayTick;
  params.next_conn_params_update_delay = kNextConnectionParamsUpdateDelayTick;
  params.max_conn_params_update_count = kMaxConnectionParamsUpdateCount;
  params.start_on_notify_cccd_handle = BLE_GATT_HANDLE_INVALID;
  params.disconnect_on_fail = false;
  params.evt_handler = OnConnectionParamsEvent;
  params.error_handler = ConnectionParamsErrorHandler;

  uint32_t err_code = ble_conn_params_init(&params);
  APP_ERROR_CHECK(err_code);
}
