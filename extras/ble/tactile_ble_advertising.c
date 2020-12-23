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

#include "tactile_ble_advertising.h"  /* NOLINT(build/include) */


#include <stdint.h>
#include <string.h>

#include "app_config.h"        /* NOLINT(build/include) */
#include "ble_advdata.h"       /* NOLINT(build/include) */
#include "ble_advertising.h"   /* NOLINT(build/include) */
#include "ble_nus.h"           /* NOLINT(build/include) */
#include "nrf_log.h"           /* NOLINT(build/include) */
#include "nrf_log_ctrl.h"      /* NOLINT(build/include) */
#include "nrf_sdh.h"           /* NOLINT(build/include) */
#include "nrf_sdh_ble.h"       /* NOLINT(build/include) */
#include "nrf_sdh_soc.h"       /* NOLINT(build/include) */
#include "tactile_ble_stack.h" /* NOLINT(build/include) */

#define kAppAdvertisingInterval MSEC_TO_UNITS(40, UNIT_0_625_MS)
/* Set to always advertising. Use MSEC_TO_UNITS(boardcast_time, UNIT_10_MS) to
 * change to broadcast time.
 */
#define kAppAdvertisingDuration BLE_GAP_ADV_TIMEOUT_GENERAL_UNLIMITED
BLE_ADVERTISING_DEF(app_advertising);

/* TODO: After adding more services, make app_advertising_uuids
 * changable during run time since different board requires different services.
 */
static ble_uuid_t app_advertising_uuids[] = {

    {BLE_UUID_BATTERY_SERVICE, BLE_UUID_TYPE_BLE},
    {BLE_UUID_NUS_SERVICE, BLE_UUID_TYPE_BLE}};

static void SleepModeEnter() {
  uint32_t err_code = sd_power_system_off();
  APP_ERROR_CHECK(err_code);
}

/* TODO: Add more types of ble_adv_evt_t as the need arises. */
static void OnAdvertisingEvent(ble_adv_evt_t ble_advertising_event) {
  switch (ble_advertising_event) {
    case BLE_ADV_EVT_FAST:
      NRF_LOG_INFO("Fast advertising.");
      break;
    case BLE_ADV_EVT_IDLE:
      SleepModeEnter();
      break;
    default:
      break;
  }
}

/* TODO: Add callbacks parameters to add advertising LED behavior. */
void AdvertisingInit() {
  ble_advertising_init_t params = {0};

  params.advdata.name_type = BLE_ADVDATA_FULL_NAME;
  params.advdata.include_appearance = true;

  if (BLE_GAP_ADV_TIMEOUT_GENERAL_UNLIMITED == kAppAdvertisingDuration) {
    params.advdata.flags = BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE;
  } else {
    params.advdata.flags = BLE_GAP_ADV_FLAGS_LE_ONLY_LIMITED_DISC_MODE;
  }

  params.srdata.uuids_complete.uuid_cnt =
      sizeof(app_advertising_uuids) / sizeof(app_advertising_uuids[0]);
  params.srdata.uuids_complete.p_uuids = app_advertising_uuids;

  params.config.ble_adv_fast_enabled = true;
  params.config.ble_adv_fast_interval = kAppAdvertisingInterval;
  params.config.ble_adv_fast_timeout = kAppAdvertisingDuration;
  params.evt_handler = OnAdvertisingEvent;

  uint32_t err_code = ble_advertising_init(&app_advertising, &params);
  APP_ERROR_CHECK(err_code);

  ble_advertising_conn_cfg_tag_set(&app_advertising, kAppBleConnCfgTag);
}

void AdvertisingStart() {
  uint32_t err_code =
      ble_advertising_start(&app_advertising, BLE_ADV_MODE_FAST);
  APP_ERROR_CHECK(err_code);
}
