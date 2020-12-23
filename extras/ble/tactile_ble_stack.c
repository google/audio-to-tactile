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

#include "tactile_ble_stack.h" /* NOLINT(build/include) */

#include "app_config.h"   /* NOLINT(build/include) */
#include "nrf_ble_qwr.h"  /* NOLINT(build/include) */
#include "nrf_log.h"      /* NOLINT(build/include) */
#include "nrf_log_ctrl.h" /* NOLINT(build/include) */
#include "nrf_sdh.h"      /* NOLINT(build/include) */
#include "nrf_sdh_ble.h"  /* NOLINT(build/include) */
#include "nrf_sdh_soc.h"  /* NOLINT(build/include) */

#define kAppBleObserverPrio 3

/* Add Queued write support. */
NRF_BLE_QWR_DEF(app_qwr);

static void TactileBleServiceQwrInit();
static void BleEventHandler(ble_evt_t const* p_ble_event, void* p_context);
static void NrfQwrErrorHandler(uint32_t nrf_error);

TactileBleStackCallback OnConnectCallback;
TactileBleStackCallback OnDisconnectCallback;

static void TactileBleServiceQwrInit() {
  nrf_ble_qwr_init_t params = {0};
  params.error_handler = NrfQwrErrorHandler;
  APP_ERROR_CHECK(nrf_ble_qwr_init(&app_qwr, &params));
}

static void NrfQwrErrorHandler(uint32_t nrf_error) {
  if (nrf_error != 0) {
    NRF_LOG_DEBUG("nrf_qwr_error: %d", nrf_error);
  }
  APP_ERROR_HANDLER(nrf_error);
}

void BleStackInit(TactileBleStackCallback OnConnect,
                  TactileBleStackCallback OnDisconnect) {
  uint32_t err_code = nrf_sdh_enable_request();
  APP_ERROR_CHECK(err_code);

  uint32_t ram_start = 0;
  err_code = nrf_sdh_ble_default_cfg_set(kAppBleConnCfgTag, &ram_start);
  APP_ERROR_CHECK(err_code);

  err_code = nrf_sdh_ble_enable(&ram_start);
  APP_ERROR_CHECK(err_code);

  NRF_SDH_BLE_OBSERVER(m_ble_observer, kAppBleObserverPrio, BleEventHandler,
                       NULL);

  OnConnectCallback = OnConnect;
  OnDisconnectCallback = OnDisconnect;

  /* Initialize Queue write module. Althogh it is a service, it is for
   * supporting queued write BLE spec.
   */
  TactileBleServiceQwrInit();
}

static void BleEventHandler(ble_evt_t const* p_ble_event, void* p_context) {
  uint32_t err_code;
  switch (p_ble_event->header.evt_id) {
    case BLE_GAP_EVT_CONNECTED:
      NRF_LOG_INFO("Connected");
      g_app_connection_handle = p_ble_event->evt.gap_evt.conn_handle;
      err_code =
          nrf_ble_qwr_conn_handle_assign(&app_qwr, g_app_connection_handle);
      APP_ERROR_CHECK(err_code);
      if (OnConnectCallback) {
        OnConnectCallback();
      }
      break;

    case BLE_GAP_EVT_DISCONNECTED:
      NRF_LOG_INFO("Disconnected (reason: 0x%x)",
                   p_ble_event->evt.gap_evt.params.disconnected.reason);
      g_app_connection_handle = BLE_CONN_HANDLE_INVALID;
      if (OnDisconnectCallback) {
        OnDisconnectCallback();
      }
      break;

    case BLE_GAP_EVT_PHY_UPDATE_REQUEST: {
      NRF_LOG_DEBUG("PHY update request.");
      ble_gap_phys_t const phys = {
          .rx_phys = BLE_GAP_PHY_AUTO,
          .tx_phys = BLE_GAP_PHY_AUTO,
      };
      err_code =
          sd_ble_gap_phy_update(p_ble_event->evt.gap_evt.conn_handle, &phys);
      APP_ERROR_CHECK(err_code);
    } break;

    case BLE_GAP_EVT_SEC_PARAMS_REQUEST:
      err_code = sd_ble_gap_sec_params_reply(
          g_app_connection_handle, BLE_GAP_SEC_STATUS_PAIRING_NOT_SUPP, NULL,
          NULL);
      APP_ERROR_CHECK(err_code);
      break;

    case BLE_GATTS_EVT_SYS_ATTR_MISSING:
      err_code = sd_ble_gatts_sys_attr_set(g_app_connection_handle, NULL, 0, 0);
      APP_ERROR_CHECK(err_code);
      break;

    case BLE_GATTC_EVT_TIMEOUT:  /* GATT client event timeout. */
      err_code =
          sd_ble_gap_disconnect(p_ble_event->evt.gattc_evt.conn_handle,
                                BLE_HCI_REMOTE_USER_TERMINATED_CONNECTION);
      APP_ERROR_CHECK(err_code);
      break;

    case BLE_GATTS_EVT_TIMEOUT:  /* GATT server event timeout. */
      err_code =
          sd_ble_gap_disconnect(p_ble_event->evt.gatts_evt.conn_handle,
                                BLE_HCI_REMOTE_USER_TERMINATED_CONNECTION);
      APP_ERROR_CHECK(err_code);
      break;

    default:
      break;
  }
}
