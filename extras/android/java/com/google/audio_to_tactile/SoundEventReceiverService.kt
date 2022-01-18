/* Copyright 2021-2022 Google LLC
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

package com.google.audio_to_tactile

import android.app.Service
import android.content.Intent
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.util.Log

/** A service for receiving sound events from remote app. */
class SoundEventReceiverService : Service() {

  private lateinit var messenger: Messenger

  internal class SoundEventHandler() : Handler(Looper.getMainLooper()) {
    override fun handleMessage(msg: Message) {
      when (msg.what) {
        MESSAGE_SOUND_EVENT -> {
          Log.i(TAG, "Receive " + msg.data.getString(KEY_SOUND_EVENT))
        }
        MESSAGE_HAS_SPEECH -> {
          Log.i(TAG, "Receive SPEECH event.")
        }
        MESSAGE_NO_SPEECH -> {
          Log.i(TAG, "Receive NO SPEECH event.")
        }
        else -> super.handleMessage(msg)
      }
    }
  }

  override fun onBind(intent: Intent?): IBinder? {
    Log.i(TAG, "onBind called.")
    messenger = Messenger(SoundEventHandler())
    return messenger.binder
  }

  private companion object {
    const val TAG = "SoundEventReceiverService"
    const val MESSAGE_SOUND_EVENT = 1
    const val MESSAGE_HAS_SPEECH = 2
    const val MESSAGE_NO_SPEECH = 3
    const val KEY_SOUND_EVENT = "sound_event"
  }
}
