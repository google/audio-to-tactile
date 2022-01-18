/* Copyright 2021 Google LLC
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

import android.app.Activity
import android.os.Bundle
import android.view.Menu
import android.view.inputmethod.InputMethodManager
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.drawerlayout.widget.DrawerLayout
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.google.android.material.navigation.NavigationView
import com.google.audio_to_tactile.ble.DisconnectReason
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint class MainActivity : AppCompatActivity() {
  private val bleViewModel: BleViewModel by viewModels()
  private lateinit var appBarConfiguration: AppBarConfiguration

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    setContentView(R.layout.activity_main)
    setSupportActionBar(findViewById<Toolbar>(R.id.toolbar))

    val drawerLayout: DrawerLayout = findViewById(R.id.drawer_layout)
    val navView: NavigationView = findViewById(R.id.nav_view)
    val navController = findNavController(R.id.nav_host_fragment_content_main)
    appBarConfiguration =
      AppBarConfiguration.Builder(
          setOf( // Top-level destinations.
            R.id.nav_home,
            R.id.nav_tuning,
            R.id.nav_channel_map,
            R.id.nav_sound_events,
            R.id.nav_pattern_editor,
            R.id.nav_dfu,
            R.id.nav_log,
          )
        )
        .setOpenableLayout(drawerLayout)
        .build()
    setupActionBarWithNavController(navController, appBarConfiguration)
    navView.setupWithNavController(navController)

    // Hide software keyboard when nav drawer opens or closes.
    drawerLayout.addDrawerListener(
      object : DrawerLayout.SimpleDrawerListener() {
        override fun onDrawerStateChanged(newState: Int) {
          hideKeyboard()
        }
      }
    )

    bleViewModel.onActivityStart(
      BuildConfig.BUILD_TIME.toLong() // Get build time from BuildConfig.
      )

    bleViewModel.isConnected.observe(this) { isConnected ->
      if (isConnected != null && isConnected) {
        Toast.makeText(this, getString(R.string.ble_connected), Toast.LENGTH_SHORT).show()
      }
    }
    bleViewModel.disconnectReason.observe(this) { reason ->
      if (reason != null && reason != DisconnectReason.APP_DISCONNECTED) {
        Toast.makeText(this, getString(reason.stringRes), Toast.LENGTH_SHORT).show()
      }
    }

    bleViewModel.flashMemoryWriteStatus.observe(this) {
      it?.let { status ->
        Toast.makeText(this, getString(status.stringRes), Toast.LENGTH_SHORT).show()
      }
    }
  }

  override fun onCreateOptionsMenu(menu: Menu): Boolean {
    menuInflater.inflate(R.menu.main, menu)
    return true
  }

  override fun onSupportNavigateUp(): Boolean {
    hideKeyboard()
    val navController = findNavController(R.id.nav_host_fragment_content_main)
    return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
  }

  /** Hides the software keyboard. */
  private fun hideKeyboard() {
    getCurrentFocus()?.let { currentFocus ->
      (getSystemService(Activity.INPUT_METHOD_SERVICE) as InputMethodManager).apply {
        if (isActive()) {
          hideSoftInputFromWindow(currentFocus.getWindowToken(), 0)
        }
      }
    }
  }
}
