[Audio-to-Tactile Representation](../README.md)

# play_buzz - simple diagnostic program

`play_buzz` is a diagnostic program for testing tactile (or sound) output. It
plays a sinusoid buzz to a specified channel of a PortAudio output device. The
sinusoid is pulsed with 50% duty cycle at a regular interval, and modulated with
a smooth envelope.

![play buzz](play-buzz.gif)


## Building

`play_buzz` is implemented in C and depends on the
[PortAudio](http://www.portaudio.com/) library. On Debian-based systems, use

``` shell
sudo apt-get install portaudio19-dev
```

to install the development headers. Then build with

``` shell
cd audio-to-tactile
make play_buzz
```

This should produce a `play_buzz` executable file.


## Running

Run as `play_buzz --output=-1` to print a numbered list of connected PortAudio
devices. Then run again with `play_buzz --output=<number>` where `number` is the
list number of the desired device.

By default, the program plays mono output. Use `--num_channels` to set the total
number of channels and, e.g., `--channel=5` option to play buzzes to the 5th
channel (where channels are counted from 1).

Command line options:

 * `--output=<int>` - Output device number to play tactor signals to.
 * `--sample_rate_hz=<int>` - Sample rate.
 * `--num_channels=<int>` - Number of output channels. (Default 1)
 * `--channel=<int>` - Which channel to play buzz on, counting from 1, or `--channel=all` for all channels. (Default 1)
 * `--amplitude=<float>` - Buzz amplitude, value in [0.0, 1.0] where 1.0 is full scale. (Default 0.2)
 * `--frequency_hz=<float>` - Buzz frequency. (Default 250.0)

