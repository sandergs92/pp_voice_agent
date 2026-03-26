#!/bin/sh
# Restore working audio routing for PinePhone Pro
# Run this if audio breaks (e.g. after a modem call)

# Speaker output via HP amp + internal speaker amp
amixer -c 0 cset name='HPO MIX HPVOL Switch' on
amixer -c 0 cset name='HP L Playback Switch' on
amixer -c 0 cset name='HP R Playback Switch' on
amixer -c 0 cset name='Internal Speaker Switch' on
amixer -c 0 sset 'Headphone' 31
amixer -c 0 sset 'Speaker' on 70%
amixer -c 0 sset 'Speaker L' on
amixer -c 0 sset 'Speaker R' on

# DAC routing
amixer -c 0 sset 'SPK MIXL DAC L1' on
amixer -c 0 sset 'SPK MIXR DAC R1' on
amixer -c 0 sset 'SPOL MIX SPKVOL L' on
amixer -c 0 sset 'SPOL MIX SPKVOL R' on

# Mic routing (DMIC via ADC2 for PPP internal mic)
amixer -c 0 sset 'Mono ADC MIXL ADC2' on
amixer -c 0 sset 'Mono ADC MIXR ADC2' on
amixer -c 0 sset 'Stereo ADC MIXL ADC2' on
amixer -c 0 sset 'Stereo ADC MIXR ADC2' on
amixer -c 0 sset 'RECMIXL BST1' off
amixer -c 0 sset 'RECMIXR BST1' off

# Or just restore the saved state:
# sudo cp ~/pp_voice_agent/asound.state.working /var/lib/alsa/asound.state
# sudo alsactl restore

echo "Audio restored."

# PipeWire capture volume
wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 1.0 2>/dev/null

echo "PipeWire capture set to 100%"
