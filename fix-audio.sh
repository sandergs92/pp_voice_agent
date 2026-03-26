#!/bin/sh
# Restore working audio routing for PinePhone Pro
# Run this if audio breaks (e.g. after a modem call, callaudiod, or PipeWire corruption)

set -e

# ── 1) Force card profile back to HiFi with speaker + mic ──
# This is critical — callaudiod and PipeWire filter experiments can switch
# the profile to "Voice Call (Earpiece, Mic)" which breaks speaker output.
# The amixer commands below are useless if the profile is wrong.
pactl set-card-profile alsa_card.platform-rt5640-sound "HiFi (Mic, Speaker)" 2>/dev/null || true

# ── 2) Restore known-good ALSA state ──
# This sets all 139 mixer controls atomically — more reliable than
# individual amixer commands which can fail silently on wrong profile.
if [ -f ~/pp_voice_agent/asound.state.working ]; then
    sudo cp ~/pp_voice_agent/asound.state.working /var/lib/alsa/asound.state
    sudo alsactl restore
    echo "ALSA state restored from asound.state.working"
else
    echo "WARNING: asound.state.working not found, using amixer fallback"
    # Fallback: individual mixer commands
    amixer -c 0 cset name='HPO MIX HPVOL Switch' on
    amixer -c 0 cset name='HP L Playback Switch' on
    amixer -c 0 cset name='HP R Playback Switch' on
    amixer -c 0 cset name='Internal Speaker Switch' on
    amixer -c 0 sset 'Headphone' 31
    amixer -c 0 sset 'Speaker' on 70%
    amixer -c 0 sset 'Speaker L' on
    amixer -c 0 sset 'Speaker R' on
    amixer -c 0 sset 'SPK MIXL DAC L1' on
    amixer -c 0 sset 'SPK MIXR DAC R1' on
    amixer -c 0 sset 'SPOL MIX SPKVOL L' on
    amixer -c 0 sset 'SPOL MIX SPKVOL R' on
    amixer -c 0 sset 'Mono ADC MIXL ADC2' on
    amixer -c 0 sset 'Mono ADC MIXR ADC2' on
    amixer -c 0 sset 'Stereo ADC MIXL ADC2' on
    amixer -c 0 sset 'Stereo ADC MIXR ADC2' on
    amixer -c 0 sset 'RECMIXL BST1' off
    amixer -c 0 sset 'RECMIXR BST1' off
fi

# ── 3) Restart PipeWire to pick up the restored state ──
systemctl --user restart pipewire pipewire-pulse wireplumber
sleep 1

# ── 4) Set PipeWire capture volume (resets to ~32% on reboot) ──
wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 1.0 2>/dev/null

echo "Audio restored. Profile: HiFi (Mic, Speaker)"
