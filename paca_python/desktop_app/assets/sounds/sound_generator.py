"""
Sound Generator for PACA Desktop Application
Purpose: Generate audio signals and sound effects for the PACA desktop interface
Author: PACA Development Team
Created: 2024-09-24
"""

import numpy as np
import wave
import struct
import os
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class SoundType(Enum):
    """Sound effect types."""
    ALERT = "alert"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    NOTIFICATION = "notification"
    CLICK = "click"
    HOVER = "hover"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    PROCESSING = "processing"


@dataclass
class SoundConfig:
    """Configuration for sound generation."""
    sample_rate: int = 44100  # Standard CD quality
    duration: float = 0.5     # Duration in seconds
    volume: float = 0.5       # Volume level (0.0 to 1.0)
    fade_in: float = 0.05     # Fade in time in seconds
    fade_out: float = 0.05    # Fade out time in seconds


class SoundGenerator:
    """
    Dynamic sound generation system for PACA desktop application.

    Generates various sound effects using mathematical wave functions.
    All sounds are synthesized to avoid copyright issues and ensure consistency.
    """

    def __init__(self):
        self.default_config = SoundConfig()
        self.generated_sounds = {}

    def generate_sine_wave(self, frequency: float, duration: float,
                          sample_rate: int = 44100, volume: float = 0.5) -> np.ndarray:
        """
        Generate a pure sine wave.

        Args:
            frequency: Wave frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            volume: Volume level (0.0 to 1.0)

        Returns:
            NumPy array of audio samples
        """
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_data = np.sin(2 * np.pi * frequency * t) * volume
        return wave_data

    def generate_triangle_wave(self, frequency: float, duration: float,
                             sample_rate: int = 44100, volume: float = 0.5) -> np.ndarray:
        """Generate a triangle wave."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_data = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t)) * volume
        return wave_data

    def generate_square_wave(self, frequency: float, duration: float,
                           sample_rate: int = 44100, volume: float = 0.5) -> np.ndarray:
        """Generate a square wave."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_data = np.sign(np.sin(2 * np.pi * frequency * t)) * volume
        return wave_data

    def generate_sawtooth_wave(self, frequency: float, duration: float,
                             sample_rate: int = 44100, volume: float = 0.5) -> np.ndarray:
        """Generate a sawtooth wave."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_data = (2 / np.pi) * np.arctan(np.tan(np.pi * frequency * t)) * volume
        return wave_data

    def generate_noise(self, duration: float, sample_rate: int = 44100,
                      volume: float = 0.1, noise_type: str = "white") -> np.ndarray:
        """
        Generate noise patterns.

        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            volume: Volume level
            noise_type: Type of noise ("white", "pink", "brown")

        Returns:
            NumPy array of noise samples
        """
        samples = int(sample_rate * duration)

        if noise_type == "white":
            noise = np.random.uniform(-1, 1, samples) * volume
        elif noise_type == "pink":
            # Simplified pink noise generation
            white_noise = np.random.uniform(-1, 1, samples)
            # Apply simple low-pass filtering for pink noise approximation
            b = np.ones(10) / 10
            noise = np.convolve(white_noise, b, mode='same') * volume
        elif noise_type == "brown":
            # Brownian noise (integrated white noise)
            white_noise = np.random.uniform(-1, 1, samples)
            noise = np.cumsum(white_noise) * volume
            noise = noise / np.max(np.abs(noise))  # Normalize
        else:
            noise = np.random.uniform(-1, 1, samples) * volume

        return noise

    def apply_envelope(self, wave_data: np.ndarray, fade_in: float = 0.05,
                      fade_out: float = 0.05, sample_rate: int = 44100) -> np.ndarray:
        """
        Apply fade in/out envelope to prevent audio clicks.

        Args:
            wave_data: Input audio samples
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Audio samples with envelope applied
        """
        samples = len(wave_data)
        fade_in_samples = int(fade_in * sample_rate)
        fade_out_samples = int(fade_out * sample_rate)

        # Apply fade in
        if fade_in_samples > 0:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            wave_data[:fade_in_samples] *= fade_in_curve

        # Apply fade out
        if fade_out_samples > 0:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            wave_data[-fade_out_samples:] *= fade_out_curve

        return wave_data

    def generate_alert_sound(self, sound_type: SoundType, config: Optional[SoundConfig] = None) -> np.ndarray:
        """
        Generate alert sounds based on type.

        Args:
            sound_type: Type of alert sound
            config: Sound configuration

        Returns:
            Generated audio samples
        """
        if config is None:
            config = self.default_config

        if sound_type == SoundType.SUCCESS:
            # Ascending chord progression (C-E-G)
            frequencies = [523.25, 659.25, 783.99]  # C5, E5, G5
            wave_data = np.zeros(int(config.sample_rate * config.duration))

            for i, freq in enumerate(frequencies):
                start_idx = i * int(config.sample_rate * config.duration / 3)
                end_idx = (i + 1) * int(config.sample_rate * config.duration / 3)
                segment = self.generate_sine_wave(freq, config.duration / 3,
                                                config.sample_rate, config.volume)
                wave_data[start_idx:start_idx + len(segment)] += segment

        elif sound_type == SoundType.ERROR:
            # Descending dissonant tones
            frequencies = [800, 600, 400]
            wave_data = np.zeros(int(config.sample_rate * config.duration))

            for i, freq in enumerate(frequencies):
                start_idx = i * int(config.sample_rate * config.duration / 3)
                segment = self.generate_square_wave(freq, config.duration / 3,
                                                  config.sample_rate, config.volume * 0.7)
                wave_data[start_idx:start_idx + len(segment)] += segment

        elif sound_type == SoundType.WARNING:
            # Alternating two-tone warning
            freq1, freq2 = 880, 440  # A5 and A4
            wave_data = np.zeros(int(config.sample_rate * config.duration))
            segment_length = int(config.sample_rate * config.duration / 4)

            for i in range(4):
                freq = freq1 if i % 2 == 0 else freq2
                start_idx = i * segment_length
                segment = self.generate_sine_wave(freq, config.duration / 4,
                                                config.sample_rate, config.volume)
                wave_data[start_idx:start_idx + len(segment)] += segment

        elif sound_type == SoundType.NOTIFICATION:
            # Gentle chime (harmonic series)
            fundamental = 440  # A4
            wave_data = np.zeros(int(config.sample_rate * config.duration))

            for harmonic in [1, 2, 3]:
                freq = fundamental * harmonic
                volume = config.volume / harmonic  # Decreasing volume for higher harmonics
                harmonic_wave = self.generate_sine_wave(freq, config.duration,
                                                      config.sample_rate, volume)
                wave_data += harmonic_wave

        elif sound_type == SoundType.ALERT:
            # Sharp attention-grabbing sound
            wave_data = self.generate_triangle_wave(1000, config.duration,
                                                  config.sample_rate, config.volume)

        else:
            # Default: simple beep
            wave_data = self.generate_sine_wave(800, config.duration,
                                              config.sample_rate, config.volume)

        # Apply envelope
        wave_data = self.apply_envelope(wave_data, config.fade_in, config.fade_out,
                                      config.sample_rate)

        return wave_data

    def generate_feedback_sound(self, action: str, config: Optional[SoundConfig] = None) -> np.ndarray:
        """
        Generate UI feedback sounds.

        Args:
            action: Type of action ("click", "hover", "select", "drag", "drop")
            config: Sound configuration

        Returns:
            Generated audio samples
        """
        if config is None:
            config = SoundConfig(duration=0.1, volume=0.3)  # Shorter, quieter for feedback

        if action == "click":
            # Sharp click sound
            wave_data = self.generate_sine_wave(2000, 0.05, config.sample_rate, config.volume)
            wave_data = np.concatenate([
                wave_data,
                self.generate_sine_wave(1000, 0.05, config.sample_rate, config.volume * 0.5)
            ])

        elif action == "hover":
            # Soft hover sound
            wave_data = self.generate_sine_wave(800, config.duration,
                                              config.sample_rate, config.volume * 0.5)

        elif action == "select":
            # Selection confirmation
            wave_data = self.generate_sine_wave(1200, config.duration,
                                              config.sample_rate, config.volume)

        elif action == "drag":
            # Dragging feedback
            wave_data = self.generate_noise(config.duration, config.sample_rate,
                                          config.volume * 0.2, "pink")

        elif action == "drop":
            # Drop confirmation
            frequencies = [400, 600]  # Low to mid tone
            wave_data = np.zeros(int(config.sample_rate * config.duration))
            for freq in frequencies:
                segment = self.generate_sine_wave(freq, config.duration / 2,
                                                config.sample_rate, config.volume)
                wave_data[:len(segment)] += segment

        else:
            wave_data = self.generate_sine_wave(1000, config.duration,
                                              config.sample_rate, config.volume)

        return self.apply_envelope(wave_data, 0.01, 0.01, config.sample_rate)

    def generate_ambient_sound(self, ambient_type: str, duration: float = 10.0,
                             config: Optional[SoundConfig] = None) -> np.ndarray:
        """
        Generate ambient background sounds.

        Args:
            ambient_type: Type of ambient sound ("calm", "focus", "energetic")
            duration: Duration in seconds
            config: Sound configuration

        Returns:
            Generated audio samples
        """
        if config is None:
            config = SoundConfig(duration=duration, volume=0.1)

        if ambient_type == "calm":
            # Soft nature-like sounds with low frequency components
            wave_data = self.generate_noise(duration, config.sample_rate,
                                          config.volume * 0.3, "pink")
            # Add gentle sine wave oscillations
            for freq in [200, 300, 450]:
                sine_component = self.generate_sine_wave(freq, duration,
                                                       config.sample_rate,
                                                       config.volume * 0.1)
                wave_data += sine_component

        elif ambient_type == "focus":
            # White noise with subtle tonal elements for concentration
            wave_data = self.generate_noise(duration, config.sample_rate,
                                          config.volume * 0.4, "white")
            # Add very subtle harmonic content
            for freq in [100, 150, 220]:
                tone = self.generate_sine_wave(freq, duration,
                                             config.sample_rate, config.volume * 0.05)
                wave_data += tone

        elif ambient_type == "energetic":
            # More dynamic ambient sound with rhythmic elements
            wave_data = np.zeros(int(config.sample_rate * duration))
            pulse_duration = 2.0  # 2-second pulses
            num_pulses = int(duration / pulse_duration)

            for i in range(num_pulses):
                start_idx = i * int(config.sample_rate * pulse_duration)
                # Generate a pulse with multiple frequencies
                pulse = np.zeros(int(config.sample_rate * pulse_duration))
                for freq in [80, 160, 240]:
                    freq_component = self.generate_sine_wave(freq, pulse_duration,
                                                           config.sample_rate,
                                                           config.volume * 0.3)
                    pulse += freq_component

                # Apply pulse envelope
                pulse_envelope = np.exp(-3 * np.linspace(0, 1, len(pulse)))
                pulse *= pulse_envelope
                wave_data[start_idx:start_idx + len(pulse)] += pulse

        else:
            # Default: gentle pink noise
            wave_data = self.generate_noise(duration, config.sample_rate,
                                          config.volume, "pink")

        return self.apply_envelope(wave_data, 1.0, 1.0, config.sample_rate)

    def save_wav_file(self, wave_data: np.ndarray, filename: str,
                     sample_rate: int = 44100) -> bool:
        """
        Save audio data as WAV file.

        Args:
            wave_data: Audio samples
            filename: Output filename
            sample_rate: Sample rate in Hz

        Returns:
            Success status
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Convert to 16-bit integers
            wave_data = np.clip(wave_data, -1.0, 1.0)
            wave_data_int = (wave_data * 32767).astype(np.int16)

            # Save as WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data_int.tobytes())

            return True
        except Exception as e:
            print(f"Error saving WAV file {filename}: {e}")
            return False

    def generate_all_sounds(self, base_dir: str) -> Dict[str, bool]:
        """
        Generate all sound files for the PACA application.

        Args:
            base_dir: Base directory for saving sound files

        Returns:
            Dictionary of generation results
        """
        results = {}

        # Alert sounds
        alert_types = [
            (SoundType.SUCCESS, "success.wav"),
            (SoundType.ERROR, "error.wav"),
            (SoundType.WARNING, "warning.wav"),
            (SoundType.NOTIFICATION, "notification.wav"),
            (SoundType.ALERT, "alert.wav")
        ]

        alerts_dir = os.path.join(base_dir, "alerts")
        for sound_type, filename in alert_types:
            wave_data = self.generate_alert_sound(sound_type)
            filepath = os.path.join(alerts_dir, filename)
            success = self.save_wav_file(wave_data, filepath)
            results[f"alerts/{filename}"] = success

        # Feedback sounds
        feedback_actions = ["click", "hover", "select", "drag", "drop"]
        feedback_dir = os.path.join(base_dir, "feedback")
        for action in feedback_actions:
            wave_data = self.generate_feedback_sound(action)
            filename = f"{action}.wav"
            filepath = os.path.join(feedback_dir, filename)
            success = self.save_wav_file(wave_data, filepath)
            results[f"feedback/{filename}"] = success

        # Ambient sounds
        ambient_types = ["calm", "focus", "energetic"]
        ambient_dir = os.path.join(base_dir, "ambient")
        for ambient_type in ambient_types:
            wave_data = self.generate_ambient_sound(ambient_type, duration=5.0)
            filename = f"{ambient_type}.wav"
            filepath = os.path.join(ambient_dir, filename)
            success = self.save_wav_file(wave_data, filepath)
            results[f"ambient/{filename}"] = success

        return results


def main():
    """Main function to generate all sound files."""
    generator = SoundGenerator()

    # Base sounds directory
    sounds_base = os.path.dirname(os.path.abspath(__file__))

    print("PACA Sound Generation System Starting...")

    try:
        results = generator.generate_all_sounds(sounds_base)

        # Count results by category
        alert_success = sum(1 for k, v in results.items() if k.startswith("alerts/") and v)
        feedback_success = sum(1 for k, v in results.items() if k.startswith("feedback/") and v)
        ambient_success = sum(1 for k, v in results.items() if k.startswith("ambient/") and v)

        total_success = alert_success + feedback_success + ambient_success
        total_sounds = len(results)

        print(f"Alert sounds generated: {alert_success}/5")
        print(f"Feedback sounds generated: {feedback_success}/5")
        print(f"Ambient sounds generated: {ambient_success}/3")
        print(f"\nTotal sounds generated: {total_success}/{total_sounds}")
        print("PACA Sound System Ready!")

    except ImportError as e:
        print(f"Warning: NumPy not available - {e}")
        print("Sound generation requires NumPy. Install with: pip install numpy")
        print("Generating sound configuration files instead...")

        # Generate configuration files for manual sound creation
        config_content = """# PACA Sound Configuration
# This file describes the sounds that should be created for the PACA desktop application

ALERT_SOUNDS:
  success: Ascending C-E-G chord, 0.5s duration, gentle fade
  error: Descending 800-600-400 Hz square waves, 0.5s duration
  warning: Alternating 880-440 Hz tones, 0.5s duration
  notification: Harmonic chime based on 440 Hz, 0.5s duration
  alert: 1000 Hz triangle wave, 0.5s duration

FEEDBACK_SOUNDS:
  click: 2000 Hz + 1000 Hz sharp click, 0.1s duration
  hover: 800 Hz soft tone, 0.1s duration
  select: 1200 Hz confirmation, 0.1s duration
  drag: Pink noise feedback, 0.1s duration
  drop: 400-600 Hz drop confirmation, 0.1s duration

AMBIENT_SOUNDS:
  calm: Pink noise with 200-450 Hz tones, 5s+ loop
  focus: White noise with subtle 100-220 Hz harmonics, 5s+ loop
  energetic: Pulsed multi-frequency ambient, 5s+ loop
"""
        config_file = os.path.join(sounds_base, "sound_config.txt")
        with open(config_file, 'w') as f:
            f.write(config_content)
        print("Sound configuration saved to sound_config.txt")


if __name__ == "__main__":
    main()