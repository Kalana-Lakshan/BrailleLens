// tools/generate_earcons.dart
// Run with: dart run tools/generate_earcons.dart
// Generates four earcon .wav files into assets/audio/
// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

const int sampleRate = 44100;

/// Generates a PCM sine wave at [frequency] Hz for [durationMs] milliseconds,
/// with an optional fade-in and fade-out envelope.
List<double> generateTone(
  double frequency,
  int durationMs, {
  double amplitude = 0.7,
}) {
  final numSamples = (sampleRate * durationMs / 1000).round();
  final samples = List<double>.filled(numSamples, 0.0);

  for (int i = 0; i < numSamples; i++) {
    final t = i / sampleRate;
    // Fade in over first 10ms, fade out over last 30ms
    double envelope = 1.0;
    final fadeInSamples = (sampleRate * 0.01).round();
    final fadeOutSamples = (sampleRate * 0.03).round();
    if (i < fadeInSamples) {
      envelope = i / fadeInSamples;
    } else if (i > numSamples - fadeOutSamples) {
      envelope = (numSamples - i) / fadeOutSamples;
    }
    samples[i] = amplitude * envelope * sin(2 * pi * frequency * t);
  }
  return samples;
}

/// Concatenates multiple tone segments into one sample list.
List<double> concat(List<List<double>> segments, {int silenceMs = 0}) {
  final result = <double>[];
  final silenceSamples = (sampleRate * silenceMs / 1000).round();
  for (final seg in segments) {
    result.addAll(seg);
    if (silenceMs > 0) {
      result.addAll(List.filled(silenceSamples, 0.0));
    }
  }
  return result;
}

/// Writes a 16-bit PCM mono WAV file.
void writeWav(String path, List<double> samples) {
  final numSamples = samples.length;
  final dataSize = numSamples * 2; // 16-bit = 2 bytes per sample
  final fileSize = 36 + dataSize;

  final buffer = BytesBuilder();

  // RIFF header
  buffer.add('RIFF'.codeUnits);
  buffer.add(_int32LE(fileSize));
  buffer.add('WAVE'.codeUnits);

  // fmt chunk
  buffer.add('fmt '.codeUnits);
  buffer.add(_int32LE(16));       // chunk size
  buffer.add(_int16LE(1));        // PCM format
  buffer.add(_int16LE(1));        // mono
  buffer.add(_int32LE(sampleRate));
  buffer.add(_int32LE(sampleRate * 2)); // byte rate
  buffer.add(_int16LE(2));        // block align
  buffer.add(_int16LE(16));       // bits per sample

  // data chunk
  buffer.add('data'.codeUnits);
  buffer.add(_int32LE(dataSize));

  // PCM samples clamped to [-1, 1] -> int16
  for (final s in samples) {
    final clamped = s.clamp(-1.0, 1.0);
    final intVal = (clamped * 32767).round().clamp(-32768, 32767);
    buffer.add(_int16LE(intVal));
  }

  File(path).writeAsBytesSync(buffer.toBytes());
  print('Written: $path (${buffer.toBytes().length} bytes)');
}

List<int> _int32LE(int value) {
  final b = Uint8List(4);
  b.buffer.asByteData().setInt32(0, value, Endian.little);
  return b;
}

List<int> _int16LE(int value) {
  final b = Uint8List(2);
  b.buffer.asByteData().setInt16(0, value, Endian.little);
  return b;
}

void main() {
  final outputDir = Directory('assets/audio');
  outputDir.createSync(recursive: true);

  // 1. earcon_start.wav - High-pitch chime (880 Hz, 300ms)
  writeWav(
    'assets/audio/earcon_start.wav',
    generateTone(880, 300, amplitude: 0.65),
  );

  // 2. earcon_stop.wav - Low double-chime (440 -> 330 Hz, 200ms each)
  writeWav(
    'assets/audio/earcon_stop.wav',
    concat([
      generateTone(440, 180, amplitude: 0.6),
      generateTone(330, 200, amplitude: 0.55),
    ], silenceMs: 40),
  );

  // 3. earcon_success.wav - Rising two-tone (523 -> 659 Hz)
  writeWav(
    'assets/audio/earcon_success.wav',
    concat([
      generateTone(523, 180, amplitude: 0.65),
      generateTone(659, 250, amplitude: 0.7),
    ], silenceMs: 30),
  );

  // 4. earcon_error.wav - Descending buzz (330 -> 220 Hz)
  writeWav(
    'assets/audio/earcon_error.wav',
    concat([
      generateTone(330, 220, amplitude: 0.65),
      generateTone(220, 280, amplitude: 0.6),
    ], silenceMs: 30),
  );

  print('\nAll earcon files generated in assets/audio/');
}
