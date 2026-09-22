import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';

import 'glass_device_service.dart';
import 'stt_onnx_service.dart';

/// Where a recorded answer came from, for status lines and logs.
enum VoiceSource { glasses, phone, none }

@immutable
class VoiceCapture {
  /// 16 kHz mono samples in [-1, 1], ready for [SttOnnxService.transcribe].
  final Float32List samples;
  final VoiceSource source;
  final String? error;

  const VoiceCapture({
    required this.samples,
    required this.source,
    this.error,
  });

  bool get hasAudio => samples.isNotEmpty;

  /// Nothing recorded, with [error] saying why.
  factory VoiceCapture.failed(String error) => VoiceCapture(
        samples: Float32List(0),
        source: VoiceSource.none,
        error: error,
      );
}

/// Records a short spoken answer as 16 kHz mono PCM.
///
/// The glasses microphone is preferred when connected — it is at the wearer's
/// head, already 16 kHz mono (the bridge configures the SDK that way), and
/// arrives over the vendor channel rather than the phone's audio input. When
/// the glasses are absent the phone mic does the same job through `record`.
///
/// Both paths return the same shape, so the STT model sees one contract.
class VoiceCaptureService {
  final GlassDeviceService _glass;
  final AudioRecorder _recorder = AudioRecorder();

  VoiceCaptureService({GlassDeviceService? glass})
      : _glass = glass ?? GlassDeviceService.instance;

  /// Records for [duration], from the glasses when connected.
  Future<VoiceCapture> record({
    Duration duration = const Duration(milliseconds: 2500),
  }) async {
    if (_glass.isConnected) {
      final capture = await _recordFromGlasses(duration);
      // The glasses can refuse (link busy, mic already streaming); the phone
      // is a usable answer path, so fall through rather than failing the round.
      if (capture.hasAudio) return capture;
      debugPrint('[VoiceCapture] glasses mic unusable: ${capture.error}');
    }
    return _recordFromPhone(duration);
  }

  /// Glasses path: `startUserVoiceInput()` on the bridge, accumulate the PCM
  /// chunks (16 kHz mono 16-bit, ~640 bytes per frame), then stop.
  Future<VoiceCapture> _recordFromGlasses(Duration duration) async {
    final chunks = <Uint8List>[];
    // Subscribe before starting so the opening frames are not dropped.
    final sub = _glass.micData.listen((e) => chunks.add(e.pcm));
    try {
      if (!await _glass.startMicrophone()) {
        return VoiceCapture.failed(
          _glass.lastError ?? 'glasses refused the microphone',
        );
      }
      await Future<void>.delayed(duration);
    } finally {
      await _glass.stopMicrophone();
      await sub.cancel();
    }

    final pcm = _concat(chunks);
    debugPrint('[VoiceCapture] glasses: ${pcm.length} bytes');
    return VoiceCapture(
      samples: pcm16ToFloat32(pcm),
      source: VoiceSource.glasses,
    );
  }

  /// Phone path: raw PCM stream, so no file is written and nothing has to be
  /// decoded afterwards.
  Future<VoiceCapture> _recordFromPhone(Duration duration) async {
    try {
      if (!await _recorder.hasPermission()) {
        return VoiceCapture.failed('microphone permission denied');
      }
      final stream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: SttOnnxService.sampleRate,
          numChannels: 1,
          // The glasses are a Bluetooth headset when connected; this keeps
          // the phone's own mic in play rather than reopening a SCO link.
          androidConfig: AndroidRecordConfig(useLegacy: false),
        ),
      );
      final chunks = <Uint8List>[];
      final sub = stream.listen(chunks.add);
      await Future<void>.delayed(duration);
      await _recorder.stop();
      await sub.cancel();

      final pcm = _concat(chunks);
      debugPrint('[VoiceCapture] phone: ${pcm.length} bytes');
      return VoiceCapture(
        samples: pcm16ToFloat32(pcm),
        source: VoiceSource.phone,
      );
    } catch (e) {
      debugPrint('[VoiceCapture] phone mic failed: $e');
      await _recorder.cancel().catchError((_) {});
      return VoiceCapture.failed('$e');
    }
  }

  static Uint8List _concat(List<Uint8List> chunks) {
    final total = chunks.fold<int>(0, (sum, c) => sum + c.length);
    final out = Uint8List(total);
    var offset = 0;
    for (final c in chunks) {
      out.setAll(offset, c);
      offset += c.length;
    }
    return out;
  }

  /// Signed 16-bit little-endian PCM → float samples in [-1, 1].
  @visibleForTesting
  static Float32List pcm16ToFloat32(Uint8List pcm) {
    final sampleCount = pcm.length ~/ 2;
    final out = Float32List(sampleCount);
    final view = ByteData.view(pcm.buffer, pcm.offsetInBytes, sampleCount * 2);
    for (var i = 0; i < sampleCount; i++) {
      out[i] = view.getInt16(i * 2, Endian.little) / 32768.0;
    }
    return out;
  }

  Future<void> dispose() async {
    await _recorder.cancel().catchError((_) {});
    await _recorder.dispose();
  }
}
