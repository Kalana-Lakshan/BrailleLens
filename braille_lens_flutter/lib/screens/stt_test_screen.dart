import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../services/glass_device_service.dart';
import '../services/stt_onnx_service.dart';
import '../services/voice_capture_service.dart';
import '../theme/app_theme.dart';
import '../utils/sinhala_phonetics.dart';

/// Bench screen for the on-device Sinhala STT model on its own: tap, speak,
/// and see exactly what the model heard, character by character.
///
/// Uses the same [VoiceCaptureService] + [SttOnnxService] path as Testing
/// Mode, so a result here is what Testing Mode will score. The optional
/// "expected" field runs Testing Mode's [sinhalaAnswerMatches] too.
///
///   flutter run -t lib/main_stt_test.dart
class SttTestScreen extends StatefulWidget {
  const SttTestScreen({super.key});

  @override
  State<SttTestScreen> createState() => _SttTestScreenState();
}

class _Attempt {
  final String text;
  final VoiceSource source;
  final double seconds;
  final double peak;
  final int inferenceMs;

  const _Attempt({
    required this.text,
    required this.source,
    required this.seconds,
    required this.peak,
    required this.inferenceMs,
  });
}

class _SttTestScreenState extends State<SttTestScreen> {
  final SttOnnxService _stt = SttOnnxService.instance;
  final VoiceCaptureService _voice = VoiceCaptureService();
  final TextEditingController _expected = TextEditingController();
  StreamSubscription<GlassButtonClicked>? _glassButtonSub;

  static const _durations = [2, 3, 5];
  int _seconds = 3;

  bool _loading = true;
  bool _recording = false;
  bool _decoding = false;
  String _status = 'Loading speech model…';
  final List<_Attempt> _history = [];

  bool get _busy => _loading || _recording || _decoding;

  @override
  void initState() {
    super.initState();
    _glassButtonSub =
        GlassDeviceService.instance.buttonClicks.listen((_) => _record());
    _load();
  }

  Future<void> _load() async {
    final sw = Stopwatch()..start();
    final ok = await _stt.initialize();
    if (!mounted) return;
    setState(() {
      _loading = false;
      _status = ok
          ? '${_stt.loadedAsset?.split('/').last} · vocab ${_stt.vocabSize}'
              ' · loaded in ${sw.elapsedMilliseconds} ms'
          : 'STT unavailable: ${_stt.lastError}';
    });
  }

  Future<void> _record() async {
    if (_busy || !_stt.isAvailable) return;
    setState(() {
      _recording = true;
      _status = 'Listening for $_seconds s — speak now';
    });
    final capture = await _voice.record(duration: Duration(seconds: _seconds));
    if (!mounted) return;
    if (!capture.hasAudio) {
      setState(() {
        _recording = false;
        _status = capture.error ?? 'No audio recorded';
      });
      return;
    }

    setState(() {
      _recording = false;
      _decoding = true;
      _status = 'Transcribing…';
    });
    final sw = Stopwatch()..start();
    final text = await _stt.transcribe(capture.samples);
    final ms = sw.elapsedMilliseconds;
    if (!mounted) return;

    setState(() {
      _decoding = false;
      _history.insert(
        0,
        _Attempt(
          text: text ?? '',
          source: capture.source,
          seconds: capture.samples.length / SttOnnxService.sampleRate,
          peak: _peak(capture.samples),
          inferenceMs: ms,
        ),
      );
      _status = text == null
          ? 'Transcription failed: ${_stt.lastError ?? 'audio too short'}'
          : 'Tap to speak again';
    });
  }

  /// Loudest sample, 0–1. Near zero means the mic delivered silence, which
  /// separates "mic problem" from "model problem" at a glance.
  static double _peak(Float32List samples) {
    var peak = 0.0;
    for (final s in samples) {
      final a = s.abs();
      if (a > peak) peak = a;
    }
    return peak;
  }

  @override
  void dispose() {
    _glassButtonSub?.cancel();
    _voice.dispose();
    _expected.dispose();
    // The STT session is app-wide and shared, so it is not disposed here.
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final latest = _history.isEmpty ? null : _history.first;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: AppTheme.primaryYellow,
        title: const Text('STT TEST'),
        actions: [
          if (_history.isNotEmpty)
            IconButton(
              tooltip: 'Clear history',
              icon: const Icon(Icons.delete_outline),
              onPressed: _busy ? null : () => setState(_history.clear),
            ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              _status,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white54, fontSize: 12),
            ),
            const SizedBox(height: 16),
            _buildTranscript(latest),
            const SizedBox(height: 16),
            if (latest != null && latest.text.isNotEmpty)
              _buildCharacters(latest.text),
            const SizedBox(height: 16),
            _buildExpected(latest),
            const SizedBox(height: 20),
            _buildControls(),
            if (_history.length > 1) ...[
              const SizedBox(height: 24),
              const Text('EARLIER',
                  style: TextStyle(
                      color: Colors.white54,
                      fontSize: 12,
                      letterSpacing: 1.1)),
              const SizedBox(height: 6),
              for (final a in _history.skip(1))
                ListTile(
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  title: Text(a.text.isEmpty ? '—' : a.text,
                      style: const TextStyle(color: Colors.white, fontSize: 20)),
                  subtitle: Text(_meta(a),
                      style: const TextStyle(color: Colors.white38, fontSize: 11)),
                ),
            ],
          ],
        ),
      ),
    );
  }

  String _meta(_Attempt a) =>
      '${a.source.name} mic · ${a.seconds.toStringAsFixed(1)} s audio · '
      'peak ${(a.peak * 100).round()}% · ${a.inferenceMs} ms';

  Widget _buildTranscript(_Attempt? latest) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: AppTheme.primaryYellow, width: 2),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          const Text('YOU SAID',
              style: TextStyle(
                  color: Colors.white70, fontSize: 12, letterSpacing: 1.1)),
          const SizedBox(height: 6),
          SelectableText(
            latest == null ? '…' : (latest.text.isEmpty ? '(nothing recognised)' : latest.text),
            textAlign: TextAlign.center,
            style: TextStyle(
              color: AppTheme.primaryYellow,
              fontSize: latest?.text.isNotEmpty == true ? 44 : 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          if (latest != null) ...[
            const SizedBox(height: 6),
            Text(_meta(latest),
                textAlign: TextAlign.center,
                style: TextStyle(
                  // A near-silent capture is the first thing to rule out.
                  color: latest.peak < 0.02 ? const Color(0xFFFF5252) : Colors.white38,
                  fontSize: 11,
                )),
          ],
        ],
      ),
    );
  }

  /// One chip per code point, so combining vowel signs and the virama the
  /// model emitted are visible rather than folded into the rendered syllable.
  Widget _buildCharacters(String text) {
    final chars = text.runes.map(String.fromCharCode).toList();
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      alignment: WrapAlignment.center,
      children: [
        for (final ch in chars)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.white10,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(ch == ' ' ? '␣' : ch,
                    style: const TextStyle(color: Colors.white, fontSize: 26)),
                Text(
                  'U+${ch.runes.first.toRadixString(16).toUpperCase().padLeft(4, '0')}',
                  style: const TextStyle(color: Colors.white38, fontSize: 10),
                ),
              ],
            ),
          ),
      ],
    );
  }

  /// Optional: type the letter you meant and see whether Testing Mode would
  /// have marked the transcript correct.
  Widget _buildExpected(_Attempt? latest) {
    final expected = _expected.text.trim();
    final verdict = latest == null || expected.isEmpty
        ? null
        : sinhalaAnswerMatches(latest.text, expected);
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _expected,
            onChanged: (_) => setState(() {}),
            style: const TextStyle(color: Colors.white, fontSize: 20),
            decoration: InputDecoration(
              labelText: 'Expected letter (optional)',
              labelStyle: const TextStyle(color: Colors.white54),
              helperText: expected.isEmpty
                  ? 'e.g. ක — Testing Mode accepts it or its name'
                  : 'Accepted: $expected or ${sinhalaLetterName(expected)}',
              helperStyle: const TextStyle(color: Colors.white38),
              enabledBorder: const OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.white24)),
            ),
          ),
        ),
        const SizedBox(width: 12),
        if (verdict != null)
          Text(
            verdict ? '✓ MATCH' : '✗ NO MATCH',
            style: TextStyle(
              color: verdict ? const Color(0xFF00E5FF) : const Color(0xFFFF5252),
              fontWeight: FontWeight.bold,
            ),
          ),
      ],
    );
  }

  Widget _buildControls() {
    return Column(
      children: [
        Wrap(
          spacing: 8,
          alignment: WrapAlignment.center,
          children: [
            for (final s in _durations)
              ChoiceChip(
                label: Text('$s s'),
                selected: _seconds == s,
                onSelected: _busy ? null : (_) => setState(() => _seconds = s),
              ),
          ],
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          height: 72,
          child: ElevatedButton.icon(
            onPressed: (_busy || !_stt.isAvailable) ? null : _record,
            icon: Icon(_recording ? Icons.mic : Icons.mic_none, size: 30),
            label: Text(
              _recording
                  ? 'LISTENING…'
                  : _decoding
                      ? 'TRANSCRIBING…'
                      : 'TAP TO SPEAK',
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor:
                  _recording ? const Color(0xFFFF5252) : AppTheme.primaryYellow,
              foregroundColor: Colors.black,
            ),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          GlassDeviceService.instance.isConnected
              ? 'Glasses connected: their mic is used · glasses button also records'
              : 'Using the phone microphone',
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.white38, fontSize: 11),
        ),
      ],
    );
  }
}
