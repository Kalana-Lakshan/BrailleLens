import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../services/audio_service.dart';
import '../services/glass_device_service.dart';
import '../services/stt_onnx_service.dart';
import '../services/voice_capture_service.dart';
import '../theme/app_theme.dart';
import '../utils/letter_ctc.dart';
import '../utils/sinhala_phonetics.dart';

/// Bench screen for measuring the spoken-letter classifier by hand: pick the
/// letter you are about to say, tap, say it, and see what the model heard.
/// A running tally gives the accuracy over everything said so far.
///
/// Uses the same [VoiceCaptureService] + [SttOnnxService.classify] path and
/// the same right/wrong rule ([sinhalaLettersSoundAlike]) as Testing Mode, so
/// the score here is the score Testing Mode would give.
///
///   flutter run -t lib/main_stt_test.dart
class SttTestScreen extends StatefulWidget {
  /// Plays the same mic-open / mic-close chimes as Testing Mode.
  final AudioService audioService;

  const SttTestScreen({super.key, required this.audioService});

  @override
  State<SttTestScreen> createState() => _SttTestScreenState();
}

class _Attempt {
  final String expected;

  /// Null when nothing usable was heard.
  final SpokenLetter? answer;
  final VoiceSource source;
  final double seconds;
  final double peak;

  const _Attempt({
    required this.expected,
    required this.answer,
    required this.source,
    required this.seconds,
    required this.peak,
  });

  bool get heard => answer != null;

  /// Testing Mode's verdict.
  bool get correct =>
      heard && sinhalaLettersSoundAlike(answer!.letter, expected);

  bool get inTopThree =>
      heard &&
      answer!.guess.ranked
          .take(3)
          .any((s) => sinhalaLettersSoundAlike(s.letter, expected));
}

class _SttTestScreenState extends State<SttTestScreen> {
  final SttOnnxService _stt = SttOnnxService.instance;
  final VoiceCaptureService _voice = VoiceCaptureService();
  StreamSubscription<GlassButtonClicked>? _glassButtonSub;

  static const _durations = [2, 3, 5];
  int _seconds = 2;

  bool _loading = true;
  bool _recording = false;
  bool _classifying = false;

  /// Move to the next letter after each attempt, to sweep the whole set.
  bool _autoAdvance = true;
  String _status = 'Loading speech model…';
  String? _expected;
  final List<_Attempt> _history = [];

  bool get _busy => _loading || _recording || _classifying;

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
      if (ok && _stt.letters.isNotEmpty) _expected = _stt.letters.first;
      _status = ok
          ? '${_stt.loadedAsset?.split('/').last} · ${_stt.letters.length} '
              'letters · loaded in ${sw.elapsedMilliseconds} ms'
          : 'STT unavailable: ${_stt.lastError}';
    });
  }

  Future<void> _record() async {
    final expected = _expected;
    if (_busy || !_stt.isAvailable || expected == null) return;
    setState(() {
      _recording = true;
      _status = 'Say "$expected" now ($_seconds s)';
    });
    await widget.audioService.playMicOpen();
    final capture = await _voice.record(duration: Duration(seconds: _seconds));
    await widget.audioService.playMicClose();
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
      _classifying = true;
      _status = 'Classifying…';
    });
    final result = await _stt.classify(capture.samples);
    if (!mounted) return;

    setState(() {
      _classifying = false;
      _history.insert(
        0,
        _Attempt(
          expected: expected,
          // An unheard clip is counted separately, as Testing Mode does.
          answer: (result != null && result.heard) ? result : null,
          source: capture.source,
          seconds: capture.samples.length / SttOnnxService.sampleRate,
          peak: _peak(capture.samples),
        ),
      );
      _status = 'Pick a letter and tap to speak';
      if (_autoAdvance) {
        final letters = _stt.letters;
        _expected = letters[(letters.indexOf(expected) + 1) % letters.length];
      }
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
        title: const Text('STT ACCURACY TEST'),
        actions: [
          if (_history.isNotEmpty)
            IconButton(
              tooltip: 'Reset score',
              icon: const Icon(Icons.restart_alt),
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
            const SizedBox(height: 12),
            _buildScore(),
            const SizedBox(height: 12),
            _buildResult(latest),
            const SizedBox(height: 16),
            _buildControls(),
            const SizedBox(height: 16),
            _buildLetterPicker(),
            if (_history.isNotEmpty) ...[
              const SizedBox(height: 20),
              _buildMistakes(),
              const SizedBox(height: 16),
              const _SectionLabel('ATTEMPTS'),
              for (final a in _history) _buildAttemptRow(a),
            ],
          ],
        ),
      ),
    );
  }

  static const _good = Color(0xFF00E5FF);
  static const _bad = Color(0xFFFF5252);

  Widget _buildScore() {
    final heard = _history.where((a) => a.heard).toList();
    final correct = heard.where((a) => a.correct).length;
    final topThree = heard.where((a) => a.inTopThree).length;
    final unheard = _history.length - heard.length;
    String pct(int n, int of) =>
        of == 0 ? '—' : '${(100 * n / of).round()}%';

    Widget stat(String label, String value, {Color color = Colors.white}) =>
        Expanded(
          child: Column(
            children: [
              Text(value,
                  style: TextStyle(
                      color: color, fontSize: 22, fontWeight: FontWeight.bold)),
              Text(label,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white54, fontSize: 11)),
            ],
          ),
        );

    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
      decoration: BoxDecoration(
        color: Colors.white10,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          stat('correct', '$correct / ${heard.length}',
              color: AppTheme.primaryYellow),
          stat('accuracy', pct(correct, heard.length), color: _good),
          stat('in top 3', pct(topThree, heard.length)),
          stat('not heard', '$unheard'),
        ],
      ),
    );
  }

  Widget _buildResult(_Attempt? latest) {
    final answer = latest?.answer;
    final color = latest == null
        ? AppTheme.primaryYellow
        : (latest.correct ? _good : _bad);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: color, width: 2),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Text(
            latest == null ? 'MODEL HEARD' : 'YOU SAID ${latest.expected} · MODEL HEARD',
            style: const TextStyle(
                color: Colors.white70, fontSize: 12, letterSpacing: 1.1),
          ),
          const SizedBox(height: 4),
          Text(
            latest == null ? '…' : (answer?.letter ?? 'nothing'),
            style: TextStyle(
              color: color,
              fontSize: answer == null ? 24 : 64,
              fontWeight: FontWeight.bold,
            ),
          ),
          if (answer != null) ...[
            Text(
              latest!.correct ? '✓ CORRECT' : '✗ WRONG',
              style: TextStyle(color: color, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            _buildRanking(answer.guess, latest.expected),
          ],
          if (latest != null) ...[
            const SizedBox(height: 6),
            Text(
              _meta(latest),
              textAlign: TextAlign.center,
              style: TextStyle(
                // A near-silent capture is the first thing to rule out.
                color: latest.peak < 0.02 ? _bad : Colors.white38,
                fontSize: 11,
              ),
            ),
          ],
        ],
      ),
    );
  }

  /// The model's five most likely letters with their share of the
  /// probability; the expected letter is highlighted wherever it lands.
  Widget _buildRanking(LetterGuess guess, String expected) {
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      alignment: WrapAlignment.center,
      children: [
        for (final s in guess.ranked.take(5))
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: sinhalaLettersSoundAlike(s.letter, expected)
                  ? _good.withValues(alpha: 0.25)
                  : Colors.white10,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              '${s.letter}  ${(s.confidence * 100).round()}%',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
          ),
      ],
    );
  }

  String _meta(_Attempt a) =>
      '${a.source.name} mic · ${a.seconds.toStringAsFixed(1)} s · '
      'peak ${(a.peak * 100).round()}%'
      '${a.answer == null ? '' : ' · ${a.answer!.elapsed.inMilliseconds} ms'}';

  Widget _buildLetterPicker() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Expanded(child: _SectionLabel('LETTER YOU WILL SAY')),
            const Text('auto-next',
                style: TextStyle(color: Colors.white54, fontSize: 12)),
            Switch(
              value: _autoAdvance,
              activeThumbColor: AppTheme.primaryYellow,
              onChanged: (v) => setState(() => _autoAdvance = v),
            ),
          ],
        ),
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final letter in _stt.letters)
              ChoiceChip(
                label: Text(letter, style: const TextStyle(fontSize: 18)),
                selected: letter == _expected,
                onSelected:
                    _busy ? null : (_) => setState(() => _expected = letter),
              ),
          ],
        ),
      ],
    );
  }

  /// "said → heard" pairs, most frequent first: which letters the model
  /// mixes up.
  Widget _buildMistakes() {
    final counts = <String, int>{};
    for (final a in _history) {
      if (a.heard && !a.correct) {
        final key = '${a.expected} → ${a.answer!.letter}';
        counts[key] = (counts[key] ?? 0) + 1;
      }
    }
    if (counts.isEmpty) return const SizedBox.shrink();
    final sorted = counts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _SectionLabel('MISTAKES (SAID → HEARD)'),
        const SizedBox(height: 4),
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final e in sorted)
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: _bad.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  e.value > 1 ? '${e.key}  ×${e.value}' : e.key,
                  style: const TextStyle(color: Colors.white, fontSize: 16),
                ),
              ),
          ],
        ),
      ],
    );
  }

  Widget _buildAttemptRow(_Attempt a) {
    final color = !a.heard ? Colors.white38 : (a.correct ? _good : _bad);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 96,
            child: Text(
              '${a.expected} → ${a.answer?.letter ?? '—'}',
              style: TextStyle(color: color, fontSize: 20),
            ),
          ),
          Expanded(
            child: Text(
              a.heard
                  ? '${a.answer!.guess.ranked.take(3).join(' · ')}\n${_meta(a)}'
                  : 'not heard\n${_meta(a)}',
              style: const TextStyle(color: Colors.white38, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          height: 72,
          child: ElevatedButton.icon(
            onPressed:
                (_busy || !_stt.isAvailable || _expected == null) ? null : _record,
            icon: Icon(_recording ? Icons.mic : Icons.mic_none, size: 30),
            label: Text(
              _recording
                  ? 'LISTENING…'
                  : _classifying
                      ? 'CLASSIFYING…'
                      : 'TAP, THEN SAY  ${_expected ?? ''}',
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: _recording ? _bad : AppTheme.primaryYellow,
              foregroundColor: Colors.black,
            ),
          ),
        ),
        const SizedBox(height: 8),
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
        const SizedBox(height: 4),
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

class _SectionLabel extends StatelessWidget {
  final String text;

  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) => Text(
        text,
        style: const TextStyle(
            color: Colors.white54, fontSize: 12, letterSpacing: 1.1),
      );
}
