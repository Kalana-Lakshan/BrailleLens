import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:audioplayers/audioplayers.dart';
import '../utils/answer_match.dart';

/// Unified audio service handling TTS, STT, earcon tones, and haptic feedback.
/// All methods fail silently so a missing earcon asset or hardware limitation
/// never crashes the app.
class AudioService {
  final FlutterTts _tts = FlutterTts();
  final stt.SpeechToText _speech = stt.SpeechToText();
  final AudioPlayer _player = AudioPlayer();
  bool _isSpeechInitialized = false;

  AudioService() {
    _initTts();
  }

  // ── TTS ──────────────────────────────────────────────────────────────────────

  Future<void> _initTts() async {
    try {
      await _tts.setLanguage('en-US');
      await _tts.setSpeechRate(0.48);
      await _tts.setVolume(1.0);
      await _tts.setPitch(1.0);
      await _tts.awaitSpeakCompletion(true);
    } catch (e) {
      debugPrint('[AudioService] TTS init error: $e');
    }
  }

  Future<void> speak(String text) async {
    try {
      await _tts.stop();
      await _tts.speak(text);
    } catch (e) {
      debugPrint('[AudioService] TTS speak error: $e');
    }
  }

  Future<void> stopSpeech() async {
    try {
      await _tts.stop();
    } catch (e) {
      debugPrint('[AudioService] TTS stop error: $e');
    }
  }

  // ── Earcons ──────────────────────────────────────────────────────────────────

  /// High-pitch chime — played when the microphone opens.
  Future<void> playStartListeningTone() async {
    try {
      await _player.play(AssetSource('audio/earcon_start.wav'));
    } catch (e) {
      debugPrint('[AudioService] Earcon start error: $e');
    }
  }

  /// Low double-chime — played when the microphone closes.
  Future<void> playStopListeningTone() async {
    try {
      await _player.play(AssetSource('audio/earcon_stop.wav'));
    } catch (e) {
      debugPrint('[AudioService] Earcon stop error: $e');
    }
  }

  /// Rising two-tone — played on a correct answer.
  Future<void> playSuccessTone() async {
    try {
      await _player.play(AssetSource('audio/earcon_success.wav'));
    } catch (e) {
      debugPrint('[AudioService] Earcon success error: $e');
    }
  }

  /// Descending buzz — played on an incorrect answer.
  Future<void> playErrorTone() async {
    try {
      await _player.play(AssetSource('audio/earcon_error.wav'));
    } catch (e) {
      debugPrint('[AudioService] Earcon error error: $e');
    }
  }

  // ── Haptic Feedback ───────────────────────────────────────────────────────────

  /// Single light tap — e.g. camera frame locked.
  Future<void> hapticLight() async {
    try {
      await HapticFeedback.lightImpact();
    } catch (_) {}
  }

  /// Single medium tap — e.g. mode selected or connection established.
  Future<void> hapticMedium() async {
    try {
      await HapticFeedback.mediumImpact();
    } catch (_) {}
  }

  /// Single heavy tap — strong confirmation.
  Future<void> hapticHeavy() async {
    try {
      await HapticFeedback.heavyImpact();
    } catch (_) {}
  }

  /// Double heavy pulse — e.g. BT device connected or correct answer.
  Future<void> hapticDouble() async {
    try {
      await HapticFeedback.heavyImpact();
      await Future.delayed(const Duration(milliseconds: 130));
      await HapticFeedback.heavyImpact();
    } catch (_) {}
  }

  /// Triple rapid heavy pulse — e.g. error or incorrect answer.
  Future<void> hapticError() async {
    try {
      await HapticFeedback.heavyImpact();
      await Future.delayed(const Duration(milliseconds: 80));
      await HapticFeedback.heavyImpact();
      await Future.delayed(const Duration(milliseconds: 80));
      await HapticFeedback.heavyImpact();
    } catch (_) {}
  }

  // ── STT ──────────────────────────────────────────────────────────────────────

  Future<bool> initStt() async {
    if (!_isSpeechInitialized) {
      try {
        _isSpeechInitialized = await _speech.initialize(
          onError: (val) => debugPrint('[AudioService] STT error: $val'),
          onStatus: (val) => debugPrint('[AudioService] STT status: $val'),
        );
      } catch (e) {
        debugPrint('[AudioService] STT init exception: $e');
        _isSpeechInitialized = false;
      }
    }
    return _isSpeechInitialized;
  }

  /// Listens for a single spoken answer with a fixed [timeout].
  /// Returns the recognized words (lower-cased & trimmed), or null on timeout.
  Future<String?> listenForAnswer({
    Duration timeout = const Duration(seconds: 6),
  }) async {
    final available = await initStt();
    if (!available) return null;

    final completer = Completer<String?>();
    Timer? timer;

    timer = Timer(timeout, () {
      if (!completer.isCompleted) {
        _speech.stop();
        completer.complete(null);
      }
    });

    try {
      if (_speech.isListening) await _speech.stop();
      await _speech.listen(
        onResult: (result) {
          if (result.finalResult && result.recognizedWords.isNotEmpty) {
            timer?.cancel();
            if (!completer.isCompleted) {
              completer.complete(
                result.recognizedWords.toLowerCase().trim(),
              );
            }
          }
        },
        listenOptions: stt.SpeechListenOptions(
          listenMode: stt.ListenMode.confirmation,
          partialResults: false,
          onDevice: true,
        ),
      );
    } catch (e) {
      debugPrint('[AudioService] listenForAnswer on-device error: $e');
      try {
        if (_speech.isListening) await _speech.stop();
        await _speech.listen(
          onResult: (result) {
            if (result.finalResult && result.recognizedWords.isNotEmpty) {
              timer?.cancel();
              if (!completer.isCompleted) {
                completer.complete(
                  result.recognizedWords.toLowerCase().trim(),
                );
              }
            }
          },
          listenOptions: stt.SpeechListenOptions(
            listenMode: stt.ListenMode.confirmation,
            partialResults: false,
            onDevice: false,
          ),
        );
      } catch (e2) {
        debugPrint('[AudioService] listenForAnswer error: $e2');
        timer.cancel();
        if (!completer.isCompleted) completer.complete(null);
      }
    }

    return completer.future;
  }

  /// Continuous listen that ends when the user says the word **"stop"** or [timeout] elapses.
  Future<ListenOutcome> listenUntilStop({
    Duration timeout = const Duration(seconds: 12),
  }) async {
    final available = await initStt();
    if (!available) {
      return const ListenOutcome(words: null, stoppedByKeyword: false, timedOut: false, sttUnavailable: true);
    }

    final completer = Completer<ListenOutcome>();
    Timer? timer;
    String latestWords = '';

    timer = Timer(timeout, () {
      if (!completer.isCompleted) {
        _speech.stop();
        completer.complete(ListenOutcome(
          words: latestWords.isNotEmpty ? latestWords : null,
          stoppedByKeyword: false,
          timedOut: true,
        ));
      }
    });

    void onResult(result) {
      final words = result.recognizedWords.toLowerCase().trim();

      if (spokenContainsStopKeyword(words)) {
        timer?.cancel();
        if (!completer.isCompleted) {
          _speech.stop();
          completer.complete(const ListenOutcome(
            words: null,
            stoppedByKeyword: true,
            timedOut: false,
          ));
        }
        return;
      }

      if (result.finalResult) {
        latestWords = words;
      }
    }

    try {
      if (_speech.isListening) await _speech.stop();
      await _speech.listen(
        onResult: onResult,
        listenOptions: stt.SpeechListenOptions(
          listenMode: stt.ListenMode.dictation,
          partialResults: true,
          onDevice: true,
        ),
      );
    } catch (e) {
      debugPrint('[AudioService] listenUntilStop on-device error: $e');
      try {
        if (_speech.isListening) await _speech.stop();
        await _speech.listen(
          onResult: onResult,
          listenOptions: stt.SpeechListenOptions(
            listenMode: stt.ListenMode.dictation,
            partialResults: true,
            onDevice: false,
          ),
        );
      } catch (e2) {
        debugPrint('[AudioService] listenUntilStop error: $e2');
        timer.cancel();
        if (!completer.isCompleted) {
          completer.complete(const ListenOutcome(
            words: null,
            stoppedByKeyword: false,
            timedOut: false,
            sttUnavailable: true,
          ));
        }
      }
    }

    return completer.future;
  }

  bool get isListening => _speech.isListening;

  void stopListening() {
    if (_speech.isListening) _speech.stop();
  }

  void dispose() {
    _tts.stop();
    _speech.stop();
    _player.dispose();
  }
}

/// Result of [AudioService.listenUntilStop].
class ListenOutcome {
  final String? words;
  final bool stoppedByKeyword;
  final bool timedOut;
  final bool sttUnavailable;

  const ListenOutcome({
    required this.words,
    required this.stoppedByKeyword,
    required this.timedOut,
    this.sttUnavailable = false,
  });
}
