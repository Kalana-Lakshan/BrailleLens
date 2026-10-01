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
    _ttsReady = _initTts();
  }

  // ── TTS ──────────────────────────────────────────────────────────────────────

  /// Google's engine ships a Sinhala (si-LK) voice. Samsung phones default
  /// to Samsung TTS, which has none — every Sinhala prompt was silent on the
  /// Galaxy M14 — so Google's is used whenever it is installed.
  static const _googleTtsEngine = 'com.google.android.tts';

  late final Future<void> _ttsReady;
  bool _sinhalaVoice = false;

  /// Whether the engine in use can speak si-LK. False means Sinhala prompts
  /// will be silent or mangled until Sinhala voice data is installed.
  bool get hasSinhalaVoice => _sinhalaVoice;

  Future<void> _initTts() async {
    if (defaultTargetPlatform == TargetPlatform.android) {
      try {
        final engines = ((await _tts.getEngines) as List?)
                ?.map((e) => '$e')
                .toList() ??
            const <String>[];
        // Completes once the new engine has finished initialising.
        if (engines.contains(_googleTtsEngine)) {
          await _tts.setEngine(_googleTtsEngine);
        }
        debugPrint('[AudioService] TTS engines $engines, using '
            '${engines.contains(_googleTtsEngine) ? _googleTtsEngine : 'system default'}');
      } catch (e) {
        debugPrint('[AudioService] TTS engine selection failed: $e');
      }
    }
    try {
      _sinhalaVoice = await _tts.isLanguageAvailable('si-LK') == true;
      final installed = await _tts.isLanguageInstalled('si-LK');
      debugPrint('[AudioService] si-LK available=$_sinhalaVoice '
          'installed=$installed');
    } catch (e) {
      debugPrint('[AudioService] si-LK check failed: $e');
    }
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

  /// Set while an in-screen command listener runs: every prompt first takes
  /// the microphone back from the recogniser, so the app neither transcribes
  /// itself nor has its own voice ducked by the recogniser's audio focus.
  bool yieldListeningToSpeech = false;

  int _speaking = 0;

  /// True while a prompt is playing or queued.
  bool get isSpeaking => _speaking > 0;

  /// Prompts play one after another. Each used to start with `stop()`, so a
  /// second prompt cut off the one playing — on the phone log an English
  /// status line replaced a Sinhala prompt 10 ms after it started, and Google
  /// TTS reported the cut sentence as TextToSpeech.ERROR.
  Future<void> _speechTail = Future<void>.value();

  /// Bumped by [stopSpeech]: queued prompts from before it are dropped.
  int _speechGen = 0;

  Future<void> _enqueue(Future<void> Function() utter) {
    final gen = _speechGen;
    _speaking++;
    final run = _speechTail.then((_) async {
      try {
        if (gen == _speechGen) await utter();
      } finally {
        _speaking--;
      }
    });
    _speechTail = run.catchError((_) {});
    return run;
  }

  /// Completes once every queued prompt has finished — the recogniser waits
  /// on this, since starting it over a Bluetooth headset tears down the
  /// audio link and kills the sentence playing on it.
  Future<void> get speechIdle => _speechTail;

  Future<void> speak(String text) {
    if (yieldListeningToSpeech) stopListening();
    return _enqueue(() => _speakNow(text));
  }

  Future<void> _speakNow(String text) async {
    try {
      await _ttsReady;
      await _tts.speak(text);
    } catch (e) {
      debugPrint('[AudioService] TTS speak error: $e');
    }
  }

  /// Cuts off the prompt playing and drops the queued ones — for leaving a
  /// screen or opening the mic, where stale speech must not carry on.
  Future<void> stopSpeech() async {
    _speechGen++;
    try {
      await _tts.stop();
    } catch (e) {
      debugPrint('[AudioService] TTS stop error: $e');
    }
  }

  /// Says a Sinhala glyph using the device's Sinhala voice when available.
  Future<void> speakSinhalaCharacter(String character) =>
      speakSinhala(character);

  /// Speaks Sinhala text, then puts the engine back on the app's default
  /// voice.
  ///
  /// Leaving the engine on `si-LK` (as this used to) means every later
  /// English prompt is read by the Sinhala voice, which mangles it or goes
  /// silent when only one of the two voices is installed.
  Future<void> speakSinhala(String text) {
    if (text.trim().isEmpty) return Future<void>.value();
    if (yieldListeningToSpeech) stopListening();
    return _enqueue(() => _speakSinhalaNow(text));
  }

  Future<void> _speakSinhalaNow(String text) async {
    try {
      await _ttsReady;
      // 0 means this engine has no Sinhala voice: say so in the log, since
      // the prompt will come out silent or read letter by letter.
      if (await _tts.setLanguage('si-LK') != 1) {
        debugPrint('[AudioService] no si-LK voice for: $text');
      }
      await _tts.speak(text);
    } catch (e) {
      debugPrint('[AudioService] Sinhala TTS error: $e');
      // Already inside the queue: speak directly, not via [speak].
      await _speakNow(text);
    } finally {
      try {
        await _tts.setLanguage('en-US');
      } catch (_) {
        // Leaving the language unrestored is not worth failing the prompt.
      }
    }
  }

  /// Chime + light haptic marking the microphone opening, and — importantly —
  /// waits for the TTS engine to release audio focus first.
  ///
  /// Android hands the mic to whoever holds focus; starting a recording while
  /// the prompt is still playing gets a truncated recording, the prompt
  /// itself recorded back, or a mic that never opens.
  Future<void> playMicOpen() async {
    await stopSpeech();
    await playStartListeningTone();
    await hapticLight();
    // Short settle so the earcon does not bleed into the recording.
    await Future<void>.delayed(const Duration(milliseconds: 180));
  }

  /// Closing chime for the microphone.
  Future<void> playMicClose() => playStopListeningTone();

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

  /// Plays [count] start earcons with [gap] between them (page / dwell lock).
  ///
  /// Returns `false` if [shouldAbort] became true mid-sequence so the host
  /// can cancel the pending capture.
  Future<bool> playCountdownBeeps(
    int count, {
    Duration gap = const Duration(milliseconds: 500),
    bool Function()? shouldAbort,
  }) async {
    for (var i = 0; i < count; i++) {
      if (shouldAbort?.call() == true) return false;
      try {
        await _player.stop();
        await _player.play(AssetSource('audio/earcon_start.wav'));
      } catch (e) {
        debugPrint('[AudioService] Countdown beep error: $e');
      }
      if (i < count - 1) {
        await Future<void>.delayed(gap);
      }
    }
    return shouldAbort?.call() != true;
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
    // Never open the recogniser over a prompt (see [speechIdle]).
    await speechIdle;

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
    // Never open the recogniser over a prompt (see [speechIdle]).
    await speechIdle;

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
