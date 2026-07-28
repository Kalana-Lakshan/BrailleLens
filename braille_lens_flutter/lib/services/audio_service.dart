import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

class AudioService {
  final FlutterTts _tts = FlutterTts();
  final stt.SpeechToText _speech = stt.SpeechToText();
  bool _isSpeechInitialized = false;

  AudioService() {
    _initTts();
  }

  Future<void> _initTts() async {
    try {
      await _tts.setLanguage("en-US");
      await _tts.setSpeechRate(0.5);
      await _tts.setVolume(1.0);
      await _tts.setPitch(1.0);
    } catch (e) {
      debugPrint('TTS Init error: $e');
    }
  }

  Future<void> speak(String text) async {
    try {
      await _tts.stop();
      await _tts.speak(text);
    } catch (e) {
      debugPrint('TTS Speak error: $e');
    }
  }

  Future<void> stopSpeech() async {
    try {
      await _tts.stop();
    } catch (e) {
      debugPrint('TTS Stop error: $e');
    }
  }

  Future<bool> initStt() async {
    if (!_isSpeechInitialized) {
      try {
        _isSpeechInitialized = await _speech.initialize(
          onError: (val) => debugPrint('STT Error: $val'),
          onStatus: (val) => debugPrint('STT Status: $val'),
        );
      } catch (e) {
        debugPrint('STT Init exception: $e');
        _isSpeechInitialized = false;
      }
    }
    return _isSpeechInitialized;
  }

  Future<String?> listenForAnswer({Duration timeout = const Duration(seconds: 5)}) async {
    bool available = await initStt();
    if (!available) {
      return null;
    }

    Completer<String?> completer = Completer<String?>();
    Timer? timer;

    timer = Timer(timeout, () {
      if (!completer.isCompleted) {
        _speech.stop();
        completer.complete(null);
      }
    });

    try {
      await _speech.listen(
        onResult: (result) {
          if (result.recognizedWords.isNotEmpty) {
            timer?.cancel();
            if (!completer.isCompleted) {
              completer.complete(result.recognizedWords.toLowerCase().trim());
            }
          }
        },
        listenOptions: stt.SpeechListenOptions(
          listenMode: stt.ListenMode.confirmation,
          partialResults: true,
          onDevice: true,
        ),
      );
    } catch (e) {
      debugPrint('STT listen error: $e');
      timer.cancel();
      if (!completer.isCompleted) {
        completer.complete(null);
      }
    }

    return completer.future;
  }

  void stopListening() {
    if (_speech.isListening) {
      _speech.stop();
    }
  }

  void dispose() {
    _tts.stop();
    _speech.stop();
  }
}
