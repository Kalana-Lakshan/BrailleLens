import 'dart:async';

import 'package:flutter/foundation.dart';

import '../utils/answer_match.dart';
import 'audio_service.dart';

/// Listens for English "back" / "retry" / "help" while a Learning or Testing
/// screen is idle.
///
/// The recogniser shares the microphone and audio focus with everything
/// else the screen does, so it only listens while [canListen] says the
/// screen is waiting for the learner (page hunt, or between letters), never
/// while the app speaks, beeps a countdown, captures or records an answer.
/// Screens call [pause] the moment one of those starts; the loop resumes by
/// itself once [canListen] is true again.
class ScreenCommandListener {
  ScreenCommandListener({
    required this.audio,
    required this.canListen,
    required this.onCommand,
  });

  final AudioService audio;
  final bool Function() canListen;
  final Future<void> Function(ScreenCommand command) onCommand;

  bool _running = false;

  static const _listenWindow = Duration(seconds: 5);
  static const _idlePoll = Duration(milliseconds: 700);

  void start() {
    if (_running) return;
    _running = true;
    audio.yieldListeningToSpeech = true;
    unawaited(_loop());
  }

  /// Hands the microphone back now; listening resumes when [canListen].
  void pause() => audio.stopListening();

  void stop() {
    _running = false;
    audio.yieldListeningToSpeech = false;
    audio.stopListening();
  }

  Future<void> _loop() async {
    while (_running) {
      if (!canListen() || audio.isSpeaking) {
        await Future<void>.delayed(_idlePoll);
        continue;
      }
      final heard = await audio.listenForAnswer(timeout: _listenWindow);
      if (!_running) break;
      final command = parseScreenCommand(heard);
      // Re-check: the screen may have moved on while the words came in.
      if (command == null || !canListen()) continue;
      debugPrint('[Commands] "$heard" -> ${command.name}');
      await onCommand(command);
    }
  }
}
