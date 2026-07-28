import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../services/classifier_service.dart';
import '../services/audio_service.dart';
import '../theme/app_theme.dart';

class InteractionScreen extends StatefulWidget {
  final Uint8List imageBytes;
  final String imageName;
  final String mode; // 'learning' or 'testing'

  const InteractionScreen({
    super.key,
    required this.imageBytes,
    required this.imageName,
    required this.mode,
  });

  @override
  State<InteractionScreen> createState() => _InteractionScreenState();
}

class _InteractionScreenState extends State<InteractionScreen> {
  final ClassifierService _classifierService = ClassifierService();
  final AudioService _audioService = AudioService();
  final TextEditingController _textController = TextEditingController();

  bool _isLoading = true;
  PredictionResult? _prediction;
  String? _spokenAnswer;
  bool _isListening = false;
  String? _feedbackText;
  bool? _isCorrect;
  bool _showKeyboardFallback = false;

  @override
  void initState() {
    super.initState();
    _runPipeline();
  }

  Future<void> _runPipeline() async {
    try {
      final prediction = await _classifierService.predict(widget.imageBytes);
      if (!mounted) return;
      setState(() {
        _prediction = prediction;
        _isLoading = false;
      });

      if (widget.mode == 'learning') {
        _executeLearningMode();
      } else {
        _executeTestingMode();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Inference error: $e')),
      );
    }
  }

  void _executeLearningMode() {
    if (_prediction == null) return;
    final text = "This is character ${_prediction!.character.toUpperCase()}";
    if (mounted) {
      setState(() {
        _feedbackText = text;
      });
    }
    _audioService.speak(text);
  }

  Future<void> _executeTestingMode() async {
    if (_prediction == null) return;

    // 1. Initial Prompt
    await _audioService.speak("Please state the character under your finger.");
    await Future.delayed(const Duration(milliseconds: 2500));

    // 2. Start STT Listening
    if (!mounted) return;
    setState(() {
      _isListening = true;
      _feedbackText = "Listening for your voice... (5s)";
    });

    final spoken = await _audioService.listenForAnswer(timeout: const Duration(seconds: 5));

    if (!mounted) return;
    setState(() {
      _isListening = false;
      _spokenAnswer = spoken;
    });

    if (spoken != null && spoken.isNotEmpty) {
      _evaluateAnswer(spoken);
    } else {
      // Trigger Fallback to keyboard input
      setState(() {
        _showKeyboardFallback = true;
        _feedbackText = "Could not detect voice. Please type your answer below:";
      });
      _audioService.speak("Microphone timeout. Please type your character.");
    }
  }

  void _evaluateAnswer(String userAnswer) {
    if (_prediction == null) return;

    final targetChar = _prediction!.character.toLowerCase();
    final cleanAnswer = userAnswer.trim().toLowerCase();

    final bool correct = cleanAnswer.contains(targetChar);

    if (mounted) {
      setState(() {
        _isCorrect = correct;
        _showKeyboardFallback = false;
      });
    }

    if (correct) {
      final text = "Correct! You identified character ${targetChar.toUpperCase()} properly.";
      if (mounted) {
        setState(() {
          _feedbackText = text;
        });
      }
      _audioService.speak(text);
    } else {
      final text = "Incorrect. You said $userAnswer, but the correct character is ${targetChar.toUpperCase()}.";
      if (mounted) {
        setState(() {
          _feedbackText = text;
        });
      }
      _audioService.speak(text);
    }
  }

  @override
  void dispose() {
    _audioService.dispose();
    _classifierService.dispose();
    _textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isLearning = widget.mode == 'learning';

    return Scaffold(
      appBar: AppBar(
        title: Text(isLearning ? 'Learning Mode' : 'Testing Mode'),
      ),
      body: _isLoading
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Processing image on-device...'),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Image & Prediction Header Card
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Row(
                        children: [
                          Container(
                            width: 90,
                            height: 90,
                            decoration: BoxDecoration(
                              color: Colors.black26,
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: AppTheme.primaryColor.withValues(alpha: 0.4)),
                            ),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(10),
                              child: Image.memory(widget.imageBytes, fit: BoxFit.contain),
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  isLearning ? 'PREDICTED CHARACTER' : 'GROUND TRUTH (TARGET)',
                                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                      ),
                                ),
                                const SizedBox(height: 4),
                                Row(
                                  children: [
                                    Text(
                                      _prediction?.character.toUpperCase() ?? '?',
                                      style: Theme.of(context).textTheme.displayLarge?.copyWith(
                                            color: AppTheme.primaryLight,
                                          ),
                                    ),
                                    const SizedBox(width: 12),
                                    Chip(
                                      label: Text(
                                        'Conf: ${((_prediction?.confidence ?? 0) * 100).toStringAsFixed(1)}%',
                                        style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
                                      ),
                                      backgroundColor: AppTheme.primaryColor.withValues(alpha: 0.2),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Mode Workflow Card
                  if (isLearning) ...[
                    Card(
                      color: AppTheme.surfaceColor,
                      child: Padding(
                        padding: const EdgeInsets.all(24.0),
                        child: Column(
                          children: [
                            const Icon(Icons.volume_up, size: 64, color: AppTheme.primaryLight),
                            const SizedBox(height: 16),
                            Text(
                              _feedbackText ?? '',
                              style: Theme.of(context).textTheme.headlineMedium,
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 24),
                            ElevatedButton.icon(
                              onPressed: _executeLearningMode,
                              icon: const Icon(Icons.replay),
                              label: const Text('Repeat Voice Announcement'),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ] else ...[
                    // Testing Mode Content
                    Card(
                      color: _isCorrect == null
                          ? AppTheme.surfaceColor
                          : (_isCorrect! ? AppTheme.secondaryColor.withValues(alpha: 0.15) : AppTheme.errorColor.withValues(alpha: 0.15)),
                      child: Padding(
                        padding: const EdgeInsets.all(24.0),
                        child: Column(
                          children: [
                            if (_isListening) ...[
                              const CircularProgressIndicator(color: AppTheme.secondaryColor),
                              const SizedBox(height: 16),
                              const Icon(Icons.mic, size: 48, color: AppTheme.secondaryColor),
                            ] else if (_isCorrect != null) ...[
                              Icon(
                                _isCorrect! ? Icons.check_circle : Icons.cancel,
                                size: 64,
                                color: _isCorrect! ? AppTheme.secondaryColor : AppTheme.errorColor,
                              ),
                            ] else ...[
                              const Icon(Icons.graphic_eq, size: 48, color: AppTheme.primaryLight),
                            ],
                            const SizedBox(height: 16),
                            Text(
                              _feedbackText ?? '',
                              style: Theme.of(context).textTheme.headlineMedium,
                              textAlign: TextAlign.center,
                            ),
                            if (_spokenAnswer != null) ...[
                              const SizedBox(height: 12),
                              Text(
                                'You said: "${_spokenAnswer!}"',
                                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                                      fontStyle: FontStyle.italic,
                                    ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Keyboard Fallback Input (if mic fails or user prefers typing)
                    if (_showKeyboardFallback || _isCorrect == null) ...[
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Keyboard Fallback Input',
                                style: Theme.of(context).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.bold),
                              ),
                              const SizedBox(height: 8),
                              Row(
                                children: [
                                  Expanded(
                                    child: TextField(
                                      controller: _textController,
                                      decoration: const InputDecoration(
                                        hintText: 'Type character (e.g. A)',
                                        border: OutlineInputBorder(),
                                        contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                      ),
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  ElevatedButton(
                                    onPressed: () {
                                      if (_textController.text.isNotEmpty) {
                                        _evaluateAnswer(_textController.text);
                                      }
                                    },
                                    child: const Text('Submit'),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],

                    const SizedBox(height: 16),
                    ElevatedButton.icon(
                      onPressed: _executeTestingMode,
                      icon: const Icon(Icons.refresh),
                      label: const Text('Retry Voice Test'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.cardColor,
                      ),
                    ),
                  ],
                ],
              ),
            ),
    );
  }
}
