import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import '../services/classifier_service.dart';
import '../services/fingertip_onnx_service.dart';
import '../theme/app_theme.dart';
import '../widgets/cell_overlay_painter.dart';

/// On-device check for the two bundled ONNX models:
/// CNN (`braille_model.onnx`) and fingertip YOLO (`*_mobile.onnx`).
class ModelCheckScreen extends StatefulWidget {
  const ModelCheckScreen({super.key});

  @override
  State<ModelCheckScreen> createState() => _ModelCheckScreenState();
}

class _ModelCheckScreenState extends State<ModelCheckScreen> {
  final ClassifierService _cnn = ClassifierService();
  final FingertipOnnxService _tip = FingertipOnnxService();
  final ImagePicker _picker = ImagePicker();

  bool _loading = true;
  bool _busy = false;
  String _log = 'Loading models…';

  Uint8List? _photo;
  FingertipDetection? _detection;
  PredictionResult? _cnnResult;
  String? _cnnExpected;

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    final cnnOk = await _cnn.initialize();
    final tipOk = await _tip.initialize();
    if (!mounted) return;
    setState(() {
      _loading = false;
      _log = [
        cnnOk
            ? 'CNN ready · ${_cnn.loadedAsset?.split('/').last}'
            : 'CNN FAILED · ${_cnn.lastError}',
        tipOk
            ? 'YOLO ready · ${_tip.loadedAsset?.split('/').last}'
            : 'YOLO FAILED · ${_tip.lastError}',
      ].join('\n');
    });
  }

  Future<void> _classifySamples() async {
    if (!_cnn.isInitialized || _busy) return;
    setState(() {
      _busy = true;
      _log = 'Classifying bundled samples…';
      _detection = null;
    });

    var ok = 0;
    const letters = 'abcdefghijklmnopqrstuvwxyz';
    final lines = <String>[];
    for (var i = 0; i < letters.length; i++) {
      final ch = letters[i];
      final asset = 'assets/samples/sample_$ch.jpg';
      try {
        final data = await rootBundle.load(asset);
        final pred = await _cnn.predict(data.buffer.asUint8List());
        final hit = pred.character.toLowerCase() == ch;
        if (hit) ok++;
        lines.add(
          '${hit ? "✓" : "✗"} $ch → ${pred.character} '
          '${(pred.confidence * 100).toStringAsFixed(0)}%',
        );
        if (i == 0) {
          _cnnResult = pred;
          _cnnExpected = ch;
          _photo = data.buffer.asUint8List();
        }
      } catch (e) {
        lines.add('✗ $ch · $e');
      }
    }

    if (!mounted) return;
    setState(() {
      _busy = false;
      _log = 'CNN samples $ok/${letters.length}\n${lines.join('\n')}';
    });
  }

  Future<void> _pickAndClassify(ImageSource source) async {
    if (!_cnn.isInitialized || _busy) return;
    final file = await _picker.pickImage(source: source, imageQuality: 85);
    if (file == null) return;
    setState(() => _busy = true);
    try {
      final bytes = await file.readAsBytes();
      final pred = await _cnn.predict(bytes);
      if (!mounted) return;
      setState(() {
        _photo = bytes;
        _cnnResult = pred;
        _cnnExpected = null;
        _detection = null;
        _log =
            'CNN: ${pred.character.toUpperCase()}  '
            '${(pred.confidence * 100).toStringAsFixed(1)}%';
        _busy = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _log = 'CNN error: $e';
      });
    }
  }

  Future<void> _pickAndDetect(ImageSource source) async {
    if (!_tip.isReady || _busy) return;
    final file = await _picker.pickImage(source: source, imageQuality: 90);
    if (file == null) return;
    setState(() => _busy = true);
    try {
      final bytes = await file.readAsBytes();
      final det = await _tip.detect(bytes);
      if (!mounted) return;
      setState(() {
        _photo = bytes;
        _detection = det;
        _cnnResult = null;
        _busy = false;
        _log = det == null
            ? 'YOLO: no fingertip (conf < 0.25)'
            : 'YOLO: fingertip  ${(det.confidence * 100).toStringAsFixed(0)}%  '
                'at (${det.contactPoint.dx.round()}, ${det.contactPoint.dy.round()})';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _log = 'YOLO error: $e';
      });
    }
  }

  @override
  void dispose() {
    _cnn.dispose();
    _tip.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundBlack,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text('ONNX check'),
        foregroundColor: AppTheme.primaryYellow,
      ),
      body: _loading
          ? const Center(
              child: CircularProgressIndicator(color: AppTheme.primaryYellow),
            )
          : Column(
              children: [
                _statusRow('CNN  braille_model.onnx', _cnn.isInitialized),
                _statusRow(
                  'YOLO  ${_tip.loadedAsset?.split('/').last ?? "not loaded"}',
                  _tip.isReady,
                ),
                Expanded(child: _preview()),
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
                  child: Text(
                    _log,
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                      fontFamily: 'monospace',
                      height: 1.35,
                    ),
                  ),
                ),
                _buttons(),
              ],
            ),
    );
  }

  Widget _statusRow(String label, bool ok) {
    return ListTile(
      dense: true,
      leading: Icon(
        ok ? Icons.check_circle : Icons.error,
        color: ok ? AppTheme.successCyan : AppTheme.errorCoral,
      ),
      title: Text(label, style: const TextStyle(color: Colors.white, fontSize: 14)),
    );
  }

  Widget _preview() {
    if (_photo == null) {
      return const Center(
        child: Text(
          'Load models, then classify samples\nor pick a photo.',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.white38),
        ),
      );
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        Image.memory(_photo!, fit: BoxFit.contain),
        if (_detection != null)
          Positioned.fill(
            child: CustomPaint(
              painter: FingertipOverlayPainter(
                tipBox: _detection!.box,
                contactPoint: _detection!.contactPoint,
                imageWidth: _detection!.imageWidth,
                imageHeight: _detection!.imageHeight,
              ),
            ),
          ),
        if (_cnnResult != null)
          Align(
            alignment: Alignment.topCenter,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Text(
                _cnnExpected == null
                    ? _cnnResult!.character.toUpperCase()
                    : '${_cnnExpected!.toUpperCase()} → ${_cnnResult!.character.toUpperCase()}',
                style: const TextStyle(
                  color: AppTheme.primaryYellow,
                  fontSize: 40,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buttons() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 20),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: _busy ? null : _classifySamples,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primaryYellow,
                    foregroundColor: Colors.black,
                  ),
                  child: const Text('CNN · 26 samples'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _pickAndClassify(ImageSource.gallery),
                  child: const Text('CNN · gallery'),
                ),
              ),
            ],
          ),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _pickAndDetect(ImageSource.camera),
                  child: const Text('YOLO · camera'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _pickAndDetect(ImageSource.gallery),
                  child: const Text('YOLO · gallery'),
                ),
              ),
            ],
          ),
          if (_busy) const LinearProgressIndicator(color: AppTheme.primaryYellow),
        ],
      ),
    );
  }
}
