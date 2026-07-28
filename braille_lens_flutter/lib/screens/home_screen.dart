import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'interaction_screen.dart';
import '../theme/app_theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Uint8List? _selectedImageBytes;
  String _selectedImageName = "sample_a.jpg";
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadSampleImage('sample_a.jpg');
  }

  Future<void> _loadSampleImage(String imageName) async {
    try {
      final ByteData data = await rootBundle.load('assets/samples/$imageName');
      if (!mounted) return;
      setState(() {
        _selectedImageBytes = data.buffer.asUint8List();
        _selectedImageName = imageName;
      });
    } catch (e) {
      debugPrint('Error loading sample image: $e');
    }
  }

  Future<void> _pickFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        final bytes = await image.readAsBytes();
        if (!mounted) return;
        setState(() {
          _selectedImageBytes = bytes;
          _selectedImageName = image.name;
        });
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Gallery pick error: $e')),
      );
    }
  }

  Future<void> _takePhoto() async {
    try {
      final XFile? photo = await _picker.pickImage(source: ImageSource.camera);
      if (photo != null) {
        final bytes = await photo.readAsBytes();
        if (!mounted) return;
        setState(() {
          _selectedImageBytes = bytes;
          _selectedImageName = photo.name;
        });
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Camera error: $e')),
      );
    }
  }

  void _navigateToMode(String mode) {
    if (_selectedImageBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select or capture an image first!')),
      );
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => InteractionScreen(
          imageBytes: _selectedImageBytes!,
          imageName: _selectedImageName,
          mode: mode,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('BrailleLens Offline'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => _showAboutDialog(context),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // App Banner Header
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: AppTheme.primaryColor.withValues(alpha: 0.15),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.fingerprint,
                        size: 48,
                        color: AppTheme.primaryLight,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'BrailleLens ML Engine',
                      style: Theme.of(context).textTheme.headlineMedium,
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 6),
                    Text(
                      '100% Offline On-Device Braille Classification',
                      style: Theme.of(context).textTheme.bodyMedium,
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),

            // Image Preview Container
            Text('Selected Braille Image', style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontSize: 18)),
            const SizedBox(height: 10),
            Container(
              height: 200,
              decoration: BoxDecoration(
                color: AppTheme.cardColor,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppTheme.primaryColor.withValues(alpha: 0.3), width: 2),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(14),
                child: _selectedImageBytes != null
                    ? Image.memory(
                        _selectedImageBytes!,
                        fit: BoxFit.contain,
                      )
                    : const Center(
                        child: Text('No Image Selected'),
                      ),
              ),
            ),
            const SizedBox(height: 16),

            // Image Source Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _takePhoto,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.cardColor,
                      foregroundColor: AppTheme.primaryLight,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickFromGallery,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.cardColor,
                      foregroundColor: AppTheme.primaryLight,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Sample Braille Picker
            Text('Or Select Bundled Sample:', style: Theme.of(context).textTheme.bodyMedium),
            const SizedBox(height: 8),
            SizedBox(
              height: 50,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: 26,
                itemBuilder: (context, index) {
                  final letter = String.fromCharCode(97 + index);
                  final imageName = 'sample_$letter.jpg';
                  final isSelected = _selectedImageName == imageName;
                  return Padding(
                    padding: const EdgeInsets.only(right: 8.0),
                    child: ChoiceChip(
                      label: Text(letter.toUpperCase()),
                      selected: isSelected,
                      selectedColor: AppTheme.primaryColor,
                      onSelected: (_) => _loadSampleImage(imageName),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 30),

            // Mode Selection Action Buttons
            Text('Choose Interaction Mode', style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontSize: 18)),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: () => _navigateToMode('learning'),
              icon: const Icon(Icons.school, size: 24),
              label: const Padding(
                padding: EdgeInsets.symmetric(vertical: 12.0),
                child: Text('LEARNING MODE\n(Listen to Character)', textAlign: TextAlign.center),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.primaryColor,
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: () => _navigateToMode('testing'),
              icon: const Icon(Icons.quiz, size: 24),
              label: const Padding(
                padding: EdgeInsets.symmetric(vertical: 12.0),
                child: Text('TESTING MODE\n(Voice Guess & Feedback)', textAlign: TextAlign.center),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.secondaryColor,
                foregroundColor: Colors.black,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showAboutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About BrailleLens'),
        content: const Text(
          'BrailleLens runs an edge-optimized PyTorch/ONNX CNN directly inside your phone.\n\n'
          '• 100% Offline execution\n'
          '• On-device TTS & Voice recognition\n'
          '• Zero external server requests',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}
