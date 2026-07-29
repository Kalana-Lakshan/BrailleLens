import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/main.dart';

void main() {
  testWidgets('BrailleLensApp smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const BrailleLensApp());
    expect(find.text('BrailleLens Offline'), findsOneWidget);
  });
}
