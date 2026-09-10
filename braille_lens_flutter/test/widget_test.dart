import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/main.dart';

void main() {
  testWidgets('UI-FL-02 home shows Learning and Testing zones', (tester) async {
    await tester.pumpWidget(const BrailleLensApp());
    await tester.pump();
    expect(find.text('BrailleLens'), findsOneWidget);
    expect(find.textContaining('LEARNING'), findsOneWidget);
    expect(find.textContaining('TESTING'), findsOneWidget);
  });
}
