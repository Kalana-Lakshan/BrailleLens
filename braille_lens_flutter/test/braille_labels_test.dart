import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:braille_lens_flutter/config/app_config.dart';

/// The CNN's class index IS the 6-dot cell code (see AppConfig.brailleCnnAsset's
/// docs) — braille_labels.json is the single source of truth for what each
/// code means, no more English-letter round-trip.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('braille_labels.json has 64 rows, one per code, sane structure', () async {
    final raw = await rootBundle.loadString(AppConfig.brailleLabelsAsset);
    final rows = jsonDecode(raw) as List;
    expect(rows.length, 64);

    final codes = rows.map((r) => (r as Map)['code'] as int).toSet();
    expect(codes, Set<int>.from(List.generate(64, (i) => i)));

    final row19 = rows.cast<Map>().firstWhere((r) => r['code'] == 19);
    expect(row19['si'], 'ක');
    expect(row19['dots'], 'dots 1, 2, 5');

    final space = rows.cast<Map>().firstWhere((r) => r['code'] == 0);
    expect(space['dots'], 'no dots');
  });
}
