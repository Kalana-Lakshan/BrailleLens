import 'package:flutter_test/flutter_test.dart';
import 'package:braille_lens_flutter/models/braille_cell.dart';
import 'package:braille_lens_flutter/services/tip_dwell_tracker.dart';

BrailleCell _cell(int id, double x) => BrailleCell(
      id: id,
      x0: x,
      y0: 0,
      x1: x + 40,
      y1: 60,
      char: 'ක',
      pattern: '13',
      code: 5,
      conf: 1,
      line: 0,
      col: id,
    );

void main() {
  // 40 px cells in a 1000 px wide scan.
  final map = CellMap(
    cells: [_cell(0, 0), _cell(1, 50), _cell(2, 100)],
    imageWidth: 1000,
    imageHeight: 1000,
  );

  test('a still or slowly drifting tip keeps its id', () {
    final t = TipDwellTracker();
    final id = t.track(const Offset(463, 1221), 1000, map);
    // Real consecutive samples of a held finger from the phone log.
    expect(t.track(const Offset(467, 1213), 1000, map), id);
    expect(t.track(const Offset(446, 1202), 1000, map), id);
    expect(t.track(const Offset(456, 1198), 1000, map), id);
  });

  test('a jump of more than a cell starts a new dwell', () {
    final t = TipDwellTracker();
    final id = t.track(const Offset(100, 100), 1000, map);
    expect(t.track(const Offset(200, 100), 1000, map), isNot(id));
  });

  test('losing the tip starts a new dwell on return', () {
    final t = TipDwellTracker();
    final id = t.track(const Offset(100, 100), 1000, map);
    t.lost();
    expect(t.track(const Offset(100, 100), 1000, map), isNot(id));
  });

  test('cell size scales with the finger frame width', () {
    final t = TipDwellTracker();
    // Finger frame half as wide as the scan → cells ~20 px there.
    final id = t.track(const Offset(100, 100), 500, map);
    expect(t.track(const Offset(125, 100), 500, map), isNot(id));
  });
}
