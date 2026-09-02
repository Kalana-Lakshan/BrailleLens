import 'dart:ui';

/// One Braille cell from the stage-1 prescan (reference frame).
class BrailleCell {
  final int id;
  final double x0;
  final double y0;
  final double x1;
  final double y1;
  final String char;
  final String pattern;
  final int code;
  final double conf;
  final int line;
  final int col;

  const BrailleCell({
    required this.id,
    required this.x0,
    required this.y0,
    required this.x1,
    required this.y1,
    this.char = '',
    this.pattern = '',
    this.code = 0,
    this.conf = 1.0,
    this.line = 0,
    this.col = 0,
  });

  Offset get center => Offset((x0 + x1) / 2, (y0 + y1) / 2);

  Rect get rect => Rect.fromLTRB(x0, y0, x1, y1);

  factory BrailleCell.fromJson(Map<String, dynamic> json) {
    final xyxy = json['xyxy'];
    if (xyxy is List && xyxy.length == 4) {
      return BrailleCell(
        id: json['id'] as int? ?? 0,
        x0: (xyxy[0] as num).toDouble(),
        y0: (xyxy[1] as num).toDouble(),
        x1: (xyxy[2] as num).toDouble(),
        y1: (xyxy[3] as num).toDouble(),
        char: json['char'] as String? ?? '',
        pattern: json['pattern'] as String? ?? '',
        code: json['code'] as int? ?? 0,
        conf: (json['conf'] as num?)?.toDouble() ?? 1.0,
        line: json['line'] as int? ?? 0,
        col: json['col'] as int? ?? 0,
      );
    }
    return BrailleCell(
      id: json['id'] as int? ?? 0,
      x0: (json['x0'] as num).toDouble(),
      y0: (json['y0'] as num).toDouble(),
      x1: (json['x1'] as num).toDouble(),
      y1: (json['y1'] as num).toDouble(),
      char: json['char'] as String? ?? '',
      pattern: json['pattern'] as String? ?? '',
      code: json['code'] as int? ?? 0,
      conf: (json['conf'] as num?)?.toDouble() ?? 1.0,
      line: json['line'] as int? ?? 0,
      col: json['col'] as int? ?? 0,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'x0': x0,
        'y0': y0,
        'x1': x1,
        'y1': y1,
        'char': char,
        'pattern': pattern,
        'code': code,
        'conf': conf,
        'line': line,
        'col': col,
      };

  /// Human-readable label for the bottom panel.
  String get displayLabel {
    if (char.trim().isNotEmpty && char != ' ') return char;
    if (code == 0) return 'space';
    return '#$code';
  }

  String get patternLabel {
    if (pattern == '0' || pattern.isEmpty) return 'no dots';
    return pattern;
  }
}

/// All cells detected during stage-1 prescan.
class CellMap {
  final List<BrailleCell> cells;
  final int imageWidth;
  final int imageHeight;

  const CellMap({
    required this.cells,
    required this.imageWidth,
    required this.imageHeight,
  });

  int get length => cells.length;

  BrailleCell? byId(int id) {
    for (final c in cells) {
      if (c.id == id) return c;
    }
    return null;
  }

  factory CellMap.fromJson(Map<String, dynamic> json) {
    final raw = json['cells'] as List<dynamic>? ?? [];
    return CellMap(
      cells: raw
          .map((e) => BrailleCell.fromJson(Map<String, dynamic>.from(e as Map)))
          .toList(),
      imageWidth: json['width'] as int? ?? json['imageWidth'] as int? ?? 0,
      imageHeight: json['height'] as int? ?? json['imageHeight'] as int? ?? 0,
    );
  }

  Map<String, dynamic> toJson() => {
        'width': imageWidth,
        'height': imageHeight,
        'cells': cells.map((c) => c.toJson()).toList(),
      };
}
