/// Compact LabelMe-style raised-dot sequence (digits 1-6 only), e.g. 124, 34.
String compactDotSequence(String? pattern) {
  if (pattern == null) return '—';
  final raw = pattern.trim().toLowerCase();
  if (raw.isEmpty || raw == '0' || raw == 'no dots' || raw == 'none') {
    return '—';
  }

  final digits = <int>[];
  for (final match in RegExp(r'[1-6]').allMatches(raw)) {
    final d = int.parse(match.group(0)!);
    if (!digits.contains(d)) digits.add(d);
  }
  if (digits.isEmpty) return '—';
  digits.sort();
  return digits.join();
}
