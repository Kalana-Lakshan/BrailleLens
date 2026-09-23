import 'dart:ui';

import 'package:flutter/foundation.dart';

import '../models/braille_cell.dart';
import '../utils/dot_sequence.dart';
import '../utils/homography.dart';
import 'cell_constellation_aligner.dart';
import 'cell_hit_test.dart';
import 'coordinate_mapper.dart';
import 'frame_registration.dart';

/// Result of covered-character lookup (geometry only — no CNN on finger photo).
class CoveredCellResult {
  final BrailleCell? cell;
  final Offset tipInPrescan;
  final Offset tipInFingerImage;
  final Rect? fingertipBox;
  final String alignMode;

  const CoveredCellResult({
    required this.cell,
    required this.tipInPrescan,
    required this.tipInFingerImage,
    this.fingertipBox,
    this.alignMode = 'scale',
  });

  bool get hasHit => cell != null;

  String get headline {
    final c = cell;
    if (c == null) return '—';
    if (c.char.trim().isNotEmpty && c.char != ' ') return c.char;
    return c.displayLabel;
  }

  /// Raised-dot sequence for the bottom terminal (e.g. 124, 34).
  String get compactDots => compactDotSequence(cell?.pattern);

  String get subtitle {
    final c = cell;
    if (c == null) {
      // 'scale' means alignment fell through to raw image ratios, which is
      // the usual reason a covered cell reads as a miss — say so instead of
      // leaving the learner to guess at their finger placement.
      return alignMode.startsWith('scale')
          ? 'no cell under your finger · page not aligned — frame it as you scanned it'
          : 'no cell under your finger · align: $alignMode';
    }
    return '$headline · align: $alignMode';
  }
}

/// Greedy live-centre → prescan-cell matches after a constellation transform.
///
/// Each unmasked live box centre is mapped into the prescan; the nearest
/// unused reference centre within [maxDistCells] cell widths claims it.
class CellIdentityMatch {
  /// live box index → matched [BrailleCell] from the prescan map.
  final Map<int, BrailleCell> byLiveIndex;

  const CellIdentityMatch(this.byLiveIndex);

  BrailleCell? forLiveIndex(int i) => byLiveIndex[i];

  /// Build matches. Live centres under [excludeLive] (expanded fingertip) are
  /// skipped so the hand does not steal a neighbour's identity, except indices
  /// listed in [forceLiveIndices] (the tip's own box).
  static CellIdentityMatch match({
    required Homography h,
    required List<Rect> liveBoxes,
    required CellMap cellMap,
    Rect? excludeLive,
    Set<int> forceLiveIndices = const {},
    double maxDistCells = 0.6,
  }) {
    final cellW = CellHitTest.medianCellWidth(cellMap);
    if (cellW <= 0 || cellMap.cells.isEmpty) {
      return const CellIdentityMatch({});
    }
    final maxDist = cellW * maxDistCells;
    final exclude = CellConstellationAligner.expandedExclude(excludeLive);
    final usedRef = <int>{};
    final out = <int, BrailleCell>{};

    // Sort by how close the mapped centre is to some ref centre so confident
    // pairs claim first.
    final candidates = <({int live, int ref, double d})>[];
    for (var i = 0; i < liveBoxes.length; i++) {
      final c = liveBoxes[i].center;
      if (exclude != null &&
          exclude.contains(c) &&
          !forceLiveIndices.contains(i)) {
        continue;
      }
      final mapped = h.transform(c);
      for (var j = 0; j < cellMap.cells.length; j++) {
        final d = (mapped - cellMap.cells[j].center).distance;
        if (d <= maxDist) {
          candidates.add((live: i, ref: j, d: d));
        }
      }
    }
    candidates.sort((a, b) => a.d.compareTo(b.d));
    final usedLive = <int>{};
    for (final c in candidates) {
      if (usedLive.contains(c.live) || usedRef.contains(c.ref)) continue;
      usedLive.add(c.live);
      usedRef.add(c.ref);
      out[c.live] = cellMap.cells[c.ref];
    }
    return CellIdentityMatch(out);
  }
}

/// Identifies which prescan cell is covered by the fingertip.
class CoveredCellService {
  /// Full stage-2 lookup: work out how the finger frame sits on the prescan,
  /// then read the label off the prescan map.
  ///
  /// Alignment order (structure first):
  /// 1. `cells` / `cells+id` / `cells+box` — constellation on cell centres,
  ///    with fingertip centres masked and optional affine refine. Tip-in-box
  ///    then prefers identity match over warping the tip alone.
  /// 2. `homography` — pixel feature registration only when constellation
  ///    cannot run (too few live cells or align failed).
  /// 3. `scale` — plain width/height ratios.
  Future<CoveredCellResult> resolveAligned({
    required Uint8List fingerJpeg,
    required Offset tipInFingerImage,
    required CellMap cellMap,
    required int fingerImageWidth,
    required int fingerImageHeight,
    required List<Rect> fingerFrameCells,
    Rect? fingertipBox,
    Uint8List? prescanJpeg,
  }) async {
    Homography? h;
    var mode = 'scale';

    if (fingerFrameCells.length >= CellConstellationAligner.minInliers) {
      final aligned = await CellConstellationAligner.align(
        liveCenters: fingerFrameCells.map((b) => b.center).toList(),
        referenceCenters: cellMap.cells.map((c) => c.center).toList(),
        cellWidth: CellHitTest.medianCellWidth(cellMap),
        excludeLive: fingertipBox,
      );
      if (aligned != null) {
        h = aligned.homography;
        mode = 'cells';
        debugPrint('[CoveredCell] cell alignment on ${aligned.inliers} cells');
      }
    }

    // Texture ORB only when constellation is unavailable or failed.
    if (h == null && prescanJpeg != null) {
      final reg = await FrameRegistration.estimate(
        referenceJpeg: prescanJpeg,
        liveJpeg: fingerJpeg,
        liveMask: fingertipBox,
      );
      h = reg.homography;
      if (h != null) mode = 'homography';
    }

    final touched = cellUnderTipInFingerFrame(
      tipInFingerImage: tipInFingerImage,
      fingerFrameCells: fingerFrameCells,
      fingerImageWidth: fingerImageWidth,
      fingerImageHeight: fingerImageHeight,
    );

    // Structure identity: tip is in live box i → use matched prescan cell.
    if (h != null && mode == 'cells' && touched != null) {
      final liveIndex = _liveBoxIndex(touched.center, fingerFrameCells);
      if (liveIndex != null) {
        final ids = CellIdentityMatch.match(
          h: h,
          liveBoxes: fingerFrameCells,
          cellMap: cellMap,
          excludeLive: fingertipBox,
          // Always match the tip's box even if its centre sits under the hand.
          forceLiveIndices: {liveIndex},
        );
        final matched = ids.forLiveIndex(liveIndex);
        if (matched != null) {
          debugPrint('[CoveredCell] cells+id live=$liveIndex → '
              'prescan id=${matched.id} ${matched.char}');
          return CoveredCellResult(
            cell: matched,
            tipInPrescan: h.transform(tipInFingerImage),
            tipInFingerImage: tipInFingerImage,
            fingertipBox: fingertipBox,
            alignMode: 'cells+id',
          );
        }
      }
    }

    if (touched != null) {
      mode = '$mode+box';
      debugPrint('[CoveredCell] tip is in finger-frame box '
          '${touched.rect} of ${fingerFrameCells.length}');
    }

    return resolve(
      tipInFingerImage: tipInFingerImage,
      cellMap: cellMap,
      fingerImageWidth: fingerImageWidth,
      fingerImageHeight: fingerImageHeight,
      fingertipBox: fingertipBox,
      homography: h,
      alignMode: mode,
      // Map the touched box's centre rather than the tip: a box centre has a
      // matching box centre to land on in the prescan, so the lookup becomes
      // nearest-cell-to-a-cell instead of point-in-a-box.
      probeInFingerImage: touched?.center,
      matchNearestCentre: touched != null,
    );
  }

  int? _liveBoxIndex(Offset center, List<Rect> boxes) {
    for (var i = 0; i < boxes.length; i++) {
      if ((boxes[i].center - center).distanceSquared < 1.0) return i;
    }
    // Fallback: nearest box centre.
    var bestI = -1;
    var bestD = double.infinity;
    for (var i = 0; i < boxes.length; i++) {
      final d = (boxes[i].center - center).distanceSquared;
      if (d < bestD) {
        bestD = d;
        bestI = i;
      }
    }
    return bestI >= 0 ? bestI : null;
  }

  /// Which of the finger frame's own detected boxes the tip is touching.
  /// Pure geometry within that one frame — no mapping involved.
  BrailleCell? cellUnderTipInFingerFrame({
    required Offset tipInFingerImage,
    required List<Rect> fingerFrameCells,
    required int fingerImageWidth,
    required int fingerImageHeight,
  }) {
    if (fingerFrameCells.isEmpty) return null;
    final frameMap = CellMap(
      cells: [
        for (var i = 0; i < fingerFrameCells.length; i++)
          BrailleCell(
            id: i,
            x0: fingerFrameCells[i].left,
            y0: fingerFrameCells[i].top,
            x1: fingerFrameCells[i].right,
            y1: fingerFrameCells[i].bottom,
          ),
      ],
      imageWidth: fingerImageWidth,
      imageHeight: fingerImageHeight,
    );
    // These boxes carry no labels, so code 0 means "unclassified" here rather
    // than "space" — skipping them would reject every one of them.
    return CellHitTest.hitTest(tipInFingerImage, frameMap, skipEmpty: false);
  }

  /// [tipInFingerImage] — contact point in stage-2 JPEG pixel coordinates.
  /// [cellMap] — stage-1 prescan with labels already assigned.
  /// [homography] — optional live→prescan 3x3; otherwise independent X/Y scale.
  /// [probeInFingerImage] — point to map instead of the tip, when something
  /// more mappable is known (see [resolveAligned]). The tip is still what
  /// gets reported.
  /// [matchNearestCentre] — the probe is a cell centre, so match it against
  /// prescan cell centres rather than testing box containment.
  CoveredCellResult resolve({
    required Offset tipInFingerImage,
    required CellMap cellMap,
    required int fingerImageWidth,
    required int fingerImageHeight,
    Rect? fingertipBox,
    Homography? homography,
    String alignMode = 'scale',
    double marginFrac = 0.12,
    /// Absolute slack for a live capture, where the mapped point carries the
    /// error of the whole alignment, not just the finger position.
    double marginPx = 15.0,
    double nearestWithinCells = 1.0,
    Offset? probeInFingerImage,
    bool matchNearestCentre = false,
  }) {
    final probe = probeInFingerImage ?? tipInFingerImage;
    final tipInPrescan = homography != null
        ? homography.transform(probe)
        : CoordinateMapper.mapFingerTipToPrescan(
            tipInFingerImage: probe,
            prescanWidth: cellMap.imageWidth,
            prescanHeight: cellMap.imageHeight,
            fingerImageWidth: fingerImageWidth,
            fingerImageHeight: fingerImageHeight,
          );

    var hit = matchNearestCentre ? CellHitTest.nearestCell(tipInPrescan, cellMap) : null;
    // Centre matching comes up empty where the prescan dropped a cell as
    // background, so the box test still gets its say.
    hit ??= CellHitTest.hitTest(
      tipInPrescan,
      cellMap,
      marginFrac: marginFrac,
      marginPx: marginPx,
      skipEmpty: true,
      nearestWithinCells: nearestWithinCells,
    );

    return CoveredCellResult(
      cell: hit,
      tipInPrescan: tipInPrescan,
      tipInFingerImage: tipInFingerImage,
      fingertipBox: fingertipBox,
      alignMode: alignMode,
    );
  }
}
