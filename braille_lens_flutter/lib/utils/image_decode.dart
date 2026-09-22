import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:image/image.dart' as img;

/// Decodes [bytes] and corrects for EXIF orientation so the returned
/// image's pixel data is always upright.
///
/// `package:image`'s `decodeImage()`/`decodeJpg()` do **not** do this
/// automatically — they decode the JPEG's raw sensor-order raster as-is and
/// expose the EXIF orientation tag as separate metadata (`image.exif`);
/// baking it into the pixels is a distinct, opt-in step (`bakeOrientation`).
/// Phone cameras routinely save a portrait photo as a landscape-native
/// raster plus an EXIF "rotate 90°" tag rather than physically rotating
/// pixels before encoding — every model in this app (cell detector,
/// classifier, fingertip detector) assumes an upright capture, so skipping
/// this is the classic reason a pipeline looks fine on a laptop (where an
/// image viewer or PIL's `exif_transpose` auto-corrects it before you look
/// at or process it) but performs badly live on a phone (where the raw,
/// still-rotated sensor pixels get fed to the model unmodified — a 90°
/// rotation scrambles which pixels are even in the same Braille cell row).
img.Image? decodeUpright(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) return null;
  return img.bakeOrientation(decoded);
}

/// Same upright guarantee as [decodeUpright], as a `dart:ui` image for
/// painting.
///
/// `ui.instantiateImageCodec` does **not** apply the EXIF orientation tag, so
/// a widget that decodes with it paints the raw sensor raster while every
/// model coordinate in this app is in `decodeUpright` space. On a phone photo
/// tagged "rotate 90°" that swaps the axes, and overlays (cell boxes, the
/// fingertip marker) and taps land nowhere near the pixels they name. Baking
/// the orientation here keeps painting and geometry in one space.
Future<ui.Image> decodeUprightUi(Uint8List bytes) async {
  final baked = decodeUpright(bytes);
  if (baked == null) {
    final codec = await ui.instantiateImageCodec(bytes);
    return (await codec.getNextFrame()).image;
  }
  final rgba = baked.getBytes(order: img.ChannelOrder.rgba);
  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(
    rgba,
    baked.width,
    baked.height,
    ui.PixelFormat.rgba8888,
    completer.complete,
  );
  return completer.future;
}
