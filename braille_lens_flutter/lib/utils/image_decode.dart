import 'dart:typed_data';

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
