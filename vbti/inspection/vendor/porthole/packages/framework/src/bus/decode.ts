/**
 * msgpack → JS decoding, including the `$nd` and `$img` tagged maps.
 *
 * Kept separate from the client so it can be tested against fixtures without a
 * socket, and so a replay tool can reuse it later.
 */

import { decode as msgpackDecode } from '@msgpack/msgpack';

import type { AnyTypedArray, EncodedImage, BusEnvelope, NdArray, NdDtype } from './types.js';

type TypedArrayCtor = {
  new (buffer: ArrayBufferLike, byteOffset: number, length: number): AnyTypedArray;
  readonly BYTES_PER_ELEMENT: number;
};

const CTORS: Record<NdDtype, TypedArrayCtor> = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
};

interface TaggedNd {
  $nd: { dtype: NdDtype; shape: number[]; data: Uint8Array };
}

interface TaggedImg {
  $img: { mime: string; w: number; h: number; data: Uint8Array };
}

/**
 * Wrap raw bytes in a TypedArray without copying when possible.
 *
 * msgpack hands back `Uint8Array` views into one big receive buffer, and their
 * `byteOffset` is wherever the field happened to land. `new Float32Array(buf,
 * offset, n)` throws unless `offset` is a multiple of 4, so a misaligned view
 * must be copied. This is the only place the decoder ever copies array data,
 * and it usually does not have to.
 */
function toTypedArray(dtype: NdDtype, bytes: Uint8Array): AnyTypedArray {
  const Ctor = CTORS[dtype];
  if (!Ctor) throw new TypeError(`unknown dtype on the wire: ${dtype}`);

  const bpe = Ctor.BYTES_PER_ELEMENT;
  const source = bytes.byteOffset % bpe === 0 ? bytes : new Uint8Array(bytes);
  return new Ctor(source.buffer, source.byteOffset, source.byteLength / bpe);
}

function isTaggedNd(value: object): value is TaggedNd {
  return '$nd' in value;
}

function isTaggedImg(value: object): value is TaggedImg {
  return '$img' in value;
}

/**
 * Walk a decoded payload and replace tagged maps with usable values.
 *
 * Recursion is depth-first over maps and arrays. Binary blobs are leaves, so
 * the walk is cheap even when the payload carries megabytes: it visits the
 * structure, never the bytes.
 */
export function revivePayload(value: unknown): unknown {
  if (value === null || typeof value !== 'object') return value;

  if (Array.isArray(value)) {
    return value.map(revivePayload);
  }

  if (ArrayBuffer.isView(value)) return value;

  if (isTaggedNd(value)) {
    const { dtype, shape, data } = value.$nd;
    const nd: NdArray = { dtype, shape, data: toTypedArray(dtype, data) };
    return nd;
  }

  if (isTaggedImg(value)) {
    const { mime, w, h, data } = value.$img;
    const image: EncodedImage = { mime, w, h, bytes: data };
    return image;
  }

  const out: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    out[key] = revivePayload(item);
  }
  return out;
}

/** Decode one binary frame into an envelope with a revived payload. */
export function decodeEnvelope(buffer: ArrayBuffer | Uint8Array): BusEnvelope {
  const raw = msgpackDecode(buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer));

  if (typeof raw !== 'object' || raw === null || !('topic' in raw)) {
    throw new TypeError('frame is not a Porthole envelope');
  }

  const { topic, seq, ts, payload } = raw as BusEnvelope;
  return { topic, seq, ts, payload: revivePayload(payload) };
}
