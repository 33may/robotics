/**
 * Wire types for the Porthole bus. Mirrors `docs/bus-protocol.md`.
 *
 * These types are the frontend half of a contract whose other half is Python.
 * Changing anything here without changing the protocol document and the Python
 * codec is how the two sides silently drift apart.
 */

export const PROTOCOL_VERSION = 1;

/** Numeric dtypes permitted on the wire (protocol §3.1). */
export type NdDtype =
  | 'float32'
  | 'float64'
  | 'int8'
  | 'uint8'
  | 'int16'
  | 'uint16'
  | 'int32'
  | 'uint32'
  | 'int64'
  | 'uint64';

export type AnyTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array;

/**
 * A decoded numeric array. `shape` is carried alongside a flat buffer — the
 * decoder never builds nested JS arrays, because doing so for a 12k-point
 * cloud would allocate 12k objects per frame.
 */
export interface NdArray<T extends AnyTypedArray = AnyTypedArray> {
  readonly dtype: NdDtype;
  readonly shape: readonly number[];
  readonly data: T;
}

/** An encoded image, left encoded. Panels decide how to decode it. */
export interface EncodedImage {
  readonly mime: string;
  readonly w: number;
  readonly h: number;
  readonly bytes: Uint8Array;
}

/** Quality of service, declared by the publisher (protocol §6). */
export type TopicQoS = 'stream' | 'event';

export interface TopicDeclaration {
  readonly topic: string;
  readonly qos: TopicQoS;
  readonly kind: string;
}

export interface BusHello {
  readonly protocol: number;
  readonly app: string;
  readonly server: string;
  readonly topics: readonly TopicDeclaration[];
}

/** One message as it arrives (protocol §2). */
export interface BusEnvelope<P = unknown> {
  readonly topic: string;
  readonly seq: number;
  readonly ts: number;
  readonly payload: P;
}

/** A command sent UI → Python (protocol §7). Inert until the backend acts. */
export interface BusCommand {
  readonly cmd: string;
  readonly id?: string;
  readonly ts?: number;
  readonly [key: string]: unknown;
}

export type ConnectionStatus = 'connecting' | 'open' | 'closed';

/** Type guards, for panels that must branch on payload shape. */
export function isNdArray(value: unknown): value is NdArray {
  return (
    typeof value === 'object' &&
    value !== null &&
    'dtype' in value &&
    'shape' in value &&
    'data' in value &&
    ArrayBuffer.isView((value as NdArray).data)
  );
}

export function isEncodedImage(value: unknown): value is EncodedImage {
  return (
    typeof value === 'object' &&
    value !== null &&
    'mime' in value &&
    'bytes' in value &&
    (value as EncodedImage).bytes instanceof Uint8Array
  );
}
