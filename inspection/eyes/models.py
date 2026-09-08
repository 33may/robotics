#!/usr/bin/env python3
"""The thin swappable model interface — cognition is cloud, tools are local.

One method, `respond(parts, schema)`, where `parts` is a list of text strings
and (label, rgb) image pairs. Everything the subagent knows about models
lives behind it, so the layer-1 bench can swap ER-2 for gemini-3.6-flash by
changing one constructor argument (Anton 2026-08-21: settle the choice on our
own cup images, not on published benchmarks).

Deliberately NOT abstracted here: retries, batching, caching. They belong to
whoever runs the bench, and inventing them now would guess at the wrong ones.
"""
import json
import os

# gemini-robotics-er-2-preview is the researched default: built on Gemini 3.5
# Flash, ~$0.16 per 5-view episode, and documented ONLY on the Interactions
# API (`client.interactions.create`) — not generateContent. The overview page
# omits it; the API-reference model enum wins. Fallback ladder if it
# disappoints on our images: gemini-3.6-flash, then gemini-3.1-pro. Do NOT
# take 3.7-flash for vision — CharXiv regressed 85.2 -> 84.5.
DEFAULT_MODEL = "gemini-robotics-er-2-preview"


class StubVlm:
    """Scripted responses for tests. Records what it was actually shown."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def respond(self, parts, schema=None):
        self.seen.append(parts)
        if not self.script:
            return {"evidence": [], "reasoning": "stub exhausted",
                    "answer": "unknown"}
        return self.script.pop(0)

    #: The convention this model speaks. Quoted to the model in tool-failure
    #: messages, so the loop never teaches a convention the adapter does not
    #: apply (that mismatch cost a real run 16 turns of background crops,
    #: 2708-aicam f04).
    BOX_CONVENTION = "[x0, y0, x1, y1] in full-frame pixels"

    @staticmethod
    def to_pixel_box(box, width, height):
        return tuple(int(v) for v in box)          # already pixels

    @staticmethod
    def to_model_box(box, width, height):
        return [int(v) for v in box]               # identity, both ways


class GeminiVlm:
    """Google Gemini behind `respond`. Reads GEMINI_API_KEY from the env.

    Notes that will bite whoever debugs this at 2am:
    - The API 403s on UNRESTRICTED keys; restrict the key in AI Studio first.
    - ER-2 returns TEXT only. Any geometry it emits is JSON-in-text, with
      points normalised 0-1000 and **y first** (`{"point": [y, x]}`), boxes as
      [ymin, xmin, ymax, xmax]. Conversion belongs here, not in the agent.
    - The streaming endpoint supports neither structured output nor code
      execution, so we do not stream.
    """

    def __init__(self, model=DEFAULT_MODEL, api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set — export it first")
        self._client = None

    def _lazy(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def to_pixels(point_yx, width, height):
        """ER-2's normalised [y, x] (0-1000, y FIRST) -> (x, y) in pixels."""
        y, x = point_yx
        return (round(x / 1000.0 * width), round(y / 1000.0 * height))

    @staticmethod
    def to_pixel_box(box, width, height):
        """ER-2's [ymin, xmin, ymax, xmax] (0-1000) -> (x0, y0, x1, y1) px.

        Confirmed against a real reply: asked to crop the cup, ER-2 emitted
        [390, 460, 715, 595], which decodes to (390, 187, 505, 343) — SAM 3
        independently boxed the same cup at (392, 188, 504, 342). Interpreting
        those numbers as raw pixels put the crop off the bottom of the frame.

        Values already outside 0-1000 cannot be normalised coordinates, so
        they are passed through as pixels — the model occasionally answers in
        the frame's own units after being told the frame size.
        """
        vals = [float(v) for v in box]
        if max(vals) > 1000.0:
            return tuple(int(v) for v in vals)
        ymin, xmin, ymax, xmax = vals
        return (round(xmin / 1000.0 * width), round(ymin / 1000.0 * height),
                round(xmax / 1000.0 * width), round(ymax / 1000.0 * height))

    #: See to_pixel_box: this is what the model natively emits, so it is also
    #: the only convention it may ever be shown.
    BOX_CONVENTION = "[ymin, xmin, ymax, xmax], normalised 0-1000, y first"

    @staticmethod
    def to_model_box(box, width, height):
        """(x0, y0, x1, y1) px -> ER-2's [ymin, xmin, ymax, xmax] (0-1000).

        The exact inverse of to_pixel_box. Every box the model READS (detect
        results, crop echoes) passes through here, so what it reads is what
        it may echo back — round-trip through the model boundary is identity.
        Without this, 2708-aicam f04 burned 16 turns: detect printed pixels,
        the model echoed them into crop, the adapter decoded them as 0-1000
        y-first, and every crop landed on background.
        """
        x0, y0, x1, y1 = [float(v) for v in box]
        return [round(y0 / height * 1000.0), round(x0 / width * 1000.0),
                round(y1 / height * 1000.0), round(x1 / width * 1000.0)]

    def respond(self, parts, schema=None):
        from google.genai import types
        client = self._lazy()
        contents = []
        for p in parts:
            if isinstance(p, str):
                contents.append(p)
            else:                                   # (label, rgb ndarray)
                label, rgb = p
                import cv2
                ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f"could not encode image {label!r}")
                contents.append(label)
                contents.append(types.Part.from_bytes(data=buf.tobytes(),
                                                      mime_type="image/png"))
        cfg = {"response_mime_type": "application/json"} if schema else {}
        out = client.models.generate_content(model=self.model, contents=contents,
                                             config=cfg)
        text = (out.text or "").strip()
        if not schema:
            return text
        obj = first_json_object(text)
        if obj is not None:
            return obj
        # Keep the raw text: a malformed answer is evidence about the model,
        # and the bench needs to see it rather than a swallowed exception.
        return {"evidence": [], "reasoning": f"unparsed model output: {text}",
                "answer": "unknown"}


def first_json_object(text):
    """First balanced {...} in a string, or None.

    ER-2 in JSON mode reliably emits a correct object and then trails extra
    closing braces after it (measured: `{"tool": "detect", ...}}\\n"}\\n}`).
    A plain json.loads throws away a perfectly good tool call because of that
    tail, so scan for the first balanced object instead. Strings are tracked
    so a brace inside a value cannot end the scan early.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
