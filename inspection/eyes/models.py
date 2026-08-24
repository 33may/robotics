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
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Keep the raw text: a malformed answer is evidence about the model,
            # and the bench needs to see it rather than a swallowed exception.
            return {"evidence": [], "reasoning": f"unparsed model output: {text}",
                    "answer": "unknown"}
