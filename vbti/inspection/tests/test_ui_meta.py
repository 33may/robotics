#!/usr/bin/env python3
"""run/meta: retained topic that tells the frontend what kind of run this is.
Run: p inspection/tests/test_ui_meta.py"""
from inspection.ui.publisher import InspectionPublisher, TOPIC_META


class BusSpy:
    def __init__(self):
        self.published = []          # (topic, payload)
        self.declared = []           # (topic, kwargs)

    def publish(self, topic, payload):
        self.published.append((topic, payload))

    def declare(self, topic, **kw):
        self.declared.append((topic, kw))

    @staticmethod
    def array_payload(a):
        return a

    @staticmethod
    def jpeg_payload(a, quality=80):
        return a

    def last(self, topic):
        return [p for t, p in self.published if t == topic][-1]


def test_declare_registers_run_meta():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    pub.declare()
    topics = dict(bus.declared)
    assert TOPIC_META in topics
    assert topics[TOPIC_META]["qos"] == "stream"
    assert topics[TOPIC_META]["kind"] == "json"


def test_publish_run_meta_live_payload():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    pub.publish_run_meta(source="live", name="cup-inspection", object="cup",
                          question="is there a logo on the cup?")
    payload = bus.last(TOPIC_META)
    assert payload == {
        "source": "live",
        "name": "cup-inspection",
        "object": "cup",
        "question": "is there a logo on the cup?",
    }


def test_publish_run_meta_data_engine_optional_fields_are_null():
    bus = BusSpy()
    pub = InspectionPublisher(bus)
    pub.publish_run_meta(source="data-engine", name="collect-run-003")
    payload = bus.last(TOPIC_META)
    assert payload["source"] == "data-engine"
    assert payload["name"] == "collect-run-003"
    assert payload["object"] is None
    assert payload["question"] is None


def main():
    test_declare_registers_run_meta()
    test_publish_run_meta_live_payload()
    test_publish_run_meta_data_engine_optional_fields_are_null()
    print("OK test_ui_meta")


if __name__ == "__main__":
    main()
