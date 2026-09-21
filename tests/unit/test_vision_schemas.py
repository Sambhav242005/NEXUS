from nexus.vision.schemas import ScreenObservation


def test_zero_size_semantic_element_is_observable_but_not_invalid():
    observation = ScreenObservation(elements=[{
        "label": "document text", "role": "text", "x": 0, "y": 0,
        "width": 0, "height": 0, "confidence": 0.8,
    }])
    assert observation.elements[0].width == 0
