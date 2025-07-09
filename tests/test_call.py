import pytest

from handler import EndpointHandler

@pytest.fixture
def html() -> str:
    with open("data/test.html", "r") as f:
        return f.read()

def test_call(html):
    eh = EndpointHandler(
        path="/"
    )
    output = eh({"inputs": html})
    print(output)
