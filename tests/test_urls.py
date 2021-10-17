import pytest
from http.client import HTTPConnection
from urllib.parse import urlparse

from pyech.utils import SURVEY_URLS, DICTIONARY_URLS


@pytest.mark.parametrize("year", range(2006, 2021))
@pytest.mark.parametrize("data", [SURVEY_URLS, DICTIONARY_URLS])
def test_urls(data, year):
    parsed = urlparse(data[year])
    conn = HTTPConnection(parsed.netloc)
    conn.request("HEAD", parsed.path)
    resp = conn.getresponse()
    assert resp.status < 400
