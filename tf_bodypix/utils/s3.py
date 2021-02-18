import logging

import urllib.request
from xml.etree import ElementTree
from typing import Iterable


LOGGER = logging.getLogger(__name__)


S3_NS = 'http://doc.s3.amazonaws.com/2006-03-01'
S3_PREFIX = '{%s}' % S3_NS
S3_CONTENTS = S3_PREFIX + 'Contents'
S3_KEY = S3_PREFIX + 'Key'
S3_NEXT_MARKER = S3_PREFIX + 'NextMarker'


def iter_s3_file_urls(base_url: str) -> Iterable[str]:
    if not base_url.endswith('/'):
        base_url += '/'
    marker = None
    while True:
        current_url = base_url
        if marker:
            current_url += '?marker=' + marker
        with urllib.request.urlopen(current_url) as url_fp:
            response_data = url_fp.read()
        LOGGER.debug('response_data: %r', response_data)
        root = ElementTree.fromstring(response_data)
        for item in root.findall(S3_CONTENTS):
            key = item.findtext(S3_KEY)
            LOGGER.debug('key: %s', key)
            if key:
                yield base_url + key
        next_marker = root.findtext(S3_NEXT_MARKER)
        if not next_marker or next_marker == marker:
            break
        marker = next_marker
