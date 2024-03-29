#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/io.py
"""

import logging
import os
import re
import sys
from urllib import request as urlrequest

from iopath.common.file_io import PathManagerFactory


pathmgr = PathManagerFactory.get()

logger = logging.getLogger(__name__)

_PYCLS_BASE_URL = ""


def cache_url(url_or_file, cache_dir, base_url=_PYCLS_BASE_URL, download=True):
    is_url = re.match(r"^(?:http)s?://", url_or_file, re.IGNORECASE) is not None
    if not is_url:
        return url_or_file
    url = url_or_file
    assert url.startswith(base_url), "url must start with: {}".format(base_url)
    cache_file_path = url.replace(base_url, cache_dir)
    if pathmgr.exists(cache_file_path):
        return cache_file_path
    cache_file_dir = os.path.dirname(cache_file_path)
    if not pathmgr.exists(cache_file_dir):
        pathmgr.mkdirs(cache_file_dir)
    if download:
        logger.info("Downloading remote file {} to {}".format(url, cache_file_path))
        download_url(url, cache_file_path)
    return cache_file_path


def _progress_bar(count, total):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    sys.stdout.write(
        "  [{}] {}% of {:.1f}MB file  \r".format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write("\n")


def download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    req = urlrequest.Request(url)
    response = urlrequest.urlopen(req)
    total_size = response.info().get("Content-Length").strip()
    total_size = int(total_size)
    bytes_so_far = 0
    with pathmgr.open(dst_file_path, "wb") as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)
    return bytes_so_far
