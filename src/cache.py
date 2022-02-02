import tempfile

import diskcache as dc

cache = dc.Cache(tempfile.gettempdir(), size_limit=100 * 2 ** 30)
