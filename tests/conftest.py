import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def wp4_data_root():
    """C11: the historical source resolves ONLY through this
    environment variable; the suite binds it here for local checkouts
    and no absolute path lives in the public code."""
    if "WP4_DATA_ROOT" not in os.environ:
        candidate = os.path.join(os.path.dirname(__file__), "..",
                                 "..", "financial-data")
        if os.path.isdir(candidate):
            os.environ["WP4_DATA_ROOT"] = os.path.abspath(candidate)
    yield
