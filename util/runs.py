import contextlib
import re
import unicodedata
from datetime import datetime
from typing import Optional

import joblib
from tqdm import tqdm


def slugify(value: str, allow_unicode: bool = False) -> str:
    """Converts a string to a valid filename (see
    https://github.com/django/django/blob/master/django/utils/text.py). Converts to
    ASCII if `allow_unicode=False.`; converts spaces or repeated dashes to single
    dashes; removes characters that aren't alphanumerics, underscores, or hyphens;
    converts to lowercase; strips leading and trailing whitespace, dashes, and
    underscores.

    Parameters
    ----------
    value : str
        String to be converted to a valid filename.
    allow_unicode : bool, optional
        If `True`, the string is converted to ASCII, by default False

    Returns
    -------
    str
        The string as a valid fileame.
    """
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def get_runname(candidate: Optional[str] = None) -> str:
    """Gets the name for this run, and makes sure it can be used as filename.

    Parameters
    ----------
    candidate : str, optional
        A candidate string as run name. If none is provided, the current datetime is
        used to build one.

    Returns
    -------
    str
        The runname as a valid fileame.
    """
    if candidate is None or not candidate:
        return f'R_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    return slugify(candidate)


@contextlib.contextmanager
def tqdm_joblib(*args, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar given as
    argument (thanks to https://stackoverflow.com/questions/24983493/
    tracking-progress-of-joblib-parallel-execution/58936697#58936697)."""
    tqdm_object = tqdm(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
