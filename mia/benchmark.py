"""Main mia benchmarking module."""

import io
import logging
import multiprocessing as mp
import os
import types
from datetime import datetime, timedelta

import humanfriendly
import psutil
import tqdm

from mia.data import load_dataset
from mia.evaluate import compute_scores
from mia.results import make_leaderboard
from mia.synthesizers.base import BaseSynthesizer

LOGGER = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
LEADERBOARD_PATH = os.path.join(BASE_DIR, 'leaderboard.csv')

DEFAULT_DATASETS = [
    "alphabank"
]


class TqdmLogger(io.StringIO):

    _buffer = ''

    def write(self, buf):
        self._buffer = buf.strip('\r\n\t ')

    def flush(self):
        LOGGER.info(self._buffer)


def _used_memory():
    process = psutil.Process(os.getpid())
    return humanfriendly.format_size(process.memory_info().rss)


def _score_synthesizer_on_dataset(name, synthesizer, iteration, args):
    try:
        LOGGER.info('Evaluating %s on dataset %s; iteration %s; %s',
                    name, args.dataset_name, iteration, _used_memory())

        train, test, meta, categoricals, ordinals = load_dataset(args.dataset_name, benchmark=True)
        

        if isinstance(synthesizer, type) and issubclass(synthesizer, BaseSynthesizer):
            synthesizer = synthesizer(dataset_name=args.dataset_name, meta=meta).fit_sample

        LOGGER.info('Running %s on dataset %s; iteration %s; %s',
                    name, args.dataset_name, iteration, _used_memory())

        synthesized = synthesizer(train, categoricals, ordinals, args)

        LOGGER.info('Scoring %s on dataset %s; iteration %s; %s',
                    name, args.dataset_name, iteration, _used_memory())

        scores = compute_scores(train, test, synthesized, meta, args)

        scores['dataset'] = args.dataset_name
        scores['iteration'] = iteration
        scores['synthesizer'] = name

        return scores

    except Exception:
        LOGGER.exception('Error running %s on dataset %s; iteration %s',
                         name, args.dataset_name, iteration)

    finally:
        LOGGER.info('Finished %s on dataset %s; iteration %s; %s',
                    name, args.dataset_name, iteration, _used_memory())


def _score_synthesizer_on_dataset_args(args):
    return _score_synthesizer_on_dataset(*args)


def _get_synthesizer_name(synthesizer):
    """Get the name of the synthesizer function or class.

    If the given synthesizer is a function, return its name.
    If it is a method, return the name of the class to which
    the method belongs.

    Args:
        synthesizer (function or method):
            The synthesizer function or method.

    Returns:
        str:
            Name of the function or the class to which the method belongs.
    """
    if isinstance(synthesizer, types.MethodType):
        synthesizer_name = synthesizer.__self__.__class__.__name__
    else:
        synthesizer_name = synthesizer.__name__

    return synthesizer_name


def _get_synthesizers(synthesizers):
    """Get the dict of synthesizers from the input value.

    If the input is a synthesizer or an iterable of synthesizers, get their names
    and put them on a dict.

    Args:
        synthesizers (function, class, list, tuple or dict):
            A synthesizer (function or method or class) or an iterable of synthesizers
            or a dict containing synthesizer names as keys and synthesizers as values.

    Returns:
        dict[str, function]:
            dict containing synthesizer names as keys and function as values.

    Raises:
        TypeError:
            if neither a synthesizer or an iterable or a dict is passed.
    """
    if callable(synthesizers):
        synthesizers = {_get_synthesizer_name(synthesizers): synthesizers}
    if isinstance(synthesizers, (list, tuple)):
        synthesizers = {
            _get_synthesizer_name(synthesizer): synthesizer
            for synthesizer in synthesizers
        }
    elif not isinstance(synthesizers, dict):
        raise TypeError('`synthesizers` can only be a function, a class, a list or a dict')

    return synthesizers


def progress(*futures):
    """Track progress of dask computation in a remote cluster.

    LogProgressBar is defined inside here to avoid having to import
    its dependencies if not used.
    """
    # Import distributed only when used
    from distributed.client import futures_of  # pylint: disable=C0415
    from distributed.diagnostics.progressbar import TextProgressBar  # pylint: disable=c0415

    class LogProgressBar(TextProgressBar):
        """Dask progress bar based on logging instead of stdout."""

        last = 0
        logger = logging.getLogger('distributed')

        def _draw_bar(self, remaining, all, **kwargs):   # pylint: disable=W0221,W0622
            done = all - remaining
            frac = (done / all) if all else 0

            if frac > self.last + 0.01:
                self.last = int(frac * 100) / 100
                bar = "#" * int(self.width * frac)
                percent = int(100 * frac)

                time_per_task = self.elapsed / (all - remaining)
                remaining_time = timedelta(seconds=time_per_task * remaining)
                eta = datetime.utcnow() + remaining_time

                elapsed = timedelta(seconds=self.elapsed)
                msg = "[{0:<{1}}] | {2}/{3} ({4}%) Completed | {5} | {6} | {7}".format(
                    bar, self.width, done, all, percent, elapsed, remaining_time, eta
                )
                self.logger.info(msg)
                LOGGER.info(msg)

        def _draw_stop(self, **kwargs):
            pass

    futures = futures_of(futures)
    if not isinstance(futures, (set, list)):
        futures = [futures]

    LogProgressBar(futures)



def run(synthesizers=None, iterations=1, workers=4, output_path=None, arguments=None):
    """Run the octgan benchmark and return a leaderboard.

    The ``synthesizers`` object can either be a single synthesizer or, an iterable of
    synthesizers or a dict containing synthesizer names as keys and synthesizers as values.

    Args:
        synthesizers (function, class, list, tuple or dict):
            The synthesizer or synthesizers to evaluate. It can be a single synthesizer
            (function or method or class), or an iterable of synthesizers, or a dict
            containing synthesizer names as keys and synthesizers as values. If the input
            is not a dict, synthesizer names will be extracted from the given object.
        iterations (int):
            Number of iterations to perform over each dataset and synthesizer. Defaults to 3.
        workers (int or str):
            If ``workers`` is given as an integer value other than 0 or 1, a multiprocessing
            Pool is used to distribute the computation across the indicated number of workers.
            If the string ``dask`` is given, the computation is distributed using ``dask``.
            In this case, setting up the ``dask`` cluster and client is expected to be handled
            outside of this function.
        output_path (str):
            If an ``output_path`` is given, the generated leaderboard will be stored in the
            indicated path as a CSV file. The given path must be a complete path including
            the ``.csv`` filename.

    Returns:
        pandas.DataFrame or None:
            If not ``output_path`` is given, a table containing one row per synthesizer and
            one column for each dataset and metric is returned. Otherwise, there is no output.
    """

    scorer_args = list()

    if synthesizers != None:
        synthesizers = _get_synthesizers(synthesizers)

        for synthesizer_name, synthesizer in synthesizers.items():
            for iteration in range(iterations):
                args = (synthesizer_name, synthesizer, iteration, arguments)
                scorer_args.append(args)

    else:
        for iteration in range(iterations):
            args = ("benchmark", None, iteration, arguments)
            scorer_args.append(args)

    if workers in (0, 1):
        scores = map(_score_synthesizer_on_dataset_args, scorer_args)
    else:
        pool = mp.Pool(mp.cpu_count())
        scores = pool.imap_unordered(_score_synthesizer_on_dataset_args, scorer_args)

    scores = tqdm.tqdm(scores, total=len(scorer_args), file=TqdmLogger())


    return make_leaderboard(
        scores,
        output_path=output_path,
        args=arguments
    )