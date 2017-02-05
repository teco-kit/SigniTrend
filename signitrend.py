"""
A simple implementation of SigniTrend [0], a scalable detection system by Erich Schubert, Michael Weiler, and
Hans-Peter Kriegel.

Instead of updating the EWM{A|Var} every time, pandas dataframes are kept, updated, and the new values are queried.
However, this implementation can be easily adapted to perfectly match the algorithm proposed in the paper by Schubert
et al. To do so, get rid of the dataframes in the buckets, choose alpha = 2 / (span + 1), and adapt the update step
in the nex_epoch method according to the paper.

[0] Erich Schubert, Michael Weiler, and Hans-Peter Kriegel. 2014. SigniTrend: scalable detection of emerging topics
     in textual streams by hashed significance thresholds. In Proceedings of the 20th ACM SIGKDD international
     conference on Knowledge discovery and data mining (KDD '14). ACM, New York, NY, USA, 871-880.
     DOI: http://dx.doi.org/10.1145/2623330.2623740*

__author__ = "Peter Bozsoky"
__copyright__ = "Copyright 2017, the author"
__credits__ = ["Peter Bozsoky"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Peter Bozsoky"
__email__ = "peter.bozsoky@student.kit.edu"
__status__ = "Testing"
"""

import hashlib
import math
import pandas as pd
import humanize
from pympler import asizeof
import logging
app_log = logging.getLogger('root')


class SigniTrend:
    """Implementation of SigniTrend [0], a scalable detection system by Erich Schubert, Michael Weiler,
     and Hans-Peter Kriegel.

     Use as follows:

     - Create a SigniTrend instance
     - index_new_tweets() for every tweet you encounter during the current epoch (or timestep)
     - Optionally, get an end_of_day_analysis()
     - Call next_epoch()
     - Goto 1

     *[0] Erich Schubert, Michael Weiler, and Hans-Peter Kriegel. 2014. SigniTrend: scalable detection of emerging topics
     in textual streams by hashed significance thresholds. In Proceedings of the 20th ACM SIGKDD international
     conference on Knowledge discovery and data mining (KDD '14). ACM, New York, NY, USA, 871-880.
     DOI: http://dx.doi.org/10.1145/2623330.2623740*
    """

    def __init__(self, window_size: int = 14, hash_table_bits: int = 20, hash_function_count: int = 4,
                 bias: float = 1.0, alerting_threshold: float = 6.0, debug: bool = False):
        """
        :param window_size: The amount of timesteps the moving window shall consider
        :param hash_table_bits: The amount of bits for our hash table (the log_2 of the number of hash table entries)
        :param hash_function_count: How many hash functions to use.
        :param bias: The bias term (> 0). Adjust to the expected level of background noise.
        :param alerting_threshold: Threshold in units of standard deviations, above which frequencies are considered
        anomalously high.
        :param debug: Set to True for progress output during hash table initialization.
        """
        # "Parameters"
        self.window_size = window_size
        self.bit_count = hash_table_bits
        self.hashmap_size = 2 ** self.bit_count
        self.hash_function_count = hash_function_count
        self.beta = bias
        self.alerting_threshold = alerting_threshold
        # We don't care for cryptographically secure hashes
        hash_functions_available = [hashlib.md5, hashlib.sha1, hashlib.sha224, hashlib.sha256, hashlib.sha384,
                                    hashlib.sha512]
        if hash_function_count >= len(hash_functions_available) or hash_function_count <= 0:
            raise RuntimeError("The given hash_function_count must be from [1, {}]. Was: {}."
                               .format(len(hash_functions_available), hash_function_count))
        self.hash_functions = hash_functions_available[:hash_function_count]

        # "Working Variables"
        self.tweet_count = 0
        self.epoch = 1
        self.frequency_map = dict()
        self.stats_map = dict()
        self.index = dict()
        self.index[self.epoch] = dict()
        self.refinement = []
        estimated_bucket_size = asizeof.asizeof({"data": pd.DataFrame([0.0 for _ in range(window_size)]),
                                                 "ewma": 0.0,
                                                 "ewmvar": 0.0})
        print("Initializing hash bucket list with {} buckets of size {}B (total {}B).".format(self.hashmap_size,
                                                                                              estimated_bucket_size,
                                                                                              self.hashmap_size * estimated_bucket_size))
        self.buckets = []
        self.trending_topics = []
        twenty_percent_size_printed = False
        for i in range(self.hashmap_size):
            self.buckets.append({"data": pd.DataFrame([0 for _ in range(window_size)]), "ewma": 0.0, "ewmvar": 0.0})
            if debug and not twenty_percent_size_printed and i / self.hashmap_size >= 0.25:
                print("Total hashmap size at 25% is {}".format(humanize.naturalsize(asizeof.asizeof(self.buckets))))
                twenty_percent_size_printed = True
            if i % 10000 == 0 or i == self.hashmap_size:
                print("{} buckets ({:2.1f}%) - {} used.".format(humanize.intword(i),
                                                                (i / self.hashmap_size) * 100,
                                                                humanize.naturalsize(i * estimated_bucket_size)))
        if debug:
            print("Getting actual hashmap size, this can take a moment and some memory...")
            print("Actual usage is {}.".format(humanize.naturalsize(asizeof.asizeof(self.buckets))))

    def _update_bucket(self, x: float, bucket: dict):
        df = bucket["data"]
        df.loc[len(df)] = x
        # Only keep the part of our rolling window that contributes to the EWM{A|Var}.
        df.drop(0, inplace=True)
        df.reset_index(inplace=True)
        del df['index']
        window = df.ewm(adjust=True, span=self.window_size)
        bucket["data"] = df
        bucket["ewma"] = window.mean().iloc[self.window_size - 1][0]
        bucket["ewmvar"] = window.var().iloc[self.window_size - 1][0]

    def index_new_tweet(self, id_str, tweet_tokens: list):
        """ Indexes a tweet represented by a list of str tokens. For checking a reverse index,
        the tweet's id_str is needed.
        """
        self.tweet_count += 1
        unique_words = set(tweet_tokens)
        unique_word_pairs = set()
        for i in unique_words:
            for j in unique_words - {i}:
                unique_word_pairs.add(tuple(sorted([i, j])))  # sorting because to us [a, b] = [b, a]
        for w in unique_words | unique_word_pairs:
            self.index[self.epoch][w] = id_str
            current_freq = self.frequency_map.get(w, 0)
            self.frequency_map[w] = current_freq + 1
            # Get word statistics from hash table
            statistics_present = w in self.stats_map
            if not statistics_present:
                (mu, sigma) = (math.inf, math.inf)
                for h in self.hash_functions:
                    c = get_hash(h(), repr(w)) % 2 ** self.bit_count
                    if self.buckets[c]["ewma"] < mu:
                        mu = self.buckets[c]["ewma"]
                        sigma = self.buckets[c]["ewmvar"]
                        self.stats_map[w] = (mu, sigma)
            (mu, sigma) = self.stats_map[w]
            # Test for significance threshold
            x = self.frequency_map[w]
            if self._is_frequency_significant(mu, sigma, x):
                self.refinement.append((w, self._get_significance(mu, sigma, x)))
        # For now this is enough
        if self.refinement:
            r = self.refinement
            self.refinement = []
            return r

    def _is_frequency_significant(self, mu, sigma, frequency):
        """ Checks whether a given frequency is significant, given the parameters of a normal distribution.
        :param mu: Normal distribution's expectation.
        :param sigma: Normal distribution's variance.
        :param frequency: The frequency in question.
        """
        return self._get_significance(mu, sigma, frequency) > self.alerting_threshold

    def _get_significance(self, mu, sigma, frequency):
        return (frequency - max(self.beta, mu)) / (sigma + self.beta)

    def end_of_day_analysis(self):
        """ To be considered work in progress, because it's currently not used by us."""
        for w in self.frequency_map:
            (mu, sigma) = self.stats_map[w]
            # Test for significance threshold
            x = self.frequency_map[w] / self.tweet_count
            if self._is_frequency_significant(mu, sigma, x):
                self.trending_topics.append((w, self._get_significance(mu, sigma, x)))
        return self.trending_topics

    def next_epoch(self):
        """To be called whenever an epoch ends."""
        update_table = [None for _ in range(2 ** self.bit_count)]
        for w in self.frequency_map:
            frequency = self.frequency_map[w]
            for h in self.hash_functions:
                c = get_hash(h(), repr(w)) % 2 ** self.bit_count
                if not update_table[c] or frequency > update_table[c]:
                    update_table[c] = frequency
        # Update statistics table
        debug_hit_something = False
        for c in range(len(update_table)):
            if update_table[c] is not None:
                debug_hit_something = True
                freq = update_table[c] / self.tweet_count
                self._update_bucket(freq, self.buckets[c])
            else:
                pass
        assert debug_hit_something is True
        self.epoch += 1
        self.index[self.epoch] = dict()
        self.trending_topics = []


def get_hash(hash_function, x: str):
    """Returns a given string's hash value, obtained by the given hashlib instance."""
    hash_function.update(x.encode())
    return int.from_bytes(hash_function.digest(), byteorder="big")