# SigniTrend

Implementation of SigniTrend [0], a scalable trend detection system by Erich Schubert, Michael Weiler, and Hans-Peter Kriegel.
It is designed to take a stream of tweet contents and detect significantly frequent-ocurring terms up to a fixed upper word count. For instance, the term "nice weather" consists of two words, and is represented by the sorted tuple (nice, weather).

## Usage instructions
Repeat for every epoch/timestep/timeslot/`<unit_of_time_discretization>`:

1. Create a SigniTrend instance
2. `index_new_tweets()` for every tweet you encounter during the current epoch (or timestep)
3. Optionally, get an `end_of_day_analysis()`
4. Call `next_epoch()`

[0] Erich Schubert, Michael Weiler, and Hans-Peter Kriegel. 2014. SigniTrend: scalable detection of emerging topics
in textual streams by hashed significance thresholds. In Proceedings of the 20th ACM SIGKDD international
conference on Knowledge discovery and data mining (KDD '14). ACM, New York, NY, USA, 871-880.
**DOI: http://dx.doi.org/10.1145/2623330.2623740**
