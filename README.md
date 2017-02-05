# SigniTrend

Implementation of SigniTrend [0], a scalable detection system by Erich Schubert, Michael Weiler, and Hans-Peter Kriegel.

## Usage instructions
Repeat for every epoch:

  1) Create a SigniTrend instance
  2) index_new_tweets() for every tweet you encounter during the current epoch (or timestep)
  3) Optionally, get an end_of_day_analysis()
  4) Call next_epoch()

[0] Erich Schubert, Michael Weiler, and Hans-Peter Kriegel. 2014. SigniTrend: scalable detection of emerging topics
in textual streams by hashed significance thresholds. In Proceedings of the 20th ACM SIGKDD international
conference on Knowledge discovery and data mining (KDD '14). ACM, New York, NY, USA, 871-880.
**DOI: http://dx.doi.org/10.1145/2623330.2623740**