Some cluster configuration parameters
* HDFS configuration parameters
	* Stored in hdfs-site.xml
	* Block size
	* Default replication count
* MapReduce configuration parameters
	* Stored In “mapred-site.xml”
	* Java heap size for mappers/reducers
	* Number of mappers/reducers per host. See http://wiki.apache.org/hadoop/HowManyMapsAndReduces
	* Job Tracker URL: http://<masterhost>:50030
	* Name Node URL: http://<masterhost>:50070

Submit pig job
* pig –f myscipt.pig
* To check the status of your job
	* Use the Job Tracker URL (easiest) OR
	* `hadoop job –list` (will print all job ids)
	* `hadoop job –status <jobid>` (will print the job status)
* To get the results: `hadoop fs –get /path/results.txt`



************************
(MapReduce Patterns) Counting and Summing, Collating, Filtering, Parsing, Sorting, Graph Processing, Cross-Correlation, Union, Join
https://highlyscalable.wordpress.com/2012/02/01/mapreduce-patterns/
