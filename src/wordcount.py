from operator import add
import sys

from pyspark import SparkContext

if __name__ == '__main__':
    print(sys.version)
    if len(sys.argv) != 3:
        print('Usage: wordcount <infile> <outfile>')
        exit(-1)
    sc = SparkContext(appName='Python Word Count')

    tf = sc.textFile(sys.argv[1])

    result = tf.flatMap(lambda line: line.split()) \
                    .map(lambda word: (word,1)) \
                    .reduceByKey(add)
    result.saveAsTextFile(sys.argv[2])
    
    sc.stop()
