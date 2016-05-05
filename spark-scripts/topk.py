import sys
from operator import add
import bisect

from pyspark import SparkContext

k = 0
def topk(itera):
    ks = [0 for x in range(k) ] # key list
    rs = [('',0) for x in range(k)] # [('',0),('',0),('',0), ... ]
    for word,n in itera:
        p = bisect.bisect_left(ks, n)
        ks.insert(p, n)
        rs.insert(p, (word,n))
        ks.pop(0)
        rs.pop(0)
    rs.reverse()
    return rs

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: topk.py <topk> <input> <output>')
        exit(-3)
    global k
    k = int(sys.argv[1])
    if k < 1:
        raise ValueError('k must be positive')

    sc = SparkContext(appName='python top k')
    tf = sc.textFile(sys.argv[2])
    result = tf.flatMap(lambda line: line.split()) \
                    .map(lambda w: (w,1)) \
                    .reduceByKey(add) \
                    .mapPartitions(topk) \
                    .repartition(1) \
                    .mapPartitions(topk) \
                    .map(lambda t: t[0] + '\t' + str(t[1])) \
                    .saveAsTextFile(sys.argv[3])
    sc.stop()
