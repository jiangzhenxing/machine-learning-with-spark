import numpy as np

from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS
from pyspark import SparkContext

from learning.spark import util

sc = SparkContext(appName='pyspark recommend')
ratingData = sc.textFile('data/ml-100k/u.data') \
               .map(lambda line: line.split())
ratings = ratingData.map(lambda l: Rating(user=int(l[0]), product=int(l[1]), rating=float(l[2])))
#print(ratings.take(10))

model = ALS.train(ratings=ratings, rank=50,iterations=10, lambda_=0.01)
userId = 789
predict = model.predict(userId, 123)
print('predict user 789 product rating is:', predict)
k = 10
recommends = model.recommendProducts(userId, k)

movies = sc.textFile('data/ml-100k/u.item') \
           .map(lambda line: line.split('|')) \
           .map(lambda l:(int(l[0]), l[1])) \
           .collectAsMap()

recommendMovies = list(map(lambda rating: (movies[rating.product],rating.rating), recommends))
print('recommends for user', userId)
for rec in recommendMovies:
    print(rec)

userRatings = ratings.filter(lambda r: r.user==userId) \
                     .sortBy(lambda r: -r.rating) \
                     .take(k)
userMovies = list(map(lambda r:(movies[r.product],r.rating), userRatings))
print('the user rating top 10 movies is')
for mv in userMovies:
    print(mv)

moviesfull = sc.textFile('data/ml-100k/u.item') \
               .map(lambda line: line.split('|')) \
               .map(lambda l: (int(l[0]), l)) \
               .collectAsMap()
itemId = 567
print('movie info is:')
print(util.movieInfo(moviesfull[itemId]))

itemv = model.productFeatures().lookup(itemId)[0]
sims = model.productFeatures() \
            .mapValues(lambda v:util.vectorCosine(v, itemv)) \
            .top(10, key=lambda ms: ms[1])

print('similar movie is:')
for m in sims:
    print(util.movieInfo(moviesfull[m[0]]), '\tsimilar is:', m[1])

