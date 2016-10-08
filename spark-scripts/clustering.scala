val movies = sc.textFile("data/ml-100k/u.item")
movies.first
val genre = sc.textFile("data/ml-100k/u.genre")
val genreMap = genre.filter(!_.trim.isEmpty).map(_.split("\\|")).map(arr => (arr(1),arr(0))).collectAsMap
val genres = genre.repartition(1).filter(!_.trim.isEmpty).map(_.split("\\|")).map(arr => (arr(1).toInt,arr(0))).sortBy(_._1).collect.map(_._2)
genres.foreach(println)
val titlesAndGenres = movies.map(_.split("\\|")).map(arr => (arr(0).toInt, (arr(1),arr.toSeq.slice(5,arr.size).zipWithIndex.filter(_._1 == "1").map(gen => genres(gen._2)))))
titlesAndGenres.first

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
val rawData = sc.textFile("data/ml-100k/u.data")

val ratings = rawData.map(_.split("\t").take(3)).map { case Array(user,movie,rating) => Rating(user.toInt,movie.toInt,rating.toDouble) }
ratings.first
ratings.cache
val alsModel = ALS.train(ratings, 50, 10, 0.1)

import org.apache.spark.mllib.linalg.Vectors
val movieFactors = alsModel.productFeatures.map { case (id,factor) => (id,Vectors.dense(factor)) }
val movieVectors = movieFactors.map(_._2)
val userFactors = alsModel.userFeatures.map { case (id,factor) => (id,Vectors.dense(factor)) }
val userVectors = userFactors.map(_._2)

import org.apache.spark.mllib.linalg.distributed.RowMatrix
val movieMatrix = new RowMatrix(movieVectors)
val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
val userMatrix = new RowMatrix(userVectors)
val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
println("movie factors mean: " + movieMatrixSummary.mean)
println("movie factors variance:\n" + movieMatrixSummary.variance)
println("user factors mean:\n" + userMatrixSummary.mean)
println("user factors variance:\n" + userMatrixSummary.variance)

import org.apache.spark.mllib.clustering.KMeans
val numClusters = 5
val numIterations = 10
val numRuns = 3
val movieClusterModel = KMeans.train(movieVectors, numClusters, numIterations, numRuns)
val userClusterModel = KMeans.train(userVectors, numClusters, numIterations, numRuns)

