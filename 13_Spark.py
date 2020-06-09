from pyspark import SparkContext, SparkConf

sc = SparkContext()

# Lambda expressione
even = lambda num: num % 2 == 0
#print(even(5))
#print(even(4))

f = 'dati/spark_example.txt'
textFile = sc.textFile(f)

print(textFile.count())
print(textFile.first())

print(textFile.filter(lambda line: 'second' in line))
sfind = textFile.filter(lambda line: 'second' in line)

print(sfind.collect())

#conf = SparkConf().setAppName(appName).setMaster(master)
#sc = SparkContext(conf=conf)
