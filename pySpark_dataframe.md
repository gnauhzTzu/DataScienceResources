Create a Spark DataFrame from Pandas
```
spark_df = sqlContext.createDataFrame(pandas_df)
```

```
df = sqlContext.read.text("README.txt")
return all the records as a list of Row
df.collect()
df.take(n)
df.show(n, truncate)
df.count() #show row number
```
Creates a DataFrame from one column:

```
ageCol = people.age
```

Select one or more columns from a DataFrame:
Selects two columns

Selects all the columns
```
df.select('*')
df.select('name', 'age')
```
Selects the name and age columns,
increments the values in the age column by 10,
and renames (alias) the age + 10 column as age
```
df.select(df.name,
    (df.age + 10).alias('age'))
```
Drop method returns a new DataFrame that drops
```
df.drop(df.age)

# [Row(name=u'Alice'), Row(name=u'Bob')]
```

User Defined Function(UDF)

Creates a DataFrame of [Row(slen=5), Row(slen=3)]
```
from pyspark.sql.types import IntegerType
slen = udf(lambda s: len(s), IntegerType())
df.select(slen(df.name).alias('slen'))

df = sqlContext.createDataFrame(data, ['name', 'age'])

# [Row(name=u'Alice', age=1), Row(name=u'Bob', age=2)]

from pyspark.sql.types import IntegerType
doubled = udf(lambda s: s * 2, IntegerType())
df2 = df.select(df.name, doubled(df.age).alias('age'))

# [Row(name=u'Alice', age=2), Row(name=u'Bob', age=4)]
```

Only keeps rows with age column greater than 3
```
df3 = df2.filter(df2.age > 3)
[Row(name=u'Bob', age=4)]
```

Only keeps rows that are distinct
```
data2 = [('Alice', 1), ('Bob', 2), ('Bob', 2)]
df = sqlContext.createDataFrame(data2, ['name', 'age'])

# [Row(name=u'Alice', age=1), Row(name=u'Bob', age=2),
# Row(name=u'Bob', age=2)]

df2 = df.distinct()

# [Row(name=u'Alice', age=1), Row(name=u'Bob', age=2)]
df3 = df2.sort("age", ascending=False)
```
Turn each element of the intlist column into a Row, alias the resulting
column to anInt, and select only that column
```
data3 = [Row(a=1, intlist=[1,2,3])]
df4 = sqlContext.createDataFrame(data3)

# [Row(a=1, intlist=[1,2,3])]
df4.select(explode(df4.intlist).alias("anInt"))

# [Row(anInt=1), Row(anInt=2), Row(anInt=3)]

data = [('Alice',1,6), ('Bob',2,8), ('Alice',3,9), ('Bob',4,7)]
df = sqlContext.createDataFrame(data, ['name', 'age', 'grade'])
df1 = df.groupBy(df.name)
df1.agg({"*": "count"}).collect()

# [Row(name=u'Alice', count(1)=2), Row(name=u'Bob', count(1)=2)]

df.groupBy(df.name).count()

# [Row(name=u'Alice', count=2), Row(name=u'Bob', count=2)]

data = [('Alice',1,6), ('Bob',2,8), ('Alice',3,9), ('Bob',4,7)]
df = sqlContext.createDataFrame(data, ['name', 'age', 'grade'])
df.groupBy().avg().collect()

# [Row(avg(age)=2.5, avg(grade)=7.5)]

df.groupBy('name').avg('age', 'grade').collect()

# [Row(name=u'Alice', avg(age)=2.0, avg(grade)=7.5),
# Row(name=u'Bob', avg(age)=3.0, avg(grade)=7.5)]
```

caching dataframes

```
linesDF = sqlContext.read.text('...')
linesDF.cache()

# save,don't recompute! if not will recompute second at print commentsDF
commentsDF = linesDF.filter(isComment)
print linesDF.count(),commentsDF.count()
```

#### Spark Program Lifecycle
* Create DataFrames from external data or [createDataFrame] from a collection in driver program
* Lazily [transform] them into new DataFrames
* **cache()** some DataFrames for reuse
* Perform [actions] to execute parallel
computation and produce results

#### Where Code Runs
Most Python code runs in driver, 
**except for** code passed to transformations
Transformations run at executors
Actions run at executors and driver

```                      
|-----------------| -> Worker (Spark executor)
|Your application | -> Worker (Spark executor)
|(driver program) | -> Worker (Spark executor)
|-----------------| -> Worker (Spark executor)

a = a + 1                  ----> driver program
linesDF.filter(isComment)  ----> spark executor
commentsDF.count()         ----> driver program & spark executor

```

**Don't Write Code like this**

Combine two DataFrames: aDF, bDF:
Wrong:
a = aDF.collect()
b = bDF.collect()
cDF = sqlContext.createDataFrame(a + b)
**all** distributed data for **a** and **b** is send to driver program

Wrong:
cDF = sqlContext.createDataFrame(a + b)
**all** data for **cDF** is sent to the executors

Best:
cDF = aDF.unionAll(bDF) 
**unionAll()**: Return a new DataFrame containing union of rows
in this frame and another frame
Runs completely at executors: **Very scalable and efficient**