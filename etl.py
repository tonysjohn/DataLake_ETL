import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, desc, sum as Fsum, row_number, to_timestamp, from_unixtime, trim, lower
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R, StructField as Fld, StringType as Str, IntegerType as Int, DoubleType as Dbl, TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS','AWS_SECRET_ACCESS_KEY')


def table_schema():
    """
    Create shema objects for tables 
    
    args:
        None
    
    return:
        schema : dictionary with schema objects of multiple table
        
    raises:
        None
    
    """
    schema = {}
    schema['songs'] =  R([
                        Fld("song_id",Str(), nullable= False),
                        Fld("title",Str()),
                        Fld("artist_id",Str(), nullable = False),
                        Fld("year",Int()),
                        Fld("duration",Dbl())])
    
    schema['artists'] =  R([
                        Fld("artist_id",Str(), nullable= False),
                        Fld("artist_name",Str()),
                        Fld("location",Str()),
                        Fld("latitude",Dbl()),
                        Fld("longitude",Dbl())])
    
    schema['users'] =  R([
                        Fld("user_id",Str(), nullable = False),
                        Fld("first_name",Str()),
                        Fld("last_name",Str()),
                        Fld("gender",Str()),
                        Fld("level",Str(), nullable = False)])

    schema['time'] =  R([
                        Fld("start_time",TimestampType(), nullable= False),
                        Fld("hour",Int()),
                        Fld("day",Int()),
                        Fld("week",Int()),
                        Fld("month",Int()),
                        Fld("year",Int()),
                        Fld("weekday",Int())])

    schema['songplays'] =  R([
                        Fld("start_time",TimestampType(), nullable= False),
                        Fld("user_id",Str(), nullable= False),
                        Fld("level",Str()),
                        Fld("song_id",Str()),
                        Fld("artist_id",Str()),
                        Fld("sessionId",Str(), nullable= False),
                        Fld("location",Str()),
                        Fld("user_agent",Str()),
                        Fld("ts_month",Int()),
                        Fld("ts_year",Int())])
    return schema

def create_spark_session():
    """
    Create spark session and limit number of shuffle partitions to limit write partitions
    
    args:
        None
    
    return:
        spark : Spark session
        
    raises:
        None
    
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions",2)
    return spark


def process_song_data(spark, input_data, output_data):
    """
    ETL song data from S3 to DataLake
    
    Loads song data from input_data location to  output_data location. The song data is manipulated and schema enforced to 
    create songs table and artists table. 
    
    args:
        spark : Spark Session
        input_data : general input file location
        output_data : general output file location
    
    return:
        None
        
    raises:
        None
    
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/A/A/A')
    song_data_write = os.path.join(output_data,f'songs/songs_{datetime.now().strftime("%d-%m-%Y")}.parquet')
    artist_data_write = os.path.join(output_data,f'artists/artists_{datetime.now().strftime("%d-%m-%Y")}.parquet')
    
    # reading schema
    schema = table_schema()
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    song_columns = ['song_id', 'title', 'artist_id', 'year', 'duration']
    songs_table = spark.createDataFrame(df[song_columns] \
                                        .dropDuplicates().rdd, schema=schema['songs'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode('overwrite').partitionBy('year', 'artist_id').parquet(song_data_write)

    # extract columns to create artists table
    artist_columns = ['artist_id', 'artist_name', 'artist_location as location', 'artist_latitude as latitude', 'artist_longitude as longitude']
    artists_table = spark.createDataFrame(df.selectExpr(artist_columns) \
                                              .dropDuplicates().rdd,
                                          schema=schema['artists'])
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(artist_data_write)


def process_log_data(spark, input_data, output_data):
    """
    ETL log data from S3 to DataLake
    
    Loads log data from input_data location to  output_data location. The log data is manipulated and schema enforced to 
    create users table, time table and songplay table. 
    
    args:
        spark : Spark Session
        input_data : general input file location
        output_data : general output file location
    
    return:
        None
        
    raises:
        None
    
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log-data/*/*/*.json')
    songplay_data_write = os.path.join(output_data,f'songplays/songplays_{datetime.now().strftime("%d-%m-%Y")}.parquet')
    users_data_write = os.path.join(output_data,f'users/users_{datetime.now().strftime("%d-%m-%Y")}.parquet')
    time_data_write = os.path.join(output_data,f'time/time_{datetime.now().strftime("%d-%m-%Y")}.parquet')
    
    # reading schema
    schema = table_schema()
    
    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter("page == 'NextSong'")

    # extract columns for users table
    user_columns = ['userId as user_id', 'firstName as first_name', 'lastName as last_name', 'gender', 'level']
    window = Window.partitionBy('userId').orderBy(desc('ts'))
    users_table = spark.createDataFrame(df.withColumn('LastLevel', row_number().over(window)) \
                                        .filter(col('LastLevel')==1) \
                                        .selectExpr(*user_columns) \
                                        .dropDuplicates()
                                        .rdd, schema=schema['users'])

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(users_data_write)
    
    # extract columns to create time table
    time_columns=['start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday']
    time_table = spark.createDataFrame(df.withColumn('start_time', to_timestamp(col('ts')/1000)) \
                                         .withColumn('hour', F.hour('start_time')) \
                                         .withColumn('day', F.dayofmonth('start_time')) \
                                         .withColumn('week', F.weekofyear('start_time')) \
                                         .withColumn('month', F.month('start_time')) \
                                         .withColumn('year', F.year('start_time')) \
                                         .withColumn('weekday', F.dayofweek('start_time')) \
                                         .dropDuplicates() \
                                         .select(time_columns)
                                         .rdd, schema=schema['time'])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').partitionBy('year', 'month').parquet(time_data_write)

    # read in song data to use for songplays table
    song_data = os.path.join(input_data, 'song_data/A/A/A')
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table 
    songplay_columns = ['start_time', 'userId as user_id', 'level', 'song_id', \
                        'artist_id', 'sessionId as session_id', 'location', 'userAgent as user_agent', \
                        'ts_month', 'ts_year']
    songplays_table = spark.createDataFrame(df.withColumn('start_time', to_timestamp(col('ts')/1000)) \
                                             .withColumn('ts_month', F.month('start_time')) \
                                             .withColumn('ts_year', F.year('start_time')) \
                                             .join(song_df, \
                                                   on=[lower(trim(df.song)) == lower(trim(song_df.title)) \
                                                               , lower(trim(df.artist)) == lower(trim(song_df.artist_name))], \
                                                   how='left') \
                                             .selectExpr(*songplay_columns)
                                             .rdd, schema=schema['songplays'])

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode('overwrite').partitionBy('ts_year', 'ts_month').parquet(songplay_data_write)


def main():
    """
    Main function to orchestrate ETL load. Loads song data and log data to Amazon S3 bucket
    
    args:
        None
    
    return:
        None
        
    raises:
        None
    
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://neela-bucket-1/sparkify"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
