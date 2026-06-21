# -*- coding: utf-8 -*-
"""基于 Spark 的澳大利亚气象大数据分布式处理脚本。

读取裁剪好的三维气象张量 X_GIS.npy，注入 Spark 进行分布式归一化、
列式存储与 SQL 查询，并导出供下游 GNADET 模型使用的标准化数据集。
运行：conda run -n GNADET python spark_run_gis.py
"""

import os
import sys
import time

# 在导入 pyspark 之前设置 JAVA_HOME / HADOOP_HOME，否则 Windows 下写 Parquet 会报 winutils 错误
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_java_home = os.path.join(sys.prefix, "Library", "lib", "jvm")
if not os.path.exists(os.path.join(_java_home, "bin", "java.exe")):
    _java_home = os.path.join(sys.prefix, "Library")
os.environ["JAVA_HOME"] = _java_home

HADOOP_HOME = os.path.join(SCRIPT_DIR, "hadoop")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["PATH"] = (
    os.path.join(HADOOP_HOME, "bin") + os.pathsep
    + os.path.join(_java_home, "bin") + os.pathsep
    + os.environ.get("PATH", "")
)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

NPY_PATH = os.path.join(SCRIPT_DIR, "GNADET", "data", "X_GIS.npy")
PARQUET_OUT = os.path.join(SCRIPT_DIR, "processed_weather.parquet")
SPARK_NPY_OUT = os.path.join(SCRIPT_DIR, "GNADET", "data", "X_GIS_spark.npy")

# 11 个特征列名，顺序与原始数据一致
FEATURE_COLS = [
    "f0_T2M", "f1_T850", "f2_RH1000", "f3_Q1000", "f4_TP", "f5_TISR",
    "f6_Z500", "f7_U10", "f8_V10", "f9_ORO", "f10_LSM",
]


def main():
    # 1. 初始化 Spark 会话，local[*] 使用本机全部核心
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("Spark-Based-Australia-Weather-BigData-Analysis-System")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("SparkSession 已启动，可访问 http://localhost:4040 查看 Web UI")
    time.sleep(10)

    # 2. 加载三维张量并展平为二维表，注入 Spark 后切分为 8 个分区
    arr = np.load(NPY_PATH, mmap_mode="r")
    T, N, F = arr.shape
    print(f"原始矩阵形状 (T,N,F) = ({T}, {N}, {F})")

    feats_all = np.asarray(arr[:, :, :], dtype=np.float64).reshape(T * N, F)
    time_step = np.repeat(np.arange(T, dtype=np.int32), N)
    node_id = np.tile(np.arange(N, dtype=np.int32), T)

    pdf = pd.DataFrame(feats_all, columns=FEATURE_COLS)
    pdf.insert(0, "node_id", node_id)
    pdf.insert(0, "time_step", time_step)
    print(f"展平后数据表规模：{len(pdf):,} 行 x {pdf.shape[1]} 列")

    # repartition(8) 触发一次 shuffle，cache 缓存便于复用
    sdf_full = spark.createDataFrame(pdf).repartition(8).cache()
    sdf_full.createOrReplaceTempView("weather_full")

    # 核心 4 指标视图，列名更友好
    spark.sql(
        """
        CREATE OR REPLACE TEMP VIEW australia_weather AS
        SELECT time_step, node_id,
               f0_T2M AS T2M, f1_T850 AS T850,
               f2_RH1000 AS RH1000, f3_Q1000 AS Q1000
        FROM weather_full
        """
    )

    total_rows = sdf_full.count()
    print(f"已注入分布式表，共 {total_rows:,} 行，切分为 8 个分区")
    sdf_full.printSchema()
    time.sleep(8)

    # 3. 计算全局均值/标准差，做 Z-Score 归一化后写入 Parquet
    stats_row = spark.sql(
        """
        SELECT avg(T2M) AS t2m_mean, stddev(T2M) AS t2m_std,
               avg(RH1000) AS rh_mean, stddev(RH1000) AS rh_std
        FROM australia_weather
        """
    ).collect()[0]

    t2m_mean, t2m_std = float(stats_row["t2m_mean"]), float(stats_row["t2m_std"])
    rh_mean, rh_std = float(stats_row["rh_mean"]), float(stats_row["rh_std"])
    print(f"T2M  mean={t2m_mean:.4f} std={t2m_std:.4f}")
    print(f"RH1000 mean={rh_mean:.4f} std={rh_std:.4f}")

    scaled_df = spark.sql(
        f"""
        SELECT time_step, node_id, T2M, RH1000,
               (T2M - {t2m_mean}) / {t2m_std} AS T2M_scaled,
               (RH1000 - {rh_mean}) / {rh_std} AS RH_scaled
        FROM australia_weather
        """
    )
    scaled_df.write.mode("overwrite").parquet(PARQUET_OUT)
    print(f"归一化结果已写入 Parquet：{PARQUET_OUT}")
    time.sleep(8)

    # 4. 缺失值校验，并导出供 GNADET 模型读取的标准化数据集
    nan_select = ", ".join(
        [f"sum(CASE WHEN {c} IS NULL OR isnan({c}) THEN 1 ELSE 0 END) AS nan_{c}" for c in FEATURE_COLS]
    )
    nan_row = spark.sql(f"SELECT {nan_select} FROM weather_full").collect()[0]
    total_nan = sum(int(nan_row[f"nan_{c}"]) for c in FEATURE_COLS)
    print(f"11 个特征缺失值总数 = {total_nan}")

    # 按 (time_step, node_id) 排序后重建为 (T,N,F) 张量，导出为新文件
    ordered_pdf = spark.sql(
        f"SELECT {', '.join(FEATURE_COLS)} FROM weather_full ORDER BY time_step, node_id"
    ).toPandas()
    arr_out = ordered_pdf[FEATURE_COLS].to_numpy(dtype=np.float32).reshape(T, N, F)
    np.save(SPARK_NPY_OUT, arr_out)
    max_diff = float(np.nanmax(np.abs(arr_out - np.asarray(arr, dtype=np.float32))))
    print(f"已导出模型数据集 X_GIS_spark.npy，与原始数据最大绝对差 = {max_diff:.6e}")
    time.sleep(8)

    # 5. Spark SQL 查询：时空极值挖掘 + 相对湿度演化趋势
    print("场景1：地表温度最高的节点与时间步")
    spark.sql(
        "SELECT time_step, node_id, T2M FROM australia_weather ORDER BY T2M DESC LIMIT 1"
    ).show(truncate=False)

    print("场景2：按时间步统计平均/最高相对湿度（前5条）")
    spark.sql(
        """
        SELECT time_step,
               ROUND(avg(RH1000), 3) AS avg_relative_humidity,
               ROUND(max(RH1000), 3) AS max_relative_humidity
        FROM australia_weather
        GROUP BY time_step ORDER BY time_step LIMIT 5
        """
    ).show(truncate=False)

    # 6. 保持会话存活，便于查看 Web UI
    print("计算完成，保持存活 120 秒以便查看 http://localhost:4040")
    time.sleep(120)

    spark.stop()


if __name__ == "__main__":
    main()
