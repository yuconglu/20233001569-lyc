# -*- coding: utf-8 -*-
"""读取 Spark 产出的 Parquet 数据，绘制 4 张分析图保存到 figures 目录。"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 与主脚本一致，导入 pyspark 前先配置 JAVA_HOME / HADOOP_HOME
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PARQUET_OUT = os.path.join(SCRIPT_DIR, "processed_weather.parquet")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "GNADET", "data")


def infer_grid_shape(n_nodes):
    """根据 lat_coords 推断网格 H x W：开头与首个纬度值相等的个数即为 W。"""
    lat_path = os.path.join(DATA_DIR, "lat_coords_GIS.npy")
    if os.path.exists(lat_path):
        lat = np.load(lat_path)
        w = int(np.sum(lat == lat[0]))
        if w > 0 and n_nodes % w == 0:
            return n_nodes // w, w
    h = int(np.sqrt(n_nodes))
    while n_nodes % h != 0 and h > 1:
        h -= 1
    return h, n_nodes // h


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("Spark-Australia-Weather-Visualization")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(PARQUET_OUT)
    df.createOrReplaceTempView("weather")

    # 图1：相对湿度时间演化（按时间步聚合后等间隔抽样 500 点）
    evo = spark.sql(
        "SELECT time_step, avg(RH1000) AS avg_rh, max(RH1000) AS max_rh "
        "FROM weather GROUP BY time_step ORDER BY time_step"
    ).toPandas()
    step = max(1, len(evo) // 500)
    evo_s = evo.iloc[::step]
    plt.figure(figsize=(11, 5))
    plt.plot(evo_s["time_step"], evo_s["avg_rh"], label="平均相对湿度", color="#1f77b4")
    plt.plot(evo_s["time_step"], evo_s["max_rh"], label="最高相对湿度", color="#d62728", alpha=0.6)
    plt.xlabel("时间步 (6小时/步, 2006-2018)")
    plt.ylabel("相对湿度 RH1000 (%)")
    plt.title("澳洲全境相对湿度时间演化趋势（Spark 分布式聚合）")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_1_humidity_evolution.png"), dpi=150)
    plt.close()

    # 图2：单一时间步的温度空间分布热力图
    target_t = 10277
    snap = spark.sql(
        f"SELECT node_id, T2M FROM weather WHERE time_step = {target_t} ORDER BY node_id"
    ).toPandas()
    H, W = infer_grid_shape(len(snap))
    grid = snap["T2M"].to_numpy(dtype=np.float64).reshape(H, W)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, cmap="coolwarm", aspect="auto", origin="lower")
    plt.colorbar(im, label="地表温度 T2M (K)")
    plt.xlabel("经度方向网格 (W=%d)" % W)
    plt.ylabel("纬度方向网格 (H=%d)" % H)
    plt.title(f"澳洲区域地表温度空间分布热力图（time_step={target_t}）")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_2_temperature_heatmap.png"), dpi=150)
    plt.close()

    # 图3：归一化前后分布对比（1% 抽样）
    sample = df.sample(fraction=0.01, seed=42).select("T2M", "T2M_scaled").toPandas()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(sample["T2M"], bins=60, color="#ff7f0e")
    axes[0].set_title("归一化前 T2M (原始, 单位 K)")
    axes[0].set_xlabel("T2M (K)")
    axes[0].set_ylabel("频数")
    axes[1].hist(sample["T2M_scaled"], bins=60, color="#2ca02c")
    axes[1].set_title("Z-Score 归一化后 T2M_scaled (均值≈0)")
    axes[1].set_xlabel("T2M_scaled")
    fig.suptitle("Spark 分布式 Z-Score 归一化前后分布对比（1% 抽样）")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_3_zscore_hist.png"), dpi=150)
    plt.close()

    # 图4：各分布式分区的数据量
    part_counts = (
        df.repartition(8)
        .withColumn("pid", F.spark_partition_id())
        .groupBy("pid").count().orderBy("pid")
        .toPandas()
    )
    plt.figure(figsize=(9, 5))
    plt.bar(part_counts["pid"].astype(str), part_counts["count"], color="#9467bd")
    plt.xlabel("分布式分区编号 (spark_partition_id)")
    plt.ylabel("记录数")
    plt.title("Spark 8 个分布式分区的数据量分布（并行度示意）")
    for i, v in enumerate(part_counts["count"]):
        plt.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_4_partition_counts.png"), dpi=150)
    plt.close()

    spark.stop()
    print(f"4 张图已保存到：{FIG_DIR}")


if __name__ == "__main__":
    main()
