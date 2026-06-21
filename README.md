# 基于 Spark 的气象大数据分布式处理与 GNADET 时空预测数据流水线的设计与实现

大数据编程课程设计-陆宇聪-20233001569;

基于 Apache Spark 对澳大利亚区域气象大数据（ERA5，2006-2018，6小时间隔，11个变量）进行分布式预处理、列式存储与 SQL 查询，并导出标准化数据供下游 GNADET 模型使用。

## 环境

- Python 3.11
- PySpark 3.5.3
- Java 17 (OpenJDK)
- NumPy / Pandas / PyArrow / Matplotlib
- Windows 下需 `hadoop/bin` 内的 winutils.exe、hadoop.dll

## 文件说明

- `spark_run_gis.py`：主脚本。加载三维张量并注入 Spark，做 repartition、Z-Score 归一化、Parquet 写出、缺失值校验、数据集导出与 Spark SQL 查询。
- `spark_viz.py`：读取 Parquet 结果绘制 4 张分析图，保存到 `figures/`。
- `GNADET/`：下游时空预测模型项目。

## 运行

```bash
conda activate GNADET
python spark_run_gis.py
python spark_viz.py
```

运行 `spark_run_gis.py` 时可访问 http://localhost:4040 查看 Spark Web UI。

## 输出

- `processed_weather.parquet`：归一化后的列式存储数据。
- `GNADET/data/X_GIS_spark.npy`：导出的标准化数据集，供模型读取。
- `figures/`：4 张可视化图。
