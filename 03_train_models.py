# Databricks notebook source
import pyspark.pandas as ps
from databricks import feature_store
import mlflow
import databricks.automl_runtime
import time

from mlflow.tracking import MlflowClient
import os
import uuid
import shutil
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ 1: Databricks Feature Store からデータを読み込む

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
features_df = fs.read_table('feature_store_implied_volatility.features')
labels_df = fs.read_table('feature_store_implied_volatility.labels')

# COMMAND ----------

features_df = features_df.toPandas()
labels_df =  labels_df.toPandas()

# COMMAND ----------

features_df = features_df.iloc[:, 1:]
features_df['target'] = labels_df['0.05_0.95']

# COMMAND ----------

target_col = "target"
training_col = list(features_df.columns)[:-1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### サポートされる列の選択
# MAGIC サポートされる列のみを選択します。これにより、トレーニングで使用されない余分な列を持つデータセットでも予測できるモデルをトレーニングできます。
# MAGIC `[]` はパイプラインでドロップされます。これらの列がドロップされる理由については、AutoML 実験ページの「アラート」タブを参照してください。

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = training_col
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 前処理

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### 数値列
# MAGIC
# MAGIC 数値列の欠損値はデフォルトで平均値で補完されます。

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), training_col))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

transformers.append(("numerical", numerical_pipeline, training_col))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ 2: 学習・検証・テスト分割
# MAGIC 入力データを 3 セットに分割します:
# MAGIC - 学習セット（モデルの学習に使用するデータセットの 60%）
# MAGIC - 検証セット（モデルのハイパーパラメータチューニングに使用するデータセットの 20%）
# MAGIC - テストセット（未知のデータセットに対するモデルの真の性能を報告するために使用するデータセットの 20%）

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = features_df.drop([target_col], axis=1)
split_y = features_df[target_col]

# 学習データを分割します
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=224145758)

# 残りのデータを検証とテストに均等に分割します
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=224145758)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ 3: 回帰モデルのトレーニング
# MAGIC - 実行を追跡するために関連メトリクスを MLflow に記録します
# MAGIC - すべての実行は [この MLflow 実験](#mlflow/experiments/3624174420594729/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false) に記録されます
# MAGIC - モデルパラメータを変更して学習セルを再実行すると、MLflow 実験に別のトライアルが記録されます
# MAGIC - チューニング可能なハイパーパラメータの全リストを確認するには、以下のセルの出力を確認してください

# COMMAND ----------

from xgboost import XGBRegressor

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display='diagram')

xgb_regressor = XGBRegressor(
  colsample_bytree=0.6385875217228281,
  learning_rate=0.10603131742006,
  max_depth=6,
  min_child_weight=8,
  n_estimators=148,
  n_jobs=100,
  subsample=0.5203076979604147,
  verbosity=0,
  random_state=224145758,
)

model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("regressor", xgb_regressor),
])

# 検証データセットを変換するための別パイプラインを作成します。早期停止に使用します。
pipeline = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])

mlflow.sklearn.autolog(disable=True)
pipeline.fit(X_train, y_train)
X_val_processed = pipeline.transform(X_val)

# COMMAND ----------

try:
  username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
except:
  username = str(uuid.uuid1()).replace("-", "")

# COMMAND ----------

# 入力サンプル、メトリクス、パラメータ、モデルの自動ロギングを有効化します
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

#experiment_id_ = mlflow.create_experiment("Implied Volatility Prediction")
experiment_name = experiment_name = f'/Users/{username}/implied_volatility'
#mlflow.set_experiment(experiment_name)
try:
  experiment_id_ = mlflow.create_experiment(experiment_name)
except:
  experiment_id_ = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id_, run_name=f"implied_volatility_{time.time()}") as mlflow_run:
    model.fit(X_train, y_train, regressor__early_stopping_rounds=5, regressor__eval_set=[(X_val_processed,y_val)], regressor__verbose=False)

    # 学習メトリクスは MLflow の自動ロギングによって記録されます
    # 検証セットのメトリクスを記録します
    xgb_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # テストセットのメトリクスを記録します
    xgb_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # 記録されたメトリクスを表示します
    xgb_val_metrics = {k.replace("val_", ""): v for k, v in xgb_val_metrics.items()}
    xgb_test_metrics = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}
    display(pd.DataFrame([xgb_val_metrics, xgb_test_metrics], index=["validation", "test"]))

# COMMAND ----------

display(spark.read.format("mlflow-experiment").load(experiment_id_))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### MLFlow UI
# MAGIC
# MAGIC MLFlow には UI コンポーネントがあり、実験の追跡が非常に簡単です。
# MAGIC
# MAGIC <img src='https://bbb-databricks-demo-assets.s3.amazonaws.com/Screenshot+2022-07-29+at+1.55.57+PM.png'  style="float: left" width="1150px" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ 4: フィーチャー重要度
# MAGIC
# MAGIC SHAP は機械学習モデルを説明するためのゲーム理論的アプローチであり、フィーチャーとモデル出力の関係の
# MAGIC サマリープロットを提供します。フィーチャーは重要度の降順にランク付けされ、
# MAGIC 影響度/色はフィーチャーとターゲット変数の相関を示します。
# MAGIC - SHAP フィーチャー重要度の生成はメモリ集約的な処理であるため、AutoML がメモリ不足にならずにトライアルを
# MAGIC   実行できるよう、デフォルトで SHAP は無効化されています。<br />
# MAGIC   以下で定義されたフラグを `shap_enabled = True` に設定してこのノートブックを再実行すると、SHAP プロットが表示されます。
# MAGIC - 各トライアルの計算オーバーヘッドを削減するため、説明のために検証セットから 1 サンプルがサンプリングされます。<br />
# MAGIC   より詳細な結果を得るには、説明のサンプルサイズを増やすか、独自のサンプルを提供してください。
# MAGIC - SHAP は null を含むデータを使用したモデルを説明できません。データセットに null が含まれる場合、
# MAGIC   バックグラウンドデータと説明対象のサンプルの両方がモード（最頻値）で補完されます。
# MAGIC   これにより、補完されたサンプルが実際のデータ分布と一致しない場合、計算された SHAP 値に影響が生じます。
# MAGIC
# MAGIC Shapley 値の読み方については、[SHAP ドキュメント](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) を参照してください。

# COMMAND ----------

# このフラグを True に設定してノートブックを再実行すると、SHAP プロットが表示されます
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # SHAP Explainer のバックグラウンドデータをサンプリングします。分散を減らすにはサンプルサイズを増やしてください。
    train_sample = X_train.sample(n=min(100, X_train.shape[0]))

    # 説明のために検証セットから数行をサンプリングします。より詳細な結果を得るにはサンプルサイズを増やしてください。
    example = X_val.sample(n=min(10, X_val.shape[0]))

    # 検証セットのサンプルに対してフィーチャー重要度を説明するために Kernel SHAP を使用します。
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example)
