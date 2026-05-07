# Databricks notebook source
# MAGIC %pip install tensorflow==2.11 tensorflow-probability==0.19.0 tf_quant_finance

# COMMAND ----------

import numpy as np
import tensorflow as tf
import tf_quant_finance as tff
from tf_quant_finance.math import *
from tf_quant_finance.math.piecewise import *

from tf_quant_finance.models import *
from tf_quant_finance.models.generic_ito_process import *

import time

import scipy.optimize as optimize

import pyspark.pandas as ps

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ 1. モデルのセットアップ
# MAGIC まず、特定のモデルパラメータの関数となる *トイモデル*（汎用 Ito プロセス）を定義します。
# MAGIC
# MAGIC このモデルを使用して、特定の *満期* と *行使価格* を持つコールオプションのプライシング、およびインプライドボラティリティ（ブラックショールズによるプライスとインプライドボラティリティの1対1マッピングを使用）を計算できます。
# MAGIC
# MAGIC 目標は、*このモデル* から計算したインプライドボラティリティが市場から取得したインプライドボラティリティと一致するようなモデルパラメータの値を見つけることです（このプロセスをキャリブレーションと呼びます）。
# MAGIC
# MAGIC ## 1.1. モデル定義
# MAGIC [対数正規分布](https://en.wikipedia.org/wiki/Log-normal_distributio) の FX、[Vasicek](https://en.wikipedia.org/wiki/Vasicek_model) の金利、[ローカルボラティリティ](https://en.wikipedia.org/wiki/Local_volatility) の FX ボラティリティに従うトイモデルを定義します。

# COMMAND ----------

class TimeSeries:
  """ 時間の区分的関数を表すコンテナ。XLA と互換性あり """
  def __init__(self,jump_locations, values):
    self.jump_locations = jump_locations
    self.values = values

  def apply(self, input):
    res = self.values[-1]
    for idx in range(len(self.jump_locations)):
      curr_jump_loc = self.jump_locations[idx]
      if input <= curr_jump_loc:
        res = self.values[idx]
    return res

class BlackScholesWithVasicelAndLocalVol(GenericItoProcess):
  """対数正規 FX・Vasicek 金利・ローカルボラティリティ FX のトイモデル"""

  def __init__(self,
               # 金利1のモデルパラメータ
               kappa_rate_1, theta_rate_1, vol_rate_1, fwd_rate_1,
               # 金利2のモデルパラメータ
               kappa_rate_2, theta_rate_2, vol_rate_2, fwd_rate_2,
               # FX ボラティリティのモデルパラメータ
               jump_strikes, local_vol_fx,
               # FX モデルパラメータ
               fx_fwd,
               # 相関行列
               corr_matrix,
               # 離散化ジャンプ dt
               step_size,
               # 数値精度設定
               dtype=None):

    # 親クラス 'GenericItoProcess' の基本変数
    self._name = 'BlackScholesWithVasicelAndLocalVol'
    self._dim = 4
    self._dtype = dtype

    # 金利1のモデルパラメータ
    self.kappa_rate_1 = kappa_rate_1;
    self.theta_rate_1 = theta_rate_1;
    self.vol_rate_1 = vol_rate_1;
    self.fwd_rate_1 = fwd_rate_1;

    # 金利2のモデルパラメータ
    self.kappa_rate_2 = kappa_rate_2;
    self.theta_rate_2 = theta_rate_2;
    self.vol_rate_2 = vol_rate_2;
    self.fwd_rate_2 = fwd_rate_2;

    # FX ボラティリティのモデルパラメータ
    self.jump_strikes = jump_strikes
    self.log_jump_strikes = tf.math.log(jump_strikes)
    self.local_vol_fx = local_vol_fx;

    # FX モデルパラメータ
    self.fx_fwd = fx_fwd

    # 離散化ジャンプ dt
    self.step_size = step_size

    # 相関行列
    self.cholesky = tf.linalg.cholesky(corr_matrix);

  def _volatility_fn(self, t, x):

    vol_fx = x[..., 2]
    zeros = tf.zeros_like(vol_fx)
    ones = tf.ones_like(vol_fx)

    vol_rate_1 = self.vol_rate_1.apply(t) * ones
    vol_rate_2 = self.vol_rate_2.apply(t) * ones
    vol_vol_fx = zeros

    vol_array = [ vol_rate_1, vol_rate_2,vol_vol_fx, vol_fx]

    columns = [];
    for col in range(self._dim):
      current_columns = []
      for row in range(self._dim):
        current_columns.append(self.cholesky[row][col] * vol_array[row])
      columns.append(tf.stack(current_columns, -1))

    result_matrix = tf.stack(columns, -1)
    return result_matrix

  def _drift_fn(self, t, x):
    rate_factor_1 = x[..., 0]
    rate_factor_2 = x[..., 1]
    vol_fx = x[..., 2]
    log_fx = x[..., 3]

    fwd_rate_1_t = self.fwd_rate_1.apply(t)
    fwd_rate_2_t = self.fwd_rate_2.apply(t)

    rate_1 = fwd_rate_1_t + rate_factor_1
    rate_2 = fwd_rate_2_t + rate_factor_2

    lv_for_current_t = self.local_vol_fx.apply(t)
    lv_func = PiecewiseConstantFunc(jump_locations=self.log_jump_strikes, values=lv_for_current_t, dtype=dtype)
    new_vol_fx = lv_func(log_fx)

    self.old_vol = new_vol_fx

    drift_rate_1 = self.kappa_rate_1.apply(t) * (self.theta_rate_1.apply(t) - rate_factor_1)
    drift_rate_2 = self.kappa_rate_2.apply(t) * (self.theta_rate_2.apply(t) - rate_factor_2)
    drift_vol_fx = (new_vol_fx - vol_fx)/self.step_size
    drift_fx = (rate_1 - rate_2) - 0.5 * vol_fx * vol_fx

    drift = tf.stack([ drift_rate_1, drift_rate_2, drift_vol_fx, drift_fx ], -1)
    return drift

  def implied_vol(self,
                  option_strikes,
                  option_maturities,
                  num_samples):

    paths = self.sample_paths(
          option_maturities,
          num_samples=num_samples,
          initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=self._dtype.name),
          time_step=self.step_size,
          random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
          seed=[42, 56])

    number_of_strikes = len(option_strikes)
    implied_vols = []
    for maturity_idx in range(len(option_maturities)):
      curr_maturity = option_maturities[maturity_idx]
      curr_paths = paths[:,maturity_idx]
      curr_fwd = self.fx_fwd.apply(curr_maturity)
      df = tf.exp(-curr_paths[:,0]*curr_maturity)
      df_mean = tf.math.reduce_mean(df)
      fx = curr_fwd * tf.exp(curr_paths[:,3])

      prices = []
      for strike_idx in range(number_of_strikes):
        curr_strike = option_strikes[strike_idx]
        price = tf.math.reduce_mean(tf.maximum(tf.constant(0.0, dtype=self._dtype), (fx - curr_strike)))
        prices.append(price)

      implied_vols_for_curr_expiry = tff.black_scholes.implied_vol(
          prices=prices,
          strikes=option_strikes,
          expiries= [curr_maturity] * number_of_strikes,
          forwards=[curr_fwd] * number_of_strikes,
          discount_factors= [df_mean] * number_of_strikes,
          is_call_options=True)

      implied_vols_for_curr_expiry_parsed = []
      for option_idx in range(0, len(implied_vols_for_curr_expiry)):
        curr_implied_vol = implied_vols_for_curr_expiry[option_idx]
        if np.isnan(curr_implied_vol):
          curr_implied_vol = min(0, prices[option_idx].numpy() - max(curr_fwd - option_strikes[option_idx], 0))
        implied_vols_for_curr_expiry_parsed.append(curr_implied_vol)

      implied_vols.append(implied_vols_for_curr_expiry_parsed)

    return implied_vols

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2. モデルの初期化
# MAGIC
# MAGIC ダミーのモデルパラメータ値でモデルを初期化します

# COMMAND ----------

# ダミーのモデルパラメータ値でモデルをインスタンス化します

dtype=tf.float64
jump_locations = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1])
jump_strikes = np.array([0.95, 0.99 , 1, 1.001])

# これは変更し続けるモデルパラメータの一つです
lv_surface = [[0.6, 0.36, 0.246, 0.546, 0.7978],
              [0.68, 0.37, 0.112, 0.476, 0.8987],
              [0.65, 0.33, 0.224, 0.676, 0.764],
              [0.634, 0.336, 0.332, 0.566, 0.907],
              [0.76, 0.456, 0.152, 0.601, 0.67],
              [0.676, 0.3745, 0.1632, 0.623, 0.788],
              [0.687, 0.243, 0.2123, 0.622, 0.7576],
              [0.576, 0.473, 0.253, 0.556, 0.7123],
              [0.56, 0.346, 0.2252, 0.786, 0.867],
              [0.786, 0.354, 0.2691, 0.634, 0.7545]]

model = BlackScholesWithVasicelAndLocalVol(
    kappa_rate_1 = TimeSeries(jump_locations=jump_locations, values=np.array([0.05, 0.02, 0.07, 0.02, 0.04, 0.06, 0.07, 0.02, 0.08, 0.09],dtype=dtype.name)),
    theta_rate_1 = TimeSeries(jump_locations=jump_locations, values=np.array([1.2, 2, 1.5, 1.7, 1, 1.3, 1.9, 3.0, 2.5, 1.0],dtype=dtype.name)),
    vol_rate_1 = TimeSeries(jump_locations=jump_locations, values=np.array([0.11, 0.15, 0.9, 0.15,  0.15, 0.3, 0.15, 0.2, 0.17, 0.4],dtype=dtype.name)),
    fwd_rate_1 = TimeSeries(jump_locations=jump_locations, values=np.array([0.02, 0.021, 0.022, 0.023, 0.019, 0.018, 0.23, 0.025, 0.015, 0.019],dtype=dtype.name)),
    kappa_rate_2 = TimeSeries(jump_locations=jump_locations, values=np.array([0.05, 0.02, 0.07, 0.02, 0.04, 0.06, 0.07, 0.02, 0.08, 0.09],dtype=dtype.name)),
    theta_rate_2 = TimeSeries(jump_locations=jump_locations, values=np.array([1.2, 2, 1.5, 1.7, 1, 1.3, 1.9, 3.0, 2.5, 1.0],dtype=dtype.name)),
    vol_rate_2 = TimeSeries(jump_locations=jump_locations, values=np.array([0.11, 0.15, 0.9, 0.15,  0.15, 0.3, 0.15, 0.2, 0.17, 0.4],dtype=dtype.name)),
    fwd_rate_2 = TimeSeries(jump_locations=jump_locations, values=np.array([0.025, 0.051, 0.052, 0.053, 0.069, 0.068, 0.53, 0.055, 0.075, 0.049],dtype=dtype.name)),
    jump_strikes = jump_strikes,
    fx_fwd = TimeSeries(jump_locations=jump_locations, values=np.array([1.0, 1.002, 0.998, 1.04, 1.035, 1.01, 0.999, 0.998, 1.003, 1.01],dtype=dtype.name)),
    local_vol_fx = TimeSeries(jump_locations=jump_locations, values=lv_surface),
    step_size=0.01,
    corr_matrix = tf.constant([[1.0, 0.2, 0.0, 0.3],
                               [0.2, 1.0, 0.0, 0.3],
                               [0.0, 0.0, 1.0, 0.8],
                               [0.3, 0.3, 0.8, 1.0]], dtype=dtype),
    dtype=dtype
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3. モデルのキャリブレーション
# MAGIC
# MAGIC シンプルに保つため、市場から得られるインプライドボラティリティ（ターゲット）に最も近くなる *lv_surface* を探します。
# MAGIC
# MAGIC 目的関数の入力次元は 50（10 満期 × 5 行使価格の 'local_vol_fx' サーフェス）であり、出力次元も同様に 50 であるため、以下のセルの実行は非常に時間がかかります。（実行を確認したい場合は、*option_maturities* をサイズ 1 の配列に縮小できます）

# COMMAND ----------

start_time_ = time.time()

# コールオプションを再プライシングする満期のリスト
option_maturities = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]

number_of_maturities = len(option_maturities)

# 上記の各満期に対するコールオプションの行使価格のリスト
option_strikes = np.array([0.95, 0.99 , 1, 1.001, 1.05])
number_of_strikes = len(option_strikes)

# モンテカルロのパス数
num_samples = 10

# 市場から得られるインプライドボラティリティ（ターゲット）
implied_vol_target = np.array([ [ 0.36165047, 0.38006929, 0.38622322, 0.38688029, 0.42223257 ],
       [ 0.49930734, 0.52086326, 0.52665937, 0.52725076, 0.5561736 ],
       [ 0.53926329, 0.56007775, 0.56558462, 0.56613531, 0.59324591 ],
       [ 0.55870456, 0.57847004, 0.5836174 , 0.58413716, 0.60937169 ],
       [ 0.58499617, 0.60206912, 0.60646861, 0.60690938, 0.62796077 ],
       [ 0.61917847, 0.63211846, 0.63541694, 0.63574893, 0.65183933 ],
       [ 0.64890557, 0.65896691, 0.66159441, 0.66185734, 0.67469815 ],
       [ 0.68506316, 0.69196701, 0.69382324, 0.69401138, 0.70335816 ],
       [ 0.74270218, 0.74599681, 0.74702723, 0.74713323, 0.75264819 ],
       [ 0.8005369 , 0.79634181, 0.79554414, 0.79546886, 0.7927459 ] ])


implied_vol_target = implied_vol_target[:number_of_maturities]

# LV サーフェスの初期推定値（モデルパラメータの一つ）
lv_surface_init_guess = np.array([[0.6, 0.36, 0.246, 0.546, 0.7978],
                         [0.68, 0.37, 0.112, 0.476, 0.8987],
                         [0.65, 0.33, 0.224, 0.676, 0.764],
                         [0.634, 0.336, 0.332, 0.566, 0.907],
                         [0.76, 0.456, 0.152, 0.601, 0.67],
                         [0.676, 0.3745, 0.1632, 0.623, 0.788],
                         [0.687, 0.243, 0.2123, 0.622, 0.7576],
                         [0.576, 0.473, 0.253, 0.556, 0.7123],
                         [0.56, 0.346, 0.2252, 0.786, 0.867],
                         [0.786, 0.354, 0.2691, 0.634, 0.7545]])

lv_surface_init_guess = lv_surface_init_guess[:number_of_maturities]

# 目的関数を定義します
def objective_fn(lv_surface_guess_flatened):
  lv_surface_guess = np.split(lv_surface_guess_flatened, number_of_maturities)
  model.local_vol_fx = TimeSeries(jump_locations=option_maturities[:-1], values=lv_surface_guess)
  implied_vols_from_model = model.implied_vol(option_maturities=option_maturities, option_strikes=option_strikes,num_samples=num_samples)
  errors = np.array((implied_vol_target - implied_vols_from_model)).flatten() * 1e4 # bps 単位
  # print("誤差:", errors)
  return errors


roots = optimize.least_squares(objective_fn,
                      x0= lv_surface_init_guess.flatten(),
                      ftol=0.05,
                      xtol=None,
                      gtol=None,)

roots.x # モデルで使用すべき最適な lv_surface。implied_vols_from_model が 'implied_vol_target' に最も近くなります


end_time_ = time.time()
durr_ = end_time_ - start_time_

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ 2: 機械学習によるキャリブレーション時間の短縮
# MAGIC
# MAGIC 次に、機械学習を使用してキャリブレーションの時間計算量を削減します。上記のキャリブレーションにおける主なボトルネックは、オプティマイザ内での *model.implied_vol* 関数の繰り返し呼び出しであり、モンテカルロシミュレーションのために非常に重い処理です。
# MAGIC
# MAGIC 何らかの方法でその関数（ローカルボラティリティモデルパラメータをインプライドボラティリティにマッピングする関数）を学習できれば、オプティマイザの内部でその関数を代替として使用でき、大幅に高速化できます！
# MAGIC
# MAGIC ## 2.1. 学習データの生成
# MAGIC 最初のステップは学習データセットの生成です

# COMMAND ----------

# コールオプションを再プライシングする満期のリスト
option_maturities = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
number_of_maturities = len(option_maturities)

# 上記の各満期に対するコールオプションの行使価格のリスト
option_strikes = np.array([0.95, 0.99 , 1, 1.001, 1.05])
number_of_strikes = len(option_strikes)

# モンテカルロのパス数
num_samples = 4_000

training_samples = 1_000
np.random.seed(0)

# COMMAND ----------

schema_ = []
for maturity in option_maturities:
  for strike in option_strikes:
    schema_.append(f'{maturity}_{strike}')

# COMMAND ----------

features = []
number_of_features = number_of_maturities * number_of_strikes
for i in range(number_of_features):
  features.append(np.random.uniform(0.0, 1.0, training_samples))
features = np.array(features).transpose()
#print("特徴量", features.shape)


import time
start_time = time.time()

# 各サーフェスを代入して、そのモデルからインプライドボラティリティを生成します
labels = []
for feature in features:
  lv_surface = np.split(feature, number_of_maturities)
  model.local_vol_fx = TimeSeries(jump_locations=option_maturities[:-1], values=lv_surface)
  implied_vols_from_model = model.implied_vol(option_maturities=option_maturities, option_strikes=option_strikes,num_samples=num_samples)
  labels.append(np.array(implied_vols_from_model).flatten())
labels = np.array(labels)
#print("ラベル", labels.shape)

# COMMAND ----------

assert len(schema_) == features.shape[1] == labels.shape[1], 'スキーマの長さとフィーチャー/ラベルの数に不一致があります。'

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2.2. フィーチャーとラベルを Koalas DataFrame に変換
# MAGIC
# MAGIC Koalas は pandas の代替として機能します。データサイエンティストに広く使われる pandas は、Python でのデータ構造とデータ分析ツールを提供する Python パッケージです。しかし、pandas はビッグデータにスケールしません。Koalas は Apache Spark 上で動作する pandas 互換の API を提供することでこのギャップを埋めます。Koalas は pandas ユーザーだけでなく PySpark ユーザーにも有用で、例えば PySpark DataFrame から直接データをプロットするなど、PySpark では難しいタスクをサポートします。
# MAGIC
# MAGIC https://docs.databricks.com/languages/koalas.html

# COMMAND ----------

features_ps = ps.DataFrame(features, columns=schema_).reset_index()
labels_ps = ps.DataFrame(labels, columns=schema_).reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC # ステップ 3: 生成したフィーチャーとラベルを Databricks Feature Store に保存

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Databricks Feature Store を使用する理由
# MAGIC Databricks Feature Store は Databricks の他のコンポーネントと完全に統合されています。
# MAGIC
# MAGIC - **リネージ**: Databricks Feature Store でフィーチャーテーブルを作成すると、フィーチャーテーブルの作成に使用したデータソースが保存され、アクセス可能になります。フィーチャーテーブルの各フィーチャーについて、そのフィーチャーを使用するモデル、ノートブック、ジョブ、エンドポイントにもアクセスできます。
# MAGIC
# MAGIC - **探索可能性**: Databricks ワークスペースからアクセスできる Databricks Feature Store UI を使用して、既存のフィーチャーを参照・検索できます。
# MAGIC
# MAGIC - **モデルスコアリングとサービングとの統合**: Databricks Feature Store のフィーチャーを使用してモデルをトレーニングすると、モデルはフィーチャーメタデータとともにパッケージ化されます。バッチスコアリングやオンライン推論にモデルを使用する場合、Feature Store からフィーチャーが自動的に取得されます。呼び出し元はフィーチャーについて知る必要も、新しいデータのスコアリングのためにフィーチャーをルックアップ・結合するロジックを含める必要もありません。これによりモデルのデプロイと更新が大幅に簡単になります。
# MAGIC
# MAGIC https://docs.databricks.com/applications/machine-learning/feature-store/index.html
# MAGIC
# MAGIC 以下のステップでは、このアクセラレーター用の Feature Store を作成または置き換えます。

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Feature Store データベースを作成します
# MAGIC CREATE DATABASE IF NOT EXISTS feature_store_implied_volatility;

# COMMAND ----------

try:
  fs.drop_table(
    name="feature_store_implied_volatility.features" # Feature Store テーブルが存在しない場合は ValueError をスローします
  )
except ValueError:
  pass

fs.create_table(
    name="feature_store_implied_volatility.features",
    primary_keys = ['index'],
    df = features_ps.to_spark(),
    description = 'インプライドボラティリティのフィーチャーセット')

# COMMAND ----------

try:
  fs.drop_table(
    name="feature_store_implied_volatility.labels" # Feature Store テーブルが存在しない場合は ValueError をスローします
  )
except ValueError:
  pass

fs.create_table(
    name="feature_store_implied_volatility.labels",
    primary_keys = ['index'],
    df = labels_ps.to_spark(),
    description = 'インプライドボラティリティのラベルセット')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3.1. 生成データの表示と分布の確認

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC use feature_store_implied_volatility

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3.1.1. 生成データの可視化と探索
# MAGIC
# MAGIC Databricks Notebooks にはダッシュボード機能が組み込まれています（以下で確認できます）。Databricks Feature Store に保存したフィーチャーとラベルを素早く可視化できます。以下は同じ満期における様々な行使価格レベルの散布図です（生成データ）。チャートには LOESS 回帰も表示でき、生成データの残差の分布についてさらなる情報を得ることができます。

# COMMAND ----------

display(spark.sql('select * from labels'))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3.1.2. データプロファイリング
# MAGIC
# MAGIC Databricks Notebooks にはデータプロファイリング機能が組み込まれています。以下のセルでは、サードパーティツールを使用したり追加コードを書いたりすることなく、新しく生成されたフィーチャーとラベルの多くの統計情報を確認できます。

# COMMAND ----------

display(spark.sql('select * from labels'))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # ステップ 4: R を使用した生成データの統計的問題の確認
# MAGIC
# MAGIC R ライブラリを使用して、新しく生成されたデータの不均一分散を自動的に検定します。

# COMMAND ----------

features_ps.iloc[:, 1:].to_spark().createOrReplaceTempView('IVfeatures_view')

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC library(SparkR)
# MAGIC sql("REFRESH TABLE IVfeatures_view")
# MAGIC features_df_r <- sql("SELECT * FROM IVfeatures_view")

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC head(features_df_r)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # ステップ 5: 各ラベルの ML モデルをトレーニング
# MAGIC
# MAGIC **このソリューションアクセラレーターの次のノートブック**を参照してください

# COMMAND ----------

