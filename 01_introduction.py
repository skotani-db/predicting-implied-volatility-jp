# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # クオンツリサーチャーの一日
# MAGIC このソリューションでは、クオンツリサーチャーが日常的に行う主要なタスクを再現します。具体的には、1. 学術論文を基にした資産配分や新しいリスク調整済みパフォーマンス指標（非標準リスクを考慮）などの定量モデルの開発、および 2. これらのモデルを検証するための実験設計です。
# MAGIC
# MAGIC 以下の学術論文（Deep Learning Volatility, 2019, Horvath et al）のロジックを実装し、提案されたモデルを構築し、各種 Databricks サービスを使用してプロダクション化を行います（アーキテクチャはノートブック末尾を参照）。

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # "Deep Learning Volatility"
# MAGIC ## *ディープニューラルネットワークの視点から見た（ラフ）ボラティリティモデルにおけるプライシングとキャリブレーション*
# MAGIC
# MAGIC
# MAGIC この論文の目的は、他の手段では表現が困難または評価に時間がかかる複雑なプライシング関数をオフライン近似するニューラルネットワークを構築することです。これにより、デリバティブ契約の低速なプライシングによるキャリブレーションのボトルネックを解消します。
# MAGIC
# MAGIC <img src="https://bbb-databricks-demo-assets.s3.amazonaws.com/Screenshot+2022-07-24+at+5.09.41+PM.png" style="float: left" width="750px" />
# MAGIC
# MAGIC 論文リンク - **https://arxiv.org/pdf/1901.09647.pdf**

# COMMAND ----------

# MAGIC %md
# MAGIC # Databricks Lakehouse が "Deep Learning Volatility" に適している理由
# MAGIC
# MAGIC 1. **スケール**: [Databricks Runtime](https://docs.databricks.com/runtime/mlruntime.html) と [Photon](https://www.databricks.com/product/photon) のバーストキャパシティにより、論文で提案されている非常に計算負荷の高い合成データ生成アルゴリズムを、迅速かつコスト効率よく実行できます。
# MAGIC 2. **DataOps - Feature Store**: [Databricks Feature Store](https://databricks.com/blog/2022/04/29/announcing-general-availability-of-databricks-feature-store.html) は、生成されたフィーチャーを高効率なフォーマット（Delta テーブル）で保持し、オンライン・オフライン両方のトレーニングに対応します。アルゴリズムの再実行を不要にし、追加コストを回避できます。
# MAGIC 3. **R と Python の連携**: 合成データを生成した後、データの品質を確認し、回帰モデルのトレーニングに使用するデータの統計的な問題（不均一分散など）を特定する必要があります。この目的には、統計的タスク専用に設計された R パッケージを使用します。これにより、同一の [Interactive Databricks Notebook](https://databricks.com/product/collaborative-notebooks) 内で Python と R を使用し、R ライブラリを書き直したり追加のノートブックやクラスターをプロビジョニングすることなく、それぞれの言語の強みを活かせることが実証されます。
# MAGIC     - Databricks Notebook のダッシュボード機能を使用して、生成データのペアプロットを可視化します。
# MAGIC     - Databricks Notebooks の自動 *データプロファイル* 機能により、生成データの分布と全体的な品質を確認します。
# MAGIC 4. **MLOps - ML 実験とデプロイ**: この論文では多数のモデルを同時にトレーニングする必要があります。これだけ多くのモデルを同時に扱う場合、各モデルのハイパーパラメータチューニング、計算時間、特徴量選択などを追跡することは非常に非効率になります。そこで [MLFlow](https://databricks.com/product/managed-mlflow) の実験トラッキングが役立ち、モデル開発を効率化します（ノートブック *Implied Volatility Prediction - 2. ML* を参照）。
# MAGIC 5. **プロダクション化**: 最後に、[Databricks Workflows](https://databricks.com/blog/2022/05/10/introducing-databricks-workflows.html) を使用してエンドツーエンドの実行とデプロイをオーケストレーションします。Databricks Workflows は、あらゆるデータ・分析・AI ニーズに対応する完全マネージドのオーケストレーションサービスです。基盤となる Lakehouse プラットフォームとの緊密な統合により、任意のクラウド上で信頼性の高い本番ワークロードを作成・実行しながら、エンドユーザーにとってシンプルな、深く一元化された監視を提供します（ノートブック *Implied Volatility Prediction - 4. Productionalizing* を参照）。

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # アーキテクチャ
# MAGIC
# MAGIC 以下は、このソリューションで構築するアーキテクチャです。
# MAGIC
# MAGIC <img src='https://bbb-databricks-demo-assets.s3.amazonaws.com/IV_arch.png' style="float: left" width="1250px" />
