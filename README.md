## Acknowledgements
このリポジトリはエラー修正と改善のために[astrid-string-selectivity](https://github.com/saravanan-thirumuruganathan/astrid-string-selectivity)をフォークしたものです

## 手順

まず`requirements.txt`に記載されているライブラリをインストールしてください

Astrid-Embedは、全体として重要な4つのステップから構成されています。


1. 必要なデータセットを準備します。生データをAstrid-Embedが想定する形式に処理するコードは`prepare_datasets.py`にあります。`load_and_save_dblp`、`load_imdb_movie_titles`、`load_and_save_tpch`の各関数がファイルを生成します。便宜上、これらのファイルも`datasets`フォルダに配置されます。

2. 頻度とトリプレットを計算します。次のステップは、接頭辞、接尾辞、部分文字列の頻度を取得するための要約データ構造を作成することです。これは、`prepare_datasets.py`の `prepare_dataset`関数によって行われます。`dataset_path`、`dataset_prefix`の 2 つの引数で呼び出します。たとえば、`prepare_dataset("datasets/dblp/", "dblp_authors")`のように呼び出します。Astrid-Embed は、各行に 1 つの文字列を含む入力ファイルを想定しています。そのため、この関数は`datasets/dblp/`フォルダー内の`dblp_authors.csv`を探します。次に、6 つのファイルを生成します。`dblp_authors_prefix_counts.csv`、`dblp_authors_substring_counts.csv`、`dblp_authors_suffix_counts.csv`は、文字列、選択性の 2 つの列を含む csv ファイルです。そのため、`dblp_authors.csv`の接頭辞/接尾辞/部分文字列の生の頻度を格納します。他の3つのファイルは、`dblp_authors_prefix_triplets.csv`、`dblp_authors_substring_triplets.csv`、`dblp_authors_suffix_triplets.csv`です。それぞれが、Anchor、Positive、Negativeの3つの列を持つCSVファイルです。

3. データセットを実行するには、`prepare_dataset`関数を実行する必要があります。コマンドは`prepare_datasets.py`に既に記載されています。中間ファイルはサイズが大きすぎるため、DropboxやGitHubで共有することはできません。ただし、IMDBは非常に大きいため例外ですが、これらの中間ファイルは数秒で作成できます。

4. 埋め込みモデルと選択性モデルの学習：Astrid-Embedは、まずトリプレットから埋め込みモデルを学習し、次に別のモデルを使用して選択性を学習します。これらの学習コードは両方とも`AstrindEmbed.py`にあります。まず、ハイパーパラメータ、入力ファイル、出力ファイルなど、両方のモデルを学習するために必要な設定で`setup_configs`関数を変更する必要があります。その後、`main`関数を呼び出して埋め込みモデルと選択性モデルを学習し、文字列のリストでテストできます。

### Comments:
- `AstridEmbed.py`: メイン関数には、モデルを最初からトレーニングするか、以前に保存したモデルを再ロードするためのコードが含まれています。埋め込みモデルの場合は、`train_astrid_embedding_model`関数または`load_embedding_model`関数のいずれかをコメントアウトすることで実行できます。選択性推定器の場合は、対応する関数は `train_selectivity_estimator`と`load_selectivity_estimation_model`です。
