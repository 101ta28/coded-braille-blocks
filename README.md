# [Deprecated] コード化点字ブロックの二次利用

金沢市のオープンデータポータルで、管理されるようになったため、このリポジトリは非推奨です。

[https://catalog-data.city.kanazawa.ishikawa.jp/dataset/https-www2-kanazawa-it-ac-jp-matsuilb](https://catalog-data.city.kanazawa.ishikawa.jp/dataset/https-www2-kanazawa-it-ac-jp-matsuilb)

## コード化点字ブロックについて

金沢工業大学 松井研究室が開発した、点字ブロックをコード化したものです。

視覚障がい者の歩行課題の軽減に加え、観光客や外国人も観光情報や災害時の避難情報などが取得できるものになっています。

松井研究室のホームページは[こちら](http://www2.kanazawa-it.ac.jp/matsuilb/index.html)です。

## このリポジトリについて

松井研究室公式の[ホームページ](http://www2.kanazawa-it.ac.jp/matsuilb/research.html#Braille_blocks)からは、コード化点字ブロックオープンデータとして3つのデータがダウンロードできます。

- コード化点字ブロックオープンデータ仕様.pdf

- 位置データ2020.csv

- 案内データ2020.csv

個人的にCSVはあまり使いやすいものではないと思っているので、JSON形式に変換しました。

また、Pythonで読み取るためのプログラムとして、`demo.py`を作成しました。

```bash
# ライブラリのインストール
$ pip install -r requirements.txt

# デモの実行
$ python demo.py xxx.{jpg,png}

# 案内データの表示
$ python guide.py xxx.{jpg,png}

```

精度は怪しいので、あくまでデモとしてご利用ください。
