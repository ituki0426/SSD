# SSD

***

物体検出の手法の一つであるSSDを実装します。

##  SSDにおける物体検出の流れ

***

- 1.画像を300x300にリサイズして前処理を実施

- 2.画像をSSDネットワークに入力する

- 3.デフォルトボックスを作成し、SSDモデルの出力とまとめて出力値とする。

- 4.損失関数により、損失値を測定

- 5.誤差逆伝播による重みの更新
