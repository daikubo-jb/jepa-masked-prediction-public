# 隠した動画の一部を JEPA は補完候補から選べるか

この実験では、**短い合成動画の中央にある空間領域を隠し、同じ見える部分を持つ4つの補完候補を順位付け**しました。画像を生成する実験でも、未来の映像を予測する実験でも、追加学習する実験でもありません。V-JEPA 2 predictor が隠れた場所の特徴表現を予測し、その予測に最も近い候補を選ぶ課題です。

**結果:** 4件の合成 pilot で masked-context predictor は4件全ての生成規則上の正解を1位にしました。ただし、単純な等速補間も4件全て正解でした。この小標本では predictor の優位性を示せません。

## 入力、mask、正解候補

各動画は64フレーム、64×64画素で、背景上を青いボールが水平方向へほぼ等速に動きます。各動画から4候補を作ります。4候補はフレーム30〜33の gap だけが異なり、gap の外側は一致します。生成規則上の正解は、隠れた区間でも横方向の等速運動と縦位置を保つ候補です。他の候補は同じ4フレームの間だけ、ボールを上下にずらします。seed は軌道と候補提示順を変えます。

### 隠すのは4フレーム全部ではありません

入力の各フレームは shortest-edge resize 292 の後に center crop 256 を適用します。前処理後の動画は256×256画素で、16×16の空間 patch grid と、2フレームずつまとめる tubelet を使います。モデルに入る64フレームは全て保持しますが、frames 30〜33に対応する**2 temporal tubelet の中央8×8空間 patch**だけを encoder context から除きます。つまり target は `2 × 8 × 8 = 128` token、残る8,064 token は context です。4つの対象フレームでも、その中央領域以外の画像部分は見えています。

この token mask は [生成コード](../scripts/generate_ball_gap.py)、[mask と順位付けの関数](../scripts/masked_prediction.py)、[pilot runner](../scripts/run_masked_prediction_pilot.py) に記録されています。resize・crop 後に候補の可視部分が一致し、候補差分が target 領域から漏れないことを、各候補の前処理 tensor で確認してから model scoring に進みます。

## 4つの比較方法

| 方法 | 何を使うか | 候補の順位をどう決めるか |
|---|---|---|
| **Masked-context predictor** | target token を除いた可視 context | predictor の target 特徴と、各候補の target encoder 特徴との誤差が小さい候補を上位にする |
| **画像空間の等速補間** | gap 前後に見えるボール位置 | 可視フレームの位置から直線軌道を当ててボール画像を補間し、候補との差が小さいものを上位にする |
| **seeded random** | 固定 seed のみ | 内容を使わず、候補をランダム順にする |
| **可視 encoder のみ** | 候補間で同じ可視 context | 全候補が同じ入力なので、全て同点になる識別不能の基準 |

JEPA は隠れたピクセルを画像として描き直しません。まず、target token を context encoder から除いて可視 token を encode し、predictor が128か所の latent feature を出します。採点時には target encoder が各候補の動画を encode し、同じ128か所の target feature を取り出します。target feature に token ごとの LayerNorm を適用し、予測 feature との mean absolute error (MAE) を計算します。**低い誤差の候補ほど上位**です。正解ラベルは全候補の score を確定した後で評価します。

等速補間 baseline は、gap の前後に見える全フレームから青いボールの中心を推定し、x/y 位置に直線を当てて gap 内のボールを画素として補間します。候補ごとの pixel MAE は、4対象フレームの64×64画像にある中央32×32領域で計算します。これは JEPA の128 latent token の誤差とは別の空間・尺度です。したがって手法間では生の score や margin の大きさを比べず、各手法内での候補順位を比べます。実装は [baseline code](../scripts/masked_prediction_baselines.py) にあります。

## 結果

4 fixture の seed は `20261101`〜`20261104` です。生成規則上の正解候補が並んだ位置は順に `[3, 3, 3, 0]`。位置は固定されていませんが、4件だけで分布は均等ではありません。

| 方法 | strict top-1 正解数 | tie期待 top-1 | tie期待 MRR |
|---|---:|---:|---:|
| Masked-context predictor | 4/4 | 1.000 | 1.000 |
| 画像空間の等速補間 | 4/4 | 1.000 | 1.000 |
| seeded random（この4件で観測） | 2/4 | 0.500 | 0.708 |
| 可視 encoder のみ（全候補同点） | 0/4 | 0.250 | 0.521 |

4択 random の理論的な top-1 chance は `0.25`、期待 MRR は約 `0.521` です。今回観測した random の2/4は4試行の値で、理論的 chance が50%という意味ではありません。可視 encoder の strict top-1 が0/4なのは全候補が同点で順位を決められないからです。同点を候補間で均等に扱うと、期待 top-1 は0.25、期待 MRR は約0.521になり、random の理論基準と一致します。

Predictor の平均 `min(wrong loss) − correct loss` は `0.032326`。等速補間の平均 `min(wrong pixel error) − correct pixel error` は `2.666504` でした。score は別尺度のため、この2つの margin は相互比較できません。順位結果では、predictor と等速補間がどちらも4/4です。

## 公式実装との限定比較

公式 [V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812) の ViT-L/16 predictor を使いました。pinned [predictor source](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/models/predictor.py) の native forward と中央 mask adapter を、**同じ native predictor instance** と固定 CPU float32 latent で比較しました。12 predictor blocks の token ID と shape が一致し、出力は `[1, 2, 1024]`、最大・平均絶対誤差はともに `0.0`（`atol=rtol=1e-5`）でした。詳しくは [parity result JSON](native-predictor-parity-20261101.json) を参照してください。

この値は固定した一つの入力条件での native forward と adapter の一致です。既存の互換 loader 全体や全 mask 配置での一致を証明するものではありません。V-JEPA 2 の公式 [16-frame pretrain config](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train/vitl16/pretrain-256px-16f.yaml) と [64-frame cooldown config](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train/vitl16/cooldown-256px-64f.yaml) で `loss_exp=1.0` を確認しました。

使用した既存 `vitl.pt` はローカルの identity/hash 記録と一致し、safe weight-only load と strict load を使用しました。実験時には取得時 URLや提供者のchecksum/signatureは記録されていませんでした。後日の公開準備点検で、identityに記録されたSHA-256とsizeが、公式Hugging Face `facebook/vjepa2-vitl-fpc64-256` の `original/model.pth` LFS記録と一致することを確認しました。保存済みファイルのbyte identityを裏付けますが、実際の取得経路や署名検証までは証明しません。比較対象の公式LFS記録は[Hugging Faceのmodel repository](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/commit/995ed017ecd861661bf87bf134adf0e76927198c)で確認できます。

## 限界と再現について

これは4件の合成データによる小規模評価です。評価用の held-out seed はなく、生成器の縦位置条件は対象領域内に差分を収めるよう制約しています。4件で位置分布も均等ではありません。結果は JEPA の一般的な優位性、実動画への汎化、物理理解、危険予測を示しません。

結果と設定は [pilot JSON](masked-prediction-pilot-20261101-v2.json)、[読みやすい結果表](masked-prediction-pilot-20261101-v2.md)、[fixture index](../fixtures/pilot-20261101-v2/INDEX.json) にあります。評価コードは [predictor pilot](../scripts/run_masked_prediction_pilot.py)、[mask/ranking helpers](../scripts/masked_prediction.py)、[baselines](../scripts/masked_prediction_baselines.py)、[fixture generator](../scripts/generate_ball_gap.py)、[parity checker](../scripts/check_native_predictor_parity.py) です。対応する回帰テストは [pilot tests](../tests/test_masked_prediction_pilot.py) です。

この repository 単独で同じ実験をすぐ再現できる状態ではありません。checkpoint、前処理設定、互換 loader、Python/ML 環境の一部は repository 外の既存資源に依存し、実行コードにもその環境向けの固定設定があります。公開用の依存 lock、portable な設定方法、checkpoint の正規取得・検証手順、単独で動く再現 command は未整備です。GitHub 公開前にそれらを別途整える必要があります。

記録 run はCPUで約207秒、peak direct-child RSSは約7.44 GBでした。これは一回の測定であり、別環境の実行時間やメモリの保証ではありません。

## 次に行うなら

複数の独立 trajectory を使い、正解候補位置を均等にしたうえで seed/trajectory 単位に開発用と held-out を分けます。4つの比較方法を同じ条件で残し、held-out の成績を見てから generator、mask、閾値を調整しない設計が必要です。今回の4件からその性能は推定できません。
