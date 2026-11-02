# JEPA masked latent prediction

この project では、V-JEPA 2 predictor が合成動画の隠れた領域に対応する latent feature を予測し、4つの補完候補を順位付けできるかを調べています。動画生成、未来予測、追加の probe 学習、実動画の評価は行っていません。

## 最新の小規模結果

4件の合成 fixture で masked-context predictor は生成規則上の正解を4/4で1位にしました。画像上の等速補間も4/4でした。したがって、この pilot は JEPA の優位性を示していません。手法の概要、mask、採点、結果、限界は [実験レポート](reports/jepa-masked-prediction-results-20261101.md) を参照してください。

固定 CPU float32 latent の一条件では、公式 pinned V-JEPA 2 native predictor と中央 mask adapter の出力誤差は0でした。これは1条件の実装照合で、全mask配置や互換 loader 全体の parity を示すものではありません。保存済みの checkpoint identity SHA-256 は、公式 Hugging Face model repo の LFS 記録と一致します。ただし取得時の経路や署名は確認していません。

## Project の内容

- `scripts/`: 合成fixture生成、mask/ranking、baseline、pilot・parity実装
- `tests/`: mask、fixture、ranking、guard、parity関連のテスト
- `vendor/vjepa2-204698b45b3712590f06245fbfba32d3be539812/`: pinned公式実装の限定されたソース、source manifest、MIT license
- `fixtures/pilot-20261101-v2/`: MIT対象の合成fixture metadataとanswer key。生成されたPPMフレームはサイズを抑えるためGitへ含めず、pilot runnerがgeneratorから作成します。
- `reports/jepa-masked-prediction-results-20261101.md`: 手法、数値、制約の統合レポート
- `reports/masked-prediction-pilot-20261101-v2.md` と `.json`: 保存済み4-fixture pilotの結果
- `reports/native-predictor-parity-20261101.json`: 固定latent parityの結果

## 再現性の現状

このrepository単独では、clean cloneからモデル実験を再現できません。実行時にPython環境、checkpoint、identity記録、前処理設定、互換loaderを指定する必要があります。これらの外部ファイルはrepositoryに含まれず、モデル用の依存lockfileもありません。実行scriptは固定したguardian scriptと出力先を使い、実行時に指定したcheckpointをidentity記録内のpath・byte count・SHA-256で照合し、processor設定と互換loaderはコード内の既知SHA-256と照合します。外部パスを指定しても、同一性検査やresource guardは無効になりません。

既存のidentity記録は `checkpoint_path` を使います。別環境へ移す場合は、そのローカルidentity記録の `checkpoint_path` を指定checkpointへ合わせ、記録済みbyte countとSHA-256を維持してください。checkpoint identity SHA-256 は公式 Hugging Face の `facebook/vjepa2-vitl-fpc64-256` が公開する `original/model.pth` の LFS SHA-256 と一致します。これは保存済みバイト列がその公開 artifact と一致する根拠ですが、実際の取得経路や署名検証を証明するものではありません。V-JEPA 2の条件は[公式repository](https://github.com/facebookresearch/vjepa2)、モデルの条件は[official model card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)を参照してください。

既存CPU pilotを起動する形は次のとおりです。値は利用者自身の環境にあるファイルへ置き換えます。モデル実験のclean environment再現は未確認です。

```sh
python scripts/run_guarded_smoke.py --pilot --device cpu --pilot-count 4 --timeout 600 \
  --verified-loss-exp 1.0 --checkpoint /path/to/vitl.pt \
  --identity-record /path/to/checkpoint-identity.json \
  --processor-config /path/to/video-preprocessor.json \
  --compatibility-loader /path/to/predictor.py
```

テスト環境は専用venvで準備し、`python -m pip install -r requirements-test.txt` の後に `python -m unittest discover -s tests -v` を実行できます。PyYAMLはmanifest contract test用に固定しています。PyTorch / Transformersを使う前処理・tensor testは、それらがない環境ではskipされます。モデル実験用の依存一式はまだ固定・検証されていません。環境固有の過去の実行結果は保存済みレポートを参照してください。このrepository準備中にモデル推論は行っていません。

## ライセンス・配布

`LICENSE` のMITは、このproject独自の `scripts/` と `tests/` のソースコード、project-authored `README.md`、公開snapshotに含める実験レポートと自作合成fixture dataに適用します。著作権表記は公開GitHub profileの表示名 `daikubo` を使用し、メールや非公開連絡先は含めていません。

vendor配下の公式V-JEPA 2ソースは同梱の上流MIT noticeとpinned source manifestに従い、このroot licenseの対象外です。V-JEPA 2上流は一部の未vendor dataset utilityに別途Apache-2.0を指定しています。Hugging Face model cardはモデルをMITとして表示し、保存済みcheckpoint identityのSHA-256/sizeは同model repoの `original/model.pth` LFS記録と一致しますが、取得経路と署名検証は不明です。checkpointはrepositoryに含めていません。

vendor内の第三者ソース、モデルcheckpoint、別途条件のある第三者素材には、このproject licenseを適用しません。V-JEPA 2の条件は[公式repository](https://github.com/facebookresearch/vjepa2)、モデルの条件は[official model card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)を参照してください。
