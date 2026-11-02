# 合成中央gap pilot

- 状態: completed
- fixture数: 4（4択）
- device: cpu
- 経過秒: 207.044
- 結果の範囲: 合成generator rule下の探索的候補識別のみ。

| 方法 | 件数 | strict top-1 | tie期待top-1 | tie期待MRR |
|---|---:|---:|---:|---:|
| seeded_random | 4 | 0.500 | 0.500 | 0.708 |
| visible_context_encoder_only_tie | 4 | 0.000 | 0.250 | 0.521 |
| image_space_constant_velocity_interpolation | 4 | 1.000 | 1.000 | 1.000 |
| masked_context_predictor | 4 | 1.000 | 1.000 | 1.000 |

- ランダム順位の理論基準: top-1 0.250, MRR 0.521。

小標本で統計的な性能・実動画汎化・物理理解・危険予測を示しません。
全候補/全baselineのscore確定後にだけanswer keyを評価しています。

## 実行・guard記録

- 固定seed 20261101–20261104の4 fixtureを完了。各fixtureはencoder 1回・predictor 1回・target encoder 4回で、計24 forwardを全て試行・返却確認。回答評価は全4件の全方式採点後に実施。
- child elapsed 207.044秒、guard duration 207.337秒。peak direct-child RSS 7,442,710,528 bytes、最低available-memory見積 52,056,522,752 bytes、402回計測。
- guard終了コード0、child wait済み、signal/timeoutなし、子孫なし、process group消滅確認。offline/network-deny childは開始時に確認。
- 固定出力progress JSONLは観測された許可イベントの制約済み記録（87行、18,740 bytes）。runnerのprogress_countsはfinal run_completed前の86イベントを計上。
- checkpoint SHA identity照合成功、CPU float32、loss exponent 1.0。これは4条件の合成pilot実測であり、CPU環境・モデル挙動の一般性能保証ではない。
