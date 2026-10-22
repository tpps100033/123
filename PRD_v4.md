# InfantPose-Bench PRD v4.0

## 1. 作品定位

本專題比較四種通用人體姿態模型在成人與嬰兒靜態影像上的表現，回答成人資料分布上的模型能力是否可零樣本泛化到嬰兒影像。它是研究型 benchmark，不是睡眠分期、嬰兒睡姿診斷或臨床決策系統。

## 2. 研究問題

**RQ1**：各模型在成人 COCO 與嬰兒 SyRIP 的 COCO-17 姿態品質、coverage 與延遲為何？

**RQ2**：成人至嬰兒的跨年齡下降幅度為何？

**RQ3**：在同一個公平協定下，準確度與延遲的取捨為何？

## 3. 固定範圍

| 項目 | 正式規格 |
|---|---|
| 成人域 | COCO SinglePerson val2017，921 張，COCO-17 ground truth |
| 嬰兒域 | SyRIP Test100，100 張，需先完成並驗證 COCO-17 mapping |
| 模型 | MediaPipe Pose Heavy、MoveNet Thunder、RTMPose-M、YOLO11x-Pose |
| 任務 | 單張、單人、零樣本 2D 姿態推論 |
| 指標 | coverage、端到端 OKS／PCK@0.10、條件式 OKS／PCK@0.10、中位／P95 latency |

下列內容明確不納入 v4：SLP、Roboflow、成人／嬰兒睡眠 benchmark、影片、睡姿分類、PPG、多模態融合、訓練／微調、Sapiens 與 OpenPose。

## 4. 公平協定

唯一正式 profile 為 `fair_reference_v1`：batch size 1；固定 10 張 warm-up；FP32；禁止 AMP、compile、quantization 或 TensorRT。每筆影像都保存 decode、preprocess、inference、postprocess 與 end-to-end 時間；GPU 非同步時必須同步後計時。輸出寫檔與指標計算不列入計時區間。

## 5. 資料與輸出契約

`00_Prepare_TwoDomain_Handoffs_v4` 是唯一可寫入 dataset handoff 的入口。每列必須含 image、bbox、17 個 ground-truth keypoints 與 visibility；資料集數量必須精確為 921 或 100。產生 `dataset_manifest.json` 與帶 SHA-256 的 `sample_index.csv`。

每次模型執行產出不可覆寫的 run package：`run_manifest.json`、`predictions.jsonl`、`metrics.jsonl`。每筆預測必須記錄狀態、COCO17 座標、關節信心分數、分段時間、原生輸出與錯誤碼；run manifest 必須記錄資料 handoff hash、模型來源／checkpoint、profile、環境、樣本數、成功數、latency 摘要與 `timing_comparability`。

## 6. 評估規則

若模型無姿態輸出，coverage 下降，且 end-to-end OKS/PCK 計為 0。條件式分數只能作診斷用途，不能取代主表的 end-to-end 指標。成人至嬰兒下降量以同模型 `adult_score - infant_score` 及相對下降百分比個別報告；不建立跨指標的總分。

## 7. 驗收標準

1. 兩個資料 handoff 皆通過數量、COCO-17、hash 驗證。
2. 每個模型至少通過 10 張 COCO 與 10 張 SyRIP smoke run，才可跑全量。
3. 正式 run 的輸出可被 `90_Merge_v4` 在未載入模型下讀取。
4. 主分析只讀取完整、hash 一致、profile 為 `fair_reference` 的 run。
5. 報告必須揭露 SyRIP 的 mapping 依據、失敗／漏偵比例、硬體與環境版本，以及不能推論到臨床睡眠場景的限制。

## 8. 執行順序

1. 00 準備並鎖定 COCO 與 SyRIP handoff。
2. 以同一規格完成 01 MediaPipe、02 MoveNet、03 RTMPose-M、04 YOLO11x-Pose 的 smoke run 與全量 run。
3. 用 90 合併，只接受 Fair reference 的跨模型表。
4. 用 91 產生兩域主表、跨年齡下降與 coverage。

## 9. 研究依據與限制

新生兒姿態估計在臨床影像中會受遮擋、尺度、環境與訓練資料差異影響；公開嬰兒資料與成人 COCO 的差距應被視為跨域泛化問題，而非臨床睡眠性能的直接代理。RTMPose 以 RTMCC/SimCC 與 MMPose 為基礎；本專題使用其官方推論路徑，並透過共同輸出與計時邊界做比較。

參考：Jiang et al., *RTMPose* (2023)；Sotoodeh et al., *Automatic pose estimation in newborn infants* (2026)；Grafton et al., *Neonatal pose estimation in the unaltered clinical environment* (2025)；Földesy et al., *Automated neonatal sleep positioning assessment* (2026)。
