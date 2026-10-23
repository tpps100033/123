# 資料來源與結果簡析

## 資料如何取得與整理

| 資料域 | 來源與本專案採用內容 | 標註處理 | 使用規模 |
|---|---|---|---:|
| COCO | COCO 2017 val 的 person keypoints 影像與標註；本專案沿用研究 Drive 中已選定的單人子集 | 原生 COCO-17，未重映射 | 921 張 |
| 嬰兒：SyRIP | SyRIP（Synthetic and Real Infant Pose）研究資料集的 Test100 實嬰兒影像；由專案研究 Drive 匯入 | 沿用已驗證的 SyRIP 到 COCO-17 對應；只用於評估，不納入訓練 | 100 張 |
| 兒童：COCO-UniHuman | COCO-UniHuman v2 val 的人工表觀年齡標註，加上 COCO 2017 val 原始影像 | 篩選表觀年齡 3–17、有效 COCO-17 標註、非 crowd 樣本；每張來源影像選一名最大合格人物，裁切人物框並加 12% 邊界，將原始關鍵點座標平移到裁切圖 | 134 張人物裁切圖 |

COCO 與 SyRIP 的檔案由使用者提供的研究 Drive 匯入，本次保留既有的選樣和標註處理。COCO 的官方 keypoints 任務使用 COCO 2017 的人物關鍵點標註；SyRIP 原始研究介紹其合成與真實嬰兒姿態資料，專案使用其中 Test100 真實影像評估。[COCO Keypoint Detection](https://cocodataset.org/#keypoints-2017)、[SyRIP 原始論文](https://arxiv.org/abs/2010.06100)、[SyRIP 專案](https://github.com/ostadabbas/Infant-Pose-Estimation)

COCO-UniHuman 是以 COCO 2017 train/val 影像為基礎增加人體屬性和關鍵點等標註的資料集。v2 val 的人物關鍵點沿用 17 點 COCO 格式；年齡欄位是人工評估的 apparent age（表觀年齡），不是經身分資料確認的出生年齡。本專案從合格標註中找到 262 名候選兒童，分布於 134 張來源影像；再依一張影像只選一人，產生 134 張人物裁切圖。[COCO-UniHuman 說明與下載](https://github.com/lishuhuai527/COCO-UniHuman/blob/main/COCO_UniHuman.md)、[標註格式](https://github.com/lishuhuai527/COCO-UniHuman/blob/main/data_format.md)

為避免重複，兒童樣本先排除既有 921 張 COCO 子集的全部 image ID；準備時再比對原始影像 SHA-256，排除與 COCO 子集檔案完全相同的影像，也排除兒童候選之間完全相同的檔案。最終 134 張來源影像的 ID 與檔案 hash 均唯一。兒童影像仍來自 COCO val，因此這是與現有 COCO 子集不重疊的兒童子集，不是另一個獨立拍攝來源。

人物裁切只改變影像範圍。每個可見關鍵點的 x、y 會減去裁切左上角座標；COCO-17 點順序、可見性值和原始人工位置保留。不可見點按 COCO 格式輸出為 `(0, 0, 0)`。轉換後的標註、來源 image ID、原始檔 SHA-256、裁切檔 SHA-256 和樣本清單都在資料包的 `data_provenance/child/`。

## 六模型結果簡析

所有 AP 均以官方 `pycocotools.COCOeval(iouType='keypoints')` 預設計算。COCO 是選定的 921 張單人子集；SyRIP 和兒童資料域採用相同 COCO OKS 評估器，但不是官方 COCO 排行榜結果。

| 模型 | COCO AP | SyRIP 嬰兒 AP | 兒童 AP | 兒童 − COCO AP |
|---|---:|---:|---:|---:|
| MediaPipe Pose Heavy | 0.418 | 0.730 | 0.578 | +0.161 |
| MoveNet Thunder | 0.103 | 0.417 | 0.407 | +0.304 |
| OpenPose COCO18 | 0.589 | 0.477 | 0.627 | +0.039 |
| RTMPose-M | 0.775 | 0.834 | 0.869 | +0.094 |
| Sapiens 1B COCO | 0.535 | 0.755 | 0.659 | +0.124 |
| YOLO11x Pose | 0.722 | 0.751 | 0.676 | −0.046 |

RTMPose-M 在三個資料域的 AP 都最高；兒童資料域 AP 為 0.869。五個模型的兒童 AP 高於本專案 COCO 子集，YOLO11x Pose 低約 0.046。這些差異不能單獨解讀為年齡造成的差異：兒童資料是 COCO 影像裁切，COCO 基準則沿用原本的單人影像，因此場景和裁切範圍也不同。SyRIP 結果則顯示嬰兒域排序與 COCO 域不完全相同；OpenPose 的 SyRIP AP 低於其 COCO AP，其餘模型較高。

此兒童集共有 134 張來源影像，適合做初步、零樣本的兒童域比較，樣本數有限；結論不代表一般兒童族群，也不可視為醫療或臨床驗證。COCO-UniHuman 標註依其研究用途條款使用；影像仍受原 COCO 來源的使用條款約束。論文引用或再散布前，請依資料集原始授權確認可使用範圍。

## 可重現檔案

- 三域完整結果：`results/final_results_all_domains.csv`
- 三域比較表與摘要：`results/result_comparison.md`、`results/three_domain_summary.csv`
- 樣本與標註來源清單：`data_provenance/{coco,syrip_test100,child}/`
- 六模型逐筆輸出和 run manifest：`run_evidence/`
- 完整引用及授權提示：`CITATIONS_AND_RIGHTS.md`
