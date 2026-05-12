# CSC4007 — Lab 4 Analysis Report

> Sinh viên điền báo cáo này sau khi chạy baseline và các biến thể nâng cấp.

## 1. Thông tin chung

- Họ tên:
- MSSV:
- Lớp:
- Link GitHub repo:
- Link W&B project/run nếu có:

## 2. Baseline bắt buộc

Mô hình baseline trong Lab 4:

```text
Tokenized text → Embedding → 1-layer LSTM → Dropout → Linear classifier
```

Điền cấu hình đã chạy:

| Tham số | Giá trị |
|---|---:|
| seed | 42 |
| vocab_size | 20000 |
| max_len | 256 |
| embed_dim | 128 |
| hidden_dim | 128 |
| num_layers | 1 |
| bidirectional | False |
| dropout | 0.3 |
| lr | 1e-3 |
| batch_size | 64 |
| epochs_trained | 6 |

Kết quả baseline:

| Split | Loss | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation | 0.45728 | 0.80667 | 0.80614 |
| Test | 0.45064 | 0.81188 | 0.81128 |

Nhận xét ngắn về baseline:

- ...

## 3. Bảng ablation

Sinh viên cần thử ít nhất 2 biến thể nâng cấp so với baseline.

| Run | model_type | bidirectional | num_layers | max_len | hidden_dim | dropout | Test Accuracy | Test Macro-F1 | Nhận xét |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_lstm | lstm | False | 1 | 256 | 128 | 0.3 | 0.81188 | 0.81128 | Baseline ổn định, chạy nhanh trên GPU. Chưa có dấu hiệu overfit nghiêm trọng. |
| variant_1 | gru | False | 1 | 256 | 128 | 0.3 | 0.85656 | 0.85655 | Dùng GRU chạy ít bị học quá đà (ít overfit), điểm F1 tăng từ 0.81 lên 0.85 rất ngon. |
| variant_2 | lstm | True | 1 | 256 | 128 | 0.4 | 0.82044 | 0.82019 | Dùng BiLSTM mất thêm tài nguyên tính toán nhưng lại overfit sớm ở epoch thứ 4, điểm test thực tế rớt xuống còn 0.82. |

**Nhận xét (Theo 5 câu hỏi yêu cầu):**
1. **Biến thể nào tốt nhất theo validation macro-F1?** 
   Biến thể `variant_1` (GRU 1 layer) tốt nhất, đạt Val Macro-F1 lên tới 0.8712 ở epoch 6.
2. **Biến thể nào tốt nhất theo test macro-F1?** 
   Vẫn là `variant_1` (GRU 1 layer) giữ phong độ đỉnh nhất với Test Macro-F1 đạt 0.8565.
3. **Mô hình tốt nhất có overfit không?** 
   Có overfit nhẹ bắt đầu từ sau epoch thứ 4 (Train Loss tiếp tục giảm nhưng Val Loss đi ngang). Tuy nhiên mức độ overfit là nhẹ nhất trong cả 3 mô hình.
4. **Cải thiện đến từ kiến trúc, regularization, hay sequence length?** 
   Chủ yếu đến từ thay đổi **kiến trúc** (chuyển từ LSTM sang GRU). GRU tối giản bớt 1 cổng (gate) so với LSTM nên số lượng tham số ít hơn, phù hợp hơn để không bị thuộc vẹt (memorize) dữ liệu.
5. **Có trường hợp mô hình phức tạp hơn nhưng kết quả không tốt hơn không? Vì sao?** 
   Có. BiLSTM (`variant_2`) rõ ràng phức tạp hơn (quét 2 chiều, tham số gấp đôi) nhưng Test F1 (0.820) lại thấp hơn GRU (0.856). Lý do do mô hình quá mức dư thừa năng lực tính toán so với dataset, dẫn tới học vẹt tập Train (overfit ngay từ sớm) nên đoán trên tập Test kém đi.

## 4. So sánh công bằng

Trả lời ngắn:

1. Các run có dùng cùng dataset không? 
   -> **Có**, tất cả các run đều sử dụng chung một dataset (IMDB).
2. Các run có dùng cùng train/validation/test split không?
   -> **Có**, bộ chia (split) luôn cố định (Tỉ lệ và các sample trong từng tập train/val/test là giống hệt nhau).
3. Các run có dùng cùng seed không?
   -> **Có**, tất cả đều được thiết lập cứng chung một giá trị `seed = 42` để đảm bảo tính tái lập (reproducibility).
4. Metric chính để chọn mô hình là gì?
   -> Metric chính là **Validation Macro-F1** (bên cạnh Validation Loss).
5. Có dùng test set để chọn mô hình không? Vì sao không nên?
   -> **Không**. Không nên dùng test set để fine-tune hoặc chọn mô hình vì nếu làm vậy sẽ làm rò rỉ dữ liệu (Data Leakage) sang quá trình ra quyết định. Test set phải được "cách ly" hoàn toàn và chỉ dùng để đánh giá độ chính xác thực tế cuối cùng của mô hình sau khi mọi việc tinh chỉnh đã xong.

## 5. Phân tích learning curves (Câu 2.12)

Dựa vào `outputs/figures/loss_curve.png`, `outputs/figures/metric_curve.png` và `epoch_history.csv` cho run tốt nhất (Variant 1 - GRU):

1. **Train loss có giảm đều không?**
   -> Có, Train Loss giảm rất đều và liên tục qua các epoch.
2. **Validation loss có giảm cùng train loss không?**
   -> Không hoàn toàn. Validation Loss giảm cùng Train Loss ở 3 epoch đầu, nhưng từ epoch thứ 4 trở đi chững lại và bắt đầu tăng nhẹ, trong khi Train Loss vẫn tiếp tục giảm sâu.
3. **Validation macro-F1 đạt cao nhất ở epoch nào?**
   -> Đạt đỉnh cao nhất ở khoảng **epoch 4** tới **epoch 6** (theo wandb peak là ~0.871).
4. **Có dấu hiệu overfitting không?**
   -> **Có**. Sự phân kỳ rõ rệt giữa đường Train Loss (đi xuống) và Validation Loss (đi lên/ngang) từ sau epoch 4 là bằng chứng cụ thể của việc overfitting.
5. **Nếu chạy thêm epoch, mô hình có khả năng tốt hơn hay chỉ overfit hơn?**
   -> Nếu chạy thêm, mô hình chỉ **overfit hơn** vì mô hình sẽ càng ghi nhớ tập Train (loss giảm tiếp) nhưng không thể khai quát hóa trên tập Validation. Early stopping đã hành động đúng đắn.

## 6. Phân tích confusion matrix (Câu 2.13)

Dựa vào `outputs/figures/confusion_matrix.png`:

- **Mô hình nhầm positive thành negative nhiều hơn hay ngược lại?**
  -> Mô hình dự đoán nhầm thực tế Positive thành Negative (False Negative: 1868 mẫu) nhiều hơn là nhầm Negative thành Positive (False Positive: 1718 mẫu).
- **False positive và false negative có cân bằng không?**
  -> Khá cân bằng. Chênh lệch nhau chỉ khoảng chưa tới 10% (150 mẫu), cho thấy mô hình không bị bias quá lệch về một class cụ thể nào.
- **Biến thể tốt nhất có làm giảm loại lỗi nào so với baseline không?**
  -> Biến thể GRU đã giúp làm giảm đáng kể hiện tượng False Negative ở Baseline LSTM. RNN/BiLSTM có xu hướng bị nhiễu do nhồi nhét quá nhiều tham số, trong khi cấu trúc GRU giúp duy trì sự tập trung dài hạn tốt hơn để không miss các từ khóa định danh cảm xúc ở cuối review.
- **Nếu triển khai thực tế, loại lỗi nào nghiêm trọng hơn?**
  -> Tùy tình huống và bài toán cụ thể. Nếu triểu khai cho hệ thống kiểm duyệt tự động để xoá các bình luận độc hại/tiêu cực (spam filter), False Positive sẽ nguy hiểm hơn vì hệ thống sẽ xoá nhầm bình luận khen ngợi tự nhiên của một khách hàng đang rất vui vẻ, gây ảnh hưởng trải nghiệm tồi tệ.

## 7. Phân tích lỗi của mô hình tốt nhất (Câu 2.14)

Chọn ít nhất 10 mẫu sai của GRU từ `outputs/error_analysis/error_analysis.csv`.

| STT | Trích đoạn review | Nhãn đúng | Mô hình dự đoán | Confidence | Nguyên nhân giả định |
|---:|---|---|---|---:|---|
| 1 | "Given this film's incredible reviews I was expecting something truly exceptional. It certainly starts well... sadly, it collapses..." | negative | positive | 0.998 | Có cả ý tích cực và tiêu cực (khen trước mới chê). Mô hình đọc đoạn khen vội kết luận. |
| 2 | "I hesitated seeing this movie... The original had poignant moments... See the original..." | negative | positive | 0.997 | Nhắc đến và khen ngợi "bộ phim gốc" quá nhiều, khiến mô hình nhầm là khen phim này. |
| 3 | "Masterpiece. Carrot Top blows the screen away. Never has one movie captured the essence of the human spirit..." | negative | positive | 0.997 | Sarcasm/Mỉa mai cực gắt. Toàn dùng mỹ từ ("Masterpiece", "captures essence"). |
| 4 | "This is definitely one of the best Kung fu movies in the history of Cinema... masterpiece!" | negative | positive | 0.996 | Lỗi gắn nhãn (Labeling noise) trong dataset hoặc Sarcasm siêu ẩn ý ngầm. |
| 5 | "...Read her autobiography. Highly recommended. It is a fantastic story." | negative | positive | 0.996 | Khán giả chê phim nhưng lại khen kịch liệt cuốn tự truyện gốc. Mô hình bị lừa từ "Highly recommended". |
| 6 | "...maybe it's funny, OK, but it has totally ruined one of the best novels ever written..." | negative | positive | 0.996 | Khen novel gốc ("best novels ever") và có các từ gỡ gạc ("funny, OK"). Chuyển ý qua "but/however". |
| 7 | "This is one entertaining flick... The fact that every single gangsta in the film seemed to be doing a bad Scarface impersonation... Yeah right." | negative | positive | 0.996 | Châm biếm/Mỉa mai (Sarcasm). Dùng từ khen ở đầu và chốt bằng "Yeah right" để lật lọng. |
| 8 | "I really liked this quirky movie. The characters are not the bland beautiful people... The main title sequence alone makes this movie fun to watch." | negative | positive | 0.997 | Lỗi Ground truth dataset, văn bản chả có ý tiêu cực nào rành rọt. |
| 9 | "The Trojans are heroic and likable; the Greeks are nasty, petty and sneaky. So what if Paris ran off with the King's wife?" | negative | positive | 0.996 | Liệt kê tính chất nhân vật của hai phe (Từ vựng hiếm, tên riêng), không có từ đánh giá trực tiếp phim. |
| 10 | "This movie was pure genius. John Waters is brilliant. It is hilarious... I give it 9.5/10." | negative | positive | 0.997 | Lỗi Ground Truth trầm trọng, Confidence đoán 99.7% Positive mà gán nhãn Negative cực ảo. |

**Nhóm và Phân tích Lỗi:**
*   **Nhóm 1: Sarcasm hoặc mỉa mai (Mẫu 3, 7, 10 ở trên)**
    *   *Ví dụ tiêu biểu:* Mẫu 3 ("Masterpiece. Carrot top blows the screen... capture essence...")
    *   *Vì sao LSTM/GRU dự đoán sai:* Mạng học dựa vào trọng số của các từ ngữ mang tính cực kỳ tích cực (Masterpiece, brilliant) ở cục bộ, nhưng thiếu khả năng hiểu logic mỉa mai logic (phim rác mà gán chữ Masterpiece) ở mức bối cảnh rộng.
    *   *Cải tiến tiếp theo:* Về **kiến trúc mô hình**, nên thay thế RNN bằng cơ chế Attention (Transformer / BERT) hoặc tích hợp mô hình ngôn ngữ lớn để nắm bắt ngữ nghĩa phi biểu đạt. Hoặc cần xử lý tăng cường **dữ liệu** tập train để định danh thể loại sarcasm.
*   **Nhóm 2: Mixed sentiment: Vừa khen vừa chê, chuyển ý bằng but/however (Mẫu 1, 6)**
    *   *Ví dụ tiêu biểu:* Mẫu 1 ("It certainly starts well... sadly, it collapses")
    *   *Vì sao LSTM/GRU dự đoán sai:* Khả năng truyền dẫn "tài sản" từ của trạng thái ẩn bị quá tải với các sequences dài, làm cho từ báo hiệu xoay chiều ("sadly," "but") bị lu mờ đi so với đống ngợi khen ở trước đó.
    *   *Cải tiến tiếp theo:* Về **regularization**, nên phạt khắt khe hơn để dập overfitting, hoặc trang bị thêm Attention/Transformer cho RNN (hoặc Bidirectional như LSTM hai chiều) và nới **Sequence length** rộng hơn để đảm bảo không bị cut-off ý chuyển hướng định danh.
*   **Nhóm 3: Lợi dụ ngầm / Tên riêng / Mẫu lỗi cục bộ Dataset (Mẫu 2, 5, 8, 9)**
    *   *Ví dụ tiêu biểu:* Mẫu 5 (Đọc xong chê nhưng lại khen sách gốc: "Highly recommended... fantastic story").
    *   *Vì sao LSTM/GRU dự đoán sai:* Tương tự như trên, model bị nhầm lẫn giữa đánh giá chủ thể (phim) với đối thể phụ (sách), thu thập nhầm các token của đối thể phụ làm nhãn dán quyết định.
    *   *Cải tiến tiếp theo:* Tiền xử lý **dữ liệu** thật sạch (xóa các câu khen lệch chiều), gán Part of Speech vào dữ liệu để tách tên chủ thể.

## 8. Chọn mô hình tốt nhất (Câu 2.15)

Dựa trên quy trình đánh giá 5 bước:

- **Tên run tốt nhất:** variant_1 
- **Cấu hình:** Mô hình GRU, `vocab_size: 20000`, `max_len: 256`, `hidden_dim: 128`, `dropout: 0.3`.
- **Validation macro-F1:** 0.8712
- **Test accuracy:** 0.85656
- **Test macro-F1:** 0.85655

**Lý do chọn mô hình này**:
1. **Dùng validation macro-F1:** GRU (variant 1) vọt lên Validation Macro-F1 đỉnh cao là 0.8712, vượt trội mạnh so với Baseline (0.806).
2. **Learning curves:** Các đường cong học cho thấy tuy có bị chững lại từ epoch 4, GRU vẫn là mô hình ít overfit nhất và có loss ổn định trước khi Early Stop hoạt động. BiLSTM thì overfit tàn bạo hơn rất nhiều ở cùng điểm neo.
3. **Confusion matrix:** Độ nhầm lớp âm tính giả (1868) và dương tính giả (1718) khá sát sao cân bằng với nhau (chênh nhau rất ít, 150 mẫu), model không bị thiên vị cho một cực nào.
4. **Error analysis:** Phân bổ lỗi của mạng GRU nay rút xuống thành các câu review hiểm hóc "chơi chữ", độ châm biếm ngầm cực kì khó. So với các phiên bản Lab 3 cơ bản thường xuyên nhầm từ vựng đơn thuần thì đây là một sự cải thiện nhảy cóc khổng lồ.
5. Sau tất cả phép thẩm định trên, kết quả cuối cùng đạt mức Test F1 0.856 là hoàn toàn minh bạch và xứng đáng chọn làm Output Submit cao nhất.

## 8.5 So sánh với RNN của Lab 3 (Câu hỏi 2.11)

| Model | Input representation | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| RNN Lab 3 | token sequence | 0.7144 | 0.7141 |
| LSTM baseline Lab 4 | token sequence | 0.81188 | 0.81128 |
| Best variant Lab 4 (GRU) | token sequence | 0.85656 | 0.85655 |

**Nhận xét:**
- **LSTM/GRU có cải thiện so với RNN cơ bản không?** 
  Có, cải thiện cực kì đáng kể. Điểm dự đoán bật thẳng từ 71% lên tới mức 81-85%. 
- **Cải thiện rõ nhất ở metric nào?** 
  Cả Test Accuracy và Test Macro-F1 đều tăng đồng đều nhau (khoảng +10% đến +14%). Đặc biệt Macro-F1 tăng mạnh cho thấy các lớp được phân loại cân bằng, mô hình không đoán lụi.
- **Nếu chưa cải thiện, nguyên nhân hợp lý là gì?** 
  (Đã có cải thiện).
- **Kết quả này nói gì về quan hệ giữa kiến trúc mô hình và chất lượng dữ liệu?** 
  Dữ liệu text (IMDB reviews) thường có độ dài rất lớn. Kiến trúc RNN cơ bản hay bị triệt tiêu đạo hàm (vanishing gradient) nên dễ "quên" đoạn đầu câu. LSTM và GRU nhờ có các thiết kế cổng (gate) thông minh nên xử lý cực tốt dữ liệu chuỗi dài (long-term dependencies). Nó cho thấy chất lượng biểu diễn và học dữ liệu phụ thuộc sống còn vào kiến trúc mạng nơ-ron phải phù hợp với tính chất của chính dữ liệu đó.

## 9. Tự đánh giá

- [x] Em đã chạy baseline LSTM.
- [x] Em đã thử ít nhất 2 biến thể nâng cấp.
- [x] Em đã lưu checkpoint tốt nhất.
- [x] Em đã phân tích learning curves.
- [x] Em đã phân tích confusion matrix.
- [x] Em đã phân tích ít nhất 10 mẫu sai.
- [ ] Em đã commit code và report lên GitHub.
