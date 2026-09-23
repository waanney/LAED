# Plan & To-do: Speech-to-Speech nhẹ trên điện thoại

Ngày lập: 23/09/2026. Trạng thái: đề xuất triển khai, chưa benchmark trên thiết bị.

## 1. Mục tiêu và giả định

Xây app cho phép người dùng nói, AI hiểu và trả lời bằng giọng nói, chạy local sau khi tải model. Bản đầu hỗ trợ hội thoại ngắn, ưu tiên tiếng Việt.

Giả định để bắt đầu: Android ARM64, RAM 6–8 GB; một lập trình viên; dùng model pretrained, chưa train/fine-tune. Đây là thiết bị mục tiêu thử nghiệm, không phải cam kết mọi máy cấu hình này đều chạy mượt. Nếu thiết bị thật là iPhone, giữ kiến trúc nhưng đổi lớp app/audio và build runtime.

**Quyết định quan trọng:** làm pipeline ASR → LLM → TTS trước. Đây là trải nghiệm speech-to-speech ở cấp ứng dụng, không phải model speech-to-speech end-to-end. Cách này dễ đo bottleneck và thay từng model. Đổi lại, chuyển qua văn bản sẽ mất một phần ngữ điệu và cảm xúc.

## 2. Phạm vi MVP

- Bấm để nói, thả để gửi; hiển thị transcript và phát câu trả lời.
- Câu nói đầu vào tối đa 15 giây; câu trả lời 1–3 câu ngắn.
- Lưu hội thoại trong phiên, giới hạn context; có nút dừng và xóa phiên.
- Hoạt động offline sau bước cài model.
- Chưa làm wake word, gọi công cụ, voice cloning hoặc vừa nghe vừa nói liên tục.

Không kỳ vọng LLM dưới 1B đạt chất lượng trợ lý cloud. Nếu chỉ cần vài lệnh cố định, dùng bộ phân loại intent + câu trả lời mẫu có thể nhẹ và đáng tin hơn LLM.

## 3. Stack khởi điểm

| Thành phần | Lựa chọn thử đầu tiên | Điều kiện quyết định |
|---|---|---|
| App | Kotlin + Jetpack Compose | Android trước; native audio và native inference |
| Thu/phát âm thanh | AudioRecord / AudioTrack | Không chạy inference trên UI/audio callback thread |
| ASR: tiếng nói thành chữ | whisper.cpp + Whisper tiny multilingual; đối chiếu base multilingual | Không dùng biến thể `.en` cho tiếng Việt; chọn theo lỗi nhận dạng và thời gian thực đo |
| LLM: sinh câu trả lời | llama.cpp + Qwen3-0.6B GGUF Q4_K_M | Tắt thinking qua chat template/runtime tương ứng; kiểm tra thực tế không phát nội dung thinking |
| TTS: chữ thành tiếng | sherpa-onnx + ứng viên VITS/Piper tiếng Việt `vi_VN-vais1000-medium` | Nghe thử số, tên riêng, dấu tiếng Việt; kiểm tra license riêng của weights/dataset |
| Phát hiện giọng nói, giai đoạn sau | Silero VAD qua sherpa-onnx | Chỉ thêm khi chuyển từ push-to-talk sang tự xác định hết lượt |

whisper.cpp và llama.cpp có hướng dẫn/example mobile; sherpa-onnx cung cấp ASR/TTS/VAD và hỗ trợ mobile. Đây là ứng viên kỹ thuật, chưa phải bằng chứng đạt KPI trên điện thoại của bạn. [1][2][3]

Qwen3-0.6B hỗ trợ chế độ non-thinking; khả năng đa ngôn ngữ trong model card không thay thế kiểm thử tiếng Việt. [4] Mẫu giọng TTS tiếng Việt nêu trên có trang nghe thử chính thức của sherpa-onnx. [5]

**Cấu hình LLM ban đầu:** context 2.048 token, tối đa 96 token trả lời, prompt yêu cầu trả lời ngắn bằng ngôn ngữ người dùng. Giữ lịch sử theo ngân sách token; không để lịch sử tăng vô hạn. Có thể tăng lên model 1–2B nếu 0.6B không đạt chất lượng và RAM/độ trễ còn dư.

## 4. Luồng xử lý

1. Người dùng bấm giữ nút; thu PCM mono và resample về định dạng ASR yêu cầu, thường 16 kHz.
2. Thả nút: kiểm tra clip rỗng/quá ngắn, đưa audio vào ASR.
3. Nếu transcript rỗng thì mời nói lại; không tự tạo câu hỏi thay người dùng.
4. Ghép transcript với system prompt và lịch sử trong giới hạn token.
5. LLM sinh token; bộ gom câu đưa từng câu hoàn chỉnh sang TTS.
6. TTS tạo PCM; phát theo thứ tự. Sample rate đầu ra theo model TTS, không ép dùng sample rate ASR.
7. Kết thúc lượt và quay về trạng thái sẵn sàng.

MVP có thể chờ toàn bộ câu trả lời rồi chạy TTS. Sau khi pipeline ổn, mới chồng lấp LLM và TTS theo câu để giảm thời gian chờ. Sinh TTS theo câu không có nghĩa model TTS hỗ trợ streaming nội bộ.

Trạng thái app: IDLE, RECORDING, TRANSCRIBING, GENERATING, SPEAKING, ERROR. Khi GENERATING có thể đã bắt đầu SPEAKING ở bản tối ưu. Mỗi lượt có `turn_id`; kết quả từ lượt đã hủy phải bị bỏ.

MVP tắt thu âm trong lúc phát để tránh AI nghe lại chính mình. Nút “Nói lượt mới” phải hủy generation, dừng playback và xóa hàng đợi trước khi mở mic. Barge-in tự động cần xử lý echo và điều phối audio riêng, để giai đoạn sau.

## 5. Ngân sách và tiêu chí đạt

Các số dưới đây là **mục tiêu thử nghiệm**, không phải hiệu năng đã đo hay thông số nhà cung cấp.

| Chỉ số | Mục tiêu ban đầu | Cách đo |
|---|---|---|
| Tổng model tải về | ≤ 1,5 GB | Tổng bytes thực tế của model, tokenizer, dữ liệu giọng |
| Peak RAM app | ≤ 2 GB trên máy thử 6–8 GB | Đo toàn pipeline, gồm KV cache, buffer và runtime |
| Trễ phản hồi warm | p50 ≤ 3 giây; p95 ≤ 6 giây | Từ lúc thả nút nói tới mẫu âm thanh trả lời đầu tiên |
| Chất lượng hội thoại | ≥ 80/100 tình huống đạt rubric | Đúng ý, đúng ngôn ngữ, không bịa, đủ ngắn; phải đạt cả 4 |
| Độ ổn định | 20 phút không crash/OOM | Hội thoại lặp lại với context được giới hạn |
| Hoạt động offline | Hoàn thành 20 lượt ở airplane mode | Model đã cài; Wi-Fi cũng tắt |

Đo cold start riêng, không trộn với warm latency. Theo dõi ASR latency, LLM time-to-first-token, thời gian sinh câu đầu, TTS time-to-first-audio, tốc độ token và pin/nhiệt. Tổng dung lượng model không bằng RAM cần dùng.

ASR: đo CER và WER với quy tắc chuẩn hóa cố định; WER tiếng Việt phụ thuộc cách tách từ nên cần công bố cách tính. Bổ sung tỷ lệ giữ đúng ý, số và tên riêng. Chuẩn bị ít nhất 100 câu, có giọng vùng miền, phòng yên tĩnh và tiếng ồn nhẹ; không dùng toàn bộ tập đánh giá để chỉnh prompt.

## 6. Roadmap và checklist

Ước lượng 3–4 tuần cho người đã quen Android/native build; cần điều chỉnh theo kinh nghiệm và thiết bị. Ưu tiên theo cổng nghiệm thu, không chạy theo lịch nếu chất lượng chưa đạt.

### P0 — Chốt bài toán và chứng minh trên máy thật

- [ ] Ghi model điện thoại, SoC, RAM, Android version và dung lượng trống.
- [ ] Chốt ngôn ngữ: Việt, Anh hoặc cả hai; chốt 3 nhóm tác vụ chính.
- [ ] Xác định app hội thoại tự do hay trợ lý lệnh cố định.
- [ ] Tạo 100 tình huống hội thoại, tách tập phát triển và đánh giá cuối.
- [ ] Chuẩn bị audio có bản chép chuẩn; không lưu audio cá nhân ngoài ý muốn.
- [ ] Pin commit runtime, revision/checksum model và lưu license của từng thành phần.
- [ ] Chạy riêng ASR, LLM, TTS trên điện thoại; ghi latency, RAM và chất lượng.
- [ ] So sánh Whisper tiny/base; kiểm tra Qwen 0.6B với câu hỏi tiếng Việt thực tế.

**Cổng A:** cả ba thành phần chạy offline trên điện thoại; kết quả đủ dùng cho phạm vi đã chọn. Nếu ASR hoặc LLM quá yếu, đổi model hoặc thu hẹp tác vụ trước khi làm UI đầy đủ.

### P1 — Pipeline hoạt động, khoảng tuần 1–2

- [ ] Tạo app Kotlin với nút thu âm, transcript, câu trả lời và trạng thái.
- [ ] Tích hợp native runtime bằng JNI/binding tương ứng; không nhúng Python vào app.
- [ ] Xử lý microphone permission, sample rate và audio focus.
- [ ] Nối audio → ASR → LLM → TTS → playback bằng worker/background jobs.
- [ ] Thêm giới hạn độ dài audio, context và output.
- [ ] Xử lý clip im lặng, lỗi load model, thiếu RAM và bị thu hồi audio focus.
- [ ] Thêm cancel theo `turn_id`; dừng cả inference lẫn audio queue.
- [ ] Chạy 20 lượt liên tiếp trong airplane mode.

**Cổng B:** người dùng nói và nhận câu trả lời bằng giọng nói ổn định; không treo UI, không nghe lại tiếng của app, không phát kết quả của lượt đã hủy.

### P2 — Tối ưu theo số đo, khoảng tuần 3

- [ ] Gắn timestamp cho từng stage, xuất báo cáo benchmark.
- [ ] Tách cold/warm, báo p50/p95; kiểm tra sau 20 phút chạy liên tục.
- [ ] Thêm sentence chunking để bắt đầu TTS trước khi LLM sinh xong.
- [ ] Giới hạn hàng đợi TTS để không tích lũy audio; giữ đúng thứ tự câu.
- [ ] Thử số thread khác nhau; chỉ dùng GPU/backend khác nếu đo thấy tốt hơn.
- [ ] Kiểm tra CPU contention khi LLM và TTS chạy đồng thời.
- [ ] Giữ model trong RAM nếu ngân sách cho phép; đo đánh đổi khi unload/reload.
- [ ] So sánh cấu hình quantization khác chỉ khi lỗi chất lượng có thể do quantization.

**Cổng C:** đạt KPI hoặc ghi rõ cấu hình nào chưa đạt và bottleneck cụ thể. Nếu chậm do ASR, đổi LLM sẽ không giải quyết đúng vấn đề.

### P3 — Đóng gói và khả năng phục hồi, khoảng tuần 4

- [ ] Tải model có tiến trình, resume, checksum và kiểm tra dung lượng trước khi cài.
- [ ] Có manifest ghi model ID, revision, quantization, license và định dạng audio.
- [ ] Chỉ chuyển sang bản model mới sau khi tải và xác minh hoàn tất.
- [ ] Xử lý app background/foreground, cuộc gọi, tai nghe và audio interruption.
- [ ] Không ghi audio/transcript mặc định; có tùy chọn debug và xóa dữ liệu.
- [ ] Đánh giá trên tập giữ lại; ghi rõ giới hạn của model nhỏ.
- [ ] Nếu có thiết bị, kiểm tra thêm một máy yếu hơn; nếu không, giới hạn tuyên bố tương thích.
- [ ] Xuất APK release, hướng dẫn cài model, benchmark và danh sách lỗi còn lại.

## 7. Sau MVP

- [ ] Thêm VAD và tự kết thúc lượt; bắt đầu thử khoảng im lặng 500–800 ms rồi chỉnh theo dữ liệu.
- [ ] Thử ASR streaming có tiếng Việt nếu ASR sau khi thả nút là bottleneck.
- [ ] Thêm barge-in sau khi có echo cancellation và cơ chế hủy đáng tin cậy.
- [ ] Port iOS: SwiftUI, AVAudioEngine và build runtime cho Apple; benchmark lại trên thiết bị thật.
- [ ] Chỉ cân nhắc fine-tune khi đã phân loại được lỗi và có dữ liệu phù hợp.
- [ ] Chỉ thử model speech-to-speech end-to-end khi cần ngữ điệu/độ trễ mà pipeline hiện tại không đáp ứng.

## 8. Thứ tự làm ngay

1. Chọn một điện thoại thật và chốt use case.
2. Đo ba thành phần riêng trên máy đó bằng 20 mẫu thử ban đầu.
3. Chọn cấu hình khả thi rồi mới nối app hoàn chỉnh.

**Sai lầm cần tránh:** benchmark trên server rồi suy ra tốc độ điện thoại; coi số tham số là thước đo duy nhất của độ nhẹ; chọn model chỉ vì có tên ngôn ngữ trong model card; tối ưu độ trễ trước khi biết hệ thống trả lời có đúng hay không.

## Nguồn kỹ thuật

Nguồn được kiểm tra ngày 23/09/2026; khi bắt đầu triển khai cần pin phiên bản cụ thể.

1. [whisper.cpp — repository và README](https://github.com/ggml-org/whisper.cpp/blob/master/README.md)
2. [llama.cpp — Android guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
3. [sherpa-onnx — documentation](https://k2-fsa.github.io/sherpa/onnx/index.html)
4. [Qwen3-0.6B — model card và non-thinking mode](https://huggingface.co/Qwen/Qwen3-0.6B)
5. [sherpa-onnx — mẫu TTS tiếng Việt vais1000](https://k2-fsa.github.io/sherpa/onnx/tts/all/Vietnamese/vits-piper-vi_VN-vais1000-medium.html)
