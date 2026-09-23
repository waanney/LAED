import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'speech_controller.dart';

class SpeechPage extends StatefulWidget {
  const SpeechPage({super.key});

  @override
  State<SpeechPage> createState() => _SpeechPageState();
}

class _SpeechPageState extends State<SpeechPage> with WidgetsBindingObserver {
  bool _pointerDown = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state != AppLifecycleState.resumed) {
      unawaited(context.read<SpeechController>().cancelTurn());
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = context.watch<SpeechController>();
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 24, 20, 28),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('LAED', style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              Text(_status(controller.phase), style: TextStyle(color: _statusColor(controller.phase))),
              const SizedBox(height: 16),
              if (controller.isDemoMode) ...[
                const _Notice(
                  'Demo mode: Rust đang dùng engine mô phỏng. Audio vẫn đi qua pipeline thật nhưng chưa chạy model AI.',
                ),
                const SizedBox(height: 12),
              ],
              if (controller.error case final error?) ...[
                _Notice(error),
                const SizedBox(height: 12),
              ],
              _MessageCard(label: 'Bạn', text: controller.transcript.isEmpty ? 'Transcript sẽ hiện ở đây.' : controller.transcript),
              const SizedBox(height: 12),
              _MessageCard(label: 'Trợ lý', text: controller.response.isEmpty ? 'Câu trả lời sẽ hiện ở đây.' : controller.response),
              const Spacer(),
              Align(
                child: Semantics(
                  button: true,
                  label: 'Giữ để nói, thả để gửi',
                  child: Listener(
                    onPointerDown: controller.canRecord
                        ? (_) async {
                            _pointerDown = true;
                            await controller.startRecording();
                            if (!_pointerDown && controller.phase == UiPhase.recording) {
                              await controller.cancelTurn();
                            }
                          }
                        : null,
                    onPointerUp: (_) {
                      _pointerDown = false;
                      unawaited(controller.finishRecording());
                    },
                    onPointerCancel: (_) {
                      _pointerDown = false;
                      unawaited(controller.cancelTurn());
                    },
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 150),
                      width: 136,
                      height: 136,
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: controller.phase == UiPhase.recording
                            ? const Color(0xFFC62828)
                            : controller.canRecord
                                ? const Color(0xFF3157D5)
                                : const Color(0xFF9EA5B5),
                      ),
                      child: Text(
                        controller.phase == UiPhase.recording ? 'Thả để gửi' : 'Giữ để nói',
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  FilledButton(onPressed: controller.phase == UiPhase.idle ? null : controller.cancelTurn, child: const Text('Dừng')),
                  const SizedBox(width: 12),
                  FilledButton.tonal(onPressed: controller.clearSession, child: const Text('Xóa phiên')),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MessageCard extends StatelessWidget {
  const _MessageCard({required this.label, required this.text});
  final String label;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: Colors.white,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label, style: const TextStyle(color: Color(0xFF3157D5), fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            Text(text),
          ],
        ),
      ),
    );
  }
}

class _Notice extends StatelessWidget {
  const _Notice(this.message);
  final String message;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(color: const Color(0xFFFFF0C2), borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Text(message, style: const TextStyle(color: Color(0xFF594600))),
      ),
    );
  }
}

String _status(UiPhase phase) => switch (phase) {
      UiPhase.idle => 'Sẵn sàng',
      UiPhase.recording => 'Đang nghe…',
      UiPhase.transcribing => 'Đang nhận dạng…',
      UiPhase.generating => 'Đang suy nghĩ…',
      UiPhase.speaking => 'Đang trả lời…',
      UiPhase.error => 'Cần thử lại',
    };

Color _statusColor(UiPhase phase) => switch (phase) {
      UiPhase.recording || UiPhase.error => const Color(0xFFC62828),
      _ => const Color(0xFF4A5160),
    };

