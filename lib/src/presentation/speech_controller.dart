import 'dart:async';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import '../rust/api/speech.dart';

enum UiPhase {
  idle,
  recording,
  transcribing,
  generating,
  speaking,
  error,
}

class SpeechController extends ChangeNotifier {
  static const _sampleRateHz = 16000;

  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  final BytesBuilder _pcm = BytesBuilder(copy: false);

  SpeechCore? _core;
  StreamSubscription<Uint8List>? _captureSubscription;
  Timer? _recordingDeadline;
  int _turnId = 0;
  bool _finishing = false;
  int _effectiveSampleRateHz = _sampleRateHz;
  int _effectiveChannels = 1;

  UiPhase phase = UiPhase.idle;
  String transcript = '';
  String response = '';
  String? error;
  bool isDemoMode = true;

  bool get canRecord => phase == UiPhase.idle || phase == UiPhase.error;

  Future<void> initialize() async {
    final support = await getApplicationSupportDirectory();
    _core = await SpeechCore.create(
      config: SpeechConfig(
        modelRoot: '${support.path}/models',
        demoMode: true,
        contextTokens: 2048,
        maxOutputTokens: 96,
      ),
    );
    isDemoMode = await _core!.isDemoMode();
    notifyListeners();
  }

  Future<void> startRecording() async {
    final core = _core;
    if (core == null || !canRecord) return;
    error = null;
    await _player.stop();
    if (!await _recorder.hasPermission()) {
      _setError('Cần quyền microphone để bắt đầu thu âm.');
      return;
    }

    _pcm.clear();
    _effectiveSampleRateHz = _sampleRateHz;
    _effectiveChannels = 1;
    await _recorder.setOnConfigChanged((config) {
      _effectiveSampleRateHz = config.sampleRate;
      _effectiveChannels = config.numChannels;
    });
    _turnId = await core.beginTurn();
    final stream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: _sampleRateHz,
        numChannels: 1,
        autoGain: false,
        echoCancel: true,
        noiseSuppress: true,
        streamBufferSize: 4096,
        androidConfig: AndroidRecordConfig(audioSource: AndroidAudioSource.voiceRecognition),
      ),
    );
    if (_effectiveChannels != 1) {
      await _recorder.cancel();
      await core.cancelTurn();
      _setError('Thiết bị không cung cấp được PCM mono cho pipeline.');
      return;
    }
    phase = UiPhase.recording;
    notifyListeners();

    _captureSubscription = stream.listen(
      (chunk) {
        final maxRecordingBytes = _effectiveSampleRateHz * _effectiveChannels * 2 * 15;
        final remaining = maxRecordingBytes - _pcm.length;
        if (remaining <= 0) {
          unawaited(finishRecording());
          return;
        }
        _pcm.add(chunk.length <= remaining ? chunk : chunk.sublist(0, remaining));
        if (_pcm.length >= maxRecordingBytes) unawaited(finishRecording());
      },
      onError: (Object cause, StackTrace stackTrace) {
        _setError('Không đọc được microphone: $cause');
      },
    );
    _recordingDeadline = Timer(const Duration(seconds: 15), () {
      unawaited(finishRecording());
    });
  }

  Future<void> finishRecording() async {
    if (phase != UiPhase.recording || _finishing) return;
    _finishing = true;
    _recordingDeadline?.cancel();
    _recordingDeadline = null;
    await _recorder.stop();
    await _captureSubscription?.cancel();
    _captureSubscription = null;
    final pcm = _pcm.takeBytes();
    final core = _core;
    final turnId = _turnId;
    _finishing = false;
    if (core == null) return;

    phase = UiPhase.transcribing;
    notifyListeners();
    try {
      await for (final event in core.processTurn(
        turnId: turnId,
        pcm16Le: pcm,
        inputSampleRateHz: _effectiveSampleRateHz,
      )) {
        if (event.turnId != _turnId) continue;
        transcript = event.transcript ?? transcript;
        response = event.response ?? response;
        error = event.error;

        final audio = event.audioWav;
        if (audio != null && audio.isNotEmpty) {
          phase = UiPhase.speaking;
          notifyListeners();
          final completed = _player.onPlayerComplete.first;
          await _player.play(BytesSource(audio, mimeType: 'audio/wav'));
          await completed;
          if (event.turnId == _turnId) phase = UiPhase.idle;
        } else {
          phase = _mapPhase(event.phase);
        }
        notifyListeners();
      }
    } catch (cause) {
      if (turnId == _turnId && phase != UiPhase.idle) {
        _setError(cause.toString().replaceFirst('Exception: ', ''));
      }
    }
  }

  Future<void> cancelTurn() async {
    _turnId += 1;
    _recordingDeadline?.cancel();
    _recordingDeadline = null;
    await _captureSubscription?.cancel();
    _captureSubscription = null;
    await _recorder.cancel();
    await _player.stop();
    await _core?.cancelTurn();
    _pcm.clear();
    _finishing = false;
    phase = UiPhase.idle;
    error = null;
    notifyListeners();
  }

  Future<void> clearSession() async {
    await cancelTurn();
    await _core?.clearSession();
    transcript = '';
    response = '';
    notifyListeners();
  }

  UiPhase _mapPhase(PipelinePhase value) => switch (value) {
        PipelinePhase.idle => UiPhase.idle,
        PipelinePhase.recording => UiPhase.recording,
        PipelinePhase.transcribing => UiPhase.transcribing,
        PipelinePhase.generating => UiPhase.generating,
        PipelinePhase.speaking => UiPhase.speaking,
        PipelinePhase.error => UiPhase.error,
      };

  void _setError(String message) {
    phase = UiPhase.error;
    error = message;
    notifyListeners();
  }

  @override
  void dispose() {
    _recordingDeadline?.cancel();
    unawaited(_captureSubscription?.cancel());
    unawaited(_recorder.dispose());
    unawaited(_player.dispose());
    super.dispose();
  }
}
