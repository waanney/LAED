import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'src/presentation/speech_controller.dart';
import 'src/presentation/speech_page.dart';
import 'src/rust/frb_generated.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  final controller = SpeechController();
  await controller.initialize();
  runApp(
    ChangeNotifierProvider.value(
      value: controller,
      child: const LaedApp(),
    ),
  );
}

class LaedApp extends StatelessWidget {
  const LaedApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LAED',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3157D5),
          surface: const Color(0xFFF7F7FA),
        ),
        scaffoldBackgroundColor: const Color(0xFFF7F7FA),
        useMaterial3: true,
      ),
      home: const SpeechPage(),
    );
  }
}

