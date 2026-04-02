import 'dart:async';

import 'package:autoscout_app/replay.dart';
import 'package:dio/dio.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData.light(),
      darkTheme: ThemeData.dark(),
      themeMode: ThemeMode.system,
      home: const UploadPage(),
    );
  }
}

class UploadPage extends StatefulWidget {
  const UploadPage({super.key});

  @override
  State<UploadPage> createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  final _matchController = TextEditingController();
  PlatformFile? pickedFile;
  
  bool isUploading = false;

  Future<void> pickVideo() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
    );

    if (result != null) {
      setState(() {
        pickedFile = result.files.single;
      });
    }
  }

  void uploadVideo() async {
    setState(() => isUploading = true);
    try {
      final dio = Dio();
      final formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(
          pickedFile!.path!,
          filename: pickedFile!.name,
        ),
        'match': int.parse(_matchController.text),
      });

      final response = await dio.post(
        'http://192.168.1.160:8000/upload',
        data: formData,
      );

      final jobId = response.data['job_id'];
      if (jobId != null && mounted) {
        Navigator.of(context).push(MaterialPageRoute(builder: (_) => JobPage(jobId: jobId)));
      }
    } on DioException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: ${e.type.toString()}')));
      }
    } finally {
      if (mounted) setState(() => isUploading = false);
    }
  }

  @override
  void dispose() {
    _matchController.dispose();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    _matchController.addListener(() => setState(() {}));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('AutoScout: upload'),
        centerTitle: true,
      ),
      body: Center(
        child: SizedBox(
          width: 400,
          child: Card(
            child: Padding(
              padding: EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  TextField(
                    keyboardType: TextInputType.number,
                    inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                    decoration: InputDecoration(labelText: 'Match Number'),
                    controller: _matchController,
                  ),
                  SizedBox(height: 16),
                  OutlinedButton.icon(
                    onPressed: pickVideo,
                    icon: Icon(Icons.video_file),
                    label: Text(pickedFile == null ? 'Select Video' : pickedFile!.name),
                  ),
                  SizedBox(height: 16),
                  FilledButton(
                    onPressed: pickedFile == null || _matchController.text.isEmpty || isUploading ? null : uploadVideo,
                    child: Text('Upload'),
                  ),
                ],
              )
            ),
          ),
        ),
      ),
    );
  }
}

class JobPage extends StatefulWidget {
  final String jobId;
  const JobPage({super.key, required this.jobId});

  @override
  State<StatefulWidget> createState() => _JobPageState();
}

class _JobPageState extends State<JobPage> with SingleTickerProviderStateMixin {
  Timer? _timer;
  Ticker? _ticker;
  int _realFrame = 0;
  double _estimatedFrame = 0;
  double _displayFrame = 0;
  double? _itPerS;
  double _eta = double.infinity;
  int? _totalFrames;
  String _status = 'processing';
  
  Duration _lastTick = Duration.zero;

  late final Map<String, dynamic> _results;

  String get _etaLabel {
    if (_eta == double.infinity) return '—';
    final s = _eta.round();
    if (s >= 60) return '${s ~/ 60}m ${s % 60}s';
    return '${s}s';
  }

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(Duration(seconds: 2), (_) => _poll());
    _ticker = createTicker(_onTick)..start();
  }

  @override
  void dispose() {
    _timer?.cancel();
    _ticker?.dispose();
    super.dispose();
  }

  void _onTick(Duration elapsed) {
    if (_itPerS == null || _totalFrames == null) return;
    final dt = (elapsed - _lastTick).inMilliseconds / 1000.0;
    _lastTick = elapsed;
    setState(() {
      _estimatedFrame = (_estimatedFrame + _itPerS! * dt)
        .clamp(0, _totalFrames!.toDouble());
      _displayFrame = ((_displayFrame + _itPerS! * dt) * 0.8 + _estimatedFrame * 0.2)
        .clamp(0, _totalFrames!.toDouble());
    });
  }

  Future<void> _poll() async {
    final response = await Dio().get('http://192.168.1.160:8000/status/${widget.jobId}');
    final data = response.data;
    setState(() {
      _status = data['status'];
      if (_status == 'processing') {
        _realFrame = data['progress']?['frame']?.toInt() ?? _realFrame;
        _estimatedFrame = _realFrame.toDouble();
        _itPerS = data['progress']?['it_per_s']?.toDouble() ?? 0;
        _totalFrames ??= data['progress']?['total'].toInt();
        _eta = data['progress']?['eta']?.toDouble() ?? double.infinity;
      } else if (_status == 'done') {
        _results = data['result'];
        _ticker?.stop();
      } else {
        _ticker?.stop();
      }
    });
    if (_status == 'done' || _status == 'error') {
      _timer?.cancel();
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: _status != 'processing',
      child: Scaffold(
        appBar: AppBar(
          title: Text('AutoScout: $_status'),
          centerTitle: true,
          automaticallyImplyLeading: _status != 'processing',
        ),
        body: Center(
          child: Padding(
            padding: EdgeInsets.all(16),
            child: _status == 'processing' ? Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Frame $_realFrame/$_totalFrames | ${_itPerS?.toStringAsFixed(1) ?? 0} it/s | ETA: $_etaLabel',
                  textAlign: TextAlign.center,
                ),
                LinearProgressIndicator(
                  value: _totalFrames != null
                      ? _displayFrame / _totalFrames!
                      : 0,
                  minHeight: 16,
                  borderRadius: BorderRadius.circular(8),
                ),
              ],
            ) : _status == 'done' ? ReplayWidget.fromResults(_results) : Text(_status),
          ),
        ),
      ),
    );
  }
}