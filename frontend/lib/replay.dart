import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

class ReplayWidget extends StatefulWidget {
  final Map<String, List<Offset>> positionMap;
  final Size frameSize;
  final double sampleRate;
  final double duration;

  const ReplayWidget({
    super.key,
    required this.positionMap,
    required this.duration,
    required this.frameSize,
    required this.sampleRate,
  });

  ReplayWidget.fromResults(Map<String, dynamic> results, {Key? key})
    : this(
        key: key,
        positionMap: _buildPositionMap(results),
        duration: (results['duration'] as num).toDouble(),
        frameSize: Size((results['frame_width'] as num).toDouble(), (results['frame_height'] as num).toDouble()),
        sampleRate: (results['sample_rate'] as num).toDouble(),
      );

  static Map<String, List<Offset>> _buildPositionMap(
    Map<String, dynamic> results,
  ) {
    return (results['trajectories'] as Map<String, dynamic>).map(
      (team, traj) => MapEntry(team, [
        for (final pos in traj)
          Offset(
            (pos[0] as num).toDouble(),
            (pos[1] as num).toDouble(),
          ),
      ]),
    );
  }

  @override
  State<StatefulWidget> createState() => _ReplayWidgetState();
}

class _ReplayWidgetState extends State<ReplayWidget> with SingleTickerProviderStateMixin {
  double _currentTime = 0;
  double _startingTime = 0;
  bool _isPlaying = false;

  Ticker? _ticker;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick);
  }

  void _togglePlay() {
    if (_isPlaying) {
      _ticker?.stop();
    } else {
      _startingTime = _currentTime;
      _ticker?.start();
    }
    setState(() => _isPlaying = !_isPlaying);
  }

  void _onTick(Duration elapsed) {
    setState(() {
      _currentTime = (_startingTime + elapsed.inMilliseconds / 1000).clamp(0.0, widget.duration);
      if (_currentTime >= widget.duration) {
        _isPlaying = false;
        _ticker?.stop();
      }
    });
  }

  @override
  void dispose() {
    _ticker?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: CustomPaint(
            painter: _ReplayPainter(
              positions: widget.positionMap,
              idx: _currentTime * widget.sampleRate,
              frameSize: widget.frameSize,
              backgroundColor: Theme.of(context).colorScheme.surfaceContainer
            ),
            child: const SizedBox.expand(),
          ),
        ),
        Slider(
          value: _currentTime.toDouble(),
          min: 0,
          max: widget.duration,
          onChanged: (v) => setState(() {
            _currentTime = v.toDouble();
            if (_isPlaying) {
              _isPlaying = false;
              _ticker?.stop();
            }
          }),
        ),
        IconButton(icon: Icon(_isPlaying ? Icons.pause : Icons.play_arrow), onPressed: _togglePlay)
      ],
    );
  }
}

class _ReplayPainter extends CustomPainter {
  final Map<String, List<Offset>> positions;
  final double idx;
  final Size frameSize;
  final Color? backgroundColor;

  const _ReplayPainter({required this.positions, required this.idx, required this.frameSize, this.backgroundColor});

  @override
  void paint(Canvas canvas, Size size) {
    final scale = math.min(
      size.width / frameSize.width,
      size.height / frameSize.height
    );

    final dx = (size.width - frameSize.width * scale) / 2;
    final dy = (size.height - frameSize.height * scale) / 2;

    if (backgroundColor != null) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(
            center: Offset(size.width / 2, size.height / 2),
            width: frameSize.width * scale,
            height: frameSize.height * scale,
          ),
          Radius.circular(16),
        ),
        Paint()
          ..color = backgroundColor!
          ..style = PaintingStyle.fill,
      );
    }

    for (final (i, entry) in positions.entries.indexed) {
      final color = i < 3 ? Colors.blue : Colors.red;
      final paint = Paint()..color = color;

      final lo = idx.floor().clamp(0, entry.value.length - 1);
      final hi = idx.ceil().clamp(0, entry.value.length - 1);
      final t = idx % 1.0;

      final pos = entry.value[lo] * (1 - t) + entry.value[hi] * t;

      final scaled = Offset(pos.dx * scale + dx, pos.dy * scale + dy);

      canvas.drawCircle(scaled, 8, paint);

      final textStyle = TextStyle(
        color: color,
        fontSize: 20
      );
      final textSpan = TextSpan(
        text: entry.key,
        style: textStyle,
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr
      );
      textPainter.layout(
        minWidth: 0,
        maxWidth: size.width,
      );
      final textPos = Offset(scaled.dx - textPainter.width / 2, scaled.dy - textPainter.height / 2 - 20);
      textPainter.paint(canvas, textPos);
    }
  }

  @override
  bool shouldRepaint(_ReplayPainter old) =>
      old.idx != idx || old.positions != positions;
}
