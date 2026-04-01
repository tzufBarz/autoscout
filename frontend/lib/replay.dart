import 'dart:math' as math;

import 'package:flutter/material.dart';

class ReplayWidget extends StatefulWidget {
  final Map<String, Map<int, Offset>> positionMap;
  final Size frameSize;
  final int totalFrames;

  const ReplayWidget({
    super.key,
    required this.positionMap,
    required this.totalFrames,
    required this.frameSize,
  });

  ReplayWidget.fromResults(Map<String, dynamic> results, {Key? key})
    : this(
        key: key,
        positionMap: _buildPositionMap(results),
        totalFrames: results['frame_count'],
        frameSize: Size((results['frame_width'] as num).toDouble(), (results['frame_height'] as num).toDouble()),
      );

  static Map<String, Map<int, Offset>> _buildPositionMap(
    Map<String, dynamic> results,
  ) {
    return (results['trajectories'] as Map<String, dynamic>).map(
      (team, traj) => MapEntry(team, {
        for (final pos in traj)
          pos['frame']: Offset(
            (pos['x'] as num).toDouble(),
            (pos['y'] as num).toDouble(),
          ),
      }),
    );
  }

  @override
  State<StatefulWidget> createState() => _ReplayWidgetState();
}

class _ReplayWidgetState extends State<ReplayWidget> {
  int _currentFrame = 0;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: CustomPaint(
            painter: _ReplayPainter(
              positions: widget.positionMap,
              frame: _currentFrame,
              frameSize: widget.frameSize,
              backgroundColor: Theme.of(context).colorScheme.surfaceContainer
            ),
            child: const SizedBox.expand(),
          ),
        ),
        Slider(
          value: _currentFrame.toDouble(),
          min: 0,
          max: widget.totalFrames.toDouble(),
          onChanged: (v) => setState(() => _currentFrame = v.toInt()),
        ),
      ],
    );
  }
}

class _ReplayPainter extends CustomPainter {
  final Map<String, Map<int, Offset>> positions;
  final int frame;
  final Size frameSize;
  final Color? backgroundColor;

  const _ReplayPainter({required this.positions, required this.frame, required this.frameSize, this.backgroundColor});

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

      final pos = entry.value[frame];
      if (pos == null) continue;

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
      old.frame != frame || old.positions != positions;
}
