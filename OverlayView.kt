package com.google.mediapipe.examples.handlandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: HolisticLandmarkerResult? = null
    private var linePaint = Paint()
    private var pointPaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    init {
        initPaints()
    }

    fun clear() {
        results = null
        linePaint.reset()
        pointPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color = ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.let { holisticResult ->
            // Draw left hand landmarks if available
            holisticResult.leftHandLandmarks()?.let {
                if (it.isNotEmpty()) drawLandmarks(canvas, it)
            }

            // Draw right hand landmarks if available
            holisticResult.rightHandLandmarks()?.let {
                if (it.isNotEmpty()) drawLandmarks(canvas, it)
            }

            // Draw pose landmarks if available
            holisticResult.poseLandmarks()?.let {
                if (it.isNotEmpty()) drawLandmarks(canvas, it)
            }
        }
    }

    private fun drawLandmarks(canvas: Canvas, landmarks: List<NormalizedLandmark>) {
        // Check that landmarks list has enough points to avoid out-of-bounds errors
        if (landmarks.isEmpty()) return

        // Draw points
        for (landmark in landmarks) {
            canvas.drawPoint(
                landmark.x() * imageWidth * scaleFactor,
                landmark.y() * imageHeight * scaleFactor,
                pointPaint
            )
        }

        // Define and draw connections (update with actual valid indices if needed)
        val connections = listOf(
            Pair(0, 1), Pair(1, 2), Pair(2, 3), // Example hand/pose connections
            Pair(4, 5), Pair(5, 6), Pair(6, 7)
        )

        for ((startIdx, endIdx) in connections) {
            if (startIdx < landmarks.size && endIdx < landmarks.size) {
                val start = landmarks[startIdx]
                val end = landmarks[endIdx]
                canvas.drawLine(
                    start.x() * imageWidth * scaleFactor,
                    start.y() * imageHeight * scaleFactor,
                    end.x() * imageWidth * scaleFactor,
                    end.y() * imageHeight * scaleFactor,
                    linePaint
                )
            }
        }
    }


    fun setResults(
        holisticLandmarkerResults: HolisticLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = holisticLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 8F
    }
}
