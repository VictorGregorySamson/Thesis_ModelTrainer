package com.example.goodsign


import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import org.json.JSONArray
import org.json.JSONException
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Interpreter.Options
import org.tensorflow.lite.flex.FlexDelegate
import java.io.ByteArrayOutputStream


class HolisticLandmarkerHelper(
    var minPoseDetectionConfidence: Float = DEFAULT_POSE_DETECTION_CONFIDENCE,
    var minPosePresenceConfidence: Float = DEFAULT_POSE_PRESENCE_CONFIDENCE,
    var minHandLandmarksConfidence: Float = DEFAULT_HAND_LANDMARKS_CONFIDENCE,
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    val context: Context,
    // this listener is only used when running in RunningMode.LIVE_STREAM
    val holisticLandmarkerHelperListener: LandmarkerListener? = null
)

{
    val landmarkBuffer = mutableListOf<FloatArray>()

    // For this example this needs to be a var so it can be reset on changes.
    // If the Holistic Landmarker will not change, a lazy val would be preferable.
    private var holisticLandmarker: HolisticLandmarker? = null

    init {
        setupHolisticLandmarker()
    }

    // For this example this needs to be a var so it can be reset on changes.
    // If the Hand Landmarker will not change, a lazy val would be preferable.
    private var handLandmarker: HandLandmarker? = null
    private var tflite: Interpreter? = null
    private var modelPredictor: ModelPredictor? = null


    init {
        setupHolisticLandmarker()
        loadModel()
        modelPredictor = ModelPredictor(context, "fsl_model.tflite", object : ModelPredictor.PredictionListener {
            override fun onPrediction(action: String) {
                //Log.d(TAG, "Prediction: $action")

                // Update the TextView with the predicted action
                val mainHandler = Handler(Looper.getMainLooper())
                mainHandler.post {
                    val activity = context as PredictActivity
                    val predictionTextView: TextView = activity.findViewById(R.id.predictionText)
                    predictionTextView.text = action
                }

            }
        })
    }

    fun clearHolisticLandmarker() {
        holisticLandmarker?.close()
        holisticLandmarker = null
    }

    // Return running status of HolisticLandmarkerHelper
    fun isClose(): Boolean {
        return holisticLandmarker == null
    }

    // Initialize the Holistic landmarker using current settings on the
    // thread that is using it. CPU can be used with Landmarker
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the
    // Landmarker
    fun setupHolisticLandmarker() {
        // Set general holistic landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        baseOptionBuilder.setModelAssetPath(MP_HOLISTIC_LANDMARKER_TASK)

        // Check if runningMode is consistent with holisticLandmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (holisticLandmarkerHelperListener == null) {
                    throw IllegalStateException(
                        "holisticLandmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Holistic Landmarker.
            val optionsBuilder =
                HolisticLandmarker.HolisticLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinPoseDetectionConfidence(minPoseDetectionConfidence)
                    .setMinPosePresenceConfidence(minPosePresenceConfidence)
                    .setMinHandLandmarksConfidence(minHandLandmarksConfidence)
                    .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::returnLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            holisticLandmarker =
                HolisticLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            holisticLandmarkerHelperListener?.onError(
                "Holistic Landmarker failed to initialize. See error logs for " +
                        "details"
            )
            Log.e(
                TAG, "MediaPipe failed to load the task with error: " + e
                    .message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            holisticLandmarkerHelperListener?.onError(
                "Holistic Landmarker failed to initialize. See error logs for " +
                        "details", GPU_ERROR
            )
            Log.e(
                TAG,
                "Image classifier failed to load model with error: " + e.message
            )
        }
    }



    fun yuvToRgb(imageProxy: ImageProxy): Bitmap? {
        // Get the Y, U, and V planes from the ImageProxy
        val yBuffer = imageProxy.planes[0].buffer // Y
        val uBuffer = imageProxy.planes[1].buffer // U
        val vBuffer = imageProxy.planes[2].buffer // V

        // Get the size of the Y, U, and V planes
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // Create a NV21 byte array
        val nv21 = ByteArray(ySize + uSize + vSize)

        // Copy Y, U, and V planes to the nv21 byte array
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize) // V comes after Y
        uBuffer.get(nv21, ySize + vSize, uSize) // U comes after V

        // Create a YuvImage and convert it to a Bitmap
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    // Load the tflite prediction model
    private fun loadModel() {
        try {
            Log.d(TAG, "Loading model file")
            val modelBuffer = FileUtil.loadMappedFile(context, "fsl_model.tflite")
            Log.d(TAG, "Model file loaded successfully")

            val options = Options()
            val flexDelegate = FlexDelegate()
            options.addDelegate(flexDelegate)
            Log.d(TAG, "FlexDelegate added to options")

            tflite = Interpreter(modelBuffer, options)
            Log.d(TAG, "Interpreter initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            tflite = null
        }
    }

    // Convert the ImageProxy to MP Image and feed it to HolisticlandmakerHelper.
    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call detectLiveStream while not using RunningMode.LIVE_STREAM"
            )
        }

        val frameTime = SystemClock.uptimeMillis()

        // Convert ImageProxy to Bitmap using yuvToRgb function
        val bitmapBuffer = yuvToRgb(imageProxy)
        if (bitmapBuffer == null) {
            Log.e(TAG, "Failed to convert YUV to RGB")
            imageProxy.close()
            return
        }

        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            if (isFrontCamera) {
                postScale(1f, 1f, bitmapBuffer.width / 2f, bitmapBuffer.height / 2f)
                // changing sx: -lf to sx: lf fixes the issue with right and left hand landmarks being reversed, but makes the drawing of landmarks reversed
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)

        val mpImage = BitmapImageBuilder(rotatedBitmap).build()

        detectAsync(mpImage, frameTime)
        imageProxy.close()
    }


    // Run holistic holistic landmark using MediaPipe Holistic Landmarker API
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        holisticLandmarker?.detectAsync(mpImage, frameTime)
        // As we're using running mode LIVE_STREAM, the landmark result will
        // be returned in returnLivestreamResult function
    }

    // Accepts the URI for a video file loaded from the user's gallery and attempts to run
    // holistic landmarker inference on the video. This process will evaluate every
    // frame in the video and attach the results to a bundle that will be
    // returned.
    fun detectVideoFile(
        videoUri: Uri,
        inferenceIntervalMs: Long
    ): ResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException(
                "Attempting to call detectVideoFile" +
                        " while not using RunningMode.VIDEO"
            )
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        val startTime = SystemClock.uptimeMillis()

        var didErrorOccurred = false

        // Load frames from the video and run the holistic landmarker.
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong()

        // Note: We need to read width/height from frame instead of getting the width/height
        // of the video directly because MediaRetriever returns frames that are smaller than the
        // actual dimension of the video file.
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height

        // If the video is invalid, returns a null detection result
        if ((videoLengthMs == null) || (width == null) || (height == null)) return null

        // Next, we'll get one frame every frameInterval ms, then run detection on these frames.
        val resultList = mutableListOf<HolisticLandmarkerResult>()
        val numberOfFrameToRead = videoLengthMs.div(inferenceIntervalMs)

        for (i in 0..numberOfFrameToRead) {
            val timestampMs = i * inferenceIntervalMs // ms

            retriever
                .getFrameAtTime(
                    timestampMs * 1000, // convert from ms to micro-s
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                ?.let { frame ->
                    // Convert the video frame to ARGB_8888 which is required by the MediaPipe
                    val argb8888Frame =
                        if (frame.config == Bitmap.Config.ARGB_8888) frame
                        else frame.copy(Bitmap.Config.ARGB_8888, false)

                    // Convert the input Bitmap object to an MPImage object to run inference
                    val mpImage = BitmapImageBuilder(argb8888Frame).build()

                    // Run holistic landmarker using MediaPipe Holistic Landmarker API
                    holisticLandmarker?.detectForVideo(mpImage, timestampMs)
                        ?.let { detectionResult ->
                            resultList.add(detectionResult)
                        } ?: run{
                        didErrorOccurred = true
                        holisticLandmarkerHelperListener?.onError(
                            "ResultBundle could not be returned" +
                                    " in detectVideoFile"
                        )
                    }
                }
                ?: run {
                    didErrorOccurred = true
                    holisticLandmarkerHelperListener?.onError(
                        "Frame at specified time could not be" +
                                " retrieved when detecting in video."
                    )
                }
        }

        retriever.release()

        val inferenceTimePerFrameMs =
            (SystemClock.uptimeMillis() - startTime).div(numberOfFrameToRead)

        return if (didErrorOccurred) {
            null
        } else {
            ResultBundle(resultList, inferenceTimePerFrameMs, height, width)
        }
    }

    // Accepted a Bitmap and runs holistic landmarker inference on it to return
    // results back to the caller
    fun detectImage(image: Bitmap): ResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException(
                "Attempting to call detectImage" +
                        " while not using RunningMode.IMAGE"
            )
        }


        // Inference time is the difference between the system time at the
        // start and finish of the process
        val startTime = SystemClock.uptimeMillis()

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(image).build()

        // Run holistic landmarker using MediaPipe Holistic Landmarker API
        holisticLandmarker?.detect(mpImage)?.also { landmarkResult ->
            val inferenceTimeMs = SystemClock.uptimeMillis() - startTime
            return ResultBundle(
                listOf(landmarkResult),
                inferenceTimeMs,
                image.height,
                image.width
            )
        }

        // If holisticLandmarker?.detect() returns null, this is likely an error. Returning null
        // to indicate this.
        holisticLandmarkerHelperListener?.onError(
            "Holistic Landmarker failed to detect."
        )
        return null
    }

    // Return the landmark result to this HolisticLandmarkerHelper's caller
    private fun returnLivestreamResult(
        result: HolisticLandmarkerResult,
        input: MPImage,
    ) {

        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTime = finishTimeMs - result.timestampMs()

        // Convert pose landmarks to a flattened array
        val poseLandmarks = result.poseLandmarks().let { landmarks ->
            if (landmarks.isNotEmpty()) {
                FloatArray(landmarks.size * 2) { index ->
                    if (index % 2 == 0) landmarks[index / 2].x() else landmarks[index / 2].y()
                }
            } else {
                FloatArray(33 * 2) // Assuming 33 pose landmarks
            }
        }

// Convert left hand landmarks to a flattened array
        val leftHandLandmarks = result.leftHandLandmarks().let { landmarks ->
            if (landmarks.isNotEmpty()) {
                FloatArray(landmarks.size * 2) { index ->
                    if (index % 2 == 0) landmarks[index / 2].x() else landmarks[index / 2].y()
                }
            } else {
                FloatArray(21 * 2) // Assuming 21 left hand landmarks
            }
        }

// Convert right hand landmarks to a flattened array
        val rightHandLandmarks = result.rightHandLandmarks().let { landmarks ->
            if (landmarks.isNotEmpty()) {
                FloatArray(landmarks.size * 2) { index ->
                    if (index % 2 == 0) landmarks[index / 2].x() else landmarks[index / 2].y()
                }
            } else {
                FloatArray(21 * 2) // Assuming 21 right hand landmarks
            }
        }

        // Assuming poseLandmarks, leftHandLandmarks, and rightHandLandmarks are already defined as FloatArray

// Concatenate all three arrays
        val concatenatedLandmarks = poseLandmarks + leftHandLandmarks + rightHandLandmarks

        landmarkBuffer.add(concatenatedLandmarks)

        

        if (landmarkBuffer.size == 120) {

                // Create a variable to hold all landmarks as a string
                val allLandmarksString = StringBuilder("[")

                for (landmarks in landmarkBuffer) {
                    if (landmarks != null && landmarks.isNotEmpty()) {
                        allLandmarksString.append(landmarks.joinToString(prefix = "[", postfix = "]")).append(",")
                    }
                }

// Remove the last comma if there were valid landmarks added
                if (allLandmarksString.length > 1) {
                    allLandmarksString.setLength(allLandmarksString.length - 1)
                }
                allLandmarksString.append("]") // Close the JSON array

                // Convert StringBuilder to String
                var finalString = allLandmarksString.toString()

                try {
                    // Wrap the string if necessary
                    if (!finalString.startsWith("[") || !finalString.endsWith("]")) {
                        finalString = "[$finalString]"
                    }

                    val jsonArray= JSONArray(finalString)
                    Log.d(TAG, jsonArray.toString())
                    modelPredictor?.predict(jsonArray.toString())
                } catch (e: JSONException) {
                    Log.e(TAG, "Error parsing landmarks JSON", e)
                }

                if (!finalString.startsWith("[") || !finalString.endsWith("]")) {
                    finalString = "[$finalString]"
                }


                landmarkBuffer.clear() // Clear buffer after sending data


        }




        holisticLandmarkerHelperListener?.onResults(
            ResultBundle(
                listOf(result),
                inferenceTime,
                input.height,
                input.width,
            )
        )
    }

    // Return errors thrown during detection to this HolisticLandmarkerHelper's
    // caller
    private fun returnLivestreamError(error: RuntimeException) {
        holisticLandmarkerHelperListener?.onError(
            error.message ?: "An unknown error has occurred"
        )
    }

    companion object {
        const val TAG = "HolisticLandmarkerHelper"
        private const val MP_HOLISTIC_LANDMARKER_TASK = "holistic_landmarker.task"

        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_POSE_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_POSE_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_HAND_LANDMARKS_CONFIDENCE = 0.5F
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
    }

    data class ResultBundle(
        val results: List<HolisticLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )

    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}
